import torch
import torch.nn as nn
from utils.utils import build_mlp_d
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class RefFrameCalc(nn.Module):
    """
    Computes antisymmetric, local orthonormal basis vectors (a, b, c) for each edge.
    
    Ensures SE(3) invariance by constructing a coordinate system based on relative 
    positions and velocities. The basis (a, b, c) allows the network to learn 
    orientation-independent physical laws.
    """
    def __init__(self):
        super(RefFrameCalc, self).__init__()
        self.eps = 1e-8

    def forward(self, edge_index, 
                senders_pos, receivers_pos, 
                senders_vel, receivers_vel, 
                senders_prev_vel, receivers_prev_vel, 
                senders_omega, receivers_omega,
                ):
        
        # Calculate relative position (Edge vector)
        rel_pos = receivers_pos - senders_pos
        dist = rel_pos.norm(dim=1, keepdim=True).clamp(min= self.eps)
        vector_a = rel_pos / dist

        # Preliminary vectors
        # 1. Relative velocity cross edge vector
        # 2. Sum of velocities
        diff_vel = receivers_vel - senders_vel
        sum_vel = senders_vel + receivers_vel

        diff_prev_vel = receivers_prev_vel - senders_prev_vel
        sum_prev_vel = senders_prev_vel + receivers_prev_vel

        diff_omega = receivers_omega - senders_omega
        sum_omega = senders_omega + receivers_omega


        # Helper for normalizing
        def normalize(tensor):
            return tensor / tensor.norm(dim=1, keepdim=True).clamp(min=self.eps)

        b_i = normalize(torch.cross(diff_vel, vector_a, dim=1))
        b_ii = normalize(sum_vel)
        b_iii = normalize(torch.cross(diff_omega, vector_a, dim=1))
        b_iv = normalize(sum_omega)
        b_v = normalize(torch.cross(diff_prev_vel, vector_a, dim=1))
        b_vi = normalize(sum_prev_vel)


        # Combine to form b
        b = b_i + b_ii + b_iii + b_iv + b_v + b_vi

        # Gram-Schmidt like orthogonalization
        # Project b onto a
        b_prl_dot = (b * vector_a).sum(dim=1, keepdim=True)
        b_prl = b_prl_dot * vector_a
        b_prp = b - b_prl

        # vector_b is perpendicular to a
        vector_b = normalize(torch.cross(b_prp, vector_a, dim=1))
        
        # vector_c is perpendicular to a and b
        vector_c = normalize(torch.cross(b_prl, vector_b, dim=1))
   
        return vector_a, vector_b, vector_c

class NodeEncoder(nn.Module):
    """
    Encodes static scalar node properties (e.g., mass, friction) into a latent vector representation.
    """
    def __init__(self, node_in_f, latent_size, mlp_layers):
        super(NodeEncoder, self).__init__()
        self.node_encoder = build_mlp_d(node_in_f, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
    
    def forward(self, node_scalar_feat):
        return self.node_encoder(node_scalar_feat)

class InteractionEncoder(nn.Module):
    """
    Projects global state vectors onto the local reference frame for SE(3)-invariant message passing.
    
    Encodes the projected relative velocities and edge attributes into a latent interaction vector.
    """

    def __init__(self, edge_in_f, latent_size,mlp_layers):
        super(InteractionEncoder, self).__init__()
        self.edge_feat_encoder = build_mlp_d(9, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
        self.edge_encoder = build_mlp_d(1+edge_in_f, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)
        self.interaction_encoder = build_mlp_d(3*latent_size, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True)

    def forward(self, edge_index, edge_dx_, edge_attr, vector_a, vector_b, vector_c,
                senders_v_t_, senders_v_tm1_, senders_w_t_, 
                receivers_v_t_, receivers_v_tm1_, receivers_w_t_,
                node_latent):
        
        senders, receivers = edge_index

        # --- Vectorized Projection ---
        # Stack basis vectors into a Rotation Matrix R = [a, b, c] of shape (E, 3, 3)
        # We want to project vectors v onto this basis: v_local = [v.a, v.b, v.c]
        
        # Basis shape: (E, 3, 3). rows are a, b, c. 
        basis = torch.stack([vector_a, vector_b, vector_c], dim=1) # (E, 3, 3)

        def project(v):
            # v: (E, 3) -> (E, 3, 1)
            # basis: (E, 3, 3)
            # result: (E, 3, 1) -> (E, 3)
            # This computes [row1.v, row2.v, row3.v] -> [v.a, v.b, v.c]
            return torch.bmm(basis, v.unsqueeze(-1)).squeeze(-1)
        
        # Project senders
        s_vt_proj   = project(senders_v_t_)
        s_vtm1_proj = project(senders_v_tm1_)
        s_wt_proj   = project(senders_w_t_)

        # Project receivers (on the antisymmetric reference frame)
        r_vt_proj   = -project(receivers_v_t_)
        r_vtm1_proj = -project(receivers_v_tm1_)
        r_wt_proj   = -project(receivers_w_t_)

        # Concatenate features (9 features per node per edge)
        senders_features = torch.cat([s_vt_proj, s_vtm1_proj, s_wt_proj], dim=1)
        receivers_features = torch.cat([r_vt_proj, r_vtm1_proj, r_wt_proj], dim=1)
        
        # Edge encodings
        edge_dx_norm = edge_dx_.norm(dim=1, keepdim=True)
        edge_latent = self.edge_encoder(torch.cat((edge_dx_norm, edge_attr), dim=1))

        # Encode features
        senders_latent = self.edge_feat_encoder(senders_features)
        receivers_latent = self.edge_feat_encoder(receivers_features)

        # Message Passing
        node_sum = node_latent[senders] + node_latent[receivers]
        msg_input = torch.cat((senders_latent + receivers_latent, node_sum, edge_latent), dim=1)
        
        return self.interaction_encoder(msg_input)

class InteractionDecoder(torch.nn.Module):
    """
    Decodes interaction latent vectors into physical impulses (forces and torques).
    
    Enforces conservation of linear and angular momentum by decomposing impulses into 
    central and spin components, ensuring action-reaction symmetry.
    """
    def __init__(self, latent_size=128,mlp_layers=2):
        '''
        Decode velocity and angular velocity impulses (equivalent to force*dt and torque*dt)
        '''
        super(InteractionDecoder, self).__init__()
        self.i1_decoder          = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.i2_decoder          = build_mlp_d(latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
        self.node_weight_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.eps = 1e-8

    def forward(self, edge_index, senders_pos, receivers_pos, vector_a, vector_b, vector_c, interaction_latent, node_latent):
        senders, receivers = edge_index
        
        # Decode coefficients 
        coeff_dp     = self.i1_decoder(interaction_latent)
        coeff_dl     = self.i2_decoder(interaction_latent)
        
        # Reconstruct change in momentum in global frame
        # Linear combination: c0*a + c1*b + c2*c
        dpij = (
            coeff_dp[:, 0:1] * vector_a + 
            coeff_dp[:, 1:2] * vector_b + 
            coeff_dp[:, 2:3] * vector_c
            ) 
        # Reconstruct chainge in totoal angular momentum in global frame (it has both sping and orbital components)
        dlij = (
            coeff_dl[:, 0:1] * vector_a + 
            coeff_dl[:, 1:2] * vector_b + 
            coeff_dl[:, 2:3] * vector_c
            )

        # Node weights for reference point for cons. of ang. momentum
        w_s =self.node_weight_decoder(node_latent[senders])
        w_r =self.node_weight_decoder(node_latent[receivers])
        
        # Weighted center r0ij
        denom = w_s + w_r + self.eps
        r0ij = (w_s * senders_pos + w_r * receivers_pos) / denom
        
        # Compute spin component
        dsij = dlij - torch.cross(receivers_pos - r0ij, dpij, dim=1)    
        
        return dpij, dsij

class Node_Internal_Dv_Decoder(torch.nn.Module):
    """
    Aggregates edge impulses and computes node state updates (dv, dw) using Newton's laws.
    
    Infers physical properties (inverse mass, inverse inertia) directly from node embeddings.
    """
    def __init__(self, latent_size=128,mlp_layers=2):
        super(Node_Internal_Dv_Decoder, self).__init__()
        self.m_inv_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)
        self.i_inv_decoder = build_mlp_d(latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False)

    def forward(self, edge_index, node_latent, fij, tij):
        senders, receivers = edge_index   
        num_nodes = node_latent.shape[0]
        
        # Decode physical properties
        m_inv = F.softplus(self.m_inv_decoder(node_latent))
        i_inv = F.softplus(self.i_inv_decoder(node_latent))
        
        # Aggregate Forces and Torques
        out_fij = node_latent.new_zeros((num_nodes, 3))
        out_tij = node_latent.new_zeros((num_nodes, 3))
        
        out_fij.index_add_(0, receivers, fij)
        out_tij.index_add_(0, receivers, tij)

        # Compute internal impulse
        node_dv_int = m_inv * out_fij
        # compute angular impulse
        node_dw_int = i_inv * out_tij

        return node_dv_int, node_dw_int

class Scaler(torch.nn.Module):
    """
    Standardizes input features based on training statistics to ensure stable gradients.
    
    Scales magnitudes of velocities and distances while preserving their direction.
    """
    def __init__(self):
        super(Scaler, self).__init__()
        self.eps = 1e-8

    def forward(self, senders_v_t, senders_v_tm1, receivers_v_t, receivers_v_tm1, senders_w_t, receivers_w_t, edge_dx, train_stats):
        stat_edge_dx, stat_node_v_t, _, _ = train_stats
        
        # Use detach on stats to ensure no gradients flow back to stats (redundant but safe)
        v_scale = stat_node_v_t[1].detach() + self.eps
        
        senders_v_t_ = senders_v_t / v_scale
        senders_v_tm1_ = senders_v_tm1 / v_scale
        receivers_v_t_ = receivers_v_t / v_scale
        receivers_v_tm1_ = receivers_v_tm1 / v_scale
        
        norm_edge_dx = edge_dx.norm(dim=1, keepdim=True)
        # Avoid division by zero
        safe_norm = norm_edge_dx + self.eps

        # Scale angular velocity.
        # Logic: v_tangential = w * r. By computing (w * r) / v_scale, we normalize 
        # the tangential velocity at the edge distance, making rotation features 
        # magnitude-compatible with linear velocity features.
        senders_w_t_ = senders_w_t * (1 / v_scale)
        receivers_w_t_ = receivers_w_t * (1 / v_scale)       
        
        min_stat, max_stat = stat_edge_dx
        scale_denom = (max_stat - min_stat).detach() + self.eps
        
        # Scale magnitude, preserve direction
        scaled_mag = (norm_edge_dx - min_stat.detach()) / scale_denom
        edge_dx_ = scaled_mag * (edge_dx / safe_norm)
        
        return senders_v_t_, senders_v_tm1_, receivers_v_t_, receivers_v_tm1_, senders_w_t_, receivers_w_t_,edge_dx_

class Interaction_Block(torch.nn.Module):
    """
    Wrapper module combining interaction encoding, force/torque decoding, and node updates.
    
    Supports residual connections for recurrent multi-step message passing.
    """
    def __init__(self, edge_in_f, latent_size,mlp_layers):
        super(Interaction_Block, self).__init__()
        self.interaction_encoder = InteractionEncoder(edge_in_f, latent_size,mlp_layers)
        self.interaction_decoder = InteractionDecoder(latent_size,mlp_layers)
        self.internal_dv_decoder = Node_Internal_Dv_Decoder(latent_size,mlp_layers)
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(self, edge_index, senders_pos, receivers_pos, edge_dx_, edge_attr, vector_a, vector_b, vector_c, 
                senders_v_t_, senders_v_tm1_, senders_w_t_, 
                receivers_v_t_, receivers_v_tm1_, receivers_w_t_,
                node_latent, residue=None, latent_history=False):
            
        interaction_latent = self.interaction_encoder(
            edge_index, edge_dx_, edge_attr,
            vector_a, vector_b, vector_c,
            senders_v_t_, senders_v_tm1_, senders_w_t_,
            receivers_v_t_, receivers_v_tm1_, receivers_w_t_,
            node_latent
        )

        # Residual connection
        if latent_history and residue is not None:
            interaction_latent = self.layer_norm(interaction_latent + residue)
        
        # Decode forces and torques
        edge_force, edge_tau = self.interaction_decoder(
            edge_index, senders_pos, receivers_pos, 
            vector_a, vector_b, vector_c, 
            interaction_latent, node_latent
        )
        
        # Decode node updates
        node_dv, node_dw = self.internal_dv_decoder(
            edge_index, node_latent, edge_force, edge_tau
        )
    
        return node_dv, node_dw, interaction_latent

class DynamicsSolver(torch.nn.Module):
    """
    Main physics simulation loop for Rigid Body Dynamics.
    
    This module iteratively evolves the system state using a Graph Neural Network (GNN)
    coupled with a symplectic integrator. It guarantees the conservation of linear 
    and angular momentum for internal interactions, while allowing for learnable 
    external forces (like drag or gravity).
    """
    def __init__(self, node_in_f, edge_in_f, 
                 time_step, 
                 train_stats, 
                 num_msgs=1, latent_size=128, mlp_layers=2):
        
        super(DynamicsSolver, self).__init__()
        
        # --- Core Components ---
        self.refframecalc = RefFrameCalc()       
        self.scaler = Scaler()                   
        self.node_encoder = NodeEncoder(node_in_f, latent_size, mlp_layers) 
        
        # --- Interaction Blocks ---
        self.interaction_init_layer = Interaction_Block(edge_in_f, latent_size, mlp_layers)
        self.interaction_proc_layer = Interaction_Block(edge_in_f, latent_size, mlp_layers)
        
        # --- External Force Model ---
        ext_interaction_layers = []
        for _ in range(num_msgs):
            ext_interaction_layers.append(
                build_mlp_d(latent_size + 1, latent_size, 3, num_layers=mlp_layers, lay_norm=False)
            )
        self.ext_interaction_layers = nn.ModuleList(ext_interaction_layers)
        
        # --- Dynamics Parameters ---
        self.num_messages = num_msgs
        self.sub_tstep = time_step / num_msgs  # dt for sub-stepping
        self.train_stats = train_stats

    def forward(self, graph):
        # 1. Unpack Graph Data
        pos = graph.pos.float()
        vel = graph.vel.float()
        prev_vel = graph.prev_vel.float()
        edge_attr = graph.edge_attr.float()
        node_type = graph.node_type.float()
        edge_index = graph.edge_index.long()
        senders, receivers = edge_index

        # 2. Initialize State Variables
        node_v_t = vel
        node_v_tm1 = prev_vel
        # If angular velocity isn't present, assume zero (point masses initially)
        node_w_t = getattr(graph, 'node_w_t', torch.zeros_like(vel))
        
        # 3. Initialize Accumulators
        # These will sum up the total changes over all message-passing steps
        sum_node_dv = torch.zeros_like(vel)
        sum_node_dx = torch.zeros_like(vel)
        
        # 4. Pre-compute Node Embeddings
        # Encodes static properties (mass, radius) once, as they don't change during steps.
        node_latent = self.node_encoder(node_type)

        # 5. Initialize Loop Variables
        current_pos = pos
        current_edge_dx = current_pos[receivers] - current_pos[senders]
        
        residue = None # For RNN-style memory between message steps
        node_dv = torch.zeros_like(vel)
        node_dw = torch.zeros_like(vel)

        # --- Message Passing & Integration Loop ---
        for i in range(self.num_messages):
            # A. Gather Current State for Edges
            s_vt = node_v_t[senders]
            r_vt = node_v_t[receivers]
            s_vtm1 = node_v_tm1[senders]
            r_vtm1 = node_v_tm1[receivers]
            s_wt = node_w_t[senders]
            r_wt = node_w_t[receivers]

            # B. Input Normalization
            # Scales features to O(1) range for stability, using training stats.
            s_vt_, s_vtm1_, r_vt_, r_vtm1_, s_wt_, r_wt_, edge_dx_ = self.scaler(
                s_vt, s_vtm1, r_vt, r_vtm1, s_wt, r_wt, current_edge_dx, self.train_stats
            )

            # C. Frame Construction
            # Builds local basis (a, b, c) to ensure SE(3) invariance (rotation independence).
            vec_a, vec_b, vec_c = self.refframecalc(
                edge_index, current_pos[senders], current_pos[receivers],
                s_vt_, r_vt_, s_vtm1_, r_vtm1_, s_wt, r_wt,
            )

            # D. Interaction Block (Internal Forces)
            # Calculates pairwise forces/torques conserving momentum.
            layer = self.interaction_init_layer if i == 0 else self.interaction_proc_layer
            history_flag = (i > 0)
            
            node_dv, node_dw, residue = layer(
                edge_index, current_pos[senders], current_pos[receivers],
                edge_dx_, edge_attr, vec_a, vec_b, vec_c,
                s_vt_, s_vtm1_, s_wt_,
                r_vt_, r_vtm1_, r_wt_, 
                node_latent, residue=residue, latent_history=history_flag
            )

            ext_input = torch.hstack((node_latent, node_v_t.norm(dim=1, keepdim=True)))
            
            node_dv_ext = self.ext_interaction_layers[i](ext_input)

            # F. Symplectic Euler Integration
            # 1. Accumulate total velocity change (Internal + External)
            sum_node_dv += node_dv + node_dv_ext
            
            # 2. Update Velocity (v_new = v_old + a * dt)
            # Note: node_dv is effectively (Force / Mass) * dt
            node_vf = node_v_t.clone()
            node_vf += node_dv + node_dv_ext
            
            # 3. Update Angular Velocity
            node_wf = node_w_t.clone()
            node_wf += node_dw

            # 4. Calculate Displacement (x_new = x_old + v_new * dt)
            # Semi-implicit Euler uses the *new* velocity for position update.
            # Using (v_old + v_new) * 0.5 is akin to Trapezoidal rule / Verlet integration.
            step_disp = (node_v_t + node_vf) * (0.5 * self.sub_tstep)
            sum_node_dx += step_disp

            # 5. Update State for Next Iteration
            current_pos = current_pos + step_disp 
            
            node_v_tm1 = node_v_t # Update the prev velocity
            node_v_t = node_vf
            node_w_t = node_wf
            
            # Update edge vectors based on new positions
            current_edge_dx = current_pos[receivers] - current_pos[senders]

        return sum_node_dv, sum_node_dx