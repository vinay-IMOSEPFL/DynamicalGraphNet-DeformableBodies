# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Intelligent Maintenance and Operations Systems (IMOS), 2025.
# Authors: Vinay Sharma and Olga Fink
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import torch
import torch.nn as nn
from utils.utils import build_mlp_d
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class RefFrameCalc(nn.Module):
    """
    Builds a local orthonormal basis (a, b, c) per edge from relative position and
    velocity.

    The basis rotates with the system, so projecting vectors onto it yields scalars
    that do not. That is what makes the network's inputs SE(3) invariant and its
    outputs equivariant.

    All three vectors flip sign when the edge is traversed the other way, which is what
    makes the decoded impulses antisymmetric and so conserves linear momentum.
    """

    def __init__(self):
        super(RefFrameCalc, self).__init__()
        self.eps = 1e-8

    def forward(
        self,
        edge_index,
        senders_pos,
        receivers_pos,
        senders_vel,
        receivers_vel,
        senders_prev_vel,
        receivers_prev_vel,
        senders_omega,
        receivers_omega,
    ):

        # Calculate relative position (Edge vector)
        rel_pos = receivers_pos - senders_pos
        dist = rel_pos.norm(dim=1, keepdim=True).clamp(min=self.eps)
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
        self.node_encoder = build_mlp_d(
            node_in_f, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True
        )

    def forward(self, node_scalar_feat):
        return self.node_encoder(node_scalar_feat)


class InteractionEncoder(nn.Module):
    """
    Projects each edge's velocities onto its local frame and encodes the result.

    Because the frame rotates with the system, the projections are plain scalars, so the
    MLPs downstream never see a global direction.
    """

    def __init__(self, edge_in_f, latent_size, mlp_layers):
        super(InteractionEncoder, self).__init__()
        self.edge_feat_encoder = build_mlp_d(
            9, latent_size, latent_size, num_layers=mlp_layers, lay_norm=True
        )
        self.edge_encoder = build_mlp_d(
            1 + edge_in_f,
            latent_size,
            latent_size,
            num_layers=mlp_layers,
            lay_norm=True,
        )
        self.interaction_encoder = build_mlp_d(
            3 * latent_size,
            latent_size,
            latent_size,
            num_layers=mlp_layers,
            lay_norm=True,
        )

    def forward(
        self,
        edge_index,
        edge_dx_,
        edge_attr,
        vector_a,
        vector_b,
        vector_c,
        senders_v_t_,
        senders_v_tm1_,
        senders_w_t_,
        receivers_v_t_,
        receivers_v_tm1_,
        receivers_w_t_,
        node_latent,
    ):

        senders, receivers = edge_index

        # Rows are a, b, c, so multiplying gives [v.a, v.b, v.c].
        basis = torch.stack([vector_a, vector_b, vector_c], dim=1)  # (E, 3, 3)

        def project(v):
            return torch.bmm(basis, v.unsqueeze(-1)).squeeze(-1)

        s_vt_proj = project(senders_v_t_)
        s_vtm1_proj = project(senders_v_tm1_)
        s_wt_proj = project(senders_w_t_)

        # Negated so that each end of the edge sees the same features as it would if it
        # were the sender, since the basis itself flips between the two directions.
        r_vt_proj = -project(receivers_v_t_)
        r_vtm1_proj = -project(receivers_v_tm1_)
        r_wt_proj = -project(receivers_w_t_)

        senders_features = torch.cat([s_vt_proj, s_vtm1_proj, s_wt_proj], dim=1)
        receivers_features = torch.cat([r_vt_proj, r_vtm1_proj, r_wt_proj], dim=1)

        edge_dx_norm = edge_dx_.norm(dim=1, keepdim=True)
        edge_latent = self.edge_encoder(torch.cat((edge_dx_norm, edge_attr), dim=1))

        senders_latent = self.edge_feat_encoder(senders_features)
        receivers_latent = self.edge_feat_encoder(receivers_features)

        # Summed rather than concatenated, so the message is the same whichever way the
        # edge is traversed. This is what keeps the decoded coefficients direction free.
        node_sum = node_latent[senders] + node_latent[receivers]
        msg_input = torch.cat(
            (senders_latent + receivers_latent, node_sum, edge_latent), dim=1
        )

        return self.interaction_encoder(msg_input)


class InteractionDecoder(torch.nn.Module):
    """
    Turns each edge's latent into a linear impulse (force * dt) and a spin impulse
    (torque * dt).

    The impulses are rebuilt as coefficients times the edge basis. Since the basis flips
    under edge reversal and the coefficients do not, dp_ji = -dp_ij holds exactly, so
    linear momentum is conserved regardless of the weights.

    For angular momentum the network emits the total angular impulse and a weighting that
    places a reference point between the two nodes. Subtracting the orbital part carried
    by dp leaves the spin.
    """

    def __init__(self, latent_size=128, mlp_layers=2):
        super(InteractionDecoder, self).__init__()
        self.i1_decoder = build_mlp_d(
            latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False
        )
        self.i2_decoder = build_mlp_d(
            latent_size, latent_size, 3, num_layers=mlp_layers, lay_norm=False
        )
        self.node_weight_decoder = build_mlp_d(
            latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False
        )
        self.eps = 1e-8

    def forward(
        self,
        edge_index,
        senders_pos,
        receivers_pos,
        vector_a,
        vector_b,
        vector_c,
        interaction_latent,
        node_latent,
    ):
        senders, receivers = edge_index

        coeff_dp = self.i1_decoder(interaction_latent)
        coeff_dl = self.i2_decoder(interaction_latent)

        # Back to the global frame as c0*a + c1*b + c2*c.
        dpij = (
            coeff_dp[:, 0:1] * vector_a
            + coeff_dp[:, 1:2] * vector_b
            + coeff_dp[:, 2:3] * vector_c
        )
        # Total angular impulse: orbital and spin parts together.
        dlij = (
            coeff_dl[:, 0:1] * vector_a
            + coeff_dl[:, 1:2] * vector_b
            + coeff_dl[:, 2:3] * vector_c
        )

        # Reference point the angular impulse is taken about, placed between the two
        # nodes by learned weights.
        w_s = self.node_weight_decoder(node_latent[senders])
        w_r = self.node_weight_decoder(node_latent[receivers])
        denom = w_s + w_r + self.eps
        r0ij = (w_s * senders_pos + w_r * receivers_pos) / denom

        # Strip the orbital contribution of dp to leave the spin.
        dsij = dlij - torch.cross(receivers_pos - r0ij, dpij, dim=1)

        return dpij, dsij


class Node_Internal_Dv_Decoder(torch.nn.Module):
    """
    Sums the impulses arriving at each node and turns them into dv and dw.

    Mass and inertia are never supplied. Their inverses are decoded from the node latent
    through a softplus, which keeps them positive.
    """

    def __init__(self, latent_size=128, mlp_layers=2):
        super(Node_Internal_Dv_Decoder, self).__init__()
        self.m_inv_decoder = build_mlp_d(
            latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False
        )
        self.i_inv_decoder = build_mlp_d(
            latent_size, latent_size, 1, num_layers=mlp_layers, lay_norm=False
        )

    def forward(self, edge_index, node_latent, fij, tij):
        senders, receivers = edge_index
        num_nodes = node_latent.shape[0]

        m_inv = F.softplus(self.m_inv_decoder(node_latent))
        i_inv = F.softplus(self.i_inv_decoder(node_latent))

        # Every edge appears in both directions, so scattering onto receivers alone
        # already collects both sides of each interaction.
        out_fij = node_latent.new_zeros((num_nodes, 3))
        out_tij = node_latent.new_zeros((num_nodes, 3))

        out_fij.index_add_(0, receivers, fij)
        out_tij.index_add_(0, receivers, tij)

        node_dv_int = m_inv * out_fij
        node_dw_int = i_inv * out_tij

        return node_dv_int, node_dw_int


class Scaler(torch.nn.Module):
    """
    Normalises velocities and edge vectors to O(1) using training-set statistics.

    Magnitudes are scaled, directions are left alone, so the reference frame built from
    these vectors is unaffected.
    """

    def __init__(self):
        super(Scaler, self).__init__()
        self.eps = 1e-8

    def forward(
        self,
        senders_v_t,
        senders_v_tm1,
        receivers_v_t,
        receivers_v_tm1,
        senders_w_t,
        receivers_w_t,
        edge_dx,
        train_stats,
    ):
        stat_edge_dx, stat_node_v_t, _, _ = train_stats

        # Stats are constants; detach so no gradient reaches them.
        v_scale = stat_node_v_t[1].detach() + self.eps

        senders_v_t_ = senders_v_t / v_scale
        senders_v_tm1_ = senders_v_tm1 / v_scale
        receivers_v_t_ = receivers_v_t / v_scale
        receivers_v_tm1_ = receivers_v_tm1 / v_scale

        norm_edge_dx = edge_dx.norm(dim=1, keepdim=True)
        safe_norm = norm_edge_dx + self.eps

        # Divided by the same velocity scale so that angular and linear features end up
        # comparable in magnitude.
        senders_w_t_ = senders_w_t * (1 / v_scale)
        receivers_w_t_ = receivers_w_t * (1 / v_scale)

        min_stat, max_stat = stat_edge_dx
        scale_denom = (max_stat - min_stat).detach() + self.eps

        # Scale magnitude, preserve direction
        scaled_mag = (norm_edge_dx - min_stat.detach()) / scale_denom
        edge_dx_ = scaled_mag * (edge_dx / safe_norm)

        return (
            senders_v_t_,
            senders_v_tm1_,
            receivers_v_t_,
            receivers_v_tm1_,
            senders_w_t_,
            receivers_w_t_,
            edge_dx_,
        )


class Interaction_Block(torch.nn.Module):
    """
    One round of message passing: encode the edges, decode impulses, scatter to nodes.

    When a previous latent is passed in, it is added and normalised, which lets the
    sub-steps share information.
    """

    def __init__(self, edge_in_f, latent_size, mlp_layers):
        super(Interaction_Block, self).__init__()
        self.interaction_encoder = InteractionEncoder(
            edge_in_f, latent_size, mlp_layers
        )
        self.interaction_decoder = InteractionDecoder(latent_size, mlp_layers)
        self.internal_dv_decoder = Node_Internal_Dv_Decoder(latent_size, mlp_layers)
        self.layer_norm = nn.LayerNorm(latent_size)

    def forward(
        self,
        edge_index,
        senders_pos,
        receivers_pos,
        edge_dx_,
        edge_attr,
        vector_a,
        vector_b,
        vector_c,
        senders_v_t_,
        senders_v_tm1_,
        senders_w_t_,
        receivers_v_t_,
        receivers_v_tm1_,
        receivers_w_t_,
        node_latent,
        residue=None,
        latent_history=False,
    ):

        interaction_latent = self.interaction_encoder(
            edge_index,
            edge_dx_,
            edge_attr,
            vector_a,
            vector_b,
            vector_c,
            senders_v_t_,
            senders_v_tm1_,
            senders_w_t_,
            receivers_v_t_,
            receivers_v_tm1_,
            receivers_w_t_,
            node_latent,
        )

        # Residual connection
        if latent_history and residue is not None:
            interaction_latent = self.layer_norm(interaction_latent + residue)

        # Decode forces and torques
        edge_force, edge_tau = self.interaction_decoder(
            edge_index,
            senders_pos,
            receivers_pos,
            vector_a,
            vector_b,
            vector_c,
            interaction_latent,
            node_latent,
        )

        # Decode node updates
        node_dv, node_dw = self.internal_dv_decoder(
            edge_index, node_latent, edge_force, edge_tau
        )

        return node_dv, node_dw, interaction_latent


class DynamicsSolver(torch.nn.Module):
    """
    The learned integrator: message passing and state update, run num_msgs times over a
    sub-step of the full time step.

    Internal interactions conserve linear momentum exactly, so an isolated system cannot
    gain or lose it. The one term that can is the external head, which scales the node's
    own velocity and so acts along the direction of travel, like drag.

    Returns the summed dv and dx over all sub-steps, not the final state.
    """

    def __init__(
        self,
        node_in_f,
        edge_in_f,
        time_step,
        train_stats,
        num_msgs=1,
        latent_size=128,
        mlp_layers=2,
        use_ext_force=True,
    ):

        super(DynamicsSolver, self).__init__()

        # --- Core Components ---
        self.refframecalc = RefFrameCalc()
        self.scaler = Scaler()
        self.node_encoder = NodeEncoder(node_in_f, latent_size, mlp_layers)

        # --- Interaction Blocks ---
        self.interaction_init_layer = Interaction_Block(
            edge_in_f, latent_size, mlp_layers
        )
        self.interaction_proc_layer = Interaction_Block(
            edge_in_f, latent_size, mlp_layers
        )

        # --- External Force Model ---
        # Only built when the system is actually open. A closed system such as the
        # n-body case has no external field, and this is the one path that could
        # change its total momentum, so leaving it out keeps conservation exact.
        if use_ext_force:
            # we aim to predict a scalar multiplier for a known equivariant vector 
            # (e.g. velocity-dir (vt/|vt|), 
            # acceleration-dir (vt-vtm1)/|vt-vtm1|, etc.) or maybe even the axis (0,1,0)
            # to produce an external force. 
            self.ext_interaction_layers = nn.ModuleList(
                build_mlp_d(
                    latent_size + 1,
                    latent_size,
                    1,
                    num_layers=mlp_layers,
                    lay_norm=False,
                )
                for _ in range(num_msgs)
            )

        # --- Dynamics Parameters ---
        self.num_messages = num_msgs
        self.sub_tstep = time_step / num_msgs  # dt for sub-stepping
        self.train_stats = train_stats

    def forward(self, graph):
        pos = graph.pos.float()
        vel = graph.vel.float()
        prev_vel = graph.prev_vel.float()
        edge_attr = graph.edge_attr.float()
        node_type = graph.node_type.float()
        edge_index = graph.edge_index.long()
        senders, receivers = edge_index

        node_v_t = vel
        node_v_tm1 = prev_vel
        # Datasets without spin (point masses) simply do not carry this field.
        node_w_t = getattr(graph, "node_w_t", torch.zeros_like(vel))

        # Totals over all sub-steps; these are what the model returns.
        sum_node_dv = torch.zeros_like(vel)
        sum_node_dx = torch.zeros_like(vel)

        # Node properties are static, so encode them once outside the loop.
        node_latent = self.node_encoder(node_type)

        current_pos = pos
        current_edge_dx = current_pos[receivers] - current_pos[senders]

        residue = None  # carries the edge latent between sub-steps
        node_dv = torch.zeros_like(vel)
        node_dw = torch.zeros_like(vel)

        for i in range(self.num_messages):
            s_vt = node_v_t[senders]
            r_vt = node_v_t[receivers]
            s_vtm1 = node_v_tm1[senders]
            r_vtm1 = node_v_tm1[receivers]
            s_wt = node_w_t[senders]
            r_wt = node_w_t[receivers]

            s_vt_, s_vtm1_, r_vt_, r_vtm1_, s_wt_, r_wt_, edge_dx_ = self.scaler(
                s_vt,
                s_vtm1,
                r_vt,
                r_vtm1,
                s_wt,
                r_wt,
                current_edge_dx,
                self.train_stats,
            )

            # Note the angular velocities go in unscaled, unlike the linear ones.
            vec_a, vec_b, vec_c = self.refframecalc(
                edge_index,
                current_pos[senders],
                current_pos[receivers],
                s_vt_,
                r_vt_,
                s_vtm1_,
                r_vtm1_,
                s_wt,
                r_wt,
            )

            # First sub-step uses its own block; later ones share a second block and
            # carry the previous latent forward as a residual.
            layer = (
                self.interaction_init_layer if i == 0 else self.interaction_proc_layer
            )
            history_flag = i > 0

            node_dv, node_dw, residue = layer(
                edge_index,
                current_pos[senders],
                current_pos[receivers],
                edge_dx_,
                edge_attr,
                vec_a,
                vec_b,
                vec_c,
                s_vt_,
                s_vtm1_,
                s_wt_,
                r_vt_,
                r_vtm1_,
                r_wt_,
                node_latent,
                residue=residue,
                latent_history=history_flag,
            )

            # The only term that can change the system's total momentum. A scalar times
            # the node's own velocity keeps it equivariant and confines it to the
            # direction of travel.
            if hasattr(self, "ext_interaction_layers"):
                ext_input = torch.hstack(
                    (node_latent, node_v_t.norm(dim=1, keepdim=True))
                )
                node_dv_ext = self.ext_interaction_layers[i](ext_input) * node_v_t
            else:
                node_dv_ext = torch.zeros_like(node_v_t)

            # The decoded quantities are already impulses, so they add straight to the
            # velocity without a further dt.
            sum_node_dv += node_dv + node_dv_ext

            node_vf = node_v_t.clone()
            node_vf += node_dv + node_dv_ext

            node_wf = node_w_t.clone()
            node_wf += node_dw

            # Position advances on the mean of the old and new velocity (trapezoidal),
            # which is where the sub-step dt enters.
            step_disp = (node_v_t + node_vf) * (0.5 * self.sub_tstep)
            sum_node_dx += step_disp

            current_pos = current_pos + step_disp

            node_v_tm1 = node_v_t
            node_v_t = node_vf
            node_w_t = node_wf

            current_edge_dx = current_pos[receivers] - current_pos[senders]

        return sum_node_dv, sum_node_dx
