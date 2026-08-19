# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Intelligent Maintenance and Operations Systems (IMOS), 2025.
# Authors: Vinay Sharma and Olga Fink
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import os
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
    
class HumanDatasetSeq(torch.utils.data.Dataset):
    def __init__(self, partition='train', max_samples=600, data_dir='', delta_frame = 30, nsteps=1):
        self.partition = partition
        self.data_dir = data_dir
        self.nsteps = nsteps

        self.delta_frame = delta_frame

        # --- load raw data --------------------------------------
        with open(os.path.join(data_dir, 'motion.pkl'), 'rb') as f:
            edges, X = pkl.load(f)

        Ps, Vs, As = self.central_diff(X)

        # trial IDs must match exactly as competitor (GMN)
        train_case_id = [20,1,17,13,14,9,4,2,7,5,16]
        val_case_id   = [3,8,11,12,15,18]
        test_case_id  = [6,19,21,0,22,10]

        # --- load or create competitor splits (fixed for central_diff) ----------
        split_path = os.path.join(data_dir, f'split_n{self.nsteps}.pkl')
        try:
            with open(split_path, 'rb') as f:
                train_mapping, val_mapping, test_mapping = pkl.load(f)
                print("Loaded competitor split!")
        except FileNotFoundError:
            print("Generating competitor split…")

            def make_map(case_ids):
                mapping = {}
                for i in case_ids:
                    core_len = Ps[i].shape[0]                   
                    max_start_index = core_len - self.nsteps*self.delta_frame - 1
                    min_start_index = self.delta_frame
                    if max_start_index< min_start_index:
                        raise ValueError(f"Trial {i} too short for look-ahead of {self.nsteps} steps.")
                    # competitor caps at 300
                    itv = min(300, max_start_index - min_start_index + 1)              
                    pool =np.arange(min_start_index, min_start_index + itv)                     
                    mapping[i] = np.random.choice(pool, size=min(100, len(pool)), replace=False)
                return mapping

            train_mapping = make_map(train_case_id)
            val_mapping   = make_map(val_case_id)
            test_mapping  = make_map(test_case_id)

            with open(split_path, 'wb') as f:
                pkl.dump((train_mapping, val_mapping, test_mapping), f)
            print("Saved competitor split!")

        # pick the mapping
        if   partition == 'train': mapping = train_mapping
        elif partition == 'val'  : mapping = val_mapping
        elif partition == 'test' : mapping = test_mapping
        else: raise ValueError(f"Unknown partition {partition!r}")

        each_len = max_samples // len(mapping)
        in_graphs = []
        for i, pool in mapping.items():
            for j in pool[:each_len]:
                
                cur_x_t   = Ps[i][j]
                cur_v_t   = Vs[i][j]
                cur_v_tm1 = Vs[i][j-self.delta_frame]
                y_dv      = Vs[i][j + self.nsteps*self.delta_frame] - Vs[i][j]
                y_dx      = Ps[i][j + self.nsteps*self.delta_frame] - Ps[i][j]
                gt_seq = [ Ps[i][j + k*self.delta_frame] for k in range(self.nsteps+1) ]   # list of (31,3) arrays
                y_pos_end = Ps[i][j + self.nsteps*self.delta_frame]
                y_vel_end = Vs[i][j + self.nsteps*self.delta_frame]

                in_graphs.append(self.create_in_graph(
                    edges,
                    x=(cur_x_t, cur_v_t, cur_v_tm1),
                    y=(y_dv, y_dx, y_pos_end, y_vel_end),
                    gt_seq = gt_seq
                ))

        self.in_graphs = in_graphs
        print(f"[HumanDataset:{partition}] built {len(in_graphs)} samples")

    def central_diff(self, Xs, dt: float = 1.0, window_length: int = 41):
        Ps, Vs, As = [], [], []
        for x in Xs:
            v      = (x[2:] - x[:-2]) / (2*dt)
            a      = (x[2:] - 2*x[1:-1] + x[:-2]) / (dt**2)
            p      = x[1:-1]                      # align to v,a
            Ps.append(p)
            Vs.append(v)
            As.append(a)
        return Ps, Vs, As

        
    def get_foot_nodes(self, nodes):
        foot_indices = np.argsort(nodes[:,1])[:6]
        foot_pos = nodes[foot_indices]
        return foot_pos, foot_indices
    
    def reflected_nodes(self, nodes, z0=0, epsilon=1e-3):
        reflected = nodes.copy()
        reflected[:,1] = 2*z0 - nodes[:,1] - epsilon
        distances = reflected[:,1] - nodes[:,1]
        return reflected, distances
    
    def find_min(self, nodes):
        return np.min(nodes, axis=0)
    

    def create_edges(self, N, edges):
        atom_edges = torch.zeros(N, N).int()
        for edge in edges:
            atom_edges[edge[0], edge[1]] = 1
            atom_edges[edge[1], edge[0]] = 1

        atom_edges2 = atom_edges @ atom_edges
        self.atom_edge = atom_edges
        self.atom_edge2 = atom_edges2
        edge_attr = []
        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(N):
            for j in range(N):
                if i != j:
                    if atom_edges[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([1])
                        assert not atom_edges2[i][j]
                    if atom_edges2[i][j]:
                        rows.append(i)
                        cols.append(j)
                        edge_attr.append([2])
                        assert not atom_edges[i][j]

        edges = [rows, cols] 
        edge_attr = torch.Tensor(np.array(edge_attr))  # [edge, 3]
        edge_idx =torch.tensor(edges, dtype=torch.long)  # [2, M]   
        return edge_idx,edge_attr     
    
    
    def create_in_graph(self, edges,x,y,gt_seq):
        pos_t, vel_t, vel_tm1 = x
        y_dv,y_dx,y_pos_end,y_vel_end = y

        edge_idx,edge_attr = self.create_edges(pos_t.shape[0], edges)


        # Get the foot node positions and indices
        foot_nodes_positions, foot_nodes_indices = self.get_foot_nodes(pos_t)
        #foot_nodes_reflected, foot_distances = self.reflected_nodes(foot_nodes_positions,z0=0.0)
        
        pos_t = torch.tensor(pos_t)
        vel_t = torch.tensor(vel_t)
        vel_tm1 = torch.tensor(vel_tm1)

        y_dv = torch.tensor(y_dv)
        y_dx = torch.tensor(y_dx)
        y_pos_end = torch.tensor(y_pos_end)
        y_vel_end = torch.tensor(y_vel_end)
        
        # Optional
        # foot_nodes_reflected = torch.tensor(foot_nodes_reflected)
        # Set the node type of feet to one
        node_type = torch.zeros(pos_t.shape[0],1)
        # node_type[foot_nodes_indices] = 0

        in_graph = Data(edge_index=edge_idx, edge_attr=edge_attr)
        in_graph.pos = pos_t
        in_graph.vel = vel_t
        in_graph.prev_vel = vel_tm1
        in_graph.y_dv = y_dv
        in_graph.y_dx = y_dx
        in_graph.end_pos = y_pos_end
        in_graph.end_vel = y_vel_end
        in_graph.node_type = node_type
        in_graph.gt_seq = gt_seq
        
        return in_graph     
        
        
    def __len__(self):
        return len(self.in_graphs)
    
    
    def __getitem__(self, index):
        return self.in_graphs[index]

# --- Dataloader Helpers ---
class GraphFromRawDataset(Dataset):
    def __init__(self, raw_dataset):
        self.raw_dataset = raw_dataset
    def __len__(self):
        return len(self.raw_dataset)
    def __getitem__(self, idx):
        return self.raw_dataset[idx]

def create_dataloaders(dataset, M, shuffle=True):
    dataset_wrapped = GraphFromRawDataset(dataset)
    return DataLoader(dataset_wrapped, batch_size=M, shuffle=shuffle, collate_fn=Batch.from_data_list)

def create_dataloaders_from_raw(dataset, M, shuffle=True):
    return create_dataloaders(dataset, M, shuffle)

def calculate_min_max_edge(train_loader):
    """
    Calculate min/max statistics for graph properties using the ConsecutiveGraphDataset.
    
    Parameters:
    ----------
    timestep_dict : dict
        Dictionary containing graphs with keys as timesteps.
    time_step_increment : int, optional
        Time step increment to use (default: 1).
        
    Returns:
    -------
    tuple:
        Min/max statistics for various physical properties.
    """
    # Initialize lists to collect data
    all_edge_dx = []
    all_node_v_t = []
    all_node_v_tm1 = []
    all_node_dv = []
    all_node_dx = []
    
    # Process all valid timesteps
    for batch in train_loader:
        batched_graph = batch
        
        senders,receivers = batched_graph.edge_index

        edge_dx = batched_graph.pos[receivers] - batched_graph.pos[senders]
        
        # Extract positions and velocities
        node_vel_t = batched_graph.vel.float()
        node_vel_tm1 = batched_graph.prev_vel.float()
        
        # Calculate displacements and acceleration changes
        mask_body = (batched_graph.node_type!=2).squeeze()
        node_dv = (batched_graph.y_dv).float()
        node_dx = (batched_graph.y_dx).float()
        
        # Collect data
        all_edge_dx.append(edge_dx)
        all_node_v_t.append(node_vel_t)
        all_node_v_tm1.append(node_vel_tm1)
        all_node_dv.append(node_dv)
        all_node_dx.append(node_dx)
    
    # Concatenate all collected data
    all_edge_dx = torch.cat(all_edge_dx, dim=0)
    all_node_v_t = torch.cat(all_node_v_t, dim=0)
    all_node_v_tm1 = torch.cat(all_node_v_tm1, dim=0)
    all_node_dv= torch.cat(all_node_dv, dim=0)
    all_node_dx = torch.cat(all_node_dx,dim=0)
    
    # Compute norms
    norm_edge_dx = all_edge_dx.norm(dim=1)
    norm_node_v_t = all_node_v_t.norm(dim=1)
    norm_node_v_tm1 = all_node_v_tm1.norm(dim=1)
    norm_node_dv = all_node_dv.norm(dim=1)
    norm_node_dx = all_node_dx.norm(dim=1)
    
    # Compute min and max values of the norms
    min_edge_dx = norm_edge_dx.min()
    max_edge_dx = norm_edge_dx.max()

    min_node_v_t = norm_node_v_t.min()
    max_node_v_t = norm_node_v_t.max()

    min_node_v_tm1 = norm_node_v_tm1.min()
    max_node_v_tm1 = norm_node_v_tm1.max()

    mean_node_dv = norm_node_dv.mean()
    std_node_dv = norm_node_dv.std()

    mean_node_dx = norm_node_dx.mean()
    std_node_dx = norm_node_dx.std()

    # Collect statistics in tuples
    stat_edge_dx = (min_edge_dx, max_edge_dx)
    stat_node_v_t = (min_node_v_t, max_node_v_t)
    stat_node_v_tm1 = (min_node_v_tm1, max_node_v_tm1)
    stat_node_dv = (mean_node_dv, std_node_dv)
    stat_node_dx = (mean_node_dx, std_node_dx)
    
    return stat_edge_dx, stat_node_v_t, stat_node_dv, stat_node_dx

def move_train_stats_to_device(train_stats, device):
    def move(stat):
        if len(stat) == 2:
            return stat[0].to(device), stat[1].to(device)
        return stat.to(device)
    return tuple(move(s) for s in train_stats)