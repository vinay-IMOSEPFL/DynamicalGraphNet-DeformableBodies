import os
import torch
import numpy as np
import pickle as pkl
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch

class NBodyMStickDataset():
    """
    NBodyDataset

    """
    def __init__(self, partition='train', max_samples=1e8, data_dir='',
                 n_isolated=5, n_stick=0, n_hinge=0, finite_diff=False, delta_frame=10, nsteps=1):
        self.partition = partition
        self.data_dir = data_dir
        self.n_isolated,  self.n_stick, self.n_hinge = n_isolated, n_stick, n_hinge
        self.finite_diff = finite_diff
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition

        self.suffix += '_charged{:d}_{:d}_{:d}'.format(n_isolated, n_stick, n_hinge)

        self.max_samples = int(max_samples)
        self.data, self.edges, self.cfg = self.load()
        self.nsteps = nsteps
        self.delta_frame = delta_frame

    def load(self):
        loc = np.load(self.data_dir + '/' + 'loc_' + self.suffix + '.npy')
        vel = np.load(self.data_dir + '/' + 'vel_' + self.suffix + '.npy')
        charges = np.load(self.data_dir + '/' + 'charges_' + self.suffix + '.npy')
        edges = np.load(self.data_dir + '/' + 'edges_' + self.suffix + '.npy')
        with open(self.data_dir + '/' + 'cfg_' + self.suffix + '.pkl', 'rb') as f:
            cfg = pkl.load(f)

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, charges)
        return (loc, vel, edge_attr, charges), edges, cfg

    def preprocess(self, loc, vel, edges, charges):
        # cast to torch and swap n_nodes <--> n_features dimensions
        # convert stick [M, 2]
        loc, vel = torch.Tensor(loc), torch.Tensor(vel)  # remove transpose this time
        n_nodes = loc.size(2)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0: self.max_samples]
        edges = edges[: self.max_samples, ...]  # add here for better consistency
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:  # remove self loop
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]

        # swap n_nodes <--> batch_size and add nf dimension
        edge_attr = torch.Tensor(np.array(edge_attr)).transpose(0, 1).unsqueeze(2)  # [B, N*(N-1), 1]

        return torch.Tensor(loc), torch.Tensor(vel), torch.Tensor(edge_attr), edges, torch.Tensor(charges)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges, self.cfg = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges = loc[i], vel[i], edge_attr[i], charges[i]
        
        frame_t = 30
        frame_t_p1 = frame_t + 1
        frame_t_m1 = frame_t - 1

        frame_tm1 = frame_t - self.delta_frame
        frame_tm1_p1 = frame_tm1 + 1
        frame_tm1_m1 = frame_tm1 - 1

        frame_end = frame_t + self.nsteps * self.delta_frame
        frame_end_p1 = frame_end + 1
        frame_end_m1 = frame_end - 1  
        
        # concat stick indicator to edge_attr (for egnn_vel)
        edges = self.edges
        # initialize the configurations
        cfg = self.cfg[i]
        cfg = {_: torch.from_numpy(np.array(cfg[_])) for _ in cfg}
        stick_ind = torch.zeros_like(edge_attr)[..., -1].unsqueeze(-1)
        for m in range(len(edges[0])):
            row, col = edges[0][m], edges[1][m]
            if 'Stick' in cfg:
                for stick in cfg['Stick']:
                    if (row, col) in [(stick[0], stick[1]), (stick[1], stick[0])]:
                    # if (row == stick[0] and col == stick[1]) or (row == stick[1] and col == stick[0]):
                        stick_ind[m] = 1
            if 'Hinge' in cfg:
                for hinge in cfg['Hinge']:
                    if (row, col) in [(hinge[0], hinge[1]), (hinge[1], hinge[0]), (hinge[0], hinge[2]), (hinge[2], hinge[0])]:
                        stick_ind[m] = 2
        edge_attr = torch.cat((edge_attr, stick_ind), dim=-1)

        if self.finite_diff:
            vel_t = (loc[frame_t_p1] - loc[frame_t_m1]) / 2.0
            vel_tm1 = (loc[frame_tm1_p1] - loc[frame_tm1_m1]) / 2.0
            vel_end = (loc[frame_end_p1] - loc[frame_end_m1]) / 2.0
        else:
            vel_tm1 = vel[frame_tm1]
            vel_t = vel[frame_t]
            vel_end = vel[frame_end]
        
        state_prev = loc[frame_tm1], vel_tm1
        state_0 = loc[frame_t], vel_t

        list_pos = []
        for i in range(self.nsteps+1):
            loc_next = loc[frame_t+i*10]
            list_pos.append(loc_next)

        return state_prev,state_0, edge_attr, edges, charges, loc[frame_end], vel_end, cfg,list_pos

    def __len__(self):
        return len(self.data[0])

    def get_edges(self, batch_size, n_nodes):
        edges = [torch.LongTensor(self.edges[0]), torch.LongTensor(self.edges[1])]
        if batch_size == 1:
            return edges
        elif batch_size > 1:
            rows, cols = [], []
            for i in range(batch_size):
                rows.append(edges[0] + n_nodes * i)
                cols.append(edges[1] + n_nodes * i)
            edges = [torch.cat(rows), torch.cat(cols)]
        return edges

    @staticmethod
    def get_cfg(batch_size, n_nodes, cfg):
        offset = torch.arange(batch_size) * n_nodes
        for type in cfg:
            index = cfg[type]  # [B, n_type, node_per_type]
            cfg[type] = (index + offset.unsqueeze(-1).unsqueeze(-1).expand_as(index)).reshape(-1, index.shape[-1])
            if type == 'Isolated':
                cfg[type] = cfg[type].squeeze(-1)
        return cfg

def create_graph_data(data, finite_diff=True):
    # Unpack corrected return values from __getitem__
    state_prev, state_curr, edge_attr, edges, charges, loc_end, vel_end, cfg, list_pos = data
    
    pos_prev, vel_prev = state_prev
    pos_curr, vel_curr = state_curr
    
    graph = Data(edge_index=torch.tensor(edges, dtype=torch.long), edge_attr=edge_attr)
    graph.pos = pos_curr
    graph.vel = vel_curr
    graph.prev_vel = vel_prev
    
    # Targets
    graph.end_pos = loc_end
    graph.end_vel = vel_end
    
    # Displacement and Velocity Change targets
    # Fixed variable name: used `loc_end` instead of `pos_end`
    graph.y_dx = loc_end - pos_curr 
    graph.y_dv = vel_end - vel_curr

    # Meta
    graph.charges = charges.unsqueeze(-1) if charges.dim() == 1 else charges
    graph.cfg = cfg
    graph.gt_seq = list_pos 
    graph.node_type = graph.charges

    return graph


class GraphFromRawDataset(Dataset):
    def __init__(self, raw_dataset, finite_diff=True):
        self.raw_dataset = raw_dataset
        self.finite_diff = finite_diff

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        data = self.raw_dataset[idx]
        return create_graph_data(data, self.finite_diff)


def create_dataloaders(raw_dataset, batch_size, shuffle=True, finite_diff=True):
    dataset_graph = GraphFromRawDataset(raw_dataset, finite_diff=finite_diff)
    return DataLoader(dataset_graph, batch_size=batch_size, shuffle=shuffle, collate_fn=Batch.from_data_list)


def calculate_min_max_edge(dataloader):
    all_edge_dx = []
    all_node_v_t = []
    all_node_v_tm1 = []
    all_node_dv = []
    all_node_disp = []

    for batch in dataloader:
        senders, receivers = batch.edge_index
        edge_dx = batch.pos[receivers] - batch.pos[senders]
        
        all_edge_dx.append(edge_dx)
        all_node_v_t.append(batch.vel.float())
        all_node_v_tm1.append(batch.prev_vel.float())
        all_node_dv.append((batch.y_dv).float())
        all_node_disp.append((batch.y_dx).float())

    norm_edge_dx = torch.cat(all_edge_dx).norm(dim=1)
    norm_node_v_t = torch.cat(all_node_v_t).norm(dim=1)
    norm_node_v_tm1 = torch.cat(all_node_v_tm1).norm(dim=1)
    norm_node_dv = torch.cat(all_node_dv).norm(dim=1)
    norm_node_disp = torch.cat(all_node_disp).norm(dim=1)

    stat_edge_dx = (norm_edge_dx.min(), norm_edge_dx.max())
    stat_node_v_t = (norm_node_v_t.min(), norm_node_v_t.max())
    stat_node_v_tm1 = (norm_node_v_tm1.min(), norm_node_v_tm1.max())
    stat_node_dv = (norm_node_dv.mean(), norm_node_dv.std())
    stat_node_disp = (norm_node_disp.mean(), norm_node_disp.std())

    return stat_edge_dx, stat_node_v_t, stat_node_dv, stat_node_disp

def move_train_stats_to_device(train_stats, device):
    def move(stat):
        if len(stat) == 2:
            return stat[0].to(device), stat[1].to(device)
        return stat.to(device)
    return tuple(move(s) for s in train_stats)