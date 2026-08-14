import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MDAnalysisDataset(Dataset):
    """
    Simplified MDAnalysisDataset (cached version
    using raw loc and vel from preprocessed .pkl files.
    """

    def __init__(
        self,
        dataset_name: str,
        partition: str = 'train',
        tmp_dir: str = None,
        train_valid_test_ratio=None,
        test_rot: bool = False,
        test_trans: bool = False,
        load_cached: bool = True,
        delta_frame= 15,
        nsteps=1
        ):
        super().__init__()
        assert load_cached, "This version only supports load_cached=True"
        self.delta_frame = delta_frame
        self.delta_total = nsteps*self.delta_frame
        self.nsteps = nsteps
        self.time_step = 1
        self.dataset = dataset_name
        self.partition = partition
        self.test_rot = test_rot
        self.test_trans = test_trans

        # Adjust tmp_dir to point at cached folder
        tmp_dir = os.path.join(tmp_dir, 'adk_backbone_processed')
        self.tmp_dir = tmp_dir

        # Default split ratio if not provided
        if train_valid_test_ratio is None:
            train_valid_test_ratio = [0.6, 0.2, 0.2]
        assert sum(train_valid_test_ratio) <= 1
        self.train_valid_test_ratio = train_valid_test_ratio

        # 1) Load the single '<dataset_name>.pkl' file
        edges_list, self.edge_attr, self.node_feat, self.n_frames = torch.load(
            os.path.join(tmp_dir, f'{dataset_name}.pkl')
        )
        self.edges = torch.stack(edges_list, dim=0)

        # 2) Precompute split-indices over usable frames
        usable = self.n_frames - self.delta_total-self.delta_frame
        n_train = int(self.train_valid_test_ratio[0] * usable)
        n_valid = int(self.train_valid_test_ratio[1] * usable)
        n_test  = usable - n_train - n_valid
        self.train_start = self.delta_frame
        self.valid_start = self.delta_frame + n_train
        self.test_start  = self.delta_frame + n_train + n_valid

        self.n_train = n_train
        self.n_valid = n_valid
        self.n_test  = n_test

        print(f"[MDAnalysisDataset] partition='{self.partition}'  "
              f"usable={usable}  train={n_train}  valid={n_valid}  test={n_test}")

    def __len__(self):
        if self.partition == 'train':
            return self.n_train
        elif self.partition in ('valid', 'val'):
            return self.n_valid
        else:
            return self.n_test

    def __getitem__(self, i):
        # Determine frame_0 index for this partition
        if self.partition == 'train':
            frame_0 = self.train_start + i
        elif self.partition in ('valid', 'val'):
            frame_0 = self.valid_start + i
        else:
            frame_0 = self.test_start + i
        
        frame_tm1 = frame_0 - self.delta_frame
        frame_end = frame_0 + self.delta_total

        # Load raw loc and vel from cached files
        loc_0, vel_0 = torch.load(
            os.path.join(self.tmp_dir, f'{self.dataset}_{frame_0}.pkl')
        )
        loc_end, vel_end = torch.load(
            os.path.join(self.tmp_dir, f'{self.dataset}_{frame_end}.pkl')
        )
        loc_tm, vel_tm = torch.load(
            os.path.join(self.tmp_dir, f'{self.dataset}_{frame_tm1}.pkl')
        )

        seq = []
        for step in range(self.nsteps + 1):
            idx = frame_0 + step * self.delta_frame
            loc_seq, _,= torch.load(
            os.path.join(self.tmp_dir, f'{self.dataset}_{idx}.pkl')
        )
            seq.append(loc_seq)


        node_charges = self.node_feat[:,-1:]
        node_type = self.node_feat[:,:-1]

        edge_attr = torch.zeros_like(self.edge_attr).view(-1,1)

        # Build PyG Data object
        graph = Data(
            edge_index=self.edges,
            edge_attr=edge_attr,
        )
        graph.node_feat = node_charges
        graph.node_type = node_type
        graph.pos     = loc_0
        graph.vel     = vel_0
        graph.prev_vel  = vel_tm
        graph.y_dv    = vel_end - vel_0
        graph.y_dx    = loc_end - loc_0
        graph.end_pos   = loc_end
        graph.end_vel   = vel_end
        graph.gt_seq     = seq

        return graph   


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