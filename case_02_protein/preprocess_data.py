
from tqdm import tqdm
import os

import numpy as np
import torch

from scipy.sparse import coo_matrix

import MDAnalysis
from MDAnalysis.analysis import distances
from MDAnalysisData import datasets

from sklearn.preprocessing import OneHotEncoder

# ─────────────────────────────────────────────────────────────────────────────
# 3) Main preprocessing
# ─────────────────────────────────────────────────────────────────────────────
# Settings (replace argparse)
tmp_dir     = 'mdanalysis/dataset/'
top_file    = None      
traj_file   = None     
backbone    = True
is_save     = True
use_sg      = False   # use savitzky golay filtering for stencil based finite diferencing : (redundant we use simple finite-diff)

# ─────────────────────────────────────────────────────────────────────────────
# 3.1) Load or fetch ADK
# ─────────────────────────────────────────────────────────────────────────────
if top_file is not None and traj_file is not None:
    top_path  = os.path.join(tmp_dir, top_file)
    traj_path = os.path.join(tmp_dir, traj_file)
    data = MDAnalysis.Universe(top_path, traj_path)
else:
    print("Warning: No topology or trajectory file given. Using default ADK dataset.")
    adk = datasets.fetch_adk_equilibrium(data_home=tmp_dir)
    data = MDAnalysis.Universe(adk.topology, adk.trajectory)

if backbone:
    ag = data.select_atoms('backbone')
else:
    ag = data.atoms

# ─────────────────────────────────────────────────────────────────────────────
# 3.2) Local (bond) graph & charges/type/masses
# ─────────────────────────────────────────────────────────────────────────────

charges = torch.tensor(data.atoms[ag.ix].charges, dtype=torch.float32).view(-1,1)

atom_types = np.array([atom.name for atom in ag]).reshape(-1, 1)
# 3. One-hot encode atom types
oh_encoder = OneHotEncoder(sparse_output=False)
atom_type_onehot = oh_encoder.fit_transform(atom_types)  # shape: [N, T]
atom_type_tensor = torch.tensor(atom_type_onehot, dtype=torch.float32)

node_feat = torch.hstack((atom_type_tensor,charges))

print(f'number of atoms = {charges.shape}')
bonds = np.stack([
    bond.indices
    for bond in data.bonds
    if (bond.indices[0] in ag.ix and bond.indices[1] in ag.ix)
])
map_dict = {orig: new for new, orig in enumerate(ag.ix)}
bonds = np.vectorize(map_dict.get)(bonds)
edges = [
    torch.tensor(bonds[:, 0], dtype=torch.long),
    torch.tensor(bonds[:, 1], dtype=torch.long)
]
edge_attr = torch.tensor([
    bond.length()
    for bond in data.bonds
    if (bond.indices[0] in ag.ix and bond.indices[1] in ag.ix)
])

# make bi-directional
src, dst = edges
edges     = [torch.cat([src, dst], dim=0),
             torch.cat([dst, src], dim=0)]
edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

# ─────────────────────────────────────────────────────────────────────────────
# 3.3) Build raw_pos_list (all frames)
# ─────────────────────────────────────────────────────────────────────────────
num_frames = len(data.trajectory)

raw_pos_list = []
for i in range(num_frames):
    raw_pos_list.append(data.trajectory[i].positions[ag.ix].copy())

# ─────────────────────────────────────────────────────────────────────────────
# 3.4)plain Central Diff. on raw pos.
# ─────────────────────────────────────────────────────────────────────────────
# We want velocities for frames 1..(num_frames-2), so allocate exactly (num_frames-2) slots.
loc = [None] * (num_frames - 2)
vel = [None] * (num_frames - 2)

for i in range(1, num_frames - 1):
    pos_i   = raw_pos_list[i]
    pos_ip1 = raw_pos_list[i + 1]
    pos_im1 = raw_pos_list[i - 1]

    # Store at index (i-1), so that:
    #   loc[0] corresponds to frame 1,
    #   loc[1] corresponds to frame 2, …,
    #   loc[num_frames-3] corresponds to frame (num_frames-2).
    loc[i - 1] = torch.tensor(pos_i, dtype=torch.float)
    vel[i - 1] = torch.tensor((pos_ip1 - pos_im1) * 0.5, dtype=torch.float)

# ─────────────────────────────────────────────────────────────────────────────
# 3.5) Save local graph data (we train on local data only)
# ─────────────────────────────────────────────────────────────────────────────
if backbone:
    save_path = os.path.join(tmp_dir, 'adk_backbone_processed', 'adk.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
else:
    save_path = os.path.join(tmp_dir, 'adk_processed', 'adk.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

if is_save:
    torch.save((edges, edge_attr, node_feat, len(loc)), save_path)


# ─────────────────────────────────────────────────────────────────────────────
# 3.7) Save per-frame data (loc, vel) for each core frame
# ─────────────────────────────────────────────────────────────────────────────
num_core_frames = len(loc)
if backbone:
    out_dir = os.path.join(tmp_dir, 'adk_backbone_processed')
else:
    out_dir = os.path.join(tmp_dir, 'adk_processed')

if is_save:
    for i in tqdm(range(num_core_frames), desc="Saving frames"):
        try:
            torch.save(
                (
                    loc[i],                  # [N,3]
                    vel[i],                  # [N,3]
                ),
                os.path.join(out_dir, f'adk_{i}.pkl')
            )
        except RuntimeError:
            print("Error saving frame", i)