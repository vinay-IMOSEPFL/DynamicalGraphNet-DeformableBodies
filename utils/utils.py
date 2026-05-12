import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic behavior often slows down training significantly; 
    # enable only if exact reproducibility is strictly required.
    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_mlp_d(in_size, hidden_size, out_size, num_layers=1, lay_norm=True, use_sigmoid=False, use_softmax=False):
    if use_sigmoid and use_softmax:
        raise ValueError("Only one of use_sigmoid or use_softmax can be true.")
    
    layers = []
    # --- 1. Input Block ---
    layers.append(nn.Linear(in_size, hidden_size))
    if lay_norm:
        layers.append(nn.LayerNorm(hidden_size))
    layers.append(nn.SiLU())
    
    # --- 2. Intermediate Hidden Blocks ---
    for _ in range(num_layers-1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if lay_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.SiLU())

    # --- 3. Output Layer ---
    layers.append(nn.Linear(hidden_size, out_size))

    # --- 4. Final Activations ---
    if use_sigmoid:
        layers.append(nn.Sigmoid())
    if use_softmax:
        layers.append(nn.Softmax(dim=-1))

    return nn.Sequential(*layers)

def evaluate(test_loader, model, device):
    """Evaluates 1-step loss."""
    model.eval()
    total_loss = 0.
    total_graphs = 0

    with torch.no_grad():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            
            # Forward pass
            # Note: We don't need a loop here for single step evaluation
            node_dv, node_dx = model(test_batch)
            
            new_pos = test_batch.pos + node_dx
            
            loss = F.mse_loss(new_pos, test_batch.end_pos)
            
            batch_num = test_batch.num_graphs
            total_loss += loss.item() * batch_num
            total_graphs += batch_num

    return total_loss / total_graphs if total_graphs > 0 else 0.0

def evaluate_rollout(test_loader, model, device, nsteps=1):
    """Evaluates multi-step rollout loss."""
    model.eval()
    total_loss = 0.
    total_graphs = 0

    with torch.no_grad():
        for test_batch in test_loader:
            graph_curr = test_batch.to(device)
            # Store ground truth end position for the final step
            end_pos_gt = graph_curr.end_pos 
            
            # Rollout
            for _ in range(nsteps):
                node_dv, node_dx= model(graph_curr)
                
                new_vel = graph_curr.vel + node_dv
                new_pos = graph_curr.pos + node_dx
                
                # Update graph state for next step
                graph_curr.prev_vel = graph_curr.vel # Usually model returns cleaned prev_vel or similar
                graph_curr.vel = new_vel
                graph_curr.pos = new_pos
                # prev_pos update is implicit in logic usually, but strict physics requires:
                # graph_curr.prev_pos = graph_curr.pos.clone() (before update)

            # Calculate loss against the ground truth of the *final* step
            # Note: Ensure the dataset provides the correct end_pos for the n-th step
            loss = F.mse_loss(new_pos, end_pos_gt)
            
            batch_num = graph_curr.num_graphs
            total_loss += loss.item() * batch_num
            total_graphs += batch_num

    return total_loss / total_graphs if total_graphs > 0 else 0.0


def move_train_stats_to_device(train_stats, device):
    def move(stat):
        if len(stat) == 2:
            return stat[0].to(device), stat[1].to(device)
        return stat.to(device)
    return tuple(move(s) for s in train_stats)