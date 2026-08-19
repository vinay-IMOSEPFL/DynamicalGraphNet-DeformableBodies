# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland,
# Laboratory of Intelligent Maintenance and Operations Systems (IMOS), 2025.
# Authors: Vinay Sharma and Olga Fink
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import imageio
import glob

def _stack_gt(data):
    """Ground truth sequence as [T+1, 31, 3], accepting tensors or arrays."""
    seq = [a.cpu().numpy() if torch.is_tensor(a) else np.array(a) for a in data.gt_seq]
    return np.stack(seq, axis=0)[:, :31, :]


def _rollout(model, data, device, n_steps):
    """Roll the model forward n_steps autoregressively, returning the final graph."""
    graph = data.clone().to(device)
    for _ in range(n_steps):
        dv, dx = model(graph.detach())
        graph.prev_vel = graph.vel
        graph.vel = graph.vel + dv
        graph.pos = graph.pos + dx
    return graph


def _final_mse(model, data, device, n_steps):
    """Rollout error at the final step, over the nodes the loss is taken on."""
    graph = _rollout(model, data, device, n_steps)
    mask = (data.node_type[:31] != 2).squeeze()
    gt = torch.from_numpy(_stack_gt(data)[n_steps]).to(device).float()
    return F.mse_loss(graph.pos[:31][mask], gt[mask]).item()


def visualize_multi_step(
    test_loader,
    results_dir,
    model: torch.nn.Module,
    device: torch.device,
    steps=(1, 2),
    num_graphs=10,
):
    """
    Plot the `num_graphs` best-predicted test sequences: for each, an initial pose and
    then pred-vs-GT at every step in `steps`.

    Graphs are ranked by rollout error at the last step and the lowest are kept, so the
    figures show the model at its best rather than an arbitrary sample. Coordinates are
    absolute, not centred.
    """
    model.eval()

    # flatten loader
    all_graphs = []
    for batch in test_loader:
        all_graphs.extend(batch.to_data_list())
    if not all_graphs:
        raise RuntimeError("No graphs in loader")

    # rank by rollout error and keep the best
    with torch.no_grad():
        scores = [_final_mse(model, g, device, steps[-1]) for g in all_graphs]
    chosen = sorted(range(len(all_graphs)), key=lambda i: scores[i])[:num_graphs]
    print(f"Selected {len(chosen)} best of {len(all_graphs)} by {steps[-1]}-step MSE "
          f"({scores[chosen[0]]:.4e} to {scores[chosen[-1]]:.4e})")

    # skeleton edges for the 31-body joints
    skeleton31 = [
      [1,0],[2,1],[3,2],[4,3],[5,4],
      [6,0],[7,6],[8,7],[9,8],[10,9],
      [11,0],[12,11],[13,12],[14,13],[15,14],
      [16,15],[17,13],[18,17],[19,18],[20,19],
      [21,20],[22,21],[23,20],[24,13],[25,24],
      [26,25],[27,26],[28,27],[29,28],[30,27]
    ]

    for idx in chosen:
        data = all_graphs[idx].to(device)
        plot_dir = os.path.join(results_dir, f"graph_{idx}")
        os.makedirs(plot_dir, exist_ok=True)

        gt_seq = _stack_gt(data)

        # initial pose (absolute)
        init31 = data.pos[:31].cpu().numpy()

        # compute axis limits from initial + all selected GT steps
        all_pts = np.vstack([init31] + [gt_seq[k] for k in steps])
        pad = 2.0
        x_min, x_max = all_pts[:,2].min() - pad, all_pts[:,2].max() + pad
        y_min, y_max = all_pts[:,0].min() - pad, all_pts[:,0].max() + pad
        z_min, z_max = all_pts[:,1].min() - pad, all_pts[:,1].max() + pad

        # — Plot initial_vs_gt.png (vs GT at step=1) —
        gt1 = gt_seq[0]
        fig = plt.figure(figsize=(6,6))
        ax  = fig.add_subplot(111, projection='3d')
        xx, yy = np.meshgrid([x_min,x_max], [y_min,y_max])
        ax.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.2, linewidth=0)
        ax.scatter(init31[:,2], init31[:,0], init31[:,1],
                   c='red', s=30, edgecolors='k', alpha=0.5, label='Initial')
        for a,b in skeleton31:
            ax.plot([gt1[a,2], gt1[b,2]],
                    [gt1[a,0], gt1[b,0]],
                    [gt1[a,1], gt1[b,1]],
                    c='red', alpha=0.6, linestyle='-', linewidth=2)

        mid_x = (x_min + x_max)/2
        mid_y = (y_min + y_max)/2
        mid_z = (z_min + z_max)/2
        
        ax.set_xlim(mid_x-20, mid_x+20)
        ax.set_ylim(mid_y-20, mid_y+20)
        ax.set_zlim(mid_z-20, mid_z+20)
        ax.set_box_aspect((1,1,1))
        ax.set_xlabel("X",fontsize = 16); ax.set_ylabel("Y",fontsize = 16); ax.set_zlabel("Z",fontsize = 16)
        ax.set_title("Initial",fontsize = 18)
        ax.legend(loc='upper left',fontsize = 18)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'initial_vs_gt.png'))
        plt.close(fig)

        # — rollout & per-step plotting —
        graph = data.clone().to(device)
        for step in steps:
            dv, dx = model(graph.detach())
            graph.prev_vel = graph.vel
            graph.vel      = graph.vel + dv
            graph.pos      = graph.pos + dx

            pred31 = graph.pos[:31].detach().cpu().numpy()
            gt_k   = gt_seq[step]

            fig = plt.figure(figsize=(6,6))
            ax  = fig.add_subplot(111, projection='3d')
            # optional ground plane: uncomment if desired
            ax.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.2, linewidth=0)

            ax.scatter(pred31[:,2], pred31[:,0], pred31[:,1],
                       c='blue', s=30, edgecolors='k', alpha=0.5,
                       label=f'Pred (step={step})')
            ax.scatter(gt_k[:,2], gt_k[:,0], gt_k[:,1],
                       c='red',  s=30, edgecolors='k', alpha=0.5,
                       label=f'GT (step={step})')
            for a,b in skeleton31:
                ax.plot([pred31[a,2], pred31[b,2]],
                        [pred31[a,0], pred31[b,0]],
                        [pred31[a,1], pred31[b,1]],
                        c='blue', alpha=0.6, linewidth=2)
                ax.plot([gt_k[a,2], gt_k[b,2]],
                        [gt_k[a,0], gt_k[b,0]],
                        [gt_k[a,1], gt_k[b,1]],
                        c='red', alpha=0.6, linestyle='-',linewidth=2)
            step_mse = np.mean((pred31 - gt_k) ** 2)

            ax.set_xlim(mid_x-20, mid_x+20)
            ax.set_ylim(mid_y-20, mid_y+20)
            ax.set_zlim(mid_z-20, mid_z+20)
            ax.set_box_aspect((1,1,1))
            ax.set_xlabel("X",fontsize = 16); ax.set_ylabel("Y",fontsize = 16); ax.set_zlabel("Z",fontsize = 16)
            ax.set_title(f"Prediction vs GT — {step} steps; MSE={step_mse:.2e}",fontsize = 18)
            ax.legend(loc='upper left',fontsize = 18)
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'pred_vs_gt_step{step}.png'))
            plt.close(fig)

        print(f"Graph {idx}, final step={steps[-1]}, MSE={scores[idx]:.4e}")



def create_gif(save_dir):
    # Iterate through each subfolder (graph_*)
    for graph_folder in os.listdir(save_dir):
        folder_path = os.path.join(save_dir, graph_folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Collect all PNG files in sorted order
        png_files = sorted(glob.glob(os.path.join(folder_path, '*.png')))
        if not png_files:
            continue
        
        # Read each image
        images = []
        for png in png_files:
            try:
                img = imageio.imread(png)
                images.append(img)
            except Exception as e:
                print(f"Warning: could not read {png}: {e}")
        
        # Save as infinite-loop GIF
        gif_path = os.path.join(folder_path, 'rollout.gif')
        imageio.mimsave(gif_path, images, fps=2, loop=0)
        print(f"Created {gif_path} with {len(images)} frames")