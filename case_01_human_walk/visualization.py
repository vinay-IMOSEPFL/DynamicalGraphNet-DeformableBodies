import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import imageio
import glob

def visualize_multi_step(
    test_loader,
    results_dir,
    model: torch.nn.Module,
    device: torch.device,
    steps=(1,2,3,4),
    num_graphs=10,
    seed=42
):
    """
    For each of `num_graphs` random graphs from test_loader:
      • Plot initial_vs_gt.png once (vs GT at step=1)
      • Then do a single graph rollout, saving pred_vs_gt_step{n}.png
        for each n in `steps`, comparing to GT at that same step.
    Uses absolute coordinates (no centering).
    Assumes each Data has `gt_seq` as a list of length T+1,
    each element shape (n_nodes,3).
    """
    random.seed(seed)
    model.eval()

    # flatten loader
    all_graphs = []
    for batch in test_loader:
        all_graphs.extend(batch.to_data_list())
    if not all_graphs:
        raise RuntimeError("No graphs in loader")

    # sample indices
    chosen = random.sample(range(len(all_graphs)), min(num_graphs, len(all_graphs)))

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

        # — stack gt_seq list → np array [T+1,31,3] —
        seq_list = data.gt_seq
        seq_np = []
        for arr in seq_list:
            if torch.is_tensor(arr):
                seq_np.append(arr.cpu().numpy())
            else:
                seq_np.append(np.array(arr))
        gt_seq = np.stack(seq_np, axis=0)[:, :31, :]  # shape [T+1,31,3]

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

        # optional: compute final MSE in absolute frame
        mask_cuda = (data.node_type[:31] != 2).squeeze()
        mask_cpu  = mask_cuda.cpu().numpy()
        final_pred = graph.pos[:31][mask_cuda]
        final_gt_np = gt_seq[steps[-1]][mask_cpu]
        final_gt    = torch.from_numpy(final_gt_np).to(device)
        mse = F.mse_loss(final_pred, final_gt).item()
        print(f"Graph {idx}, final step={steps[-1]}, MSE={mse:.4e}")



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