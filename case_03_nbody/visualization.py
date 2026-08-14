import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import torch

def plot_comparison(predicted, ground_truth, cfg, flag=1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lw = 1.7

    diff = predicted - ground_truth
    mse = np.mean(diff**2)

    title = f'{flag} step MSE = {mse*100:.2f}e-2'

    # Nodes
    if flag==0:
        ax.scatter(*ground_truth.T, color="r", label="GT nodes", s=50)
    else:
        ax.scatter(*ground_truth.T, color="r", label="GT nodes", s=50)
        ax.scatter(*predicted.T,   color="b", label="Pred nodes", s=50)

    # Sticks
    gt_stick = pred_stick = False
    for s in cfg.get('Stick', []):
        pts_gt = ground_truth[s.cpu().numpy()]
        ax.plot(pts_gt[:,0], pts_gt[:,1], pts_gt[:,2],
                color='red', linewidth=lw,
                label="GT sticks" if not gt_stick else None)
        gt_stick = True
        if flag!=0:
            pts_pr = predicted[s.cpu().numpy()]
            ax.plot(pts_pr[:,0], pts_pr[:,1], pts_pr[:,2],
                    color='blue', linestyle='--', linewidth=lw, alpha=0.6,
                    label="Pred sticks" if not pred_stick else None)
            pred_stick = True

    # Hinges
    gt_hinge = pred_hinge = False
    for h in cfg.get('Hinge', []):
        i, j, k = h
        for pair in [(i, j), (i, k)]:
            xs_gt = np.array([ground_truth[pair[0]], ground_truth[pair[1]]])
            ax.plot(xs_gt[:,0], xs_gt[:,1], xs_gt[:,2],
                    color='orange', linestyle='-',linewidth=lw,
                    label="GT hinges" if not gt_hinge else None)
            gt_hinge = True
            if flag!=0:
                xs_pr = np.array([predicted[pair[0]], predicted[pair[1]]])
                ax.plot(xs_pr[:,0], xs_pr[:,1], xs_pr[:,2],
                        color='green', linestyle='--', linewidth=lw, alpha=0.6,
                        label="Pred hinges" if not pred_hinge else None)
                pred_hinge = True

    # View & clean
    ax.view_init(elev=15., azim=160)
    ax.set_title(title)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
        for line in axis.get_ticklines():
            line.set_visible(False)

    ax.legend(loc='upper right',
              bbox_to_anchor=(0.2, 1.0),
              borderaxespad=0.)
    # plt.tight_layout()

    plt.savefig(f'fig_{flag}.png', bbox_inches='tight')

def vis_result_rollout(graph, model, device):
    model.eval()
    with torch.no_grad():
        res = 0.
        res_counter = 0

        pred = []
        pred.append(graph.pos)
        for i in range(3):
            if i == 0:
                graph_t0 = graph.clone()
            node_dv,node_dx = model(graph_t0.detach())
            new_vel =  graph_t0.vel + node_dv
            new_pos =  graph_t0.pos + node_dx
            pred.append(new_pos)
            graph_t0.prev_vel = graph_t0.vel.clone()
            graph_t0.pos = new_pos.clone()
            graph_t0.vel = new_vel.clone()
    return pred
