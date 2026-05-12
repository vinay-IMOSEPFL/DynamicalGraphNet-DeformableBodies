import MDAnalysis as mda
import numpy as np
import torch

def backbone_to_pdb_file(positions: np.ndarray,
                         psf_path: str,
                         dcd_path: str,
                         output_path: str) -> None:
    u = mda.Universe(psf_path, dcd_path)
    u.trajectory[0]

    bb = u.select_atoms("backbone")
    full = u.atoms.positions.copy()
    full[bb.indices] = positions
    u.atoms.positions = full

    # write _all_ atoms, not just the backbone
    u.atoms.write(output_path)



def evaluate_rollout_vis(vis_graph, model, device, nsteps=1):
    model.eval()
    pos_pred = []
    with torch.no_grad():
        graph_curr = vis_graph.to(device)
        # Store ground truth end position for the final step
        end_pos_gt = graph_curr.end_pos 
            
        # Rollout
        pos_pred.append(graph_curr.pos)
        for _ in range(nsteps):
            node_dv, node_dx = model(graph_curr)
            
            new_vel = graph_curr.vel + node_dv
            new_pos = graph_curr.pos + node_dx
            
            # Update graph state for next step
            graph_curr.prev_vel = graph_curr.vel # Usually model returns cleaned prev_vel or similar
            graph_curr.vel = new_vel
            graph_curr.pos = new_pos
            pos_pred.append(new_pos)
    return pos_pred


    