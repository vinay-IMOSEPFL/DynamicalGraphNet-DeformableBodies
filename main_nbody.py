# (c) All rights reserved. ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE (EPFL),
# Switzerland, 2025.
# Authors: Vinay Sharma and Olga Fink, Laboratory of Intelligent Maintenance and
# Operations Systems (IMOS).
# Released under the Non-Commercial License Agreement in LICENSE.txt.

import os
import shutil
import argparse
import glob
import re
import torch
import torch.optim as optim
from tqdm import tqdm
import time

# --- Import Shared Logic ---
from utils.utils import set_seed, evaluate_rollout
from utils.trainer import Trainer
from model.model import DynamicsSolver 

# --- Import Case Specific Logic ---
from case_03_nbody.config import BASE_DIR,MODEL_SETTINGS, SEED, SAVED_MODELS_DIR, DEVICE_ID, RESULTS_DIR
from case_03_nbody.dataset import NBodyMStickDataset, create_dataloaders, calculate_min_max_edge
from case_03_nbody.visualization import plot_comparison, vis_result_rollout

def find_best_model(model_dir):
    pattern = os.path.join(model_dir, "Val_Loss_*.pth")
    files = glob.glob(pattern)
    if not files: return None
    def extract_loss(fn):
        # Regex matches the format: Val_Loss_0.1234.pth
        m = re.search(r'Val_Loss_([\d\.]+)', fn) 
        return float(m.group(1)) if m else float('inf')
    return min(files, key=extract_loss)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visual'])
    parser.add_argument('--test_config', type=str, default='3,2,1', 
                        help='Dataset composition in format: n_isolated,n_stick,n_hinge (default: 3,2,1)')
    args = parser.parse_args()    

    # --- Parse the Composition Config ---
    try:
        # Splits "3,2,1" into [3, 2, 1]
        n_iso_test, n_stick_test, n_hinge_test = map(int, args.test_config.split(','))
        print(f"Configuration set to: Isolated={n_iso_test}, Stick={n_stick_test}, Hinge={n_hinge_test}")
    except ValueError:
        raise ValueError("Invalid format for --test_config. Please use format 'Int,Int,Int' (e.g., '3,2,1')")

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode}")

    # --- Data Setup ---
    # nsteps=1 for training (single step prediction)
    ds_train = NBodyMStickDataset('train', 
                                  max_samples=MODEL_SETTINGS["max_training_samples"], 
                                  data_dir=MODEL_SETTINGS["data_dir"],
                                  n_isolated=MODEL_SETTINGS["n_isolated"],
                                  n_stick=MODEL_SETTINGS["n_stick"],
                                  n_hinge=MODEL_SETTINGS["n_hinge"],
                                  finite_diff=MODEL_SETTINGS["finite_diff"],
                                  delta_frame = MODEL_SETTINGS["delta_frame"],
                                  nsteps=1)
    
    ds_val = NBodyMStickDataset('val', 
                                data_dir=MODEL_SETTINGS["data_dir"],
                                n_isolated=MODEL_SETTINGS["n_isolated"],
                                n_stick=MODEL_SETTINGS["n_stick"],
                                n_hinge=MODEL_SETTINGS["n_hinge"],
                                finite_diff=MODEL_SETTINGS["finite_diff"],
                                delta_frame = MODEL_SETTINGS["delta_frame"],
                                nsteps=1)
    
    # Initial test dataset (re-initialized in 'test' mode loop)
    ds_test = NBodyMStickDataset('test', 
                                data_dir=MODEL_SETTINGS["data_dir"],
                                n_isolated=MODEL_SETTINGS["n_isolated"],
                                n_stick=MODEL_SETTINGS["n_stick"],
                                n_hinge=MODEL_SETTINGS["n_hinge"],
                                finite_diff=MODEL_SETTINGS["finite_diff"],
                                delta_frame = MODEL_SETTINGS["delta_frame"],
                                nsteps=1)

    train_loader = create_dataloaders(ds_train, MODEL_SETTINGS["batch_size"], finite_diff=MODEL_SETTINGS["finite_diff"])
    val_loader   = create_dataloaders(ds_val,   MODEL_SETTINGS["batch_size"], shuffle=False, finite_diff=MODEL_SETTINGS["finite_diff"])
    test_loader  = create_dataloaders(ds_test,  MODEL_SETTINGS["batch_size"], shuffle=False, finite_diff=MODEL_SETTINGS["finite_diff"])

    # --- Stats ---
    print("Calculating stats...")
    raw_stats = calculate_min_max_edge(train_loader)
    def to_dev(stat):
        return (stat[0].to(device), stat[1].to(device)) if isinstance(stat, tuple) else stat.to(device)
    train_stats = tuple(to_dev(s) for s in raw_stats)

    # --- Model Setup ---
    # NBody dataset uses delta_frame=10 (hardcoded in dataset.py)
    step_interval = MODEL_SETTINGS["delta_frame"]
    
    # Calculate time step for the solver
    t_step = step_interval * MODEL_SETTINGS["time_step"]

    # Feature dimensions for N-Body:
    # node_in_f = 1 (Charge)
    # edge_in_f = 2 (Edge distance + Stick/Hinge indicator)
    node_in_f = 1 
    edge_in_f = 2 

    model = DynamicsSolver(node_in_f, 
                           edge_in_f, 
                           t_step, 
                           train_stats, 
                           num_msgs=4, 
                           latent_size=MODEL_SETTINGS["nf"], 
                           mlp_layers=MODEL_SETTINGS["n_layers"],
                           use_ext_force=MODEL_SETTINGS["use_ext_force"]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=MODEL_SETTINGS["lr"])
    trainer = Trainer(model, optimizer, device, train_stats, step_interval, SAVED_MODELS_DIR)

    # --- Execution Modes ---
    if args.mode == 'train':
        print(f"Training for {MODEL_SETTINGS['epochs']} epochs...")

        # Clear checkpoint folder once per run
        if os.path.exists(trainer.model_dir):
            shutil.rmtree(trainer.model_dir)
        os.makedirs(trainer.model_dir, exist_ok=True)

        with tqdm(range(1, MODEL_SETTINGS["epochs"]+1)) as pbar:
            for epoch in pbar:
                for batch in train_loader:
                    trainer.train(batch)
                
                # Validation every 5 epochs
                if epoch % 5 == 0:
                    trainer.test(val_loader, mode='val', epoch=epoch)
                    # Optional: uncomment if you want frequent test set checks during training
                    # trainer.test(test_loader, mode='test')
                
                pbar.set_postfix({
                    'Loss': f'{trainer.loss:.4e}',
                    'Val': f'{trainer.best_val_loss:.4f} (Ep{trainer.best_epoch})'
                })
        
        best_path = trainer.best_model_path or find_best_model(SAVED_MODELS_DIR)

    elif args.mode == 'test':
        best_path = find_best_model(SAVED_MODELS_DIR)
        print(f"Loading {best_path}...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print('\n EVALUATING...')
        test_data_dir =  os.path.join(BASE_DIR, "case_03_nbody", f'data_{n_iso_test}{n_stick_test}{n_hinge_test}')

        for nstep in [1, 2, 3, 4]:
            # Re-initialize dataset to get correct ground truth for n-step rollout
            dataset_eval = NBodyMStickDataset(
                partition='test', 
                data_dir= test_data_dir,
                n_isolated=n_iso_test,
                n_stick=n_stick_test,
                n_hinge=n_hinge_test,
                finite_diff=MODEL_SETTINGS["finite_diff"],
                delta_frame = MODEL_SETTINGS["delta_frame"],
                nsteps=nstep)
            
            dataloader_eval = create_dataloaders(dataset_eval, 200, shuffle=False, finite_diff=MODEL_SETTINGS["finite_diff"])
            
            eval_error = evaluate_rollout(dataloader_eval, trainer.model, device, nsteps=nstep)
            
            print(f'loss for rollout {nstep} steps {eval_error:.4e}')
        
        print("DONE.")

    elif args.mode == 'visual':
        set_seed(int(time.time())) # re-seed for randomly selecting a graph
        best_path = find_best_model(SAVED_MODELS_DIR)
        print(f"Loading {best_path}...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print('\n GENERATING VISUALS...')
        test_data_dir =  os.path.join(BASE_DIR, "case_03_nbody", f'data_{n_iso_test}{n_stick_test}{n_hinge_test}')
        
        # Load test set with enough steps (e.g. 4) for visualization
        dataset_eval = NBodyMStickDataset(
            partition='test', 
            max_samples=10,  # Limit samples since we just want to visualize one
            data_dir=test_data_dir,
            n_isolated=n_iso_test,
            n_stick=n_stick_test,
            n_hinge=n_hinge_test,
            finite_diff=MODEL_SETTINGS["finite_diff"],
            delta_frame = MODEL_SETTINGS["delta_frame"],
            nsteps=3)
            
        loader = create_dataloaders(dataset_eval, batch_size=1, shuffle=True, finite_diff=MODEL_SETTINGS["finite_diff"])      

        for i in range(4):
            # 3. Get a single example graph
            ex_graph = next(iter(loader)).to(device)
            print("Running rollout...")
            pred_list = vis_result_rollout(ex_graph, model, device)
            gt_list = ex_graph.gt_seq        

            # 5. Plot and save
            os.makedirs(RESULTS_DIR, exist_ok=True)
            graph_dir = os.path.join(RESULTS_DIR, f'graph{i}')
            os.makedirs(graph_dir, exist_ok=True)
            print(f"Saving plots to {graph_dir}...")
            
            for i, (pred, gt) in enumerate(zip(pred_list, gt_list)):
                # plot_comparison saves as "fig_{i}.png" in the current directory
                plot_comparison(pred.detach().cpu().numpy(), gt.detach().cpu().numpy(), ex_graph.cfg, flag=i)
                
                # Move the generated file to the results directory
                src = f'fig_{i}.png'
                dst = os.path.join(graph_dir, f'fig_{i}.png')
                if os.path.exists(src):
                    shutil.move(src, dst)
                    print(f"Saved {dst}")
        
        print("Visualization complete.")

    else:
        print("Not a valid mode.")

if __name__ == "__main__":
    main()