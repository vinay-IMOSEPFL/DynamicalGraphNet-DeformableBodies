import os
import shutil
import argparse
import glob
import re
import torch
import torch.optim as optim
from tqdm import tqdm


# --- Import Shared Logic ---
from utils.trainer import Trainer
from model.model import DynamicsSolver
from utils.utils import set_seed, evaluate, evaluate_rollout

# --- Import Case Specific Logic ---
from case_01_human_walk.config import MODEL_SETTINGS, SEED, SAVED_MODELS_DIR, DEVICE_ID
from case_01_human_walk.dataset import HumanDatasetSeq, create_dataloaders, calculate_min_max_edge, create_dataloaders_from_raw
from case_01_human_walk.visualization import visualize_multi_step, create_gif

def find_best_model(model_dir):
    pattern = os.path.join(model_dir, "Val_Loss_*.pth")
    files = glob.glob(pattern)
    if not files: return None
    def extract_loss(fn):
        m = re.search(r'GenLoss_([\d\.]+)mm', fn)
        return float(m.group(1)) if m else float('inf')
    return min(files, key=extract_loss)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'visual'])
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE_ID
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {args.mode}")

    # Data Setup
    ds_train = HumanDatasetSeq('train', 
                            max_samples=MODEL_SETTINGS["max_training_samples"], 
                            data_dir=MODEL_SETTINGS["data_dir"], delta_frame=MODEL_SETTINGS["delta_frame"], nsteps=1)
    ds_val = HumanDatasetSeq('val', 
                          max_samples=MODEL_SETTINGS["max_testing_samples"], 
                          data_dir=MODEL_SETTINGS["data_dir"], delta_frame=MODEL_SETTINGS["delta_frame"], nsteps=1)
    # Test dataset for metrics
    ds_test = HumanDatasetSeq('test', 
                           max_samples=MODEL_SETTINGS["max_testing_samples"], 
                           data_dir=MODEL_SETTINGS["data_dir"], delta_frame=MODEL_SETTINGS["delta_frame"], nsteps=1)

    train_loader = create_dataloaders(ds_train, MODEL_SETTINGS["batch_size"])
    val_loader = create_dataloaders(ds_val, MODEL_SETTINGS["batch_size"], shuffle=False)
    test_loader = create_dataloaders(ds_test, MODEL_SETTINGS["batch_size"], shuffle=False)

    # Stats
    raw_stats = calculate_min_max_edge(train_loader)
    def to_dev(stat):
        return (stat[0].to(device), stat[1].to(device)) if isinstance(stat, tuple) else stat.to(device)
    train_stats = tuple(to_dev(s) for s in raw_stats)

    step_interval = MODEL_SETTINGS["delta_frame"]

    # Model time step
    t_step = step_interval * MODEL_SETTINGS["time_step"]

    node_in_f = 1
    edge_in_f = 1

    model = DynamicsSolver(node_in_f, 
                           edge_in_f, 
                           t_step, 
                           train_stats, 
                           num_msgs=4, 
                           latent_size=MODEL_SETTINGS["nf"], 
                           mlp_layers=MODEL_SETTINGS["n_layers"]).to(device) 
    
    optimizer = optim.Adam(model.parameters(), lr=MODEL_SETTINGS["lr"])
    trainer = Trainer(model, optimizer, device, train_stats, step_interval, SAVED_MODELS_DIR)

    if args.mode == 'train':
        print(f"Training for {MODEL_SETTINGS['epochs']} epochs...")

        # ---- CLEAR CHECKPOINT FOLDER ONCE PER TRAIN RUN ----
        if os.path.exists(trainer.model_dir):
            shutil.rmtree(trainer.model_dir)
        os.makedirs(trainer.model_dir, exist_ok=True)
        # ---------------------------------------------------

        with tqdm(range(1, MODEL_SETTINGS["epochs"]+1)) as pbar:
            for epoch in pbar:
                for batch in train_loader:
                    trainer.train(batch)
                
                # Validation every 5 epochs as per notebook cell 54
                if epoch % 2 == 0:
                    trainer.test(val_loader, mode='val', epoch=epoch)
                    trainer.test(test_loader, mode='test')
                
                pbar.set_postfix({
                    'Loss': f'{trainer.loss:.4e}',
                    'Val': f'{trainer.best_val_loss:.4f} (Ep{trainer.best_epoch})'
                })
        
        best_path = trainer.best_model_path or find_best_model(SAVED_MODELS_DIR)


    elif args.mode=='test':
        best_path = find_best_model(SAVED_MODELS_DIR)
        print(f"Loading {best_path}...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print('\n EVALUATING...')

        for nstep in [1,2]:
            
            dataset_eval = HumanDatasetSeq(
                partition='test', 
                max_samples=MODEL_SETTINGS["max_testing_samples"], 
                data_dir=MODEL_SETTINGS["data_dir"],delta_frame=MODEL_SETTINGS["delta_frame"], nsteps=nstep)
            
            dataloader_eval = create_dataloaders_from_raw(dataset_eval,200,shuffle=False)
            
            eval_error = evaluate_rollout(dataloader_eval, trainer.model, device, nsteps=nstep)
            
            print(f'loss for rollout {nstep} steps {eval_error}')
        
            print("DONE.")


    elif args.mode=='visual':
        best_path = find_best_model(SAVED_MODELS_DIR)
        print(f"Loading {best_path}...")
        model.load_state_dict(torch.load(best_path, map_location=device))
        print('\n EVALUATING...')
        dataset_eval = HumanDatasetSeq(
            partition='test', 
            max_samples=MODEL_SETTINGS["max_testing_samples"], 
            data_dir=MODEL_SETTINGS["data_dir"], delta_frame=MODEL_SETTINGS["delta_frame"],
            nsteps=2)
        loader = create_dataloaders_from_raw(dataset_eval,200,shuffle=False)
        visualize_multi_step(
                            loader,
                            MODEL_SETTINGS["results_dir"],
                            trainer.model,
                            device,
                            steps=[1,2],
                            num_graphs=5
                            )
        create_gif(MODEL_SETTINGS["results_dir"])        

    else:
        print("not a valid mode.")

if __name__ == "__main__":
    main()