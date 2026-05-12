import os

# Base paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "case_02_protein", "mdanalysis", "dataset")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "case_02_protein", "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "case_02_protein", "results")

# Settings derived from the notebook
MODEL_SETTINGS ={
        "batch_size": 8,
        "epochs": 600,
        "lr": 5e-4,
        "nf": 64,
        "model": "dgn",
        "n_layers":2,
        "data_dir": DATA_DIR,
        "weight_decay": 1e-10,
        "finite_diff":True,
        "time_step":1.0,
        "delta_frame": 15, # int
        "results_dir": RESULTS_DIR,
}

SEED = 90
DEVICE_ID = "1" # From notebook: os.environ['CUDA_VISIBLE_DEVICES'] = '1'