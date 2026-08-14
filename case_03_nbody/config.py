import os

# Base paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(BASE_DIR, "case_03_nbody", "data_321")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "case_03_nbody", "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "case_03_nbody", "results")

# Ensure directories exist
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Settings derived from the notebook and your requirements
MODEL_SETTINGS = {
    "batch_size": 100,
    "epochs": 600,  # matching GMN code : https://github.com/hanjq17/GMN/blob/main/spatial_graph/n_body_system/dataset_nbody.py
    "lr": 5e-4,
    "nf": 64,  # Latent size
    "model": "dgn",
    "n_layers": 2,
    "max_training_samples": 500,  # can use higher 1000, 1500.... etc.
    "data_dir": DATA_DIR,
    "finite_diff": True,
    # Time steps
    "delta_frame": 10,  # matching GMN dataset.py : https://github.com/hanjq17/GMN/blob/main/spatial_graph/n_body_system/dataset_nbody.py
    "time_step": 1,  # Used for finite diff calculation
    # N-Body specific configuration
    "n_isolated": 3,
    "n_stick": 2,
    "n_hinge": 1,
    "results_dir": RESULTS_DIR,
}

SEED = 90
# Which GPU to run on. Override from the shell with CUDA_VISIBLE_DEVICES=<n>;
# the default picks the first visible device.
DEVICE_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
