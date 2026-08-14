import os

# Base paths
BASE_DIR = os.getcwd()
DATA_DIR = os.path.join(
    BASE_DIR, "case_01_human_walk", "data"
)  # Ensure motion.pkl is here
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "case_01_human_walk", "saved_models")
RESULTS_DIR = os.path.join(BASE_DIR, "case_01_human_walk", "results")

MODEL_SETTINGS = {
    "batch_size": 100,
    # epoch : set 500 for 1 step comparison with GMN with loss on only positions for one to one comparison with GMN, TFN etc.
    # epoch : set 1500-2000 for rollout comparison (even 1 step comparison). (better convergence at 1000 epoch)
    "epochs": 600,
    "lr": 5e-4,
    "nf": 128,
    "model": "dgn",
    "n_layers": 2,
    # Learn an external influence on each node. Off for a closed system.
    "use_ext_force": True,
    "max_testing_samples": 600,
    "max_training_samples": 200,
    "data_dir": DATA_DIR,
    "finite_diff": True,  # always true, we calculate the velocities by finite differencing positions
    "time_step": 1.0,  # for finite differencing we take a delta_t = 1
    "delta_frame": 30,  # int
    "results_dir": RESULTS_DIR,
}

SEED = 90

# Which GPU to run on. Override from the shell with CUDA_VISIBLE_DEVICES=<n>;
# the default picks the first visible device.
DEVICE_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
