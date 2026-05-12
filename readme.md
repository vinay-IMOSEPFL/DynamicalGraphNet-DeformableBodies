# A Physics-Informed Graph Neural Network Conserving Linear and Angular Momentum for Dynamical Systems

This repository contains the official **PyTorch** implementation of the paper:

**A Physics-Informed Graph Neural Network Conserving Linear and Angular Momentum for Dynamical Systems**,  
published in *Nature Communications*.

The project implements a **Dynamics Solver (DygnNet)** that integrates physical laws directly into the graph message-passing mechanism. It is designed to simulate dynamical systems while strictly adhering to conservation laws.

---

## 🌟 Key Features

- **Momentum Conservation**  
  Explicit enforcement of linear and angular momentum conservation via a custom *InteractionDecoder* and symplectic integration schemes.

- **Frame Covariance**  
  Uses a local reference frame calculation (*RefFrameCalc*) to ensure predictions are robust to global rotations and translations.

- **Bi-directional Edges & Reference Frames**  
  All graph edges, including virtual edges, are modeled as bi-directional to correctly represent symmetric physical interactions (Newton’s Third Law).

---

## 📂 Project Structure

```text
DYNGNET/
├── case_01_human_walk/         # Human Motion Capture Experiment
│   ├── data/                   # motion.pkl and split files
│   ├── results/                # Output visualizations (GIFs, PNGs)
│   ├── saved_models/           # Model checkpoints (Val_Loss_*.pth)
│   ├── config.py               # Hyperparameters (lr, epochs, batch_size)
│   ├── dataset.py              # HumanDatasetSeq & graph construction
│   ├── visualization.py        # 3D plotting & GIF generation
│   └── training.log            # Training logs
│
├── case_02_protein/            # Molecular Dynamics (Protein) Experiment
│   ├── mdanalysis/dataset/     # Raw molecular data
│   │   └── adk_equilibrium/    # Protein data (e.g. 1ake)
│   ├── config.py
│   ├── dataset.py              # MDAnalysis dataset wrapper
│   └── preprocess_data.py      # Preprocessing scripts
│
├── case_03_nbody/              # N-Body / Stick / Hinge Experiment
│   ├── data_321/               # Dataset storage
│   ├── results/
│   ├── saved_models/
│   ├── simulate_sys/           # Ground-truth physics simulator
│   │   ├── physical_objects.py
│   │   └── system.py
│   ├── config.py
│   ├── dataset.py
│   ├── generate_data.py        # Synthetic data generation
│   └── visualization.py
│
├── model/                      # Core Architecture
│   └── model.py                # DynamicsSolver, InteractionBlock, RefFrameCalc
│
├── utils/                      # Shared Utilities
│   ├── trainer.py              # Training & checkpointing
│   └── utils.py                # Math helpers, evaluation, seeds
│
├── main_human_walk.py          # Entry point: Human Walk
├── main_nbody.py               # Entry point: N-Body
├── main_protein.py             # Entry point: Protein
└── README.md
```

---

## 🛠️ Installation & Prerequisites

Python **3.8+** is recommended.

```bash
# Core deep learning framework
pip install torch torchvision

# Graph Neural Networks
pip install torch-geometric

# Utilities & visualization
pip install numpy matplotlib tqdm imageio

# Domain-specific dependency (Protein experiment only)
pip install MDAnalysis
```

---

## 🚀 Usage

Each experiment case has its own entry script (`main_*.py`).  
Scripts support three modes: **train**, **test**, and **visual**.

> **Note**: Hyperparameters (learning rate, batch size, epochs) are defined in the corresponding  
> `config.py` file inside each experiment folder.

---

### 1. Human Walk Experiment

**Train**
```bash
python main_human_walk.py --mode train
```

- Saves the best model to `case_01_human_walk/saved_models/`
- Clears previous checkpoints before training

**Test**
```bash
python main_human_walk.py --mode test
```

- Evaluates rollout error for 1–4 prediction steps

**Visualize**
```bash
python main_human_walk.py --mode visual
```

- Generates 3D prediction vs. ground-truth plots
- Saves `rollout.gif` in `case_01_human_walk/results/`

---

### 2. N-Body Experiment

Supports an additional argument `--test_config` specifying the system composition.

**Format**
```text
"n_isolated,n_stick,n_hinge"
```

**Train**
```bash
python main_nbody.py --mode train
```

**Test**
```bash
python main_nbody.py --mode test --test_config "3,2,1"
```

**Visualize**
```bash
python main_nbody.py --mode visual --test_config "3,2,1"
```

---

### 3. Protein Experiment

**Train**
```bash
python main_protein.py --mode train
```

**Test**
```bash
python main_protein.py --mode test
```

---

## 🧠 Model Details

### Dynamics Solver

The core model (`model/model.py`) acts as a **learnable physics simulator**.

- **Encoder**  
  Embeds node types (e.g. charges, indicators) and edge attributes.

- **Interaction Block**
  - **Reference Frame Calculation**: Computes a local frame \((a, b, c)\) from relative positions and velocities.
  - **Projection**: Projects vector inputs into the local frame for invariance.
  - **Interaction Decoder**: Predicts scalar coefficients combined with basis vectors to construct forces \(F_{ij}\) and torques \(\tau_{ij}\), enforcing  
    \[ F_{ij} = -F_{ji} \]  
    by design.

- **Integrator**
  Uses a semi-implicit Euler scheme to update positions \(x\) and velocities \(v, \omega\), with explicit checks on momentum conservation.

---

## 📊 Data Format

- **Input**  
  - Current position \(x_t\)  
  - Current velocity \(v_t\)  
  - Previous velocity \(v_{t-1}\)

- **Target**  
  - Next-step displacement \(\Delta x\)  
  - Velocity change \(\Delta v\)

---

## 📄 Citation

If you use this code, please cite:

```bibtex
@article{dyngnet2025,
  title   = {A physics-informed graph neural network conserving linear and angular momentum for dynamical systems},
  journal = {Nature Communications},
  year    = {2025},
  url     = {https://www.nature.com/articles/s41467-025-67802-5}
}
```

**Preprint**  
*A physics-informed graph neural network conserving linear and angular momentum for dynamical systems.*  
arXiv:2501.07373 (2025)
