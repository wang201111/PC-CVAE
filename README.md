# PC-CVAE: Physics-Constrained Conditional Variational Autoencoder for Wide-Range Extrapolation of Macroscopic Properties in Multicomponent Systems

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

This repository contains the official implementation of **PC-CVAE**, a physics-constrained generative framework for reliable wide-range extrapolation of macroscopic thermophysical properties in multicomponent systems from limited low-temperature, ambient-pressure experimental data.

---

## Overview

Reliable wide-range extrapolation of macroscopic properties such as solubility and dynamic viscosity in multicomponent systems from limited experimental data remains a critical challenge in chemical engineering. PC-CVAE addresses this by embedding three fundamental thermodynamic laws directly into the architectural design and training objectives of a Conditional Variational Autoencoder (CVAE):

1. **Phase rule-driven latent space dimensionality** — The Gibbs phase rule directly sets `dim(z) = F − n_cond`, ensuring manifold reconstruction in a space consistent with the intrinsic degrees of freedom of the system.
2. **Boundary collocation constraint** — A collocation loss anchors the learned manifold boundaries to frozen binary subsystem models, suppressing manifold drift in data-scarce regions.
3. **Inverse Manifold Mapping structure with cycle consistency** — A deterministic mapping φ from operating conditions to latent space coordinates eliminates prediction variance from random sampling and enables stable predictions beyond the training set.

The framework requires no explicit governing equations and is validated on two systems:

- **Na₂SO₄–MgSO₄–H₂O ternary aqueous salt system** (solid–liquid solubility, SLE)
- **MCH–cis-Decalin–HMN ternary organic system** (liquid-phase dynamic viscosity)

PC-CVAE achieves R² of **0.892** and **0.946** in the far-range extrapolation domain, reducing RMSE by **56%** and **51%** relative to classical semi-empirical mechanistic models.

---

## Repository Structure

```
PC-CVAE/
├── src/
│   └── models/
│       ├── pc_cvae_solubility.py         # PC-CVAE for solubility system
│       ├── pc_cvae_viscosity.py          # PC-CVAE for viscosity system
│       └── low_dim_model.py              # LowDimEnsemble (binary subsystem models)
├── experiments/
│   ├── solubility/
│   │   ├── ablation/
│   │   │   └── pc_cvae_experiment.py     # K-fold ablation study (solubility)
│   │   ├── small_sample/
│   │   │   └── Small_Sample_Sensitivity_Experiment_-_PC-CVAE.py
│   │   └── noise/
│   │       └── Noise_Robustness_Experiment_-_PC-CVAE.py
│   └── viscosity/
│       ├── ablation/
│       │   └── pc_cvae_experiment.py     # K-fold ablation study (viscosity)
│       └── small_sample/
│           └── Small_Sample_Sensitivity_Experiment_-_PC-CVAE__Viscosity_System_.py
├── data/
│   ├── solubility/
│   │   ├── split_by_temperature/         # Train / near-range / far-range splits
│   │   └── fixed_splits/                 # Fixed train / val splits
│   └── viscosity/
│       ├── split_by_temperature/
│       └── fixed_splits/
├── models/
│   └── Low_dim_model/
│       ├── solubility/                   # Pretrained binary solubility models
│       │   ├── Na2SO4-H2O.pth
│       │   └── MgSO4-H2O.pth
│       └── viscosity/                    # Pretrained binary viscosity models
│           ├── MCH_HMN.pth
│           ├── MCH_cis_Decalin.pth
│           └── cis_Decalin_HMN.pth
├── results/                              # Output directory (generated at runtime)
├── environment.yml
├── README.md
├── LICENSE
└── CITATION.cff
```

---

## Installation

**Step 1.** Clone the repository:

```bash
git clone https://github.com/your-username/PC-CVAE.git
cd PC-CVAE
```

**Step 2.** Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate pc-cvae
```

**Step 3.** Verify the installation:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## Quick Start

### Training the Solubility Model

```python
from src.models.pc_cvae_solubility import CVAEConfig, CVAEPhysicsModel, LowDimInfo
from src.models.low_dim_model import LowDimEnsemble
import numpy as np

# Load data
X_train = ...  # shape (N, 2): [T/°C, w(MgSO4)/%]
y_train = ...  # shape (N,):   w(Na2SO4)/%

# Load pretrained binary subsystem models
model_na = LowDimEnsemble.load('models/Low_dim_model/solubility/Na2SO4-H2O.pth')
model_mg = LowDimEnsemble.load('models/Low_dim_model/solubility/MgSO4-H2O.pth')

low_dim_list = [
    LowDimInfo(model=model_na, name='Na2SO4_H2O', constraint_type='Na2SO4'),
    LowDimInfo(model=model_mg, name='MgSO4_H2O',  constraint_type='MgSO4'),
]

# Configure and train PC-CVAE
config = CVAEConfig(
    LATENT_DIM=1,
    N_EPOCHS=500,
    LAMBDA_KL=0.001,
    LAMBDA_COLLOCATION_Na2SO4=1.0,
    LAMBDA_COLLOCATION_MgSO4=1.0,
    LAMBDA_CYCLE=1.0,
    CYCLE_T_RANGE=(-10.0, 200.0),
)

cvae = CVAEPhysicsModel(input_dim=3, condition_dim=1, config=config)
cvae.fit(X_train, y_train, low_dim_list=low_dim_list)

# Deterministic inference
X_test = ...  # shape (M, 2): [T/°C, w(MgSO4)/%]
y_pred = cvae.predict(X_test)  # shape (M,): predicted w(Na2SO4)/%
```

### Training the Viscosity Model

```python
from src.models.pc_cvae_viscosity import CVAEConfig, CVAEPhysicsModel, LowDimInfo
from src.models.low_dim_model import LowDimEnsemble

# Load data
X_train = ...  # shape (N, 4): [T/°C, P/Pa, x(MCH)/%, x(cis-Decalin)/%]
y_train = ...  # shape (N, 1): dynamic viscosity μ (mPa·s)

# Load pretrained binary subsystem models
model_mch_hmn = LowDimEnsemble.load('models/Low_dim_model/viscosity/MCH_HMN.pth')
model_mch_dec = LowDimEnsemble.load('models/Low_dim_model/viscosity/MCH_cis_Decalin.pth')
model_dec_hmn = LowDimEnsemble.load('models/Low_dim_model/viscosity/cis_Decalin_HMN.pth')

low_dim_list = [
    LowDimInfo(model=model_dec_hmn, name='cis_Decalin_HMN', boundary_type='mch_zero'),
    LowDimInfo(model=model_mch_hmn, name='MCH_HMN',         boundary_type='dec_zero'),
    LowDimInfo(model=model_mch_dec, name='MCH_cis_Decalin', boundary_type='hmn_zero'),
]

config = CVAEConfig(
    LATENT_DIM=2,
    N_EPOCHS=500,
    LAMBDA_KL=0.001,
    LAMBDA_COLLOCATION_MCH=1.0,
    LAMBDA_COLLOCATION_DEC=1.0,
    LAMBDA_COLLOCATION_HMN=1.0,
    LAMBDA_CYCLE=1.0,
    CYCLE_T_RANGE=(20.0, 80.0),
    CYCLE_P_RANGE=(1e5, 1e8),
)

cvae = CVAEPhysicsModel(config=config)
cvae.fit(X_train, y_train, low_dim_list=low_dim_list)

X_test = ...  # shape (M, 4)
y_pred = cvae.predict(X_test)  # shape (M, 1): predicted viscosity
```

---

## Reproducing Experiments

All experiment scripts are located under `experiments/`. Run from the project root:

### K-Fold Ablation Study (Solubility)

```bash
python experiments/solubility/ablation/pc_cvae_experiment.py
```

### K-Fold Ablation Study (Viscosity)

```bash
python experiments/viscosity/ablation/pc_cvae_experiment.py
```

### Small-Sample Sensitivity (Solubility)

```bash
python experiments/solubility/small_sample/Small_Sample_Sensitivity_Experiment_-_PC-CVAE.py
```

### Small-Sample Sensitivity (Viscosity)

```bash
python experiments/viscosity/small_sample/Small_Sample_Sensitivity_Experiment_-_PC-CVAE__Viscosity_System_.py
```

### Noise Robustness (Solubility)

```bash
python experiments/solubility/noise/Noise_Robustness_Experiment_-_PC-CVAE.py
```

Results are saved to `results/` with per-fold metrics, predictions, and training histories in Excel format.

---

## Data

Experimental data are sourced from published literature. The data files are organized by system and split type:

| Path | Description |
|---|---|
| `data/solubility/split_by_temperature/` | Solubility data split by temperature (training set ≤ 50 °C; near-range 50–100 °C; far-range ≥ 100 °C) |
| `data/solubility/fixed_splits/` | Fixed train/val split for noise robustness experiments |
| `data/viscosity/split_by_temperature/` | Viscosity data split by temperature (training set 20–30 °C; near-range 30–60 °C; far-range 60–80 °C) |
| `data/viscosity/fixed_splits/` | Fixed train/val split for small-sample experiments |

Data sources are listed in the paper (see Citation below).

---

## Pretrained Binary Subsystem Models

Pretrained LowDimEnsemble models for all binary subsystems are provided under `models/Low_dim_model/`. These are trained on the full experimental datasets of the respective binary systems and their parameters are frozen during PC-CVAE training.

To retrain a binary subsystem model from scratch:

```python
from src.models.low_dim_model import LowDimEnsemble, LowDimConfig
import numpy as np

X = ...  # shape (N, 1): temperature T/°C  (or [T, P] for viscosity binaries)
y = ...  # shape (N, 1): solubility or viscosity

config = LowDimConfig(N_ENSEMBLE=5, N_EPOCHS=1000)
model = LowDimEnsemble(input_dim=1, config=config)
model.fit(X, y, verbose=True)
model.save('models/Low_dim_model/solubility/Na2SO4-H2O.pth')
```

---

## System Requirements

| Component | Version |
|---|---|
| Python | 3.9 |
| PyTorch | 1.9.0 |
| scikit-learn | 1.0.1 |
| NumPy | ≥ 1.21 |
| pandas | ≥ 1.3 |
| openpyxl | ≥ 3.0 |
| BayesianOptimization | 1.4.2 |
| CUDA (optional) | ≥ 11.1 |

Hardware used in this work: NVIDIA RTX 4070 Super GPU, Intel i5-13600KF CPU (3.5 GHz), 32 GB RAM.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{your_citation_key,
  title   = {Physics-Constrained Conditional Variational Autoencoder for Wide-Range Extrapolation of Macroscopic Properties in Multicomponent Systems},
  author  = {Your Name and Co-authors},
  journal = {Journal Name},
  year    = {2025},
  doi     = {10.xxxx/xxxxxx}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
