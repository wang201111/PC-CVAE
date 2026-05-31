# PC-CVAE: Physics-Constrained Conditional Variational Autoencoder for Wide-Range Extrapolation of Thermophysical Properties in Multicomponent Systems

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3.1-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.19716264-blue.svg)](https://doi.org/10.5281/zenodo.19716263)

Official implementation of **PC-CVAE**, a physics-constrained generative framework for reliable wide-range extrapolation of macroscopic thermophysical properties in multicomponent systems from limited, easily accessible experimental data.

---

## Overview

The macroscopic properties of multicomponent systems are deterministic, continuous functions of state and composition that lie on a thermodynamic manifold of fixed dimensionality. Instead of fitting property values point by point, PC-CVAE reconstructs this global manifold and embeds three physical priors directly into the architecture and training objective of a Conditional Variational Autoencoder (CVAE):

1. **Phase rule-driven latent dimensionality.** The Gibbs phase rule provides a physics-informed prior for the intrinsic dimensionality of the manifold, which is used to set the latent dimensionality `dim(z) = F - n_cond`, replacing empirical hyperparameter search.
2. **Boundary collocation constraint.** A collocation loss anchors the boundaries of the reconstructed manifold to frozen low-dimensional (binary) subsystem models, fixing the manifold geometry where experimental data are absent.
3. **Inverse manifold mapping with cycle consistency.** A deterministic mapping from operating conditions to latent coordinates, supervised by a cycle-consistency loss over the full domain, removes the prediction variance of random sampling and yields a unique, physically consistent state for any condition.

The framework requires no property-specific governing equation. It is validated on three systems governed by fundamentally different thermodynamic mechanisms:

| System | Property | Mechanism |
|---|---|---|
| Na₂SO₄–MgSO₄–H₂O (ternary aqueous salt) | Solid–liquid solubility | Phase equilibrium |
| MCH–cis-Decalin–HMN (ternary organic) | Liquid-phase dynamic viscosity | Momentum transport |
| NaCl–CaCl₂–H₂O (ternary aqueous salt) | Thermal conductivity | Energy transport |

Across all three, PC-CVAE holds far-range extrapolation accuracy close to the in-domain level, with far-range **R² of 0.892, 0.981, and 0.851**. RMSE is reduced by **60%** (vs. the Pitzer model, solubility) and **70%** (vs. the Eyring-NRTL model, viscosity), and by more than **90%** relative to a purely data-driven baseline for thermal conductivity over a **200 °C** extrapolation span.

---

## Repository Structure

```
PC-CVAE/
├── src/
│   └── models/
│       ├── pc_cvae_solubility.py            # PC-CVAE for the solubility system
│       ├── pc_cvae_viscosity.py             # PC-CVAE for the viscosity system
│       ├── pc_cvae_thermal_conductivity.py  # PC-CVAE for the thermal-conductivity system
│       ├── utils_solubility.py              # Physics evaluators / helpers (solubility)
│       ├── utils_viscosity.py               # Physics evaluators / helpers (viscosity)
│       └── low_dim_model.py                 # LowDimEnsemble (binary subsystem models)
├── experiments/
│   ├── solubility/
│   │   ├── pc_cvae_experiment.py            # K-fold ablation / main extrapolation study
│   │   ├── small_sample_sensitivity_experiment.py
│   │   └── noise_robustness_experiment.py
│   ├── viscosity/
│   │   ├── pc_cvae_experiment.py
│   │   ├── small_sample_sensitivity_experiment.py
│   │   └── noise_robustness_experiment.py
│   └── Thermal_conductivity/
│       ├── pc_cvae_experiment.py
│       ├── pc_cvae_ablation_study.py
│       ├── baseline_ablation_experiment.py
│       ├── gpr_baseline.py
│       └── ternary_boundary_correspondence_analysis.py
├── data/
│   ├── solubility/
│   │   ├── raw/{binary,ternary}/            # Source experimental data
│   │   ├── cleaned/                         # Cleaned dataset
│   │   ├── split_by_temperature/            # Train / near-range / far-range splits
│   │   └── fixed_splits/                    # Fixed train / val splits
│   ├── viscosity/                           # (same layout)
│   └── Thermal_conductivity/                # (same layout)
├── models/
│   └── Low_dim_model/
│       ├── solubility/                      # Pretrained binary models (Na2SO4-H2O, MgSO4-H2O)
│       ├── viscosity/                       # Pretrained binary models (MCH_HMN, MCH_cis_Decalin, cis_Decalin_HMN)
│       └── Thermal_conductivity/            # Pretrained binary models (NaCl-H2O, CaCl2-H2O)
├── results/                                 # Output directory (generated at runtime)
├── requirements.txt                         # Minimal pip dependencies (recommended)
├── environment.yml                          # Exact conda export (full lock)
├── README.md
├── LICENSE
└── CITATION.cff
```

> **Note.** The scripts under `experiments/Thermal_conductivity/` import a module
> `utils_thermal_conductivity` (providing `ThermalConductivityPhysicsEvaluator`).
> Place `utils_thermal_conductivity.py` in `src/models/` before running them; the
> solubility and viscosity experiments run as-is.

---

## Installation

Python 3.10 is recommended.

**Option A — pip (recommended, portable):**

```bash
git clone https://github.com/wang201111/PC-CVAE.git
cd PC-CVAE
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

**Option B — conda (exact environment used in this work):**

```bash
conda env create -f environment.yml
conda activate electrolyte-pytorch
```

`environment.yml` is a full export of the environment used to produce the results
and contains build-pinned dependencies (and may list mirror channels you can
remove if they are not reachable). For most users `requirements.txt` is simpler.

Verify the installation:

```bash
python -c "import torch; print(torch.__version__)"
```

---

## Quick Start

### Solubility system

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

### Viscosity system

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

The thermal-conductivity model exposes the same interface as the viscosity model
(`pc_cvae_thermal_conductivity`).

---

## Reproducing the Experiments

All experiment scripts read data and pretrained models via paths relative to the
project root and are intended to be run from the project root:

```bash
# Solubility
python experiments/solubility/pc_cvae_experiment.py
python experiments/solubility/small_sample_sensitivity_experiment.py
python experiments/solubility/noise_robustness_experiment.py

# Viscosity
python experiments/viscosity/pc_cvae_experiment.py
python experiments/viscosity/small_sample_sensitivity_experiment.py
python experiments/viscosity/noise_robustness_experiment.py

# Thermal conductivity (requires src/models/utils_thermal_conductivity.py)
python experiments/Thermal_conductivity/pc_cvae_experiment.py
python experiments/Thermal_conductivity/pc_cvae_ablation_study.py
python experiments/Thermal_conductivity/baseline_ablation_experiment.py
python experiments/Thermal_conductivity/gpr_baseline.py
python experiments/Thermal_conductivity/ternary_boundary_correspondence_analysis.py
```

Outputs (per-fold metrics, predictions, training histories) are written to `results/`.
The training seed is fixed at 42 for reproducibility.

---

## Data

Experimental data are compiled from published literature and organized by system
and split. Data sources are listed in the associated paper.

| Path | Description |
|---|---|
| `data/<system>/raw/binary/` | Binary subsystem source data (for the boundary models) |
| `data/<system>/raw/ternary/` | Ternary source data |
| `data/<system>/cleaned/` | Cleaned dataset |
| `data/<system>/split_by_temperature/` | Train / near-range / far-range splits by temperature |
| `data/<system>/fixed_splits/` | Fixed train / val split |

Temperature partitions: solubility (train ≤ 50 °C; near 50–100 °C; far ≥ 100 °C);
viscosity (train 20–30 °C; near 30–60 °C; far 60–80 °C); thermal conductivity
(train 20–100 °C; far 200–300 °C).

---

## Pretrained Binary Subsystem Models

Pretrained `LowDimEnsemble` models for every binary subsystem are provided under
`models/Low_dim_model/`. They are trained on the full datasets of their respective
binary systems and frozen during PC-CVAE training. To retrain one:

```python
from src.models.low_dim_model import LowDimEnsemble, LowDimConfig

X = ...  # shape (N, 1): temperature T/°C  (or [T, P] for viscosity binaries)
y = ...  # shape (N, 1): solubility / viscosity / thermal conductivity

config = LowDimConfig(N_ENSEMBLE=5, N_EPOCHS=1000)
model = LowDimEnsemble(input_dim=1, config=config)
model.fit(X, y, verbose=True)
model.save('models/Low_dim_model/solubility/Na2SO4-H2O.pth')
```

---

## System Requirements

| Component | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.3.1 |
| scikit-learn | 1.7.0 |
| NumPy | 2.2.6 |
| pandas | 2.3.0 |
| SciPy | 1.15.2 |
| openpyxl | 3.1.5 |
| plotly | 6.1.2 |
| bayesian-optimization | 3.1.0 |
| networkx | 3.4.2 |

Hardware used in this work: NVIDIA RTX 4070 Super GPU, Intel i5-13600KF CPU
(3.5 GHz), 32 GB RAM. A GPU is optional; the models are small and train in under a
minute on the above hardware.

---

## Data and Code Availability

All datasets and complete model predictions are publicly available on GitHub
(<https://github.com/wang201111/PC-CVAE>) and permanently archived on Zenodo
(<https://doi.org/10.5281/zenodo.19716264>).

---

## Citation

If you use this code, please cite the software (see `CITATION.cff`) and the
associated paper:

```bibtex
@article{wang_pccvae_2026,
  title   = {Reliable Extrapolation of Multicomponent Thermophysical Properties
             to Extreme Operating Conditions via Physics-Constrained Generative Learning},
  author  = {Wang, Yuan and Yuan, Shuaiying and Ming, Hongxin and Li, Song and
             Zhang, Weidong and Li, Hui and Liu, Dahuan},
  year    = {2026},
  note    = {Manuscript under review. Code and data archived at
             \url{https://doi.org/10.5281/zenodo.19716264}}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
