# RIMD to DeltaXY CVAE

RIMD-based Conditional Variational Autoencoder for predicting coordinate corrections in blank deformation analysis.

## Overview

This project implements a machine learning approach to predict coordinate corrections (Δx, Δy) for 1-step blank deformation analysis using RIMD (Rotational Invariant Mesh Descriptors) features and Conditional Variational Autoencoders (CVAE).

## Features

- RIMD-based geometric feature extraction
- Conditional Variational Autoencoder implementation
- Flexible experiment configuration system
- Jupyter notebook-based workflow
- Comprehensive evaluation and visualization tools

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd blank_deformation_rimd_cvae

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

## Quick Start

1. **Data Preprocessing**: Run `notebooks/01_preprocessing.ipynb`
2. **Training**: Run `notebooks/02_training.ipynb`
3. **Evaluation**: Run `notebooks/03_evaluation.ipynb`
4. **Analysis**: Run `notebooks/04_analysis.ipynb`

## Project Structure

```
blank_deformation_rimd_cvae/
├── config/                 # Configuration files
├── src/                    # Source code
│   ├── data/               # Data processing
│   ├── models/             # Model definitions
│   ├── training/           # Training logic
│   ├── evaluation/         # Evaluation modules
│   └── utils/              # Utilities
├── notebooks/              # Jupyter notebooks
├── experiments/            # Experiment results
└── data/                   # Dataset storage
```

## Configuration

Experiments are configured using the config system:

```python
from config.experiment_configs import get_baseline_config, get_cvae_config

# Baseline experiment
config = get_baseline_config()

# CVAE experiment
config = get_cvae_config()

# Custom experiment
config = get_baseline_config()
config.exp_id = "my_experiment"
config.model.hidden_dim = 256
```

## License

MIT License