```markdown
# ST-FSFusion: Meta Cross-Domain Fusion with Cross-City Knowledge Transfer (MCD-CKT)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**ST-FSFusion is a Meta Cross-Domain Fusion framework with Cross-City Knowledge Transfer for few-shot traffic flow prediction. It addresses the challenge of data scarcity in new cities by combining meta-learning, cross-domain fusion, and physics-informed constraints to enable rapid adaptation with minimal target data.

### Key Features

- **Meta Cross-Domain Fusion**: Combines temporal and spatial processors with multi-head attention for rich spatio-temporal representations
- **Cross-City Knowledge Transfer**: Transfers learned patterns from data-rich source cities to data-scarce target cities
- **Few-Shot Adaptation**: Fine-tunes on as few as 3 days of target city data
- **Physics-Informed Constraints**: Low-rank meta-knowledge regularisation, proximity constraints, and hybrid FSL loss
- **Robust Transformer**: Noise injection mechanism for improved generalisation under distribution shift
- **Uncertainty Quantification**: Built-in uncertainty estimation for reliable predictions
- **Research Question Analyses**: Automated RQ evaluation covering horizons, convergence, uncertainty, few-shot efficiency, and spatial transfer

---

## Architecture

```
Input Features
      │
      ▼
Node Feature Encoder
      │
      ├──► Temporal Processor (TP)    — multi-scale temporal patterns
      └──► Spatial Processor  (SP)    — graph-based spatial dependencies
                    │
                    ▼
        Meta Cross-Domain Fusion
        (Multi-Head Attention + Low-Rank Meta-Knowledge)
                    │
                    ▼
        RobustTransformer
        (noise injection τ, α for distribution robustness)
                    │
                    ▼
        Prediction Head
        (horizon-specific outputs + uncertainty)
```

**Two-Phase Training:**
```
Phase 1 — Standard Training     Phase 2 — Fine-Tuning
Source city data (200 epochs) → Target city data (300 epochs, 3 days)
```

---

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- CUDA (optional, recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/afofanah/ST_FSFusion.git
cd ST_FSFusion

# Create conda environment
conda create -n stfsfusion python=3.8
conda activate stfsfusion

# Install PyTorch (adjust cuda version as needed)
pip install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Install PyTorch Geometric
pip install torch-geometric

# Install remaining dependencies
pip install numpy pyyaml matplotlib scikit-learn tqdm
```

---

## Data Preparation

Download datasets and place them in the `data/` directory:

| Dataset   | Nodes | Source                                      |
|-----------|-------|---------------------------------------------|
| METR-LA   | 207   | [DCRNN](https://github.com/liyaguang/DCRNN) |
| PEMS-BAY  | 325   | [DCRNN](https://github.com/liyaguang/DCRNN) |
| Chengdu-M | 524   | Available upon request                      |
| Shenzhen  | 627   | Available upon request                      |

Expected directory structure:

```
data/
├── metr-la/
│   ├── dataset.npy       # Node feature matrix [T, N, F]
│   └── matrix.npy        # Adjacency matrix [N, N]
├── pems-bay/
│   ├── dataset.npy
│   └── matrix.npy
├── chengdu_m/
│   ├── dataset.npy
│   └── matrix.npy
└── shenzhen/
    ├── dataset.npy
    └── matrix.npy
```

---

## Usage

### Quick Start

```bash
# Train with default settings (Shenzhen target, 3 target days)
python main.py

# Train on a specific target dataset
python main.py --test_dataset metr-la

# Custom few-shot settings
python main.py --test_dataset shenzhen --target_days 3 --source_epochs 200 --target_epochs 300

# Evaluation only (load existing checkpoint)
python main.py --eval_only True --resume_from experiments/your_exp/models/best_model.pth
```

### Full CLI Options

```bash
python main.py \
  --config config.yaml \           # Path to config file
  --test_dataset shenzhen \        # Target dataset: metr-la, pems-bay, chengdu_m, shenzhen
  --target_days 3 \                # Few-shot target days
  --experiment_name mcd_ckt \      # Experiment name prefix
  --save_dir ./experiments \       # Output directory
  --device auto \                  # auto | cpu | cuda
  --seed 42 \                      # Random seed
  --source_epochs 200 \            # Phase 1 training epochs
  --target_epochs 300 \            # Phase 2 fine-tuning epochs
  --hidden_dim 64 \                # Hidden dimension
  --meta_dim 32 \                  # Meta-knowledge dimension
  --message_dim 8 \                # Message/feature dimension
  --num_heads 4 \                  # Attention heads
  --dropout 0.1 \                  # Dropout rate
  --learning_rate 1e-4 \           # Learning rate
  --tp True \                      # Use temporal processor
  --sp True \                      # Use spatial processor
  --batch_size 1 \                 # DataLoader batch size
  --minibatch_size 4096            # Mini-batch for gradient computation
```

---

## Training Pipeline

### Phase 1 — Standard Training
- Trains on source city data for `source_epochs` epochs
- Optimises combined loss: flow prediction + meta-knowledge regularisation + physics constraints
- Saves best model checkpoint based on validation loss

### Phase 2 — Fine-Tuning
- Adapts to target city using only `target_days` of observations
- Lower learning rate preserves source domain knowledge
- Evaluates on held-out test set every epoch

### Loss Function (Paper Eq. 14)
```
L = L_pred + λ₁ · L_nuclear + λ₂ · L_proximity
```
- `L_pred` — prediction MSE/MAE
- `L_nuclear` — low-rank meta-knowledge regularisation (nuclear norm)
- `L_proximity` — cross-city proximity constraint

---

## Output Structure

Each experiment saves to `experiments/{name}_{dataset}_{timestamp}/`:

```
experiments/mcd_ckt_shenzhen_YYYYMMDD_HHMMSS/
├── config.yaml              # Full configuration
├── args.yaml                # CLI arguments
├── results.yaml             # Performance summary
├── summary.txt              # Human-readable results
├── models/
│   ├── best_model.pth       # Best checkpoint (Phase 1)
│   └── finetuned_model.pth  # Fine-tuned checkpoint (Phase 2)
├── predictions/
│   ├── predictions.npy      # Model predictions
│   ├── targets.npy          # Ground truth
│   └── metrics.json         # Per-horizon metrics
├── plots/                   # Visualisations
│   ├── rq1_horizon.png
│   ├── rq2_convergence.png
│   ├── rq3_uncertainty.png
│   ├── rq5_graph.png
│   ├── rq6_fewshot.png
│   └── rq7_spatial.png
└── research_results/        # RQ analysis outputs
```

---

## Evaluation Metrics

Results reported across prediction horizons:

| Horizon | MAE | RMSE | MAPE (%) |
|---------|-----|------|----------|
| Step 1  | —   | —    | —        |
| Step 2  | —   | —    | —        |
| Step 3  | —   | —    | —        |
| Step 6  | —   | —    | —        |
| Avg 1-6 | —   | —    | —        |

*Full results available in the paper.*

---

## Project Structure

```
ST_FSFusion/
├── data/                    # Dataset files (download separately)
├── models/
│   ├── meta_model.py        # MetaCrossDomainFusion architecture
│   └── adaptive_fsl.py      # Adaptive few-shot learning components
├── datasets.py              # Data loading and preprocessing
├── train.py                 # MCDTrainer (standard + fine-tuning)
├── main.py                  # Entry point + CLI argument parsing
├── rq_analyser.py           # Research question analysis (RQ1-RQ7)
├── utils.py                 # Metrics, logging, plotting utilities
└── config.yaml              # Default configuration
```

---

## Model Configuration

Key settings in `config.yaml`:

```yaml
model:
  type: MetaCrossDomainFusion
  hidden_dim: 64
  meta_dim: 32
  message_dim: 8
  num_heads: 4
  dropout: 0.1
  tp: true                      # Temporal processor
  sp: true                      # Spatial processor
  nuclear_norm_weight: 0.001    # Eq. 4: low-rank regularisation
  lambda1: 0.1                  # Eq. 14: FSL hybrid loss weight 1
  lambda2: 0.01                 # Eq. 14: FSL hybrid loss weight 2
  epsilon: 0.1                  # Eq. 16: proximity constraint
  noise_tau: 0.1                # Eq. 8: noise injection threshold
  noise_alpha: 0.05             # Eq. 8: noise injection scale
  enable_noise: true

training:
  learning_rate: 0.0001
  source_epochs: 200
  target_epochs: 300
  target_days: 3
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{FOFANAH2026100797,
title = {Spatio-temporal feature fusion and few-shot learning framework with cross-city knowledge transfer for traffic flow prediction},
journal = {Energy and AI},
volume = {25},
pages = {100797},
year = {2026},
issn = {2666-5468},
doi = {https://doi.org/10.1016/j.egyai.2026.100797},
url = {https://www.sciencedirect.com/science/article/pii/S2666546826001230},
author = {Abdul Joseph Fofanah and Lian Wen and David Chen and Shaoyang Zhang and Alpha Alimamy Kamara},
keywords = {Spatio-temporal, Feature fusion, Few-shot learning, Graph neural networks, Knowledge transfer, Traffic flow forecasting}
```
