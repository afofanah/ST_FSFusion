# ST-FSFusion: Spatio-Temporal Few-Shot Fusion for Cross-City Traffic Prediction

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![IEEE IoT Journal](https://img.shields.io/badge/IEEE%20IoT%20Journal-2025-green.svg)](https://ieee.org/)

**A meta-learning framework for accurate traffic flow prediction in new cities using only 3 days of local data.**

[Overview](#overview) • [Architecture](#architecture) • [Results](#results) • [Quick Start](#quick-start) • [Citation](#citation)

</div>

---

## Overview

ST-FSFusion addresses a fundamental challenge in traffic forecasting: deploying accurate prediction models in cities where little historical data is available. By learning transferable spatio-temporal meta-knowledge from data-rich source cities, the framework adapts to new cities in hours rather than months.

### Key Capabilities

| Capability | Detail |
|---|---|
| **Cross-City Transfer** | Transfers meta-knowledge between cities with different road topologies |
| **Few-Shot Adaptation** | Accurate predictions from just 3 days of local data (5–10% of typical requirements) |
| **Adaptive Precision** | Dynamic FP16/FP32 switching for optimal accuracy–efficiency trade-off |
| **Physical Consistency** | Graph reconstruction regularisation preserves traffic flow physics |
| **Robustness** | Maintains accuracy under 20% missing data and sensor noise |

---

## Architecture

ST-FSFusion is composed of five coordinated components:

```
┌─────────────────────────────────────────────────────────────┐
│                    ST-FSFusion Framework                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ST-MKL  ·  Spatio-Temporal Meta-Knowledge Learner         │
│  ├─ Chunked processing with dimension-aligned attention     │
│  ├─ Gated fusion with topological priors                    │
│  └─ Low-rank regularisation for transferable patterns       │
│                                                             │
│  TEWrapper  ·  Temporal Encoding                            │
│  ├─ Dual-path transformer: MetaT (efficient) + RobustT     │
│  ├─ Adaptive precision switching based on graph norm        │
│  └─ Noise injection for sensor-failure robustness           │
│                                                             │
│  CDFusion  ·  Cross-Domain Fusion                           │
│  ├─ Hierarchical cross-domain attention                     │
│  ├─ Spatio-temporal role specialisation                     │
│  └─ Precision-adaptive execution                            │
│                                                             │
│  CAdapter  ·  Cross-City Adapter                            │
│  ├─ Gated cross-attention for selective transfer            │
│  ├─ Similarity-based node matching                          │
│  └─ Adaptive fusion weights                                 │
│                                                             │
│  FSL  ·  Few-Shot Learner                                   │
│  ├─ Traffic-aware binning for prototype learning            │
│  ├─ Graph-aware regularisation                              │
│  └─ Physical constraint preservation                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Results

Evaluated on METR-LA, PEMS-BAY, Chengdu, and Shenzhen datasets across multiple prediction horizons (5–120 min).

| Metric | Improvement over Baselines |
|---|---|
| Prediction accuracy | **8–18%** better MAE/RMSE |
| Inference speed | **33.6%** faster than FP32 |
| Energy consumption | **44.1%** reduction |
| Adaptation time | **94%** less than full retraining |

---

## Project Structure

```
st-fsfusion/
├── config/                    # Configuration files
├── data/                      # Traffic datasets
│   ├── metr-la/
│   ├── pems-bay/
│   ├── chengdu/
│   └── shenzhen/
├── models/                    # Model architectures
│   ├── adaptive_fsl.py        # Main model components
│   ├── st_mkl.py              # Spatio-temporal meta-learner
│   └── tewrapper.py           # Temporal encoder wrapper
├── datasets/                  # Data loaders and preprocessing
├── scripts/                   # Helper scripts
│   ├── download_data.py
│   └── visualize_results.py
├── utils/                     # Utility functions
├── configs/
│   └── default.yaml           # Default hyperparameters
├── train.py                   # Training pipeline
├── main.py                    # Entry point
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/st-fsfusion.git
cd st-fsfusion
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
python scripts/download_data.py --dataset metr-la pems-bay chengdu shenzhen
```

### 3. Train and Evaluate

**Step 1 — Train source model:**
```bash
python main.py --train_source --source_city metr-la
```

**Step 2 — Transfer to a new city:**
```bash
python main.py --train_target --perform_few_shot \
               --source_city metr-la --target_city pems-bay \
               --target_epochs 50 --k_shot 5
```

**Step 3 — Full pipeline in one command:**
```bash
python main.py --config configs/default.yaml \
               --source_city metr-la --target_city pems-bay \
               --train_source --train_target --perform_few_shot
```

---

## Configuration

Edit `configs/default.yaml` to adjust model and training parameters:

```yaml
model_args:
  hidden_dim: 16          # Feature dimension
  meta_lr: 0.01           # Meta-learning rate
  n_heads: 8              # Attention heads
  low_rank_dim: 10        # Meta-knowledge rank

training:
  source_epochs: 100      # Source training epochs
  target_epochs: 50       # Target adaptation epochs
  batch_size: 5
  early_stop_patience: 15

evaluation:
  target_days: 3          # Days of local data for few-shot
  k_shot: 5
  time_horizons: [5, 15, 30, 60, 120]   # Prediction horizons (minutes)
```

---

## Visualisation

```bash
python scripts/visualize_results.py --results_dir ./results
```

---

## Citation

If ST-FSFusion contributes to your research, please cite:

```bibtex
@article{stfsfusion2025,
  title   = {ST-FSFusion: Spatio-Temporal Few-Shot Fusion for Cross-City Traffic Prediction},
  author  = {Fofanah, Abdul Joseph and Wen, Lian and Chen, David and
             Zhang, Shaoyang and Kamara, Alpha Alimamy},
  journal = {IEEE Internet of Things Journal},
  year    = {2025}
}
```

---

## License

Released under the [MIT License](LICENSE).

---

## Contact

| Channel | Detail |
|---|---|
| Email | a.fofanah@griffith.edu.au |
| Bug reports | [Open a GitHub issue](https://github.com/yourusername/st-fsfusion/issues) |

> **Note:** This repository contains research code. Additional testing and optimisation are recommended before production deployment.
