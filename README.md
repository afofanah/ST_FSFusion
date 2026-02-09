```markdown
# ST-FSFusion: Spatio-Temporal Few-Shot Fusion for Cross-City Traffic Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

A novel meta-learning framework for accurate traffic flow prediction in new cities with minimal adaptation data (only 3 days). Achieves state-of-the-art performance through coordinated spatio-temporal feature learning and cross-city knowledge transfer.

## 🚀 Key Features

- **Cross-City Transfer**: Meta-knowledge learning enables effective transfer between cities with different topologies
- **Few-Shot Adaptation**: Achieves accurate predictions with only 5-10% of typical training data (3 days)
- **Adaptive Precision**: Dynamic FP16/FP32 switching for optimal accuracy-efficiency trade-off
- **Physical Consistency**: Graph reconstruction regularization preserves traffic flow physics
- **Hierarchical Architecture**: Coordinated components for robust cross-domain processing

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ST-FSFusion Framework                     │
├─────────────────────────────────────────────────────────────┤
│  ST-MKL (Meta-Knowledge Learning)                           │
│  ├─ Chunked processing with dimension-aligned attention     │
│  ├─ Gated fusion with topological priors                    │
│  └─ Low-rank regularization for transferable patterns       │
│                                                             │
│  TEWrapper (Temporal Encoding)                              │
│  ├─ Dual-path transformer: MetaT (efficient) & RobustT      │
│  ├─ Adaptive precision switching based on graph norm        │
│  └─ Noise injection for sensor failure robustness           │
│                                                             │
│  CDFusion (Cross-Domain Fusion)                             │
│  ├─ Hierarchical cross-domain attention                     │
│  ├─ Spatio-temporal role specialization                     │
│  └─ Precision-adaptive execution                            │
│                                                             │
│  CAdapter (Cross-City Adapter)                              │
│  ├─ Gated cross-attention for selective transfer            │
│  ├─ Similarity-based node matching                          │
│  └─ Adaptive fusion weights                                 │
│                                                             │
│  FSL (Few-Shot Learner)                                     │
│  ├─ Traffic-aware binning for prototype learning            │
│  ├─ Graph-aware regularization                              │
│  └─ Physical constraint preservation                        │
└─────────────────────────────────────────────────────────────┘
```

## 📊 Performance Highlights

- **Accuracy**: Outperforms baselines by 8-18% on METR-LA, PEMS-BAY, Chengdu, and Shenzhen datasets
- **Efficiency**: 33.6% faster inference than FP32 with 44.1% energy reduction
- **Adaptation**: 94% reduction in adaptation time compared to full retraining
- **Robustness**: Maintains accuracy under 20% missing data and noisy conditions

## 📁 Project Structure

```
st-fsfusion/
├── config/                    # Configuration files
├── data/                      # Traffic datasets
│   ├── metr-la/
│   ├── pems-bay/
│   ├── chengdu/
│   └── shenzhen/
├── models/                    # Model architectures
│   ├── adaptive_fsl.py       # Main model components
│   ├── st_mkl.py            # Spatio-temporal meta-learner
│   └── tewrapper.py         # Temporal encoder wrapper
├── datasets/                  # Data loaders and preprocessing
├── utils/                     # Utility functions
├── train.py                  # Training pipeline
├── main.py                   # Main entry point (provided above)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/yourusername/st-fsfusion.git
cd st-fsfusion
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
# Download and preprocess traffic datasets
python scripts/download_data.py --dataset metr-la pems-bay chengdu shenzhen
```

### 3. Train and Evaluate

```bash
# Train source model on METR-LA
python main.py --train_source --source_city metr-la

# Transfer to PEMS-BAY with few-shot adaptation
python main.py --train_target --perform_few_shot \
               --source_city metr-la --target_city pems-bay \
               --target_epochs 50 --k_shot 5
```

### 4. Run Complete Pipeline

```bash
python main.py --config configs/default.yaml \
               --source_city metr-la --target_city pems-bay \
               --train_source --train_target --perform_few_shot
```

## ⚙️ Configuration

Modify `configs/default.yaml` to adjust model parameters:

```yaml
model_args:
  hidden_dim: 16          # Feature dimension
  meta_lr: 0.01          # Meta-learning rate
  n_heads: 8             # Attention heads
  low_rank_dim: 10       # Meta-knowledge rank

training:
  source_epochs: 100     # Source training epochs
  target_epochs: 50      # Target adaptation epochs
  batch_size: 5          # Batch size
  early_stop_patience: 15 # Early stopping patience

evaluation:
  target_days: 3         # Days for few-shot adaptation
  k_shot: 5             # Few-shot samples
  time_horizons: [5, 15, 30, 60, 120] # Prediction horizons
```

## 📈 Results Visualization

```python
# Generate performance plots
python scripts/visualize_results.py --results_dir ./results
```

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{stfsfusion2025,
  title={ST-FSFusion: Spatio-Temporal Few-Shot Fusion for Cross-City Traffic Prediction},
  author={Abdul Joseph Fofanah, Lian Wen, David Chen, Shaoyang
Zhang, and Alpha Alimamy Kamara},
  journal={IEEE Internet of Things Journal},
  year={2025}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or issues, please:
- Open an issue on GitHub
- Contact: a.fofanah@griffith.edu.au

---

**Note**: This repository contains research code. For production deployment, additional testing and optimization are recommended.
```
