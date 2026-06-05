import os
import json
import yaml
import pickle
import time
import warnings
import numpy as np
import torch

def _get_spatial_metrics(graph, predictions):
    """Compute graph smoothness via mcd_model.compute_spatial_metrics if available."""
    try:
        from models.meta_model import compute_spatial_metrics
        if graph is not None and predictions is not None and predictions.size > 0:
            flow = torch.from_numpy(predictions[:, 0, :]) if predictions.ndim == 3                    else torch.from_numpy(predictions[:, 0])
            adj  = torch.from_numpy(graph) if not isinstance(graph, torch.Tensor) else graph
            if flow.shape[-1] == adj.shape[0]:
                return compute_spatial_metrics(flow, adj)
    except Exception:
        pass
    return {}
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def enhanced_metric_calculation(pred: np.ndarray, y: np.ndarray, times: int) -> Dict[str, np.ndarray]:
    result = {k: np.zeros(times) for k in ['MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE', 'WAPE']}

    def _mape(p, t):
        threshold = max(0.05, float(np.mean(np.abs(t))) * 0.1)
        m = np.abs(t) > threshold
        return float(np.mean(np.abs(t[m] - p[m]) / np.abs(t[m]))) if m.any() else 0.0

    def _smape(p, t):
        d = (np.abs(p) + np.abs(t)) / 2; m = d != 0
        return float(np.mean(np.abs(p[m] - t[m]) / d[m])) if m.any() else 0.0

    def _wape(p, t):
        s = np.sum(np.abs(t))
        return float(np.sum(np.abs(p - t)) / s) if s != 0 else 0.0

    if pred.shape != y.shape:
        n = min(pred.shape[0], y.shape[0])
        pred, y = pred[:n], y[:n]

    for i in range(min(times, 12)):
        if pred.ndim == 3:
            yi, pi = y[:, i, :].flatten(), pred[:, i, :].flatten()
        else:
            yi, pi = y[:, i], pred[:, i]
        mn = min(len(yi), len(pi)); yi, pi = yi[:mn], pi[:mn]
        mask = np.isfinite(yi) & np.isfinite(pi); yi, pi = yi[mask], pi[mask]
        if len(yi) == 0: continue
        result['MSE'][i]   = mean_squared_error(yi, pi)
        result['RMSE'][i]  = np.sqrt(result['MSE'][i])
        result['MAE'][i]   = mean_absolute_error(yi, pi)
        result['MAPE'][i]  = _mape(pi, yi)
        result['SMAPE'][i] = _smape(pi, yi)
        result['WAPE'][i]  = _wape(pi, yi)
    return result


def result_print(metrics: Dict, tag: str = 'Eval'):
    n = min(6, len(metrics.get('MAE', [])))
    print(f"===== {tag} =====")
    print(" MAE:  " + " / ".join(f"{metrics['MAE'][i]:.3f}"       for i in range(n)))
    print("MAPE:  " + " / ".join(f"{metrics['MAPE'][i]*100:.3f}%" for i in range(n)))
    print("RMSE:  " + " / ".join(f"{metrics['RMSE'][i]:.3f}"      for i in range(n)))
    print(f"Avg  MAE={np.mean(metrics['MAE'][:n]):.3f}  "
          f"MAPE={np.mean(metrics['MAPE'][:n])*100:.3f}%  "
          f"RMSE={np.mean(metrics['RMSE'][:n]):.3f}")
    print("-" * 40)


def load_config(path: str = 'config.yaml') -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(cfg: Dict, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if path.endswith(('.yaml', '.yml')):
        with open(path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
    elif path.endswith('.json'):
        with open(path, 'w') as f:
            json.dump(cfg, f, indent=2)
    else:
        with open(path, 'wb') as f:
            pickle.dump(cfg, f)


def set_random_seed(seed: int = 42):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def get_device(s: str = 'auto') -> torch.device:
    if s == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(s)


def create_experiment_dir(base: str, name: str) -> str:
    import datetime
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = os.path.join(os.path.abspath(base), f"{name}_{ts}")
    for sub in ['plots', 'models', 'predictions', 'logs']:
        os.makedirs(os.path.join(exp, sub), exist_ok=True)
    print(f"Experiment dir: {exp}")
    return exp


def fmt_time(s: float) -> str:
    if s < 60:    return f"{s:.0f}s"
    if s < 3600:  return f"{s/60:.1f}m"
    if s < 86400: return f"{s/3600:.1f}h"
    return f"{s/86400:.1f}d"


def save_predictions(predictions: np.ndarray, targets: np.ndarray,
                      metrics: Dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(save_dir, 'targets.npy'),     targets)
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in metrics.items()}, f, indent=2)
    print(f"Saved predictions to: {save_dir}")


class EarlyStopping:
    def __init__(self, patience=30, min_delta=1e-4, restore_best=True, min_epochs=0):
        self.patience      = patience
        self.min_delta     = min_delta
        self.restore_best  = restore_best
        self.min_epochs    = min_epochs
        self.best_score    = float('inf')
        self.counter       = 0
        self.best_weights  = None
        self.current_epoch = 0

    def __call__(self, score, model, epoch=None):
        self.current_epoch = epoch if epoch is not None else self.current_epoch + 1
        if self.current_epoch < self.min_epochs:
            if score < self.best_score - self.min_delta:
                self.best_score   = score
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()} \
                                    if self.restore_best else None
            return False
        if score < self.best_score - self.min_delta:
            self.best_score = score; self.counter = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()} \
                                if self.restore_best else None
        else:
            self.counter += 1
        if self.counter >= self.patience:
            if self.restore_best and self.best_weights:
                dev = next(model.parameters()).device
                model.load_state_dict({k: v.to(dev) for k, v in self.best_weights.items()})
            return True
        return False


class MetricsTracker:
    def __init__(self):
        self.history = {}

    def update(self, metrics: Dict, stage: str = 'train'):
        self.history.setdefault(stage, {})
        for k, v in metrics.items():
            self.history[stage].setdefault(k, []).append(v)


def _mpl():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.size': 15, 'axes.titlesize': 16, 'axes.labelsize': 15,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
        'legend.fontsize': 13, 'figure.titlesize': 17,
    })
    return plt


_C = {
    'blue': '#1D4ED8', 'orange': '#EA580C', 'green': '#15803D',
    'red':  '#DC2626', 'teal':   '#0E7490', 'gray':  '#6B7280',
    'purple': '#6D28D9', 'text': '#1E293B',
    'grid': '#E2E8F0', 'spine': '#CBD5E1',
}
_STEPS = ['5m','10m','15m','20m','25m','30m','35m','40m','45m','50m','55m','60m']


def _ax(ax, plt, title='', xlabel='', ylabel='', grid=True):
    ax.set_facecolor('white')
    ax.tick_params(colors=_C['text'], labelsize=13)
    for sp in ax.spines.values():
        sp.set_edgecolor(_C['spine'])
    if title:  ax.set_title(title,  color=_C['text'], fontweight='bold', pad=10)
    if xlabel: ax.set_xlabel(xlabel, color=_C['text'])
    if ylabel: ax.set_ylabel(ylabel, color=_C['text'])
    if grid:
        ax.grid(True, color=_C['grid'], linewidth=0.8)
        ax.set_axisbelow(True)


def plot_rq1_horizon(metrics: Dict, save_dir: str):
    plt = _mpl()
    n   = min(12, len(metrics.get('MAE', [])))
    if n == 0 or not np.any(np.array(metrics['MAE'][:n]) > 0): return
    steps = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('white')
    fig.suptitle('RQ1 — Prediction Accuracy Across Horizons',
                 color=_C['text'], fontweight='bold', y=1.02)
    for ax, vals, color, ylabel, title in zip(
        axes,
        [metrics['MAE'][:n], metrics['RMSE'][:n], np.array(metrics['MAPE'][:n])*100],
        [_C['blue'], _C['orange'], _C['green']],
        ['MAE (mph)', 'RMSE (mph)', 'MAPE (%)'],
        ['Mean Absolute Error', 'Root Mean Squared Error', 'Mean Absolute % Error'],
    ):
        bars = ax.bar(steps, vals, color=color, alpha=0.85, width=0.6, edgecolor=_C['spine'])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vals)*0.01,
                    f'{v:.2f}', ha='center', va='bottom', color=_C['text'], fontsize=11)
        v = np.array(vals, dtype=float)
        if np.all(np.isfinite(v)) and len(np.unique(v)) > 1:
            z = np.polyfit(steps, v, 1)
            ax.plot(steps, np.poly1d(z)(steps), '--', color=_C['gray'], lw=1.2, alpha=0.7)
        ax.set_xticks(steps); ax.set_xticklabels(_STEPS[:n], rotation=35, ha='right')
        _ax(ax, plt, title=title, ylabel=ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq1_horizon.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[Plot] RQ1 saved")


def plot_rq2_convergence(train_losses: List, val_losses: List, save_dir: str):
    plt = _mpl()
    if not train_losses: return
    ep = np.arange(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(11, 5)); fig.patch.set_facecolor('white')
    ax.plot(ep, train_losses, color=_C['blue'],   lw=2, label='Train', alpha=0.9)
    if val_losses and len(val_losses) == len(train_losses):
        ax.plot(ep, val_losses, color=_C['orange'], lw=2, label='Val', linestyle='--', alpha=0.9)
    ax.fill_between(ep, train_losses, alpha=0.07, color=_C['blue'])
    ax.legend(framealpha=0.85, labelcolor=_C['text'])
    _ax(ax, plt, title='RQ2 — Training Convergence', xlabel='Epoch', ylabel='Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq2_convergence.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[Plot] RQ2 saved")


def plot_rq3_uncertainty(uncertainty: np.ndarray, mae_per_step: np.ndarray, save_dir: str):
    plt = _mpl()
    if uncertainty is None or len(uncertainty) == 0: return
    n = min(len(uncertainty), len(mae_per_step), 12)
    steps = np.arange(n)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); fig.patch.set_facecolor('white')
    fig.suptitle('RQ3 — Uncertainty vs Prediction Error', color=_C['text'], fontweight='bold')
    axes[0].plot(steps, mae_per_step[:n],  color=_C['blue'],   lw=2, marker='o', ms=5, label='MAE')
    axes[0].plot(steps, uncertainty[:n],   color=_C['orange'], lw=2, marker='s', ms=5, label='Uncertainty', linestyle='--')
    axes[0].set_xticks(steps); axes[0].set_xticklabels(_STEPS[:n], rotation=35, ha='right')
    axes[0].legend(framealpha=0.85, labelcolor=_C['text'])
    _ax(axes[0], plt, title='MAE and Uncertainty per Step', ylabel='Value')
    corr = float(np.corrcoef(uncertainty[:n], mae_per_step[:n])[0, 1]) \
           if len(np.unique(uncertainty[:n])) > 1 else 0.0
    axes[1].scatter(uncertainty[:n], mae_per_step[:n], c=_C['blue'], s=80, alpha=0.8, edgecolors=_C['spine'])
    if len(np.unique(uncertainty[:n])) > 1:
        z = np.polyfit(uncertainty[:n], mae_per_step[:n], 1)
        xs = np.linspace(uncertainty[:n].min(), uncertainty[:n].max(), 50)
        axes[1].plot(xs, np.poly1d(z)(xs), '--', color=_C['orange'], lw=1.5)
    axes[1].set_title(f'Uncertainty vs MAE  (r={corr:.3f})', color=_C['text'], fontweight='bold', pad=10)
    _ax(axes[1], plt, xlabel='Uncertainty', ylabel='MAE (mph)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq3_uncertainty.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[Plot] RQ3 saved")


def plot_rq4_meta_knowledge(meta_features: np.ndarray, save_dir: str):
    plt = _mpl()
    if meta_features is None or meta_features.size == 0: return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); fig.patch.set_facecolor('white')
    fig.suptitle('RQ4 — Meta-Knowledge Feature Analysis', color=_C['text'], fontweight='bold')
    mean_cf = meta_features.mean(axis=0) if meta_features.ndim > 1 else meta_features
    axes[0].bar(np.arange(len(mean_cf)), mean_cf, color=_C['teal'], alpha=0.85, edgecolor=_C['spine'])
    _ax(axes[0], plt, title='Mean Meta-Knowledge Values', xlabel='Feature Index', ylabel='Value')
    if meta_features.ndim == 2 and meta_features.shape[0] > 1:
        corr = np.corrcoef(meta_features.T)
        im = axes[1].imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        cb = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=_C['text'], labelsize=12)
        _ax(axes[1], plt, title='Meta-Knowledge Correlation', grid=False)
    else:
        axes[1].text(0.5, 0.5, 'Insufficient samples', ha='center', va='center',
                     color=_C['gray'], transform=axes[1].transAxes)
        _ax(axes[1], plt, title='Meta-Knowledge Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq4_meta_knowledge.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[Plot] RQ4 saved")


def plot_rq5_graph_structure(graph: np.ndarray, save_dir: str):
    plt = _mpl()
    if graph is None or graph.size == 0: return
    sm = _get_spatial_metrics(graph, None)
    rho_str = f"  ρ(f)={sm['graph_smoothness']:.3f}" if 'graph_smoothness' in sm else ''
    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); fig.patch.set_facecolor('white')
    fig.suptitle(f'RQ5 — Learned Graph Structure{rho_str}', color=_C['text'], fontweight='bold')
    im = axes[0].imshow(graph[:50, :50] if graph.shape[0] > 50 else graph,
                        cmap='Blues', aspect='auto')
    cb = plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cb.ax.tick_params(colors=_C['text'], labelsize=12)
    _ax(axes[0], plt, title='Adjacency Matrix (first 50 nodes)', grid=False)
    flat = graph.flatten(); flat = flat[flat > 0]
    if len(flat):
        axes[1].hist(flat, bins=40, color=_C['blue'], alpha=0.8, edgecolor=_C['spine'])
        axes[1].axvline(flat.mean(), color=_C['orange'], lw=1.5, linestyle='--',
                        label=f'Mean: {flat.mean():.3f}')
        axes[1].legend(framealpha=0.85, labelcolor=_C['text'])
    _ax(axes[1], plt, title='Edge Weight Distribution', xlabel='Weight', ylabel='Count')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq5_graph.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[Plot] RQ5 saved")


def plot_rq6_fewshot(pretrain_metrics: Dict, finetune_metrics: Dict, save_dir: str):
    plt = _mpl()
    if not pretrain_metrics or not finetune_metrics: return
    n = min(12, len(pretrain_metrics.get('MAE', [])), len(finetune_metrics.get('MAE', [])))
    if n == 0: return
    steps = np.arange(n)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5)); fig.patch.set_facecolor('white')
    fig.suptitle('RQ6 — Few-Shot Adaptation: Before vs After Fine-Tuning',
                 color=_C['text'], fontweight='bold')
    w = 0.35
    for ax, (key, scale, ylabel) in zip(axes, [('MAE',1,'MAE (mph)'),('RMSE',1,'RMSE (mph)'),('MAPE',100,'MAPE (%)')]):
        pre  = np.array(pretrain_metrics[key][:n]) * scale
        post = np.array(finetune_metrics[key][:n]) * scale
        ax.bar(steps - w/2, pre,  width=w, color=_C['gray'], alpha=0.8, label='Pretrained', edgecolor=_C['spine'])
        ax.bar(steps + w/2, post, width=w, color=_C['blue'], alpha=0.9, label='Fine-tuned',  edgecolor=_C['spine'])
        ax.set_xticks(steps); ax.set_xticklabels(_STEPS[:n], rotation=35, ha='right')
        ax.legend(framealpha=0.85, labelcolor=_C['text'])
        _ax(ax, plt, title=ylabel, ylabel=ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq6_fewshot.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[Plot] RQ6 saved")


def plot_rq7_spatial(predictions: np.ndarray, targets: np.ndarray, save_dir: str):
    plt = _mpl()
    if predictions.size == 0: return

    def _to2d(a):
        if a.ndim == 3:
            if a.shape[2] == 1: return a[:, :, 0]
            if a.shape[1] == 1: return a[:, 0, :]
            return a.mean(axis=2)
        return a

    p, t = _to2d(predictions), _to2d(targets)
    n = min(p.shape[0], t.shape[0]); m = min(p.shape[1], t.shape[1])
    p, t = p[:n, :m], t[:n, :m]
    if p.shape != t.shape or p.size == 0: return
    node_mae = np.mean(np.abs(p - t), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5)); fig.patch.set_facecolor('white')
    fig.suptitle('RQ7 — Spatial Error Distribution', color=_C['text'], fontweight='bold')
    sorted_mae = node_mae[np.argsort(node_mae)]
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(sorted_mae)))
    axes[0].bar(np.arange(len(sorted_mae)), sorted_mae, color=colors, alpha=0.85, width=1.0, edgecolor='none')
    axes[0].axhline(node_mae.mean(), color=_C['orange'], lw=1.5, linestyle='--',
                    label=f'Mean: {node_mae.mean():.2f}')
    axes[0].legend(framealpha=0.85, labelcolor=_C['text'])
    _ax(axes[0], plt, title='Sorted Step MAE', xlabel='Step rank', ylabel='MAE (mph)')
    cdf = np.arange(1, len(sorted_mae) + 1) / len(sorted_mae)
    axes[1].plot(sorted_mae, cdf, color=_C['blue'], lw=2)
    axes[1].fill_between(sorted_mae, cdf, alpha=0.1, color=_C['blue'])
    axes[1].axvline(node_mae.mean(), color=_C['orange'], lw=1.5, linestyle='--',
                    label=f'Mean: {node_mae.mean():.2f}')
    axes[1].legend(framealpha=0.85, labelcolor=_C['text'])
    _ax(axes[1], plt, title='CDF of Step MAE', xlabel='MAE (mph)', ylabel='Fraction')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'rq7_spatial.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[Plot] RQ7 saved")


def plot_rq8_dashboard(metrics: Dict, train_losses: List, val_losses: List,
                        predictions: np.ndarray, targets: np.ndarray, save_dir: str):
    plt = _mpl()
    import matplotlib.gridspec as gsl
    fig = plt.figure(figsize=(18, 11)); fig.patch.set_facecolor('white')
    fig.suptitle('MCD-CKT Research Summary', color=_C['text'], fontweight='bold', y=0.98)
    gs  = gsl.GridSpec(3, 4, figure=fig, hspace=0.52, wspace=0.38)

    ax_a = fig.add_subplot(gs[0, :2])
    n = min(12, len(metrics.get('MAE', [])))
    if n > 0 and np.any(np.array(metrics['MAE'][:n]) > 0):
        s = np.arange(n)
        ax_a.plot(s, metrics['MAE'][:n],  color=_C['blue'],   lw=2, marker='o', ms=4, label='MAE')
        ax_a.plot(s, metrics['RMSE'][:n], color=_C['orange'], lw=2, marker='s', ms=4, label='RMSE')
        ax_a.fill_between(s, metrics['MAE'][:n], metrics['RMSE'][:n], alpha=0.08, color=_C['blue'])
        ax_a.set_xticks(s); ax_a.set_xticklabels(_STEPS[:n], rotation=30, ha='right')
        ax_a.legend(framealpha=0.85, labelcolor=_C['text'])
    _ax(ax_a, plt, 'A — MAE & RMSE per Horizon', ylabel='mph')

    ax_b = fig.add_subplot(gs[0, 2:])
    if n > 0 and np.any(np.array(metrics['MAPE'][:n]) > 0):
        mape = np.array(metrics['MAPE'][:n]) * 100
        mx = mape.max() if mape.max() > 0 else 1
        colors_b = [plt.cm.RdYlGn_r(float(v)/mx) for v in mape]
        bars = ax_b.bar(np.arange(n), mape, color=colors_b, alpha=0.85, edgecolor=_C['spine'])
        for bar, v in zip(bars, mape):
            ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mx*0.02,
                      f'{v:.1f}%', ha='center', va='bottom', color=_C['text'], fontsize=11)
        ax_b.set_xticks(np.arange(n)); ax_b.set_xticklabels(_STEPS[:n], rotation=30, ha='right')
    _ax(ax_b, plt, 'B — MAPE per Horizon', ylabel='MAPE (%)')

    ax_c = fig.add_subplot(gs[1, :2])
    if train_losses:
        ep = np.arange(1, len(train_losses)+1)
        ax_c.plot(ep, train_losses, color=_C['blue'],   lw=1.8, label='Train')
        if val_losses and len(val_losses) == len(train_losses):
            ax_c.plot(ep, val_losses, color=_C['orange'], lw=1.8, label='Val', linestyle='--')
        ax_c.legend(framealpha=0.85, labelcolor=_C['text'])
    _ax(ax_c, plt, 'C — Training Convergence', xlabel='Epoch', ylabel='Loss')

    ax_d = fig.add_subplot(gs[1, 2:])
    if predictions.size > 0 and targets.size > 0:
        def _seq(a):
            if a.ndim == 3: return a[:100, :, 0].flatten()
            return a[:100].flatten()
        ps, ts = _seq(predictions), _seq(targets); l = min(len(ps), len(ts), 500)
        ax_d.plot(np.arange(l), ts[:l], color=_C['teal'],   lw=1.5, label='Target')
        ax_d.plot(np.arange(l), ps[:l], color=_C['orange'], lw=1.5, label='Pred', linestyle='--')
        ax_d.fill_between(np.arange(l), ts[:l], ps[:l], alpha=0.07, color=_C['red'])
        ax_d.legend(framealpha=0.85, labelcolor=_C['text'])
    _ax(ax_d, plt, 'D — Sample Prediction vs Target', xlabel='Time step', ylabel='Speed (mph)')

    ax_e = fig.add_subplot(gs[2, :2]); ax_e.axis('off')
    if n > 0 and np.any(np.array(metrics['MAE'][:n]) > 0):
        h_idx = [i for i in [0, 2, 5, 11] if i < n]
        tdata = [[f"{metrics['MAE'][i]:.3f}", f"{metrics['RMSE'][i]:.3f}",
                  f"{metrics['MAPE'][i]*100:.2f}%"] for i in h_idx]
        tbl = ax_e.table(cellText=tdata, rowLabels=[_STEPS[i] for i in h_idx],
                         colLabels=['MAE', 'RMSE', 'MAPE'],
                         cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        tbl.auto_set_font_size(False); tbl.set_fontsize(12)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor('#DBEAFE' if r == 0 else 'white')
            cell.set_text_props(color='#1E293B'); cell.set_edgecolor(_C['spine'])
    ax_e.set_title('E — Key Metrics', color=_C['text'], fontweight='bold', pad=6)

    ax_f = fig.add_subplot(gs[2, 2:])
    if predictions.size > 0 and targets.size > 0:
        pf, tf = predictions.flatten(), targets.flatten(); l = min(len(pf), len(tf))
        errors = pf[:l] - tf[:l]; errors = errors[np.isfinite(errors)]
        if len(errors):
            ax_f.hist(errors, bins=60, color=_C['blue'], alpha=0.75, edgecolor=_C['spine'], density=True)
            ax_f.axvline(0, color=_C['orange'], lw=1.5, linestyle='--', label='Zero error')
            ax_f.axvline(errors.mean(), color=_C['red'], lw=1.2, linestyle=':',
                         label=f'Bias={errors.mean():.3f}')
            ax_f.legend(framealpha=0.85, labelcolor=_C['text'])
    _ax(ax_f, plt, 'F — Error Distribution', xlabel='Error (mph)', ylabel='Density')

    plt.savefig(os.path.join(save_dir, 'rq8_dashboard.png'), dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("[Plot] RQ8 dashboard saved")


def create_comprehensive_plots(predictions: np.ndarray, targets: np.ndarray,
                               metrics: Dict, train_losses: List, val_losses: List,
                               save_dir: str, uncertainty: np.ndarray = None,
                               graph: np.ndarray = None, meta_features: np.ndarray = None,
                               pretrain_metrics: Dict = None):
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Plots] Generating visualisations in {save_dir}")
    if metrics.get('MAE') is not None and np.any(np.array(metrics.get('MAE', [0])) > 0):
        plot_rq1_horizon(metrics, save_dir)
    if train_losses:
        _vl = val_losses if val_losses and len(val_losses) == len(train_losses) else None
        plot_rq2_convergence(train_losses, _vl, save_dir)
    if uncertainty is not None:
        plot_rq3_uncertainty(uncertainty, np.array(metrics.get('MAE', [])), save_dir)
    if meta_features is not None:
        plot_rq4_meta_knowledge(meta_features, save_dir)
    if graph is not None:
        plot_rq5_graph_structure(graph, save_dir)
    if pretrain_metrics and metrics:
        plot_rq6_fewshot(pretrain_metrics, metrics, save_dir)
    if predictions is not None and predictions.size > 0:
        plot_rq7_spatial(predictions, targets, save_dir)
        plot_rq8_dashboard(metrics, train_losses or [], val_losses or [],
                           predictions, targets, save_dir)
    print("[Plots] Done.")