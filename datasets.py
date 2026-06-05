import os
import pickle
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from scipy import stats
from typing import Dict, Tuple

warnings.filterwarnings('ignore')


def get_normalized_adj(A: np.ndarray) -> np.ndarray:
    A = A.astype(np.float32) + np.eye(A.shape[0], dtype=np.float32)
    D = np.maximum(np.sum(A, axis=1), 1e-5).astype(np.float32)
    d = np.reciprocal(np.sqrt(D))
    return np.multiply(np.multiply(d.reshape(-1, 1), A), d.reshape(1, -1)).astype(np.float32)


def robust_traffic_normalization(data: np.ndarray, method: str = 'improved_standard',
                                  epsilon: float = 1e-6, clip_outliers: bool = True,
                                  outlier_threshold: float = 2.5) -> Tuple[np.ndarray, Dict]:
    original_shape = data.shape
    flat = data.reshape(-1, data.shape[-1]) if data.ndim == 3 else data.copy()

    if clip_outliers:
        q1, q3 = np.percentile(flat, [25, 75], axis=0)
        iqr    = q3 - q1
        flat   = np.clip(flat, q1 - outlier_threshold * iqr, q3 + outlier_threshold * iqr)

    out   = np.zeros_like(flat, dtype=np.float32)
    nstats = {}

    for fi in range(flat.shape[-1]):
        fd = flat[:, fi]
        if method == 'improved_standard':
            mu  = np.mean(fd); sig = np.std(fd) or 1.0
            nf  = (fd - mu) / sig
            nstats[f'f{fi}'] = {'mean': float(mu), 'std': float(sig)}
        elif method == 'robust_standard':
            med = np.median(fd); mad = np.median(np.abs(fd - med))
            scale = max(mad * 1.4826, epsilon)
            nf = (fd - med) / scale
            nstats[f'f{fi}'] = {'mean': float(med), 'std': float(scale)}
        else:
            mu  = np.mean(fd); sig = np.std(fd) or 1.0
            nf  = (fd - mu) / sig
            nstats[f'f{fi}'] = {'mean': float(mu), 'std': float(sig)}
        out[:, fi] = np.clip(nf, -10.0, 10.0).astype(np.float32)

    if data.ndim == 3:
        out = out.reshape(original_shape)
    return out, nstats


def choose_normalization_method(data: np.ndarray) -> str:
    sample = data[:min(1000, data.shape[0]), :, 0].flatten() if data.ndim == 3 \
             else data.flatten()[:10000]
    if np.mean(sample == 0) > 0.15:
        return 'robust_standard'
    mu, sig = np.mean(sample), np.std(sample)
    if abs(mu) > 3 * sig:
        return 'improved_standard'
    q25, q75 = np.percentile(sample, [25, 75])
    iqr = q75 - q25
    if iqr > 0 and np.mean((sample < q25 - 3*iqr) | (sample > q75 + 3*iqr)) > 0.1:
        return 'robust_standard'
    return 'improved_standard'


def load_traffic_data(dataset_path: str, adjacency_path: str,
                       normalize: bool = True, dataset_config: dict = None) -> dict:
    os.makedirs('cache', exist_ok=True)
    cache_key  = (f"mcd_{os.path.basename(dataset_path)}_"
                  f"{os.path.basename(adjacency_path)}_{normalize}_v2")
    cache_path = os.path.join('cache', f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    traffic  = np.load(dataset_path).astype(np.float32)
    adj      = np.load(adjacency_path).astype(np.float32)

    # Coerce to (T, N, F)
    if traffic.ndim == 3:
        s = traffic.shape
        if s[0] >= s[1] and s[0] >= s[2]:
            pass
        elif s[2] >= s[0] and s[2] >= s[1]:
            traffic = traffic.transpose(2, 0, 1)
        elif s[1] >= s[0] and s[1] >= s[2]:
            traffic = traffic.transpose(1, 0, 2)
        else:
            traffic = traffic.transpose(2, 0, 1)
    elif traffic.ndim == 2:
        traffic = traffic[:, :, np.newaxis]
    else:
        raise ValueError(f"Unexpected shape: {traffic.shape}")

    # Paper §4.2 — linear interpolation for missing values before normalisation
    if not np.all(np.isfinite(traffic)):
        T_, N_, F_ = traffic.shape
        for n in range(N_):
            for f in range(F_):
                col = traffic[:, n, f]
                nan_mask = ~np.isfinite(col)
                if nan_mask.any() and not nan_mask.all():
                    idx = np.arange(T_)
                    traffic[:, n, f] = np.interp(idx, idx[~nan_mask], col[~nan_mask])
    finite_mask = np.isfinite(traffic)
    p1  = np.percentile(traffic[finite_mask], 1)
    p99 = np.percentile(traffic[finite_mask], 99)
    traffic = np.nan_to_num(traffic, nan=0.0, posinf=p99, neginf=p1)

    num_timesteps, num_nodes, num_features = traffic.shape
    print(f"  Loaded {os.path.basename(dataset_path)}: {traffic.shape}  "
          f"mean={np.mean(traffic):.4f}  std={np.std(traffic):.4f}")

    # Prefer config speed_mean/speed_std for exact denorm
    if dataset_config and 'speed_mean' in dataset_config and 'speed_std' in dataset_config:
        denorm_mean = float(dataset_config['speed_mean'])
        denorm_std  = float(dataset_config['speed_std'])
    else:
        denorm_mean = float(np.mean(traffic[:, :, 0]))
        denorm_std  = float(np.std(traffic[:, :, 0]))

    if normalize:
        method  = choose_normalization_method(traffic)
        traffic, norm_stats = robust_traffic_normalization(
            traffic, method=method, clip_outliers=True, outlier_threshold=2.5)
        print(f"  Normalized ({method}): mean={np.mean(traffic):.4f}  std={np.std(traffic):.4f}")

    result = {
        'data':        traffic,
        'adjacency':   adj,
        'num_nodes':   num_nodes,
        'num_timesteps': num_timesteps,
        'num_features':  num_features,
        'raw_mean':    denorm_mean,
        'raw_std':     denorm_std,
    }
    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    return result


def build_edge_index(adj: np.ndarray, sigma: float = 1.0) -> tuple:
    """
    Build edge_index and edge_weight using exponential kernel (paper dataset §4.2):
        A_ij = exp(-d²/σ²)  where d_ij = 1 - adj_ij (adj already in [0,1])
    """
    rows, cols = np.nonzero(adj)
    raw_w      = adj[rows, cols].astype(np.float32)
    # Exponential kernel: treat raw weight as similarity → convert to distance
    distances  = 1.0 - raw_w.clip(0, 1)
    weights    = np.exp(-(distances ** 2) / (sigma ** 2 + 1e-8)).astype(np.float32)
    if len(rows) > 50_000:
        top = np.argsort(weights)[-50_000:]
        rows, cols, weights = rows[top], cols[top], weights[top]
    ei = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    ew = torch.tensor(weights, dtype=torch.float32)
    return ei, ew


class MCDTrafficDataset(Dataset):
    def __init__(self, traffic_data: np.ndarray, adj: np.ndarray,
                 his_num: int = 12, pred_num: int = 12):
        self.traffic_data = traffic_data
        self.adj          = adj
        self.his_num      = his_num
        self.pred_num     = pred_num

        T, N, F = traffic_data.shape
        self.num_nodes    = N
        self.num_features = F
        self.num_sequences = T - his_num - pred_num + 1

        if self.num_sequences <= 0:
            raise ValueError(f"Not enough data: T={T} his={his_num} pred={pred_num}")

        self.edge_index, self.edge_weight = build_edge_index(adj)
        # raw_adj stored for graph_recon_loss in FewShotTrafficLearner (paper Eq. 15)
        self.raw_adj   = torch.from_numpy((adj / (adj.max() + 1e-8)).astype(np.float32))
        print(f"  Edge index: {self.edge_index.shape[1]} edges, {N} nodes")

        self.raw_mean: float = 0.0
        self.raw_std:  float = 1.0
        # Sensor positions for compute_spatial_metrics (§4.5); set externally if available
        self.sensor_positions = None

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Data:
        # x: (N, his_num, F) — model expects (batch, node, seq, feature)
        x = self.traffic_data[idx:idx + self.his_num]           # (T, N, F)
        x = torch.from_numpy(x.transpose(1, 0, 2)).float()      # (N, T, F)

        y = self.traffic_data[idx + self.his_num:
                               idx + self.his_num + self.pred_num, :, 0]  # (pred, N)
        y = torch.from_numpy(y).float()                          # (pred_num, N)

        return Data(
            x           = x,                                      # (N, T, F)
            y           = y,                                      # (pred_num, N)
            edge_index  = self.edge_index,
            edge_weight = self.edge_weight,                       # (E,) kernel weights
            adjacency   = self.raw_adj,                           # (N, N) for graph_recon_loss
            num_nodes   = self.num_nodes,
        )


def collate_pyg(batch):
    xs  = torch.stack([b.x         for b in batch], dim=0)   # (B, N, T, F)
    ys  = torch.stack([b.y         for b in batch], dim=0)   # (B, pred_num, N)
    # Shared topology (same for all samples in dataset)
    ei  = batch[0].edge_index
    ew  = batch[0].edge_weight
    adj = batch[0].adjacency                                  # (N, N)
    return Data(x=xs, y=ys, edge_index=ei, edge_weight=ew,
                adjacency=adj, num_nodes=batch[0].num_nodes)


class DataManager:
    def __init__(self, data_args: dict, task_args: dict):
        self.data_args   = data_args
        self.task_args   = task_args
        self.datasets    = {}
        self._cache      = {}
        self.actual_num_features: int = 1

    def _load(self, name: str) -> dict:
        if name in self._cache:
            return self._cache[name]
        cfg  = self.data_args[name]
        info = load_traffic_data(
            cfg['dataset_path'], cfg['adjacency_matrix_path'],
            normalize=True, dataset_config=cfg)
        self._cache[name] = info
        return info

    def create_all_dataloaders(self, test_data: str, target_days: int = 3,
                               minibatch_size: int = 4096) -> Dict:
        info    = self._load(test_data)
        traffic = info['data']
        adj     = get_normalized_adj(info['adjacency'])
        T       = traffic.shape[0]

        his  = self.task_args.get('his_num', self.task_args.get('seq_len', 12))
        pred = self.task_args['pred_num']
        mbs  = int(self.task_args.get('model_batch_size', 8))

        train_end = int(0.7 * T)
        val_end   = int(0.8 * T)
        tgt_steps = min(target_days * 288, train_end)

        def _ds(d):
            ds = MCDTrafficDataset(d, adj, his, pred)
            ds.raw_mean = info['raw_mean']
            ds.raw_std  = info['raw_std']
            return ds

        src_ds  = _ds(traffic[:train_end])
        val_ds  = _ds(traffic[train_end:val_end])
        tgt_ds  = _ds(traffic[:tgt_steps])
        tst_ds  = _ds(traffic[val_end:])

        # Pass sensor positions if stored (for compute_spatial_metrics §4.5)
        sensor_pos = info.get('sensor_positions', None)
        for k, ds in [('source', src_ds), ('validation', val_ds),
                       ('target', tgt_ds),  ('test', tst_ds)]:
            if sensor_pos is not None:
                ds.sensor_positions = torch.from_numpy(sensor_pos.astype(np.float32))
            self.datasets[k] = ds
            print(f"  {k:12s}: {len(ds):6d} sequences, nodes={ds.num_nodes}, features={ds.num_features}")

        self.actual_num_features = src_ds.num_features

        kw = dict(num_workers=0, pin_memory=torch.cuda.is_available(),
                  collate_fn=collate_pyg)

        src_bs = min(mbs, max(1, len(src_ds)))
        val_bs = min(mbs, max(1, len(val_ds)))
        tgt_bs = min(mbs, max(1, len(tgt_ds)))
        tst_bs = min(mbs, max(1, len(tst_ds)))

        print(f"  model_bs={mbs} | "
              f"src_batches={max(1, len(src_ds)//src_bs)} | "
              f"val_batches={max(1, len(val_ds)//val_bs)}")

        return {
            'source':     DataLoader(src_ds, batch_size=src_bs, shuffle=True,  **kw),
            'validation': DataLoader(val_ds, batch_size=val_bs, shuffle=False, **kw),
            'target':     DataLoader(tgt_ds, batch_size=tgt_bs, shuffle=True,  **kw),
            'test':       DataLoader(tst_ds, batch_size=tst_bs, shuffle=False, **kw),
        }

    def get_dataset_statistics(self) -> dict:
        return {k: {'num_sequences': len(ds), 'num_nodes': ds.num_nodes,
                    'num_features': ds.num_features}
                for k, ds in self.datasets.items()}