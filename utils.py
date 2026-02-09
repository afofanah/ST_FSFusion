import os
import zipfile
import numpy as np
import torch
import time
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional
from scipy.stats import pearsonr
from dtaidistance import dtw
from scipy.spatial.distance import euclidean

def safe_divide(a, b, eps=1e-10):
    return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > eps)

def spatial_correlation(pred, y, adj_matrix=None):
    pred = np.squeeze(pred)
    y = np.squeeze(y)
    if pred.ndim > 2:
        pred = pred.reshape(-1, pred.shape[-1])
    if y.ndim > 2:
        y = y.reshape(-1, y.shape[-1])
    corr_coeffs = []
    for i in range(pred.shape[0]):
        pred_node = pred[i]
        y_node = y[i]
        if np.allclose(pred_node, pred_node[0]) or np.allclose(y_node, y_node[0]):
            continue
        if np.array_equal(pred_node, y_node):
            pred_node = pred_node + np.random.normal(0, 1e-8, pred_node.shape)
        
        r, _ = pearsonr(pred_node, y_node)
        
        if not np.isnan(r):
            corr_coeffs.append(r)
    return np.mean(corr_coeffs) if corr_coeffs else 0.0

def dtw_distance(x, y):
    dtw_matrix = np.zeros((len(x) + 1, len(y) + 1))
    
    for i in range(1, len(x) + 1):
        dtw_matrix[i][0] = float('inf')
    for j in range(1, len(y) + 1):
        dtw_matrix[0][j] = float('inf')
    
    dtw_matrix[0][0] = 0
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            cost = euclidean(x[i-1], y[j-1])
            dtw_matrix[i][j] = cost + min(
                dtw_matrix[i-1][j],
                dtw_matrix[i][j-1],
                dtw_matrix[i-1][j-1]
            )
    return dtw_matrix[len(x)][len(y)]

def temporal_alignment_metric(pred, y):
    pred = np.squeeze(pred)
    y = np.squeeze(y)

    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    if y.ndim == 1:
        y = y.reshape(1, -1)
    
    if pred.shape != y.shape:
        raise ValueError(f"Pred and y shapes must match. Got {pred.shape} and {y.shape}")
    
    distances = []
    for i in range(pred.shape[0]):
        pred_seq = pred[i].astype(np.float64)
        y_seq = y[i].astype(np.float64)
        
        pred_seq = (pred_seq - np.mean(pred_seq)) / (np.std(pred_seq) + 1e-10)
        y_seq = (y_seq - np.mean(y_seq)) / (np.std(y_seq) + 1e-10)
        
        distance = dtw_distance(pred_seq.reshape(-1, 1), y_seq.reshape(-1, 1))
        distances.append(distance)
    return np.mean(distances) if distances else np.inf

def feature_alignment_metric(fused_feat, src_feat, tgt_feat):
    def cosine_sim(a, b):
        if isinstance(a, torch.Tensor):
            a = a.detach().cpu().numpy()
        if isinstance(b, torch.Tensor):
            b = b.detach().cpu().numpy()

        a = a.reshape(-1)
        b = b.reshape(-1)
        
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
    
    return {
        'src_alignment': cosine_sim(fused_feat, src_feat),
        'tgt_alignment': cosine_sim(fused_feat, tgt_feat),
        'transfer_gap': cosine_sim(src_feat, tgt_feat)
    }

def graph_consistency_metrics(pred_adj, true_adj):
    if isinstance(pred_adj, torch.Tensor):
        pred_adj = pred_adj.detach().cpu().numpy()
    if isinstance(true_adj, torch.Tensor):
        true_adj = true_adj.detach().cpu().numpy()
    pred_adj = (pred_adj > 0.5).astype(float)
    true_adj = (true_adj > 0.5).astype(float)
    tp = np.sum(pred_adj * true_adj)
    fp = np.sum(pred_adj * (1 - true_adj))
    fn = np.sum((1 - pred_adj) * true_adj)
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    return {
        'edge_precision': precision,
        'edge_recall': recall,
        'edge_f1': f1
    }

def physical_constraint_violation(pred, capacities):
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(capacities, torch.Tensor):
        capacities = capacities.detach().cpu().numpy()
    violations = np.sum(pred > capacities[:, None], axis=1)
    return np.mean(violations) * 100

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_size_mb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def get_normalized_adj(A):
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    
    return A_wave

def generate_dataset(X, num_timesteps_input, num_timesteps_output, means, stds):
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j] * stds[0] + means[0])
    return np.array(features), np.array(target)

def metric_func(pred, y, times=None, adj_matrix=None, capacities=None, 
                features=None, rq_flags=None):
    if rq_flags is None:
        rq_flags = {f'rq{i}': False for i in range(1, 6)}
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()
    if pred.ndim == 2:
        pred = pred[:, np.newaxis, :]
    if y.ndim == 2:
        y = y[:, np.newaxis, :]
    if y.shape[0] > pred.shape[0] and y.shape[1:] == pred.shape[1:]:
        y = y[:pred.shape[0]]
    assert pred.shape == y.shape, f"Shape mismatch: pred {pred.shape}, y {y.shape}"
    if times is None:
        times = pred.shape[1]
    else:
        times = min(times, pred.shape[1])
    
    result = {
        'MSE': np.zeros(times),
        'RMSE': np.zeros(times),
        'MAE': np.zeros(times),
        'MAPE': np.zeros(times),
        'R2': np.zeros(times),
        'PCC': np.zeros(times),
        'SMAPE': np.zeros(times)
    }
    
    if rq_flags.get('rq1', False):
        result['SPATIAL_CORR'] = np.zeros(times)
    
    if rq_flags.get('rq2', False):
        result['DTW_DIST'] = np.zeros(times)
    
    if rq_flags.get('rq4', False) and capacities is not None:
        result['PHYS_VIOLATION'] = np.zeros(times)

    def cal_MAPE(pred, y, threshold=0.0):
        mask = np.abs(y) > threshold
        if not np.any(mask):
            return np.nan
        return np.mean(np.abs((y[mask] - pred[mask]) / y[mask]))
    
    def cal_SMAPE(pred, y, threshold=0.0):
        mask = (np.abs(y) + np.abs(pred)) > threshold
        if not np.any(mask):
            return np.nan
        return 2.0 * np.mean(np.abs(pred[mask] - y[mask]) / (np.abs(pred[mask]) + np.abs(y[mask])))

    for i in range(times):
        pred_i = pred[:, i, :]
        y_i = y[:, i, :]
        if np.isnan(pred_i).any() or np.isnan(y_i).any():
            for key in result:
                result[key][i] = np.nan
            continue
        
        mse = mean_squared_error(y_i, pred_i)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_i, pred_i)
        mape = cal_MAPE(pred_i, y_i)
        smape = cal_SMAPE(pred_i, y_i)
        
        y_flat = y_i.reshape(-1)
        pred_flat = pred_i.reshape(-1)
        r2 = r2_score(y_flat, pred_flat)
        pcc, _ = pearsonr(y_flat, pred_flat)
        result['MSE'][i] = mse
        result['RMSE'][i] = rmse
        result['MAE'][i] = mae
        result['MAPE'][i] = mape
        result['SMAPE'][i] = smape
        result['R2'][i] = r2
        result['PCC'][i] = pcc
        
        if rq_flags.get('rq1', False):
            result['SPATIAL_CORR'][i] = spatial_correlation(pred_i.T, y_i.T, adj_matrix)
        
        if rq_flags.get('rq2', False):
            result['DTW_DIST'][i] = temporal_alignment_metric(pred_i.T, y_i.T)
        
        if rq_flags.get('rq4', False) and capacities is not None:
            result['PHYS_VIOLATION'][i] = physical_constraint_violation(pred_i, capacities)
    
    if rq_flags.get('rq3', False) and features is not None:
        result['FEATURE_ALIGNMENT'] = feature_alignment_metric(
            features['fused'], features['src'], features['tgt']
        )
    
    return result

def compute_horizon_metrics(pred, y, horizons):
    results = {}
    
    for h in horizons:
        if h >= pred.shape[1] or h >= y.shape[1]:
            continue

        pred_h = pred[:, h:h+1, :]
        y_h = y[:, h:h+1, :]
        metrics = metric_func(pred_h, y_h, times=1)
        results[h] = metrics
    
    return results

def result_print(result, info_name='Evaluate', time_horizons=None, rq_flags=None):
    if not time_horizons:
        time_horizons = [5, 10, 15, 30, 60, 120]
    
    if rq_flags is None:
        rq_flags = {f'rq{i}': False for i in range(1,6)}

    core_metrics = ['MAE', 'MAPE', 'RMSE', 'MSE', 'R2']
    for metric in core_metrics:
        if metric not in result:
            result[metric] = np.zeros(len(time_horizons))
    for k in result:
        if k == 'FEATURE_ALIGNMENT':  
            continue
            
        if not isinstance(result[k], np.ndarray):
            result[k] = np.array(result[k])
            
        if len(result[k]) < len(time_horizons):
            result[k] = np.pad(result[k], (0, len(time_horizons) - len(result[k])), 
                              'constant', constant_values=np.nan)
    
    print(f"\n========== {info_name} RESULTS ==========")
    print("\n[Core Prediction Metrics]")
    
    metrics_text = "MAE: " + "/".join([f"{m:.3f}" for m in result['MAE'][:len(time_horizons)]])
    print(metrics_text)
    mape_text = "MAPE: " + "/".join([f"{m*100:.1f}%" for m in result['MAPE'][:len(time_horizons)]])
    print(mape_text)
    rmse_text = "RMSE: " + "/".join([f"{m:.3f}" for m in result['RMSE'][:len(time_horizons)]])
    print(rmse_text)
  
    if 'R2' in result:
        r2_text = "R2: " + "/".join([f"{m:.3f}" for m in result['R2'][:len(time_horizons)]])
        print(r2_text)
    
    print("Horizons: " + "/".join([f"{h}min" for h in time_horizons]))
    
    print("\n[Average Metrics]")
    print(f"Avg MAE:  {np.nanmean(result['MAE']):.3f} ± {np.nanstd(result['MAE']):.3f}")
    print(f"Avg RMSE: {np.nanmean(result['RMSE']):.3f} ± {np.nanstd(result['RMSE']):.3f}")
    print(f"Avg MAPE: {np.nanmean(result['MAPE'])*100:.1f}% ± {np.nanstd(result['MAPE'])*100:.1f}%")
    
    if rq_flags.get('rq1', False) and 'SPATIAL_CORR' in result:
        print("\n[RQ1: Spatial Dependency]")
        print(f"Spatial Correlation: {np.nanmean(result['SPATIAL_CORR']):.3f} ± {np.nanstd(result['SPATIAL_CORR']):.3f}")
    
    if rq_flags.get('rq2', False) and 'DTW_DIST' in result:
        print("\n[RQ2: Temporal Alignment]")
        print(f"DTW Distance: {np.nanmean(result['DTW_DIST']):.3f} ± {np.nanstd(result['DTW_DIST']):.3f}")
    
    if rq_flags.get('rq3', False) and 'FEATURE_ALIGNMENT' in result:
        print("\n[RQ3: Feature Fusion]")
        align = result['FEATURE_ALIGNMENT']
        print(f"Source Alignment: {align['src_alignment']:.3f}")
        print(f"Target Alignment: {align['tgt_alignment']:.3f}")
        print(f"Transfer Gap: {align['transfer_gap']:.3f}")
    
    if rq_flags.get('rq4', False) and 'PHYS_VIOLATION' in result:
        print("\n[RQ4: Physical Consistency]")
        print(f"Constraint Violations: {np.nanmean(result['PHYS_VIOLATION']):.1f}% ± {np.nanstd(result['PHYS_VIOLATION']):.1f}%")
    
    print("---------------------------------------")

    return f"{metrics_text}\n{mape_text}\n{rmse_text}"

def result_prints(result, info_name='Evaluate', time_horizons=None):
    if not time_horizons:
        time_horizons = [5, 10, 15, 30, 60, 120]
    
    metrics_text = "MAE: " + "/".join([f"{m:.3f}" for m in result['MAE'][:len(time_horizons)]])
    print(metrics_text)
    mape_text = "MAPE: " + "/".join([f"{m*100:.1f}%" for m in result['MAPE'][:len(time_horizons)]])
    print(mape_text)
    rmse_text = "RMSE: " + "/".join([f"{m:.3f}" for m in result['RMSE'][:len(time_horizons)]])
    print(rmse_text)
  
    if 'R2' in result:
        r2_text = "R2: " + "/".join([f"{m:.3f}" for m in result['R2'][:len(time_horizons)]])
        print(r2_text)

    total_MSE, total_RMSE, total_MAE, total_MAPE = result['MSE'], result['RMSE'], result['MAE'], result['MAPE']
    if len(total_MAE) < 6 or len(total_RMSE) < 6 or len(total_MAPE) < 6 or len(total_MSE) < 6:
        min_length = min(len(total_MAE), len(total_RMSE), len(total_MAPE), len(total_MSE))
        
        print(f"========== {info_name} results ==========")
        mae_str = "/ ".join([f"{mae:.3f}" for mae in total_MAE[:min_length]])
        mape_str = "/ ".join([f"{mape*100:.3f}" for mape in total_MAPE[:min_length]])
        rmse_str = "/ ".join([f"{rmse:.3f}" for rmse in total_RMSE[:min_length]])
        print(f" MAE: {mae_str}")
        print(f"MAPE: {mape_str}")
        print(f"RMSE: {rmse_str}")
    else:
        print(f"========== {info_name} results ==========")
        print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
        print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
        print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
    
    print("---------------------------------------")

    if info_name == 'Best':
        print("========== Best results ==========")
        if len(total_MAE) < 6:
            min_length = min(len(total_MAE), len(total_RMSE), len(total_MAPE))
            mae_str = "/ ".join([f"{mae:.3f}" for mae in total_MAE[:min_length]])
            mape_str = "/ ".join([f"{mape*100:.3f}" for mape in total_MAPE[:min_length]])
            rmse_str = "/ ".join([f"{rmse:.3f}" for rmse in total_RMSE[:min_length]])
            print(f" MAE: {mae_str}")
            print(f"MAPE: {mape_str}")
            print(f"RMSE: {rmse_str}")
        else:
            print(" MAE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAE[0], total_MAE[1], total_MAE[2], total_MAE[3], total_MAE[4], total_MAE[5]))
            print("MAPE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_MAPE[0] * 100, total_MAPE[1] * 100, total_MAPE[2] * 100, total_MAPE[3] * 100, total_MAPE[4] * 100, total_MAPE[5] * 100))
            print("RMSE: %.3f/ %.3f/ %.3f/ %.3f/ %.3f/ %.3f"%(total_RMSE[0], total_RMSE[1], total_RMSE[2], total_RMSE[3], total_RMSE[4], total_RMSE[5]))
        print("---------------------------------------")
    
    print("Horizons: " + "/".join([f"{h}min" for h in time_horizons]))
    print("---------------------------------------")
    
    return f"{metrics_text}\n{mape_text}\n{rmse_text}"

def load_data(dataset_name, stage, cache_dir=None):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"{dataset_name}_{stage}.npz")
        if os.path.exists(cache_file):
            cached = np.load(cache_file)
            return (
                torch.from_numpy(cached['A']), 
                cached['X'], 
                cached['means'], 
                cached['stds']
            )
    A = np.load(f"data/{dataset_name}/matrix.npy")
    
    A = get_normalized_adj(A)
    A = torch.from_numpy(A)
    X = np.load(f"data/{dataset_name}/dataset.npy")
    
    X = X.transpose((1, 2, 0))
    X = X.astype(np.float32)
    means = np.mean(X, axis=(0, 2))
    stds = np.std(X, axis=(0, 2))
    stds[stds < 1e-10] = 1.0  
    X = (X - means.reshape(1, -1, 1)) / stds.reshape(1, -1, 1)
    if stage == 'train':
        X = X[:, :, :int(X.shape[2]*0.7)]
    elif stage == 'validation':
        X = X[:, :, int(X.shape[2]*0.7):int(X.shape[2]*0.8)]
    elif stage == 'test':
        X = X[:, :, int(X.shape[2]*0.8):]
    elif stage == 'source':
        X = X
    elif stage.startswith('target_'):
        if stage == 'target_1day':
            days = 1
        elif stage == 'target_3day':
            days = 3
        elif stage == 'target_1week':
            days = 7
        else:
            days = int(stage.split('_')[1].replace('day', ''))
    
        X = X[:, :, :288 * days]
    else:
        raise ValueError(f"Error: unsupported data stage {stage}")

    if cache_dir:
        np.savez(
            cache_file, 
            A=A.numpy(), 
            X=X, 
            means=means, 
            stds=stds
        )

    return A, X, means, stds

def measure_inference_time(model, data, device, num_runs=100):
    model.eval()
    if hasattr(data, 'to'):
        data = data.to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(data)
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(data)
    
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - start_time) * 1000 / num_runs
    return avg_time

def plot_prediction_results(pred, y, node_idx=0, time_horizons=None, save_path=None):
    if not time_horizons:
        time_horizons = [5, 10, 15, 30, 60, 120]

    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, horizon in enumerate(time_horizons[:min(6, len(time_horizons))]):
        ax = axes[i]
        
        horizon_idx = i
        if horizon_idx >= pred.shape[1]:
            continue
        y_horizon = y[:, horizon_idx, node_idx]
        pred_horizon = pred[:, horizon_idx, node_idx]
        
        ax.plot(y_horizon, 'b-', label='Ground Truth')
        ax.plot(pred_horizon, 'r-', label='Prediction')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Traffic Flow')
        ax.set_title(f'{horizon}min Horizon - Node {node_idx}')
        ax.legend()
        mse = mean_squared_error(y_horizon, pred_horizon)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_horizon, pred_horizon)

        ax.text(
            0.05, 0.95, 
            f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}', 
            transform=ax.transAxes, 
            fontsize=9,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
        )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show() 
    plt.close()

def compute_baseline_metrics(y_true, method='historical_average', num_timesteps_input=12):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    
    batch_size, horizon, num_nodes = y_true.shape
    y_pred = np.zeros_like(y_true)
    
    if method == 'last_value':
        for i in range(batch_size):
            y_pred[i, :, :] = y_true[i, 0, :]
    
    elif method == 'historical_average':
        time_of_day = 288
        
        for i in range(batch_size):
            for h in range(horizon):
                time_idx = (i + h) % time_of_day
                historical_values = y_true[time_idx::time_of_day, 0, :]
                if len(historical_values) > 0:
                    y_pred[i, h, :] = np.mean(historical_values, axis=0)
                else:
                    y_pred[i, h, :] = y_true[i, 0, :]
    
    elif method == 'seasonal':
        day_pattern = 288
        
        for i in range(batch_size):
            for h in range(horizon):
                prev_day_idx = i - day_pattern + h
                if prev_day_idx >= 0 and prev_day_idx < batch_size:
                    y_pred[i, h, :] = y_true[prev_day_idx, 0, :]
                else:
                    y_pred[i, h, :] = y_true[i, 0, :]
    
    else:
        raise ValueError(f"Unknown baseline method: {method}")
    
    return y_pred