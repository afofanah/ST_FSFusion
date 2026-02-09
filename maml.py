import torch
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
import time
from copy import deepcopy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import psutil
from torch_geometric.data import DataLoader
from models.adaptive_fsl import MetaCrossDomainFusion, TransformerBasedPredictor

from utils import (
    count_parameters, 
    get_model_size_mb,
    metric_func, 
    result_print, 
    result_prints,
    spatial_correlation, 
    temporal_alignment_metric, 
    feature_alignment_metric, 
    graph_consistency_metrics, 
    physical_constraint_violation
)

class ResultVisualizer:
    def __init__(self, save_dir="results", experiment_id=None):
        if experiment_id is None:
            experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.save_dir = os.path.join(save_dir, experiment_id)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        print(f"Results will be saved to {self.save_dir}")
    
    def plot_core_metrics(self, metrics_history):
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        if 'MAE' in metrics_history and len(metrics_history['MAE']) > 0:
            sns.lineplot(x=range(len(metrics_history['MAE'])), 
                         y=metrics_history['MAE'], ax=axes[0], color='royalblue')
            axes[0].set_title('MAE Over Epochs')
            axes[0].set_ylabel('MAE')
            axes[0].set_xlabel('Epoch')
        
        if 'RMSE' in metrics_history and len(metrics_history['RMSE']) > 0:
            sns.lineplot(x=range(len(metrics_history['RMSE'])), 
                         y=metrics_history['RMSE'], ax=axes[1], color='darkorange')
            axes[1].set_title('RMSE Over Epochs')
            axes[1].set_ylabel('RMSE')
            axes[1].set_xlabel('Epoch')
        
        if 'MAPE' in metrics_history and len(metrics_history['MAPE']) > 0:
            sns.lineplot(x=range(len(metrics_history['MAPE'])), 
                         y=metrics_history['MAPE'], ax=axes[2], color='forestgreen')
            axes[2].set_title('MAPE Over Epochs')
            axes[2].set_ylabel('MAPE (%)')
            axes[2].set_xlabel('Epoch')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "core_metrics.png"))
        plt.close()
        
    def plot_rq1_metrics(self, spatial_corrs):
        if len(spatial_corrs) == 0:
            return
            
        fig = plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(spatial_corrs)), y=spatial_corrs,
                     color='purple', marker='o')
        plt.title('Spatial Correlation Preservation (RQ1)')
        plt.ylabel('Pearson Correlation')
        plt.xlabel('Evaluation Step')
        plt.savefig(os.path.join(self.save_dir, "rq1_spatial_correlation.png"))
        plt.close()
        
    def plot_rq2_metrics(self, temporal_dtws):
        if len(temporal_dtws) == 0:
            return
            
        fig = plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(temporal_dtws)), y=temporal_dtws,
                     color='teal', marker='s')
        plt.title('Temporal Alignment (Dynamic Time Warping) (RQ2)')
        plt.ylabel('DTW Distance')
        plt.xlabel('Evaluation Step')
        plt.savefig(os.path.join(self.save_dir, "rq2_temporal_alignment.png"))
        plt.close()
        
    def plot_rq3_metrics(self, feature_alignments):
        if not feature_alignments:
            return
            
        src_alignments = [fa['src_alignment'] for fa in feature_alignments]
        tgt_alignments = [fa['tgt_alignment'] for fa in feature_alignments]
        transfer_gaps = [fa['transfer_gap'] for fa in feature_alignments]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(src_alignments))
        
        sns.lineplot(x=x, y=src_alignments, color='blue', marker='o', label='Source Alignment')
        sns.lineplot(x=x, y=tgt_alignments, color='green', marker='s', label='Target Alignment')
        sns.lineplot(x=x, y=transfer_gaps, color='red', marker='^', label='Transfer Gap')
        
        plt.title('Cross-Domain Feature Alignment (RQ3)')
        plt.ylabel('Cosine Similarity')
        plt.xlabel('Evaluation Step')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "rq3_feature_alignment.png"))
        plt.close()
        
    def plot_rq4_metrics(self, graph_metrics):
        if not graph_metrics or len(graph_metrics.get('edge_precision', [])) == 0:
            return
            
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        if 'edge_precision' in graph_metrics and 'edge_recall' in graph_metrics:
            sns.lineplot(x=range(len(graph_metrics['edge_precision'])), 
                         y=graph_metrics['edge_precision'], 
                         ax=axes[0], color='crimson', label='Precision')
            sns.lineplot(x=range(len(graph_metrics['edge_recall'])), 
                         y=graph_metrics['edge_recall'], 
                         ax=axes[0], color='darkblue', label='Recall')
            
            if 'edge_f1' in graph_metrics:
                sns.lineplot(x=range(len(graph_metrics['edge_f1'])), 
                             y=graph_metrics['edge_f1'], 
                             ax=axes[0], color='purple', label='F1 Score')
                
            axes[0].set_title('Edge Prediction Performance (RQ4)')
            axes[0].set_ylabel('Score')
            axes[0].legend()
        
        if 'phys_violation' in graph_metrics and len(graph_metrics['phys_violation']) > 0:
            sns.lineplot(x=range(len(graph_metrics['phys_violation'])), 
                         y=graph_metrics['phys_violation'], 
                         ax=axes[1], color='darkred')
            axes[1].set_title('Physical Constraint Violations (RQ4)')
            axes[1].set_ylabel('Violation Rate (%)')
        else:
            axes[1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "rq4_graph_metrics.png"))
        plt.close()
        
    def plot_loss_curves(self, train_losses, val_losses=None):
        if len(train_losses) == 0:
            return
            
        fig = plt.figure(figsize=(10, 6))
        sns.lineplot(x=range(len(train_losses)), y=train_losses,
                     label='Training Loss', color='royalblue')
        if val_losses and len(val_losses) > 0:
            sns.lineplot(x=range(len(val_losses)), y=val_losses,
                         label='Validation Loss', color='darkorange')
        plt.title('Training and Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(self.save_dir, "loss_curves.png"))
        plt.close()
        
    def plot_prediction_examples(self, predictions, ground_truth, node_idx=0, timesteps=24):
        predictions = np.asarray(predictions)
        ground_truth = np.asarray(ground_truth)

        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
            ground_truth = ground_truth.reshape(1, -1)
        elif predictions.ndim == 3:
            predictions = predictions[0]
            ground_truth = ground_truth[0]
        pred_sample = predictions[node_idx] if node_idx < predictions.shape[0] else predictions[0]
        true_sample = ground_truth[node_idx] if node_idx < ground_truth.shape[0] else ground_truth[0]

        max_steps = min(timesteps, pred_sample.shape[0], true_sample.shape[0])

        pred_sample = pred_sample[:max_steps]
        true_sample = true_sample[:max_steps]

        fig = plt.figure(figsize=(14, 6))
        plt.plot(range(max_steps), true_sample, label='Ground Truth', color='navy', linewidth=2)
        plt.plot(range(max_steps), pred_sample, label='Prediction', color='crimson', linestyle='--')
        plt.title(f'Prediction vs Ground Truth (Node {node_idx})')
        plt.ylabel('Traffic Flow')
        plt.xlabel('Time Steps')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, f"prediction_example_node_{node_idx}.png"))
        plt.close()
        
        if predictions.shape[0] > 1:
            for i in range(1, min(3, predictions.shape[0])):
                pred_sample = predictions[i][:max_steps]
                true_sample = ground_truth[i][:max_steps]
                
                fig = plt.figure(figsize=(14, 6))
                plt.plot(range(max_steps), true_sample, label='Ground Truth', color='navy', linewidth=2)
                plt.plot(range(max_steps), pred_sample, label='Prediction', color='crimson', linestyle='--')
                plt.title(f'Prediction vs Ground Truth (Node {i})')
                plt.ylabel('Traffic Flow')
                plt.xlabel('Time Steps')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(self.save_dir, f"prediction_example_node_{i}.png"))
                plt.close()

    def plot_rq5_metrics(self, sample_sizes, performance_metrics):
        if not sample_sizes or not performance_metrics:
            return
        fig = plt.figure(figsize=(10, 6))
        sns.lineplot(x=sample_sizes, y=performance_metrics['MAE'], 
                     color='royalblue', marker='o', label='MAE')
        ax2 = plt.twinx()
        sns.lineplot(x=sample_sizes, y=performance_metrics['R2'], 
                     color='forestgreen', marker='s', label='R²', ax=ax2)
        
        plt.title('Model Performance vs Sample Size (RQ5)')
        plt.xlabel('Sample Size')
        plt.ylabel('MAE')
        ax2.set_ylabel('R² Score')
        lines, labels = plt.gca().get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines + lines2, labels + labels2, loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "rq5_sample_efficiency.png"))
        plt.close()
            
    def plot_cross_domain_tsne(self, features, mode='spatial', node_ids=None):
        if 'raw' not in features or 'fused' not in features:
            return
            
        fig, ax = plt.subplots(figsize=(12, 10))

        raw = np.nan_to_num(features['raw'])
        if raw.shape[0] > 2:
            perplexity = min(30, int(np.sqrt(raw.shape[0])))
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            raw_tsne = tsne.fit_transform(raw)
            ax.scatter(raw_tsne[:, 0], raw_tsne[:, 1], 
                      c='blue', marker='o', s=80, alpha=0.7, 
                      label='Raw Features')
            fused = np.nan_to_num(features['fused'])
            if fused.shape[0] > 2:
                fused_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(fused)
                ax.scatter(fused_tsne[:, 0], fused_tsne[:, 1], 
                          c='red', marker='s', s=80, alpha=0.7,
                          label='Fused Features')
                if node_ids is not None:
                    for i, node_id in enumerate(node_ids):
                        if i < raw_tsne.shape[0]:
                            ax.annotate(str(node_id), (raw_tsne[i, 0], raw_tsne[i, 1]), 
                                       fontsize=8, alpha=0.7)
                        if i < fused_tsne.shape[0]:
                            ax.annotate(str(node_id), (fused_tsne[i, 0], fused_tsne[i, 1]), 
                                       fontsize=8, alpha=0.7)
                raw_centroid = np.mean(raw_tsne, axis=0)
                fused_centroid = np.mean(fused_tsne, axis=0)
                
                ax.scatter(raw_centroid[0], raw_centroid[1], 
                          c='blue', marker='*', s=300, alpha=1.0,
                          label='Raw Centroid')
                ax.scatter(fused_centroid[0], fused_centroid[1], 
                          c='red', marker='*', s=300, alpha=1.0,
                          label='Fused Centroid')
            
                ax.plot([raw_centroid[0], fused_centroid[0]], 
                       [raw_centroid[1], fused_centroid[1]], 
                       'k--', alpha=0.5)
                centroid_dist = np.sqrt(np.sum((raw_centroid - fused_centroid)**2))
                plt.title(f'{mode.capitalize()} Feature Space (t-SNE)\nCentroid Distance: {centroid_dist:.3f}')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                plt.legend()
                plt.grid(alpha=0.3)
                plt.savefig(os.path.join(self.save_dir, f"tsne_{mode}_features.png"))
                plt.close()


class STMAML(nn.Module):
    def __init__(self, data_args, task_args, model_args, config, model='MetaCross_DomainFusion', device='cuda'):
        super(STMAML, self).__init__()
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        self.config = config
        self.rq_flags = config.get('rq_flags', {f'rq{i}': False for i in range(1,6)})

        self.node_num = task_args.get('node_num', None)
        if not self.node_num and len(data_args['data_keys']) > 0:
            first_key = data_args['data_keys'][0]
            if first_key in data_args and 'node_num' in data_args[first_key]:
                self.node_num = data_args[first_key]['node_num']
            else:
                for dataset_key in data_args['data_keys']:
                    if dataset_key in data_args and 'node_num' in data_args[dataset_key]:
                        self.node_num = data_args[dataset_key]['node_num']
                        print(f"Using node_num={self.node_num} from dataset {dataset_key}")
                        break
                
                if not self.node_num:
                    for dataset_key in data_args['data_keys']:
                        if dataset_key in data_args and 'adjacency_matrix_path' in data_args[dataset_key]:
                            import numpy as np
                            adj_path = data_args[dataset_key]['adjacency_matrix_path']
                            adj_matrix = np.load(adj_path)
                            self.node_num = adj_matrix.shape[0]
                            print(f"Inferred node_num={self.node_num} from adjacency matrix in {dataset_key}")
                            break
                
                if not self.node_num:
                    test_dataset = config.get('test_dataset', 'metr-la')
                    if test_dataset == 'metr-la':
                        self.node_num = 207
                    elif test_dataset == 'pems-bay':
                        self.node_num = 325
                    elif test_dataset == 'chengdu_m':
                        self.node_num = 524
                    elif test_dataset == 'shenzhen':
                        self.node_num = 627
                    else:
                        self.node_num = 100
                    print(f"Warning: node_num not found, using default value {self.node_num} based on test dataset")
                
        self.hidden_dim = model_args['hidden_dim']
        self.message_dim = model_args['message_dim']
        self.update_lr = model_args.get('update_lr', 0.001)
        self.meta_lr = model_args.get('meta_lr', 0.001)
        self.update_step = model_args.get('update_step', 5)
        self.update_step_test = model_args.get('update_step_test', 10)
        self.task_num = task_args.get('task_num', 4)
        self.model_name = model
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        self.loss_lambda = model_args.get('loss_lambda', 0.1)
        
        if model == 'TransformerBasedPredictor':
            self.model = TransformerBasedPredictor(model_args, task_args)
        elif model == 'MetaCross_DomainFusion':
            self.model = MetaCrossDomainFusion(model_args, task_args)
        else:
            raise ValueError(f"Unknown model type: {model}")
            
        self.model = self.model.to(self.device)
        param_count = count_parameters(self.model)
        model_size = get_model_size_mb(self.model)
        print(f"Initialized {model} with {param_count} parameters ({model_size:.2f} MB)")
        self.meta_optim = optim.Adam(self.model.parameters(), lr=self.meta_lr, weight_decay=1e-5)
        self.loss_criterion = nn.MSELoss()
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{model}"
        self.visualizer = ResultVisualizer(save_dir=config.get('save_dir', 'results'), 
                                          experiment_id=experiment_id)
        self.metrics_history = {
            'MAE': [],
            'RMSE': [],
            'MAPE': [],
            'TRAIN_LOSS': [],
            'VAL_LOSS': []
        }
        if self.rq_flags.get('rq1', False):
            self.metrics_history['SPATIAL_CORR'] = []
        if self.rq_flags.get('rq2', False):
            self.metrics_history['TEMPORAL_DTW'] = []
        if self.rq_flags.get('rq3', False):
            self.metrics_history['FEATURE_ALIGNMENT'] = []
        if self.rq_flags.get('rq4', False):
            self.metrics_history.update({
                'EDGE_PRECISION': [],
                'EDGE_RECALL': [],
                'EDGE_F1': [],
                'PHYS_VIOLATION': []
            })
        if self.rq_flags.get('rq5', False):
            self.metrics_history.update({
                'SAMPLE_SIZES': [],
                'PERFORMANCE': {}
            })

    def graph_reconstruction_loss(self, meta_graph, adj_graph):
        if meta_graph.dim() == 3 and adj_graph.dim() == 2:
            adj_graph = adj_graph.unsqueeze(0).float()
            matrix = adj_graph.repeat(meta_graph.shape[0], 1, 1)
        else:
            matrix = adj_graph.float()
        return F.mse_loss(meta_graph, matrix)

    def calculate_loss(self, pred, target, meta_graph=None, adj_matrix=None, 
                       stage='target', capacities=None):
        if pred.size(1) != target.size(1):
            min_nodes = min(pred.size(1), target.size(1))
            pred = pred[:, :min_nodes]
            target = target[:, :min_nodes]
        
        if pred.dim() == 2 and target.dim() == 3:
            target = target.mean(dim=1)
        elif pred.dim() == 3 and target.dim() == 2:
            pred = pred.mean(dim=1)
            
        prediction_loss = self.loss_criterion(pred, target)
        total_loss = prediction_loss
        
        if self.rq_flags.get('rq4', False) and meta_graph is not None and adj_matrix is not None:
            if stage in ['source', 'target_maml']:
                if isinstance(meta_graph, torch.Tensor) and isinstance(adj_matrix, torch.Tensor):
                    if meta_graph.size() != adj_matrix.size():
                        meta_graph_flat = meta_graph.mean()
                        graph_loss = torch.nn.functional.mse_loss(
                            meta_graph_flat.expand_as(adj_matrix), adj_matrix
                        )
                    else:
                        graph_loss = self.graph_reconstruction_loss(meta_graph, adj_matrix)
                else:
                    graph_loss = self.graph_reconstruction_loss(meta_graph, adj_matrix)
                    
                total_loss += self.loss_lambda * graph_loss
        
        if self.rq_flags.get('rq4', False) and capacities is not None:
            if isinstance(capacities, torch.Tensor) and pred.size(1) != capacities.size(0):
                min_size = min(pred.size(1), capacities.size(0))
                pred_for_violation = pred[:, :min_size]
                capacities_for_violation = capacities[:min_size]
            else:
                pred_for_violation = pred
                capacities_for_violation = capacities
                
            pred_np = pred_for_violation.detach().cpu().numpy()
            constraints_np = capacities_for_violation.detach().cpu().numpy() if isinstance(capacities_for_violation, torch.Tensor) else capacities_for_violation
            
            violation_pct = physical_constraint_violation(pred_np, constraints_np)
            violation_penalty = torch.tensor(violation_pct * 0.001, device=pred.device)
            total_loss += violation_penalty
            
        return total_loss
    
    def meta_train(self, data_spt, matrix_spt, data_qry, matrix_qry, dim=3, capacities=None):
        task_losses = []
        task_results = []
        init_model = deepcopy(self.model).to(self.device)
        init_model.train()
        
        for i in range(self.task_num):
            maml_model = deepcopy(init_model).to(self.device)
            task_optimizer = optim.Adam(maml_model.parameters(), lr=self.update_lr)
            data_spt_i = data_spt[i].to(self.device)
            matrix_spt_i = matrix_spt[i].to(self.device)
            data_qry_i = data_qry[i].to(self.device)
            matrix_qry_i = matrix_qry[i].to(self.device)
            
            for k in range(self.update_step):
                adj_mx = self._prepare_adj_matrix(matrix_spt_i)
                if self.model_name == 'MetaCross_DomainFusion':
                    pred, uncertainty, meta_graph = maml_model(data_spt_i, adj_mx)
                else:
                    pred, meta_graph = maml_model(data_spt_i, adj_mx, dim=dim)
                
                if self.model_name in ['MetaCross_DomainFusion', 'TransformerBasedPredictor']:
                    if data_spt_i.y.dim() > 2 and pred.dim() == 2:
                        target = data_spt_i.y[:, 0, :]
                    else:
                        target = data_spt_i.y
                else:
                    target = data_spt_i.y
                
                if pred.size(1) != target.size(1):
                    min_nodes = min(pred.size(1), target.size(1))
                    pred = pred[:, :min_nodes]
                    target = target[:, :min_nodes]
                
                loss = self.calculate_loss(pred, target, meta_graph, matrix_spt_i, 
                                        'source', capacities)

                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()
            
            adj_mx = self._prepare_adj_matrix(matrix_qry_i)

            if self.model_name == 'MetaCross_DomainFusion':
                pred, uncertainty, meta_graph = maml_model(data_qry_i, adj_mx)
            else:
                pred, meta_graph = maml_model(data_qry_i, adj_mx, dim=dim)
            
            if data_qry_i.y.dim() > 2 and pred.dim() == 2:
                target = data_qry_i.y[:, 0, :]
            else:
                target = data_qry_i.y
            
            if pred.size(1) != target.size(1):
                min_nodes = min(pred.size(1), target.size(1))
                pred = pred[:, :min_nodes]
                target = target[:, :min_nodes]
                
            loss_q = self.calculate_loss(pred, target, meta_graph, matrix_qry_i, 
                                      'target_maml', capacities)
            task_losses.append(loss_q)
            
            y_for_eval = data_qry_i.y
            if pred.size(1) != y_for_eval.size(1) and y_for_eval.dim() >= 2:
                min_nodes = min(pred.size(1), y_for_eval.size(1))
                pred_for_eval = pred[:, :min_nodes]
                y_for_eval = y_for_eval[:, :min_nodes]
            else:
                pred_for_eval = pred
            
            task_metrics = self._evaluate_rq_metrics(pred_for_eval, y_for_eval, meta_graph, 
                                                  matrix_qry_i, capacities)
            task_results.append(task_metrics)
        
        meta_loss = torch.stack(task_losses).mean()
        self.meta_optim.zero_grad()
        meta_loss.backward()
        self.meta_optim.step()
        
        rq_metrics = self._aggregate_rq_metrics(task_results)
        self.metrics_history['TRAIN_LOSS'].append(meta_loss.detach().cpu().item())
        return meta_loss.detach().cpu().item(), rq_metrics
    
    def _evaluate_rq_metrics(self, pred, y, meta_graph, adj_matrix, capacities=None):
        metrics = {}
        if isinstance(pred, torch.Tensor):
            pred_np = pred.detach().cpu().numpy()
        else:
            pred_np = pred
                
        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = y
        if pred_np.ndim == 2 and y_np.ndim == 3:
            y_np = y_np[:, 0, :]

        if self.rq_flags.get('rq1', False):
            metrics['spatial_corr'] = spatial_correlation(pred_np, y_np, adj_matrix.cpu().numpy())
                
        if self.rq_flags.get('rq2', False):
            metrics['temporal_dtw'] = temporal_alignment_metric(pred_np, y_np)
                
        if self.rq_flags.get('rq3', False) and hasattr(self.model, 'cross_domain_fusion'):
            fused_features = getattr(self.model, 'fused_features', None)
            src_features = getattr(self.model, 'spatial_features', None)
            tgt_features = getattr(self.model, 'temporal_features', None)
                
            if fused_features is not None and src_features is not None and tgt_features is not None:
                metrics['feature_alignment'] = feature_alignment_metric(
                    fused_features.detach().cpu().numpy(),
                    src_features.detach().cpu().numpy(),
                    tgt_features.detach().cpu().numpy()
                )
        
        return metrics

    def _prepare_adj_matrix(self, matrix):
        if hasattr(self, 'model_name') and self.model_name == 'TransformerBasedPredictor':
            return [matrix.float(), matrix.float().t()]
        elif self.model_name == 'MetaCross_DomainFusion':
            return matrix.float()
        return matrix.float()
    
    def _aggregate_rq_metrics(self, task_results):
        task_results = [result for result in task_results if result]
        
        if not task_results:
            return {}
        keys = task_results[0].keys()
        
        aggregated = {key: [] for key in keys}
        
        for result in task_results:
            for key in keys:
                if key in result:
                    aggregated[key].append(result[key])
        
        return {k: np.mean(v) for k, v in aggregated.items() if v}
        
    def finetuning(self, target_dataloader, test_dataloader, target_epochs=100, dim=3, capacities=None):
        print(f"Starting fine-tuning for {target_epochs} epochs...")
        maml_model = deepcopy(self.model).to(self.device)
        optimizer = optim.Adam(maml_model.parameters(), lr=self.update_lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
        best_metrics = {'MAE': float('inf')}
        best_model = None
        best_epoch = 0
        train_losses = []
        patience_counter = 0
        patience = 50
        
        for epoch in tqdm(range(target_epochs)):
            epoch_start_time = time.time()
            maml_model.train()
            batch_losses = []
            for step, (data, A_wave) in enumerate(target_dataloader):
                data, A_wave = data.to(self.device), A_wave.to(self.device)
                adj_mx = self._prepare_adj_matrix(A_wave)
                if self.model_name == 'MetaCross_DomainFusion':
                    pred, uncertainty, meta_graph = maml_model(data, adj_mx)
                else:
                    pred, meta_graph = maml_model(data, adj_mx, dim=dim)
                if data.y.dim() > 2 and pred.dim() == 2:
                    target = data.y[:, 0, :]
                else:
                    target = data.y
                loss = self.calculate_loss(pred, target, meta_graph, A_wave, 
                                         'target', capacities)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(maml_model.parameters(), 5.0)
                optimizer.step()
                batch_losses.append(loss.item())
            avg_loss = np.mean(batch_losses)
            train_losses.append(avg_loss)
            self.metrics_history['TRAIN_LOSS'].append(avg_loss)
            
            if epoch % 5 == 0 or epoch == target_epochs - 1:
                maml_model.eval()
                eval_metrics = self._evaluate_model(maml_model, test_dataloader, dim, capacities)
                scheduler.step(eval_metrics['MAE'])
                
                for key, value in eval_metrics.items():
                    if key in self.metrics_history:
                        if isinstance(value, np.ndarray):
                            value = np.mean(value)
                        self.metrics_history[key].append(value)
                
                current_mae = eval_metrics['MAE']
                if isinstance(current_mae, np.ndarray):
                    current_mae = np.mean(current_mae)
                    
                if current_mae < best_metrics['MAE']:
                    best_metrics = eval_metrics
                    best_model = deepcopy(maml_model)
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(best_model.state_dict(), os.path.join(self.visualizer.save_dir, "best_model.pt"))
                else:
                    patience_counter += 1
                epoch_time = time.time() - epoch_start_time
                print(f"Epoch {epoch+1}/{target_epochs} | Loss: {avg_loss:.4f} | MAE: {current_mae:.4f} | Time: {epoch_time:.2f}s")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        print(f"\nBest model found at epoch {best_epoch+1}")
        final_metrics = self._evaluate_model(best_model, test_dataloader, dim, capacities)
        self.visualizer.plot_loss_curves(train_losses)
        self.visualize_results()
        self._generate_prediction_examples(best_model, test_dataloader, dim)
        
        return final_metrics
    
    def _evaluate_model(self, model, dataloader, dim=3, capacities=None):
        model.eval()
        all_preds = []
        all_targets = []
        meta_graphs = []
        
        with torch.no_grad():
            for step, (data, A_wave) in enumerate(dataloader):
                data, A_wave = data.to(self.device), A_wave.to(self.device)
                adj_mx = self._prepare_adj_matrix(A_wave)
                if self.model_name == 'MetaCross_DomainFusion':
                    pred, uncertainty, meta_graph = model(data, adj_mx)
                else:
                    pred, meta_graph = model(data, adj_mx, dim=dim)
                if data.y.dim() > 2 and pred.dim() == 2:
                    target = data.y[:, 0, :]
                else:
                    target = data.y
                all_preds.append(pred.cpu())
                all_targets.append(target.cpu())
                if meta_graph is not None and self.rq_flags.get('rq4', False):
                    meta_graphs.append(meta_graph.cpu())
    
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        times = self.task_args.get('pred_num', all_preds.shape[1] if all_preds.ndim > 1 else 1)
        metrics = metric_func(all_preds, all_targets, times=times)
        
        if self.rq_flags.get('rq1', False) or self.rq_flags.get('rq2', False):
            if len(dataloader) > 0:
                A_wave = dataloader.dataset[0][1]
                if self.rq_flags.get('rq1', False):
                    metrics['SPATIAL_CORR'] = spatial_correlation(all_preds, all_targets, A_wave.numpy())
                
                if self.rq_flags.get('rq2', False):
                    metrics['TEMPORAL_DTW'] = temporal_alignment_metric(all_preds, all_targets)
        
        if self.rq_flags.get('rq4', False) and len(meta_graphs) > 0:
            avg_meta_graph = torch.stack(meta_graphs).mean(dim=0).numpy()
            A_wave = dataloader.dataset[0][1].numpy()
            graph_metrics = graph_consistency_metrics(avg_meta_graph, A_wave)
            metrics.update({
                'EDGE_PRECISION': graph_metrics['edge_precision'],
                'EDGE_RECALL': graph_metrics['edge_recall'],
                'EDGE_F1': graph_metrics['edge_f1']
            })
            if capacities is not None:
                metrics['PHYS_VIOLATION'] = physical_constraint_violation(all_preds, capacities)
        return metrics
    
    def _generate_prediction_examples(self, model, dataloader, dim=3):
        model.eval()
        with torch.no_grad():
            for data, A_wave in dataloader:
                data, A_wave = data.to(self.device), A_wave.to(self.device)
                adj_mx = self._prepare_adj_matrix(A_wave)
    
                if self.model_name == 'MetaCross_DomainFusion':
                    pred, uncertainty, _ = model(data, adj_mx)
                else:
                    pred, _ = model(data, adj_mx, dim=dim)
                if data.y.dim() > 2 and pred.dim() == 2:
                    target = data.y[:, 0, :]
                else:
                    target = data.y
                
                self.visualizer.plot_prediction_examples(
                    pred.cpu().numpy(),
                    target.cpu().numpy(),
                    node_idx=0,
                    timesteps=min(24, pred.shape[1] if pred.ndim > 1 else 1)
                )
                break
    
    def visualize_results(self):
        if all(len(self.metrics_history[k]) > 0 for k in ['MAE', 'RMSE', 'MAPE']):
            self.visualizer.plot_core_metrics({
                'MAE': self.metrics_history['MAE'],
                'RMSE': self.metrics_history['RMSE'],
                'MAPE': self.metrics_history['MAPE']
            })
        if len(self.metrics_history['TRAIN_LOSS']) > 0:
            self.visualizer.plot_loss_curves(
                self.metrics_history['TRAIN_LOSS'],
                self.metrics_history.get('VAL_LOSS', [])
            )
        if self.rq_flags.get('rq1', False) and 'SPATIAL_CORR' in self.metrics_history:
            if len(self.metrics_history['SPATIAL_CORR']) > 0:
                self.visualizer.plot_rq1_metrics(self.metrics_history['SPATIAL_CORR'])
            
        if self.rq_flags.get('rq2', False) and 'TEMPORAL_DTW' in self.metrics_history:
            if len(self.metrics_history['TEMPORAL_DTW']) > 0:
                self.visualizer.plot_rq2_metrics(self.metrics_history['TEMPORAL_DTW'])
            
        if self.rq_flags.get('rq4', False):
            rq4_metrics = {}
            if 'EDGE_PRECISION' in self.metrics_history and len(self.metrics_history['EDGE_PRECISION']) > 0:
                rq4_metrics['edge_precision'] = self.metrics_history['EDGE_PRECISION']
            if 'EDGE_RECALL' in self.metrics_history and len(self.metrics_history['EDGE_RECALL']) > 0:
                rq4_metrics['edge_recall'] = self.metrics_history['EDGE_RECALL']
            if 'PHYS_VIOLATION' in self.metrics_history and len(self.metrics_history['PHYS_VIOLATION']) > 0:
                rq4_metrics['phys_violation'] = self.metrics_history['PHYS_VIOLATION']
            
            if rq4_metrics:
                self.visualizer.plot_rq4_metrics(rq4_metrics)
    
    def evaluate_sample_efficiency(self, dataset, test_dataloader, sample_sizes, dim=3):
        if not self.rq_flags.get('rq5', False):
            print("RQ5 is not enabled. Skipping sample efficiency evaluation.")
            return
        performances = {'MAE': [], 'RMSE': [], 'R2': []}
        for size in sample_sizes:
            print(f"Evaluating sample size: {size}")
            indices = np.random.choice(len(dataset), size=min(size, len(dataset)), replace=False)
            subset_loader = DataLoader(
                [dataset[i] for i in indices],
                batch_size=self.task_args.get('batch_size', 32),
                shuffle=True
            )
            temp_model = deepcopy(self.model).to(self.device)
            optimizer = optim.Adam(temp_model.parameters(), lr=self.update_lr)

            for epoch in range(15):
                temp_model.train()
                epoch_losses = []
                
                for step, (data, A_wave) in enumerate(subset_loader):
                    data, A_wave = data.to(self.device), A_wave.to(self.device)
                    adj_mx = self._prepare_adj_matrix(A_wave)
                
                    if self.model_name == 'MetaCross_DomainFusion':
                        pred, uncertainty, meta_graph = temp_model(data, adj_mx)
                    else:
                        pred, meta_graph = temp_model(data, adj_mx, dim=dim)
                
                    if data.y.dim() > 2 and pred.dim() == 2:
                        target = data.y[:, 0, :]
                    else:
                        target = data.y
                    loss = self.calculate_loss(pred, target, meta_graph, A_wave, 'target')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    epoch_losses.append(loss.item())
                
                if (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch+1}/15 | Loss: {np.mean(epoch_losses):.4f}")
            metrics = self._evaluate_model(temp_model, test_dataloader, dim)
            for key in performances.keys():
                if key in metrics:
                    value = metrics[key]
                    if isinstance(value, np.ndarray):
                        value = np.mean(value)
                    performances[key].append(value)
            del temp_model
            torch.cuda.empty_cache()

        self.metrics_history['SAMPLE_SIZES'] = sample_sizes
        self.metrics_history['PERFORMANCE'] = performances
        self.visualizer.plot_rq5_metrics(sample_sizes, performances)
        
        return performances