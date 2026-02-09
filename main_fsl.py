import os
import argparse
import yaml
import json
import torch
import numpy as np
import random
import time
import sys
import math
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
import torch.nn as nn
from yaml_config import yaml_to_framework_config
from train import TrafficPredictionFramework
from datasets import TrafficDataset
from models.adaptive_fsl import CrossCityAdapter, MetaCrossDomainFusion, FewShotTrafficLearner
from types import MethodType
from utils import count_parameters, get_model_size_mb, metric_func, result_print
from torch_geometric.data import Batch as PyGBatch

def move_model_to_device(model, device):
    model = model.to(device)
    for module in model.modules():
        if hasattr(module, 'device'):
            module.device = torch.device(device)
    return model

def clear_gpu_memory():
    if torch.backends.mps.is_available():
        torch.backends.mps.empty_cache()

def normalize_city_name(city_name):
    if isinstance(city_name, torch.Tensor):
        city_name = str(city_name.item())
    if isinstance(city_name, str) and '/' in city_name:
        city_name = city_name.split('/')[-1].split('.')[0]
    return city_name

def print_metrics(metrics):
    for horizon, values in metrics.items():
        print(f"  Horizon {horizon}: RMSE={values.get('rmse', float('nan')):.4f}, MAE={values.get('mae', float('nan')):.4f}, MAPE={values.get('mape', float('nan')):.4f}")

def prepare_data_args(data_args, source_city, target_city):
    base_path = data_args.get('base_path', './data')

    data_keys = data_args.get('data_keys', ['metr-la', 'pems-bay', 'chengdu_m', 'shenzhen'])
    for city in data_keys:
        if city not in data_args:
            data_args[city] = {
                'adjacency_matrix_path': f'{base_path}/{city}/matrix.npy',
                'dataset_path': f'{base_path}/{city}/dataset.npy'
            }

    data_args[source_city] = {
        'adjacency_matrix_path': f'{base_path}/{source_city}/matrix.npy',
        'dataset_path': f'{base_path}/{source_city}/dataset.npy'
    }

    data_args[target_city] = {
        'adjacency_matrix_path': f'{base_path}/{target_city}/matrix.npy',
        'dataset_path': f'{base_path}/{target_city}/dataset.npy'
    }

    return data_args

def initialize_target_model(framework, device):
    if not hasattr(framework, 'model_args') or not framework.model_args:
        raise ValueError("Framework missing model_args")

    if not hasattr(framework, 'task_args') or not framework.task_args:
        raise ValueError("Framework missing task_args")

    target_model = MetaCrossDomainFusion(
        model_args=framework.model_args,
        task_args=framework.task_args,
        device=device
    )

    if hasattr(framework, 'source_model'):
        source_params = sum(p.numel() for p in framework.source_model.parameters())
        target_params = sum(p.numel() for p in target_model.parameters())

        if source_params != target_params:
            print(f"Warning: Source model has {source_params} parameters but target model has {target_params} parameters")

    target_model = move_model_to_device(target_model, device)

    print(f"Target model initialized with {sum(p.numel() for p in target_model.parameters())} parameters on {device}")
    return target_model

def check_and_fix_models(framework):
    print("\n==== CHECKING MODEL DEVICES ====")
    device = framework.device

    print("\nMoving all models to device:", device)
    framework.source_model = framework.source_model.to(device)
    framework.adapter = framework.adapter.to(device)

    for module in framework.source_model.modules():
        if hasattr(module, 'device'):
            module.device = device

    for module in framework.adapter.modules():
        if hasattr(module, 'device'):
            module.device = device

    return framework

def safe_calculate_metrics(truth, pred):
    metrics = {'rmse': float('nan'), 'mae': float('nan'), 'mape': float('nan')}

    truth = truth.cpu().contiguous()
    pred = pred.cpu().contiguous()

    if truth.numel() == 0 or pred.numel() == 0:
        return metrics
    
    truth_size = truth.numel()
    pred_size = pred.numel()

    if truth_size != pred_size:
        min_size = min(truth_size, pred_size)
        truth = truth.reshape(-1)[:min_size]
        pred = pred.reshape(-1)[:min_size]
    else:
        truth = truth.reshape(-1)
        pred = pred.reshape(-1)

    mse = torch.mean((truth - pred)**2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()

    mae = torch.mean(torch.abs(truth - pred)).item()

    epsilon = 1e-10  
    non_zero_mask = truth.abs() > epsilon

    if torch.sum(non_zero_mask) > 0:
        mape = torch.mean(torch.abs((truth[non_zero_mask] - pred[non_zero_mask]) /
                                  (truth[non_zero_mask].abs() + epsilon)) * 100).item()
    else:
        mape = float('nan')

    if not (math.isfinite(rmse) and math.isfinite(mae) and
            (math.isnan(mape) or math.isfinite(mape))):
        return metrics

    metrics['rmse'] = rmse
    metrics['mae'] = mae
    metrics['mape'] = mape

    return metrics

def initialize_adapter(framework, config, device):
    embedding_dim = config['model_args'].get('hidden_dim', 16)
    hidden_dim = max(8, embedding_dim // 2)  
    adapter_model = CrossCityAdapter(
        source_dim=embedding_dim,
        target_dim=embedding_dim,
        hidden_dim=config['model_args'].get('adapter_dim', hidden_dim),
        device=device
    )

    adapter_model = move_model_to_device(adapter_model, device)
    print(f"Created adapter with source_dim={embedding_dim}, target_dim={embedding_dim}")

    return adapter_model

def fix_tensor_dimensions(tensor, expected_dims=3):
    if tensor is None:
        return None
    
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f"Expected tensor, got {type(tensor)}")
    
    current_dims = tensor.dim()
    
    if current_dims == expected_dims:
        return tensor
    
    if current_dims < expected_dims:
        for _ in range(expected_dims - current_dims):
            tensor = tensor.unsqueeze(0)
        return tensor
    
    if current_dims > expected_dims:
        new_shape = list(tensor.shape[-(expected_dims-1):])
        flattened_dims = 1
        for d in tensor.shape[:-(expected_dims-1)]:
            flattened_dims *= d
        new_shape = [flattened_dims] + new_shape
        return tensor.reshape(*new_shape)
    
    return tensor

def patch_crosscityadapter_forward():
    from models.adaptive_fsl import CrossCityAdapter
    
    original_forward = CrossCityAdapter.forward
    
    def patched_forward(self, source_features, target_features):
        if source_features.dim() != 3:
            print(f"Converting source_features from {source_features.dim()}D to 3D")
            if source_features.dim() == 2:
                source_features = source_features.unsqueeze(0)
            elif source_features.dim() == 1:
                source_features = source_features.unsqueeze(0).unsqueeze(0)
            elif source_features.dim() > 3:
                original_shape = source_features.shape
                batch_dim = original_shape[0]
                node_dim = 1
                for d in original_shape[1:-1]:
                    node_dim *= d
                feature_dim = original_shape[-1]
                source_features = source_features.reshape(batch_dim, node_dim, feature_dim)
        
        if target_features.dim() != 3:
            print(f"Converting target_features from {target_features.dim()}D to 3D")
            if target_features.dim() == 2:
                target_features = target_features.unsqueeze(0)
            elif target_features.dim() == 1:
                target_features = target_features.unsqueeze(0).unsqueeze(0)
            elif target_features.dim() > 3:
                original_shape = target_features.shape
                batch_dim = original_shape[0]
                node_dim = 1
                for d in original_shape[1:-1]:
                    node_dim *= d
                feature_dim = original_shape[-1]
                target_features = target_features.reshape(batch_dim, node_dim, feature_dim)
        
        if source_features.size(0) != target_features.size(0):
            print(f"Batch size mismatch: source {source_features.size(0)} vs target {target_features.size(0)}")
            if source_features.size(0) == 1 and target_features.size(0) > 1:
                source_features = source_features.expand(target_features.size(0), -1, -1)
            elif target_features.size(0) == 1 and source_features.size(0) > 1:
                target_features = target_features.expand(source_features.size(0), -1, -1)
        
        return original_forward(self, source_features, target_features)
    
    CrossCityAdapter.forward = patched_forward
    print("Successfully patched CrossCityAdapter.forward")
    
    return CrossCityAdapter

def patch_transport_meta_knowledge():
    from models.adaptive_fsl import CrossCityAdapter
    
    original_method = CrossCityAdapter.transfer_meta_knowledge
    
    def patched_transfer_meta_knowledge(self, source_meta, target_meta):
        if not isinstance(source_meta, torch.Tensor) or not isinstance(target_meta, torch.Tensor):
            raise ValueError("Meta features must be torch.Tensor")
        
        if source_meta.dim() < 3:
            print(f"Reshaping source_meta from {source_meta.dim()}D to 3D")
            while source_meta.dim() < 3:
                source_meta = source_meta.unsqueeze(0)
        
        if target_meta.dim() < 3:
            print(f"Reshaping target_meta from {target_meta.dim()}D to 3D")
            while target_meta.dim() < 3:
                target_meta = target_meta.unsqueeze(0)
        
        if source_meta.size(0) != target_meta.size(0):
            print(f"Batch size mismatch in meta features: {source_meta.size(0)} vs {target_meta.size(0)}")
            if source_meta.size(0) == 1 and target_meta.size(0) > 1:
                source_meta = source_meta.expand(target_meta.size(0), -1, -1)
            elif target_meta.size(0) == 1 and source_meta.size(0) > 1:
                target_meta = target_meta.expand(source_meta.size(0), -1, -1)
        
        return original_method(self, source_meta, target_meta)
    
    CrossCityAdapter.transfer_meta_knowledge = patched_transfer_meta_knowledge
    print("Successfully patched CrossCityAdapter.transfer_meta_knowledge")
    
    return CrossCityAdapter

def patch_all_adapter_instances(framework):
    from models.adaptive_fsl import CrossCityAdapter
    
    patch_crosscityadapter_forward()
    patch_transport_meta_knowledge()
    
    if hasattr(framework, 'adapter') and isinstance(framework.adapter, CrossCityAdapter):
        original_forward = framework.adapter.forward
        
        def instance_forward(self, source_features, target_features):
            device = next(self.parameters()).device
            source_features = source_features.to(device)
            target_features = target_features.to(device)
            
            if source_features.dim() != 3:
                source_features = fix_tensor_dimensions(source_features, 3)
            if target_features.dim() != 3:
                target_features = fix_tensor_dimensions(target_features, 3)
            
            return CrossCityAdapter.forward(self, source_features, target_features)
        
        framework.adapter.forward = MethodType(instance_forward, framework.adapter)
        print("Patched framework.adapter.forward")
    
    return framework

def geometric_collate(batch):
    data_list = [item[0] for item in batch]
    A_wave_list = [item[1] for item in batch]
    
    batched_data = PyGBatch.from_data_list(data_list)
    if all(isinstance(A, torch.Tensor) for A in A_wave_list):
        batched_A_wave = torch.stack(A_wave_list)
    else:
        batched_A_wave = A_wave_list
        
    return batched_data, batched_A_wave

def convert_for_json(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(i) for i in obj]
    else:
        return obj

def set_random_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def geometric_collate_safe(batch):
    import torch
    from torch_geometric.data import Data, Batch

    if all(isinstance(item, Data) for item in batch):
        return Batch.from_data_list(batch)

    if all(isinstance(item, torch.Tensor) for item in batch):
        return torch.stack(batch)

    if all(isinstance(item, tuple) for item in batch):
        data_list, a_wave_list = zip(*batch)

        if all(isinstance(item, Data) for item in data_list):
            data_batch = Batch.from_data_list(data_list)
        else:
            data_batch = torch.stack(data_list) if all(isinstance(item, torch.Tensor) for item in data_list) else data_list

        if all(isinstance(item, torch.Tensor) for item in a_wave_list):
            a_wave_batch = torch.stack(a_wave_list)
        else:
            a_wave_batch = a_wave_list

        return data_batch, a_wave_batch

    return batch

def patched_init(self, model_args, task_args, source_dataset, target_dataset,
                device='mps', learning_rate=0.01, weight_decay=1e-4,
                scheduler_factor=0.5, scheduler_patience=5, clip_grad_norm=5.0, log_dir='logs'):
    self.device = device if torch.backends.mps.is_available() else 'cpu'
    self.model_args = model_args
    self.task_args = task_args
    self.source_dataset = source_dataset
    self.target_dataset = target_dataset

    self.learning_rate = float(learning_rate)
    self.weight_decay = float(weight_decay)
    self.scheduler_factor = float(scheduler_factor)
    self.scheduler_patience = int(scheduler_patience)
    self.clip_grad_norm = float(clip_grad_norm)
    self.log_dir = log_dir

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(os.path.join(log_dir, 'checkpoints')):
        os.makedirs(os.path.join(log_dir, 'checkpoints'))

    self.source_model = MetaCrossDomainFusion(
        model_args=model_args,
        task_args=task_args,
        device=self.device
    )

    self.adapter = CrossCityAdapter(
        source_dim=model_args['hidden_dim'],
        target_dim=model_args['hidden_dim'],
        hidden_dim=model_args.get('adapter_dim', 16),
        device=self.device
    )

    self.few_shot_learner = FewShotTrafficLearner(
        base_model=self.source_model,
        device=self.device
    )

    self.metrics = defaultdict(list)

    self._setup_optimizer_and_scheduler()

    self._save_configs()

    self.patch_train_cross_city_adapter()

def train_target_model(self, batch_size=5, epochs=200, early_stopping_patience=150, 
                        target_city='pems-bay', val_ratio=0.2, with_source_knowledge=True):
    print(f"Starting target model training for {epochs} epochs on {target_city}...")
    target_city = self._normalize_city_name(target_city)
    
    self.target_model = MetaCrossDomainFusion(
        model_args=self.model_args,
        task_args=self.task_args,
        device=self.device
    )
    
    if with_source_knowledge:
        print("Initializing target model with source model weights...")
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', 'source_model_best.pth')
        if os.path.exists(checkpoint_path):
            self.source_model.load_state_dict(torch.load(checkpoint_path))
            self.target_model.load_state_dict(self.source_model.state_dict())
            print("Successfully initialized target model with source model weights")
        else:
            print("Source model checkpoint not found. Training target model from scratch")
    
    self.target_model = self.target_model.to(self.device)
    
    target_optimizer = optim.Adam(
        self.target_model.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay
    )
    
    target_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        target_optimizer,
        mode='min',
        factor=self.scheduler_factor,
        patience=self.scheduler_patience,
        verbose=True
    )
    
    total_samples = len(self.target_dataset)
    val_size = int(total_samples * val_ratio)
    train_size = total_samples - val_size
    
    indices = list(range(total_samples))
    random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    
    for epoch in range(epochs):
        self.target_model.train()
        train_loss = 0.0
        train_samples = 0
        
        train_loader = DataLoader(
            self.target_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=geometric_collate  
        )
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            data, A_wave = batch_data
            data = data.to(self.device)
            if A_wave is not None:
                A_wave = A_wave.to(self.device)
            
            target_optimizer.zero_grad()
            
            if A_wave is not None:
                predictions, uncertainties, _ = self.target_model(data, A_wave)
            else:
                predictions, uncertainties, _ = self.target_model(data)
            
            if uncertainties is not None:
                loss = self._compute_uncertainty_loss(predictions, data.y, uncertainties)
            else:
                loss = nn.MSELoss()(predictions, data.y)
            
            loss.backward()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.target_model.parameters(), self.clip_grad_norm)
            target_optimizer.step()
            
            batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
            train_loss += loss.item() * batch_size
            train_samples += batch_size
        
        avg_train_loss = train_loss / train_samples
        self.metrics['target_train_loss'].append(avg_train_loss)
        
        val_loader = DataLoader(
            self.target_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=0,
            pin_memory=True,
            collate_fn=geometric_collate  
        )
        
        val_loss = self._validate_model(self.target_model, val_loader)
        self.metrics['target_val_loss'].append(val_loss)
        
        target_scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            self._save_model(self.target_model, f'target_model_{target_city}_best.pth')
        else:
            early_stopping_counter += 1
        
        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print("\nEvaluating current model on target city...")
            eval_metrics = self.evaluate_multi_horizon(
                model=self.target_model,
                dataset=self.target_dataset,
                city_name=target_city,
                horizons=[5, 15, 30, 60, 120]
            )
            
            for horizon, metrics in eval_metrics.items():
                for metric_name, value in metrics.items():
                    self.metrics[f'target_{metric_name}_h{horizon}_{epoch+1}'] = value
        
        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    self._load_model(self.target_model, f'target_model_{target_city}_best.pth')
    
    print("\nFinal Evaluation on target city...")
    final_metrics = self.evaluate_multi_horizon(
        model=self.target_model,
        dataset=self.target_dataset,
        city_name=target_city,
        horizons=[5, 15, 30, 60, 120]
    )
    
    for horizon, metrics in final_metrics.items():
        for metric_name, value in metrics.items():
            self.metrics[f'target_final_{metric_name}_h{horizon}'] = value
    
    return self.target_model

def patch_train_cross_city_adapter(self):
    if hasattr(self.target_dataset, 'get_cross_city_batch'):
        return

    def get_cross_city_batch(self_dataset, source_city, target_city, batch_size=32):
        from torch_geometric.data import Data, Batch

        source_city = source_city.lower().replace('-', '_')
        target_city = target_city.lower().replace('-', '_')

        source_key = None
        target_key = None

        for key in self_dataset.x_list.keys():
            key_lower = key.lower().replace('-', '_')
            if source_city in key_lower:
                source_key = key
            if target_city in key_lower:
                target_key = key

        if not source_key:
            source_key = list(self_dataset.x_list.keys())[0]
            print(f"Warning: Source city {source_city} not found, using {source_key}")

        if not target_key:
            target_key = list(self_dataset.x_list.keys())[0]
            print(f"Warning: Target city {target_city} not found, using {target_key}")

        source_idx = torch.randint(0, len(self_dataset.x_list[source_key]), (batch_size,))
        source_x = self_dataset.x_list[source_key][source_idx]

        target_idx = torch.randint(0, len(self_dataset.x_list[target_key]), (batch_size,))
        target_x = self_dataset.x_list[target_key][target_idx]

        source_data_list = []
        target_data_list = []

        for i in range(batch_size):
            source_data = Data(x=source_x[i].unsqueeze(0))
            if source_key in self_dataset.edge_index_list:
                source_data.edge_index = self_dataset.edge_index_list[source_key]
            if source_key in self_dataset.edge_attr_list:
                source_data.edge_attr = self_dataset.edge_attr_list[source_key]
            if source_key in self_dataset.y_list and i < len(self_dataset.y_list[source_key]):
                source_data.y = self_dataset.y_list[source_key][i]
            source_data_list.append(source_data)

            target_data = Data(x=target_x[i].unsqueeze(0))
            if target_key in self_dataset.edge_index_list:
                target_data.edge_index = self_dataset.edge_index_list[target_key]
            if target_key in self_dataset.edge_attr_list:
                target_data.edge_attr = self_dataset.edge_attr_list[target_key]
            if target_key in self_dataset.y_list and i < len(self_dataset.y_list[target_key]):
                target_data.y = self_dataset.y_list[target_key][i]
            target_data_list.append(target_data)

        source_batch = Batch.from_data_list(source_data_list)
        target_batch = Batch.from_data_list(target_data_list)

        return source_batch, target_batch

    self.target_dataset.get_cross_city_batch = MethodType(get_cross_city_batch, self.target_dataset)

TrafficPredictionFramework.__init__ = patched_init
TrafficPredictionFramework.train_target_model = train_target_model
TrafficPredictionFramework.patch_train_cross_city_adapter = patch_train_cross_city_adapter

def parse_args():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Traffic Prediction with Cross-City Knowledge Transfer")

    parser.add_argument('--config', type=str, required=False,
                      default='/Users/s5273738/Paper Four/Crosscity_Prediction/config.yaml',
                      help='Path to YAML config file')
    parser.add_argument('--source_city', type=str,
                      default='/Users/s5273738/Paper Four/Crosscity_Prediction/data/metr-la',
                      help='Source city dataset')
    parser.add_argument('--target_city', type=str,
                      default='/Users/s5273738/Paper Four/Crosscity_Prediction/data/pems-bay',
                      help='Target city dataset')

    parser.add_argument('--train_source', action='store_true', help='Train source model')
    parser.add_argument('--train_target', action='store_true', help='Train target model directly')
    parser.add_argument('--train_adapter', action='store_true', help='Train cross-city adapter')
    parser.add_argument('--perform_few_shot', action='store_true', help='Perform few-shot adaptation')
    parser.add_argument('--analyze_research_questions', action='store_true', help='Analyze research questions')

    parser.add_argument('--target_epochs', type=int, default=2, help='Maximum epochs for target model training')
    parser.add_argument('--target_patience', type=int, default=15, help='Early stopping patience for target model')
    parser.add_argument('--with_source_knowledge', action='store_true', help='Initialize target model with source model weights')

    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--seed', type=int, help='Override random seed')
    parser.add_argument('--target_days', type=int, help='Override target days')
    parser.add_argument('--k_shot', type=int, help='Override k-shot value')
    parser.add_argument('--n_heads', type=int, default=8, help='Override target epochs')

    parser.add_argument('--use_gpu', action='store_true', help='Use GPU even if not specified in config')
    parser.add_argument('--output_dir', type=str, help='Override output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with more verbose output')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation data ratio')

    if 'ipykernel' in sys.modules:
        return parser.parse_args([])
    return parser.parse_args()

def main():
    print("\n====== SPATIO-TEMPORAL TRAFFIC PREDICTION FRAMEWORK ======\n")

    args = parse_args()
    debug_mode = args.debug

    if debug_mode:
        print("DEBUG MODE ENABLED - Detailed logging will be shown")

    print("Loading configuration...")
    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)
    config = yaml_to_framework_config(args.config)

    if args.learning_rate is not None:
        config['model_args']['meta_lr'] = args.learning_rate

    if args.batch_size is not None:
        config['task_args']['batch_size'] = args.batch_size

    if args.seed is not None:
        config['system']['seed'] = args.seed

    if args.target_days is not None:
        config['evaluation']['target_days'] = args.target_days

    if args.k_shot is not None:
        config['evaluation']['k_shot'] = args.k_shot
        print(f"Override: k_shot = {args.k_shot}")

    if 'training' not in config:
        config['training'] = {}

    config['training']['target_epochs'] = args.target_epochs
    config['training']['target_patience'] = args.target_patience
    config['training']['with_source_knowledge'] = args.with_source_knowledge

    if args.use_gpu:
        config['system']['use_gpu'] = True

    use_gpu = config['system'].get('use_gpu', True) and torch.backends.mps.is_available()
    device = 'mps' if use_gpu else 'cpu'
    print(f"Using device: {device}")

    seed = config['system'].get('seed', 42)
    set_random_seeds(seed)
    print(f"Random seed set to {seed}")

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_dir = config['system'].get('results_dir', './results')
    if args.output_dir:
        results_dir = args.output_dir
        print(f"Override: output_dir = {args.output_dir}")

    output_dir = os.path.join(results_dir, f"traffic_prediction_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    print("Configuration saved")

    if not any([args.train_source, args.train_target, args.train_adapter, args.perform_few_shot, args.analyze_research_questions]):
        args.train_source = True
        args.train_target = True
        args.train_adapter = True
        args.perform_few_shot = True
        args.analyze_research_questions = True
        print("No stages specified, running all stages")

    print("\n" + "="*50)
    print("INITIALIZING DATASETS")
    print("="*50)

    source_city = normalize_city_name(args.source_city)
    target_city = normalize_city_name(args.target_city)
    print(f"Using source city: {source_city}")
    print(f"Using target city: {target_city}")

    data_args = prepare_data_args(config['data_args'], source_city, target_city)
    config['data_args'] = data_args

    print("Initializing source city dataset...")
    source_dataset = TrafficDataset(
        data_args=config['data_args'],
        task_args=config['task_args'],
        model_args=config['model_args'],
        stage='source',
        test_data=target_city,
        add_target=False,
        cache_dir=config['system'].get('cache_dir', './cache'),
        use_weather=False,
        use_time_features=True
    )
    print(f"Source dataset initialized with {len(source_dataset)} samples")
    print(f"Available source city keys: {source_dataset.data_list}")

    print("\nInitializing target city dataset...")
    target_dataset = TrafficDataset(
        data_args=config['data_args'],
        task_args=config['task_args'],
        model_args=config['model_args'],
        stage='target',
        test_data=target_city,
        add_target=True,
        target_days=config['evaluation'].get('target_days', 3),
        cache_dir=config['system'].get('cache_dir', './cache'),
        use_weather=False,
        use_time_features=True
    )
    print(f"Target dataset initialized with {len(target_dataset)} samples")
    print(f"Available target city keys: {target_dataset.data_list}")

    print("\nInitializing traffic prediction framework...")
    framework = TrafficPredictionFramework(
        model_args=config['model_args'],
        task_args=config['task_args'],
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        device=device,
        learning_rate=config['model_args'].get('meta_lr', 0.01),
        weight_decay=config['training'].get('weight_decay', 1e-5),
        scheduler_factor=config['training'].get('lr_scheduler_factor', 0.5),
        scheduler_patience=config['training'].get('lr_scheduler_patience', 10),
        clip_grad_norm=5.0,
        log_dir=output_dir
    )

    setattr(framework, 'geometric_collate', geometric_collate_safe)

    print("Framework initialized successfully")

    results = {
        'source_metrics': None,
        'direct_target_metrics': None,
        'target_metrics': None,
        'few_shot_metrics': None,
        'few_shot_eval_metrics': None
    }

    source_model = None
    if args.train_source:
        print("\n" + "="*50)
        print("STAGE 1: SOURCE MODEL TRAINING")
        print("="*50)

        move_model_to_device(framework.source_model, device)

        source_model = framework.train_source_model(
            batch_size=config['task_args'].get('batch_size', 5),
            epochs=config['training'].get('source_epochs', 100),
            early_stopping_patience=config['training'].get('early_stop_patience', 15),
            source_city=source_city,
            val_ratio=args.val_ratio
        )
        print("Source model training completed")

        print("\nEvaluating source model on source city...")

        eval_model = move_model_to_device(source_model, 'cpu')
        clear_gpu_memory()

        source_metrics = framework.evaluate_multi_horizon(
            model=eval_model,
            dataset=source_dataset,
            city_name=source_city,
            horizons=config['evaluation'].get('time_horizons', [5, 15, 30, 60, 120])
        )

        print("\nSource model evaluation results:")
        print_metrics(source_metrics)

        results['source_metrics'] = source_metrics

        source_model = move_model_to_device(source_model, device)
        framework.source_model = source_model

    target_direct_model = None
    direct_target_metrics = None

    if args.train_target:
        print("\n" + "="*50)
        print("STAGE 2: TARGET MODEL DIRECT TRAINING")
        print("="*50)

        target_direct_model = framework.train_target_model(
            batch_size=config['task_args'].get('batch_size', 5),
            epochs=config['training'].get('target_epochs', 100),
            early_stopping_patience=config['training'].get('target_patience', 15),
            target_city=target_city,
            val_ratio=args.val_ratio,
            with_source_knowledge=config['training'].get('with_source_knowledge', True)
        )

        print("Target model direct training completed")

        print("\nEvaluating directly trained target model on target city...")

        eval_model = move_model_to_device(target_direct_model, 'cpu')
        clear_gpu_memory()

        direct_target_metrics = framework.evaluate_multi_horizon(
            model=eval_model,
            dataset=target_dataset,
            city_name=target_city,
            horizons=config['evaluation'].get('time_horizons', [5, 15, 30, 60, 120])
        )

        print("\nDirectly trained target model evaluation results:")
        print_metrics(direct_target_metrics)

        results['direct_target_metrics'] = direct_target_metrics

        target_direct_model = move_model_to_device(target_direct_model, device)
        framework.target_direct_model = target_direct_model

    adapted_model = None
    few_shot_metrics = None
    few_shot_eval_metrics = None

    if args.perform_few_shot and source_model is not None:
        print("\n" + "="*50)
        print("STAGE 4: FEW-SHOT ADAPTATION")
        print("="*50)

        k_shot = config['evaluation'].get('k_shot', 5)
        query_size = config['evaluation'].get('query_size', 120)

        print(f"Performing few-shot adaptation, {k_shot}-shot learning...")
        print(f"Using {query_size} query samples")

        framework.few_shot_learner.base_model = framework.source_model
        framework.few_shot_learner.device = device

        adapted_model, few_shot_metrics = framework.few_shot_adaptation(
            target_city=args.target_city,
            k_shot=args.k_shot,
            query_size=120
        )

        print("Few-shot adaptation completed")
        results['few_shot_metrics'] = few_shot_metrics

        print("\nEvaluating few-shot adapted model...")
        eval_model = move_model_to_device(adapted_model, 'cpu')
        clear_gpu_memory()

        few_shot_eval_metrics = framework.evaluate_multi_horizon(
            model=eval_model,
            dataset=target_dataset,
            city_name=target_city,
            horizons=config['evaluation'].get('time_horizons', [5, 15, 30, 60, 120])
        )

        print("\nFew-shot adapted model evaluation results:")
        print_metrics(few_shot_eval_metrics)

        results['few_shot_eval_metrics'] = few_shot_eval_metrics

        adapted_model = move_model_to_device(adapted_model, device)
        framework.few_shot_adapted_model = adapted_model

    print("\n" + "="*50)
    print("COMPARATIVE ANALYSIS OF TRAINING METHODS")
    print("="*50)

    methods = []
    method_metrics = {}

    if results['source_metrics'] is not None:
        methods.append("Source Model")
        method_metrics["Source Model"] = results['source_metrics']

    if results['direct_target_metrics'] is not None:
        methods.append("Direct Target Training")
        method_metrics["Direct Target Training"] = results['direct_target_metrics']

    if results['target_metrics'] is not None:
        methods.append("Adapter-Based Transfer")
        method_metrics["Adapter-Based Transfer"] = results['target_metrics']

    if results['few_shot_eval_metrics'] is not None:
        methods.append("Few-Shot Adaptation")
        method_metrics["Few-Shot Adaptation"] = results['few_shot_eval_metrics']

    if len(methods) > 1:
        print("\nComparative Results (RMSE):")

        first_method = methods[0]
        first_metrics = method_metrics[first_method]
        horizons = sorted([int(h) for h in first_metrics.keys()])

        header = "Horizon |"
        for method in methods:
            header += f" {method} |"
        print(header)
        print("-" * len(header))

        for horizon in horizons:
            row = f"{horizon:7d} |"
            for method in methods:
                if method in method_metrics and str(horizon) in method_metrics[method]:
                    rmse = method_metrics[method][str(horizon)].get('rmse', float('nan'))
                    row += f" {rmse:12.4f} |"
                else:
                    row += f" {'N/A':12s} |"
            print(row)

        print("\nBest method per horizon (RMSE):")
        for horizon in horizons:
            best_rmse = float('inf')
            best_method = "None"

            for method in methods:
                if method in method_metrics and str(horizon) in method_metrics[method]:
                    rmse = method_metrics[method][str(horizon)].get('rmse', float('inf'))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_method = method

            print(f"  Horizon {horizon}: {best_method} (RMSE = {best_rmse:.4f})")

    if hasattr(framework, 'plot_training_metrics'):
        framework.plot_training_metrics()
        print("Training metrics plot generated")

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(convert_for_json(results), f, indent=4)
    print(f"Results saved to {os.path.join(output_dir, 'results.json')}")

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print(f"Results saved to {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()