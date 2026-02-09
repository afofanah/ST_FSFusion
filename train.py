import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import os
import math
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random
import argparse
import json
from collections import defaultdict
from models.adaptive_fsl import MetaCrossDomainFusion, FewShotTrafficLearner, CrossCityAdapter
from torch_geometric.data import Batch as PyGBatch
from types import MethodType
from datasets import TrafficDataset

def geometric_collate(batch):
    data_list = [item[0] for item in batch]  
    A_wave_list = [item[1] for item in batch]  
    
    batched_data = PyGBatch.from_data_list(data_list)
    if all(isinstance(A, torch.Tensor) for A in A_wave_list):
        batched_A_wave = torch.stack(A_wave_list)
    else:
        batched_A_wave = A_wave_list
        
    return batched_data, batched_A_wave

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
        
        if not target_key:
            target_key = list(self_dataset.x_list.keys())[0]
        
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

class TrafficPredictionFramework:
    def __init__(self, 
                model_args, 
                task_args,
                source_dataset,
                target_dataset,
                device='mps',
                learning_rate=0.01,
                weight_decay=1e-5,
                scheduler_factor=0.5,
                scheduler_patience=5,
                clip_grad_norm=5.0,
                log_dir='logs'):
    
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
        
    def _save_configs(self):
        config = {
            'model_args': self.model_args,
            'task_args': self.task_args,
            'training_params': {
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'scheduler_factor': self.scheduler_factor,
                'scheduler_patience': self.scheduler_patience,
                'clip_grad_norm': self.clip_grad_norm
            }
        }
        
        with open(os.path.join(self.log_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
    def _setup_optimizer_and_scheduler(self):
        self.source_optimizer = optim.Adam(
            self.source_model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.adapter_optimizer = optim.Adam(
            self.adapter.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.source_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.source_optimizer,
            mode='min',
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            verbose=True
        )
        
        self.adapter_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.adapter_optimizer,
            mode='min',
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            verbose=True
        )
    
    def train_source_model(self, batch_size=5, epochs=200, early_stopping_patience=15, 
                           source_city='metr-la', val_ratio=0.2):
        from torch import autocast, GradScaler

        scaler = GradScaler()
        
        total_samples = len(self.source_dataset)
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
            self.source_model.train()
            train_loss = 0.0
            train_samples = 0
            
            train_loader = DataLoader(
                self.source_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=0,
                pin_memory=True,
                collate_fn=geometric_collate  
            )
            
            for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                data, A_wave = batch_data
                data = data.to(self.device)
                A_wave = A_wave.to(self.device)
                
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                use_autocast = True

                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device_type = 'mps'
                    use_autocast = False

                if use_autocast:
                    with autocast(device_type=device_type):
                        predictions, uncertainties, _ = self.source_model(data, A_wave)
                        loss = self._compute_uncertainty_loss(predictions, data.y, uncertainties)
                else:
                    predictions, uncertainties, _ = self.source_model(data, A_wave)
                    loss = self._compute_uncertainty_loss(predictions, data.y, uncertainties)
                    
                self.source_optimizer.zero_grad()
                if uncertainties is not None:
                    loss = self._compute_uncertainty_loss(predictions, data.y, uncertainties)
                else:
                    loss = nn.MSELoss()(predictions, data.y)
                
                scaler.scale(loss).backward()
                if self.clip_grad_norm > 0:
                    scaler.unscale_(self.source_optimizer)
                    torch.nn.utils.clip_grad_norm_(self.source_model.parameters(), self.clip_grad_norm)
                scaler.step(self.source_optimizer)
                scaler.update()

                batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
                train_loss += loss.item() * batch_size
                train_samples += batch_size
            
            avg_train_loss = train_loss / train_samples
            self.metrics['source_train_loss'].append(avg_train_loss)
            
            val_loader = DataLoader(
                self.source_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                num_workers=0,
                pin_memory=True,
                collate_fn=geometric_collate  
            )
            
            val_loss = self._validate_model(self.source_model, val_loader)
            self.metrics['source_val_loss'].append(val_loss)
            
            self.source_scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                self._save_model(self.source_model, 'source_model_best.pth')
            else:
                early_stopping_counter += 1
            
            if early_stopping_counter >= early_stopping_patience:
                break
        
        self._load_model(self.source_model, 'source_model_best.pth')
        return self.source_model

    def train_target_model(self, batch_size=5, epochs=150, early_stopping_patience=50, 
                        target_city='pems-bay', val_ratio=0.2, with_source_knowledge=True):
        target_city = self._normalize_city_name(target_city)
        
        self.target_model = MetaCrossDomainFusion(
            model_args=self.model_args,
            task_args=self.task_args,
            device=self.device
        )
        
        if with_source_knowledge:
            checkpoint_path = os.path.join(self.log_dir, 'checkpoints', 'source_model_best.pth')
            if os.path.exists(checkpoint_path):
                self.source_model.load_state_dict(torch.load(checkpoint_path))
                self.target_model.load_state_dict(self.source_model.state_dict())
        
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
            
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
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
                break
        
        self._load_model(self.target_model, f'target_model_{target_city}_best.pth')
        
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
    
    def few_shot_adaptation(self, target_city='pems-bay', k_shot=5, query_size=120, 
                   adaptation_steps=100, adaptation_lr=0.01, meta_batch_size=4):
        from copy import deepcopy
        adapted_model = deepcopy(self.source_model)
        adapted_model = adapted_model.to(self.device)
        target_city = self._normalize_city_name(target_city)
 
        self._load_model(self.source_model, 'source_model_best.pth')
        self.source_model = self.source_model.to(self.device)

        self.few_shot_learner = FewShotTrafficLearner(
            base_model=self.source_model,
            device=self.device
        )
        
        adaptation_metrics = {
            'support_loss': [],
            'query_loss': [],
            'rmse': [],
            'mae': []
        }
        
        best_query_loss = float('inf')
        best_adapted_model = None
        
        meta_tasks = []
        
        for i in range(meta_batch_size):
            support_data, support_labels, query_data, query_labels = self.target_dataset.get_few_shot_support_query(
                dataset_name=target_city,
                k_shot=k_shot,
                query_size=query_size
            )
            
            if support_data is not None and query_data is not None:
                meta_tasks.append((support_data, support_labels, query_data, query_labels))
        
        if not meta_tasks:
            return self.source_model, {'rmse': float('nan'), 'mae': float('nan')}
        
        from copy import deepcopy
        adapted_model = deepcopy(self.source_model)
        adapted_model = adapted_model.to(self.device)
        adapted_model.train()

        adaptation_optimizer = optim.Adam(
            adapted_model.parameters(),
            lr=adaptation_lr,
            weight_decay=self.weight_decay * 0.1  
        )
        
        adaptation_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            adaptation_optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        for state in adaptation_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

        with tqdm(total=adaptation_steps, desc="Adaptation Progress") as pbar:
            for step in range(adaptation_steps):
                step_support_loss = 0.0
                step_query_loss = 0.0
                
                for task_idx, (support_data, support_labels, query_data, query_labels) in enumerate(meta_tasks):
                    if isinstance(support_data, torch.Tensor):
                        support_data = support_data.to(self.device)
                    if isinstance(support_labels, torch.Tensor):
                        support_labels = support_labels.to(self.device)
                    if isinstance(query_data, torch.Tensor):
                        query_data = query_data.to(self.device)
                    if isinstance(query_labels, torch.Tensor):
                        query_labels = query_labels.to(self.device)
                
                    support_pyg = self._convert_to_pyg_if_needed(support_data, support_labels)
                    query_pyg = self._convert_to_pyg_if_needed(query_data, query_labels)
                    
                    if hasattr(support_pyg, 'x'):
                        support_pyg = support_pyg.to(self.device)
                    if hasattr(query_pyg, 'x'):
                        query_pyg = query_pyg.to(self.device)
                    
                    adaptation_optimizer.zero_grad()
                    
                    if isinstance(support_pyg, torch.Tensor):
                        support_pred, support_uncertainty, _ = adapted_model(support_pyg)
                        support_loss = nn.MSELoss()(support_pred, support_labels)
                    elif hasattr(support_pyg, 'x') and hasattr(support_pyg, 'y'):
                        support_pred, support_uncertainty, _ = adapted_model(support_pyg)
                        
                        if support_uncertainty is not None:
                            support_loss = self._compute_uncertainty_loss(support_pred, support_pyg.y, support_uncertainty)
                        else:
                            support_loss = nn.MSELoss()(support_pred, support_pyg.y)
                    else:
                        continue

                    support_loss.backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(adapted_model.parameters(), self.clip_grad_norm)
                    adaptation_optimizer.step()
                    
                    step_support_loss += support_loss.item()
                    
                    with torch.no_grad():
                        adapted_model.eval()
                        
                        if isinstance(query_pyg, torch.Tensor):
                            query_pred, query_uncertainty, _ = adapted_model(query_pyg)
                            query_loss = nn.MSELoss()(query_pred, query_labels)
                        elif hasattr(query_pyg, 'x') and hasattr(query_pyg, 'y'):
                            query_pred, query_uncertainty, _ = adapted_model(query_pyg)
                            
                            if query_uncertainty is not None:
                                query_loss = self._compute_uncertainty_loss(query_pred, query_pyg.y, query_uncertainty)
                            else:
                                query_loss = nn.MSELoss()(query_pred, query_pyg.y)
                        else:
                            continue
                            
                        step_query_loss += query_loss.item()
                    
                    adapted_model.train()
                    
                avg_support_loss = step_support_loss / len(meta_tasks)
                avg_query_loss = step_query_loss / len(meta_tasks)
                
                adaptation_metrics['support_loss'].append(avg_support_loss)
                adaptation_metrics['query_loss'].append(avg_query_loss)
                
                adaptation_scheduler.step(avg_query_loss)
                
                if avg_query_loss < best_query_loss:
                    best_query_loss = avg_query_loss
                    best_adapted_model = deepcopy(adapted_model)
                    self._save_model(adapted_model, f'few_shot_adapted_{target_city}.pth')

                pbar.update(1)
                pbar.set_postfix({
                    'support_loss': f'{avg_support_loss:.6f}',
                    'query_loss': f'{avg_query_loss:.6f}'
                })
            
                if (step + 1) % 10 == 0 or step == adaptation_steps - 1:
                    all_preds = []
                    all_truths = []
                    
                    with torch.no_grad():
                        adapted_model.eval()
                        
                        for _, (_, _, query_data, query_labels) in enumerate(meta_tasks):
                            if isinstance(query_data, torch.Tensor):
                                query_data = query_data.to(self.device)
                            
                            query_pyg = self._convert_to_pyg_if_needed(query_data, None)
                            if hasattr(query_pyg, 'x'):
                                query_pyg = query_pyg.to(self.device)
                            
                            if isinstance(query_pyg, torch.Tensor):
                                pred, _, _ = adapted_model(query_pyg)
                            elif hasattr(query_pyg, 'x'):
                                pred, _, _ = adapted_model(query_pyg)
                            else:
                                continue
                            
                            all_preds.append(pred.cpu())
                            all_truths.append(query_labels.cpu() if isinstance(query_labels, torch.Tensor) else query_pyg.y.cpu())
                    
                    if all_preds and all_truths:
                        all_preds = torch.cat(all_preds, dim=0)
                        all_truths = torch.cat(all_truths, dim=0)
                        rmse, mae, _ = self._calculate_metrics(all_truths, all_preds)
                        
                        adaptation_metrics['rmse'].append(rmse)
                        adaptation_metrics['mae'].append(mae)

        if best_adapted_model is not None:
            adapted_model = best_adapted_model
        
        adapted_model.eval()
        all_query_preds = []
        all_query_labels = []
        
        with torch.no_grad():
            for _, (_, _, query_data, query_labels) in enumerate(meta_tasks):
                if isinstance(query_data, torch.Tensor):
                    query_data = query_data.to(self.device)
                
                query_pyg = self._convert_to_pyg_if_needed(query_data, None)
                if hasattr(query_pyg, 'x'):
                    query_pyg = query_pyg.to(self.device)
                
                if isinstance(query_pyg, torch.Tensor):
                    pred, _, _ = adapted_model(query_pyg)
                    all_query_preds.append(pred.cpu())
                    all_query_labels.append(query_labels.cpu())
                elif hasattr(query_pyg, 'x') and hasattr(query_pyg, 'y'):
                    pred, _, _ = adapted_model(query_pyg)
                    all_query_preds.append(pred.cpu())
                    all_query_labels.append(query_pyg.y.cpu())
        
        if all_query_preds and all_query_labels:
            all_query_preds = torch.cat(all_query_preds, dim=0).numpy()
            all_query_labels = torch.cat(all_query_labels, dim=0).numpy()
            
            rmse = np.sqrt(mean_squared_error(all_query_labels, all_query_preds))
            mae = mean_absolute_error(all_query_labels, all_query_preds)
            
            final_metrics = {'rmse': rmse, 'mae': mae}
        else:
            final_metrics = {'rmse': float('nan'), 'mae': float('nan')}
        
        self.metrics['few_shot_rmse'] = final_metrics['rmse']
        self.metrics['few_shot_mae'] = final_metrics['mae']
        self.metrics['adaptation_support_loss'] = adaptation_metrics['support_loss']
        self.metrics['adaptation_query_loss'] = adaptation_metrics['query_loss']
        self._plot_adaptation_metrics(adaptation_metrics, target_city)
        self._save_model(adapted_model, f'adapted_model_{target_city}_final.pth')
        
        return adapted_model, final_metrics

    def evaluate_multi_horizon(self, model, dataset, city_name, horizons=None):
        
        if hasattr(dataset, 'x_list'):
            data_source = dataset
        else:
            class TempDataset:
                def __init__(self, tensor):
                    self.x_list = {'default': tensor}
                    self.y_list = {'default': torch.zeros_like(tensor)}
                    self.edge_index_list = {}
                    self.edge_attr_list = {}
                    self.time_granularity = 5
            data_source = TempDataset(dataset)
        
        if horizons is None:
            horizons = [5, 10, 15, 30, 60, 120]
            
        default_results = {h: {'rmse': float('nan'), 'mae': float('nan'), 'mape': float('nan')} for h in horizons}
        
        model = model.cpu()
        for module in model.modules():
            if hasattr(module, 'device'):
                module.device = torch.device('cpu')
        
        model.eval()
        city_name = self._normalize_city_name(city_name)
        
        actual_key = self._find_matching_key(data_source.y_list.keys(), city_name)
        if not actual_key:
            return default_results
            
        batch_size = min(100, len(data_source.x_list[actual_key]))
        x_data = data_source.x_list[actual_key][:batch_size]
        
        from torch_geometric.data import Data, Batch
        
        data_list = []
        for i in range(batch_size):
            edge_index = data_source.edge_index_list.get(actual_key, None)
            edge_attr = data_source.edge_attr_list.get(actual_key, None)
            
            data_i = Data(x=x_data[i].unsqueeze(0))
            if edge_index is not None:
                data_i.edge_index = edge_index
            if edge_attr is not None:
                data_i.edge_attr = edge_attr
                
            data_list.append(data_i)
            
        batch_data = Batch.from_data_list(data_list)

        with torch.no_grad():
            result = model(batch_data)
            if isinstance(result, tuple) and len(result) >= 1:
                predictions = result[0]
            else:
                predictions = result
            
        time_granularity = getattr(data_source, 'time_granularity', 5)
        results = {}
        
        for horizon in horizons:
            
            horizon_steps = min(horizon // time_granularity, predictions.size(1))
            
            if horizon_steps <= 0:
                results[horizon] = default_results[horizon]
                continue
            
            horizon_truth_list = []
            valid_indices = []
            
            for i in range(batch_size):
                horizon_key = f"{actual_key}_{horizon}"
                horizon_y = None
                
                if horizon_key in data_source.y_list and i < len(data_source.y_list[horizon_key]):
                    horizon_y = data_source.y_list[horizon_key][i]
                elif hasattr(data_source, 'get_prediction_for_horizon'):
                    horizon_y = data_source.get_prediction_for_horizon(horizon, i, actual_key)
                elif actual_key in data_source.y_list and i < len(data_source.y_list[actual_key]):
                    y_data = data_source.y_list[actual_key][i]
                    if y_data.size(0) >= horizon_steps:
                        horizon_y = y_data[:horizon_steps]
                
                if horizon_y is not None:
                    horizon_truth_list.append(horizon_y.cpu())
                    valid_indices.append(i)
            
            if not horizon_truth_list:
                results[horizon] = default_results[horizon]
                continue
            
            valid_predictions = predictions[valid_indices]
            horizon_pred = valid_predictions[:, :horizon_steps]
            horizon_truth = torch.stack(horizon_truth_list)
            horizon_pred = horizon_pred.cpu()
            
            if horizon_truth.dim() != horizon_pred.dim():
                if horizon_truth.dim() > horizon_pred.dim():
                    for _ in range(horizon_truth.dim() - horizon_pred.dim()):
                        horizon_pred = horizon_pred.unsqueeze(-1)
                else:
                    for _ in range(horizon_pred.dim() - horizon_truth.dim()):
                        horizon_truth = horizon_truth.unsqueeze(-1)
                        
            rmse, mae, mape = self._calculate_metrics(horizon_truth, horizon_pred)
            
            results[horizon] = {
                'rmse': rmse,
                'mae': mae,
                'mape': mape
            }
        
        return results

    def train_cross_city_adapter(self, batch_size=32, epochs=150, early_stopping_patience=10,
                            source_city='metr-la', target_city='pems-bay'):
        source_city = self._normalize_city_name(source_city)
        target_city = self._normalize_city_name(target_city)
        
        if 'hidden_dim' in self.model_args and self.model_args['hidden_dim'] <= 1:
            self.model_args['hidden_dim'] = 16
        
        if 'message_dim' in self.model_args and self.model_args['message_dim'] <= 1:
            self.model_args['message_dim'] = 16
        
        if 'meta_dim' in self.model_args and self.model_args['meta_dim'] <= 1:
            self.model_args['meta_dim'] = 16
        
        for param in self.source_model.parameters():
            param.requires_grad = False
        
        target_model = MetaCrossDomainFusion(
            model_args=self.model_args,
            task_args=self.task_args,
            device=self.device
        )
        target_model = target_model.to(self.device)

        target_optimizer = optim.Adam(
            list(target_model.parameters()) + list(self.adapter.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        source_len = max(1, len(self.source_dataset))
        target_len = max(1, len(self.target_dataset))
        source_steps = max(1, source_len // batch_size)
        target_steps = max(1, target_len // batch_size)
        steps_per_epoch = min(source_steps, target_steps)
        
        best_val_loss = float('inf')
        early_stopping_counter = 0

        source_indices = list(range(source_len))
        target_indices = list(range(target_len))
        random.shuffle(source_indices)
        random.shuffle(target_indices)
        
        source_train_idx = source_indices[:int(0.8 * source_len)]
        source_val_idx = source_indices[int(0.8 * source_len):]
        target_train_idx = target_indices[:int(0.8 * target_len)]
        target_val_idx = target_indices[int(0.8 * target_len):]
        
        source_train_sampler = torch.utils.data.SubsetRandomSampler(source_train_idx)
        source_val_sampler = torch.utils.data.SubsetRandomSampler(source_val_idx)
        target_train_sampler = torch.utils.data.SubsetRandomSampler(target_train_idx)
        target_val_sampler = torch.utils.data.SubsetRandomSampler(target_val_idx)
        
        source_train_loader = DataLoader(
            self.source_dataset,
            batch_size=batch_size,
            sampler=source_train_sampler,
            num_workers=0,
            collate_fn=geometric_collate
        )
        
        source_val_loader = DataLoader(
            self.source_dataset,
            batch_size=batch_size,
            sampler=source_val_sampler,
            num_workers=0,
            collate_fn=geometric_collate
        )
        
        target_train_loader = DataLoader(
            self.target_dataset,
            batch_size=batch_size,
            sampler=target_train_sampler,
            num_workers=0,
            collate_fn=geometric_collate
        )
        
        target_val_loader = DataLoader(
            self.target_dataset,
            batch_size=batch_size,
            sampler=target_val_sampler,
            num_workers=0,
            collate_fn=geometric_collate
        )
        
        for epoch in range(epochs):
            self.source_model.eval()  
            self.adapter.train()
            target_model.train()
            
            train_loss = 0.0
            train_samples = 0
            
            source_iter = iter(source_train_loader)
            target_iter = iter(target_train_loader)
            
            with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{epochs} [Train]") as pbar:
                for step in range(steps_per_epoch):
                    source_batch = next(source_iter) if len(source_train_loader) > step else next(iter(source_train_loader))
                    target_batch = next(target_iter) if len(target_train_loader) > step else next(iter(target_train_loader))
                    
                    source_data, source_A_wave = source_batch
                    target_data, target_A_wave = target_batch
                    
                    source_data = source_data.to(self.device)
                    if source_A_wave is not None:
                        source_A_wave = source_A_wave.to(self.device)
                        
                    target_data = target_data.to(self.device)
                    if target_A_wave is not None:
                        target_A_wave = target_A_wave.to(self.device)
                    
                    with torch.no_grad():
                        if source_A_wave is not None:
                            _, _, source_features = self.source_model(source_data, source_A_wave)
                        else:
                            _, _, source_features = self.source_model(source_data)
                    
                    target_optimizer.zero_grad()
                    
                    if target_A_wave is not None:
                        target_pred, target_uncertainty, target_features = target_model(target_data, target_A_wave)
                    else:
                        target_pred, target_uncertainty, target_features = target_model(target_data)
                    
                    adapted_features, adaptation_loss = self.adapter(source_features, target_features)
                    
                    if hasattr(target_data, 'y') and target_data.y is not None:
                        if target_uncertainty is not None:
                            prediction_loss = self._compute_uncertainty_loss(target_pred, target_data.y, target_uncertainty)
                        else:
                            prediction_loss = nn.MSELoss()(target_pred, target_data.y)
                        loss = prediction_loss + 0.1 * adaptation_loss
                    else:
                        loss = adaptation_loss
                    
                    loss.backward()
                    if self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(target_model.parameters()) + list(self.adapter.parameters()),
                            self.clip_grad_norm
                        )
                    target_optimizer.step()
                    
                    batch_size = target_data.num_graphs if hasattr(target_data, 'num_graphs') else 1
                    train_loss += loss.item() * batch_size
                    train_samples += batch_size
                    
                    pbar.update(1)
                    pbar.set_postfix({'loss': loss.item()})
            
            avg_train_loss = train_loss / train_samples if train_samples > 0 else float('inf')
            
            self.source_model.eval()
            self.adapter.eval()
            target_model.eval()
            
            val_loss = 0.0
            val_samples = 0
            
            source_val_iter = iter(source_val_loader)
            target_val_iter = iter(target_val_loader)
            
            with torch.no_grad():
                for step in range(min(len(source_val_loader), len(target_val_loader))):
                    source_batch = next(source_val_iter) if len(source_val_loader) > step else next(iter(source_val_loader))
                    target_batch = next(target_val_iter) if len(target_val_loader) > step else next(iter(target_val_loader))
                    
                    source_data, source_A_wave = source_batch
                    target_data, target_A_wave = target_batch
                    
                    source_data = source_data.to(self.device)
                    if source_A_wave is not None:
                        source_A_wave = source_A_wave.to(self.device)
                        
                    target_data = target_data.to(self.device)
                    if target_A_wave is not None:
                        target_A_wave = target_A_wave.to(self.device)
                    
                    if source_A_wave is not None:
                        _, _, source_features = self.source_model(source_data, source_A_wave)
                    else:
                        _, _, source_features = self.source_model(source_data)
                    
                    if target_A_wave is not None:
                        target_pred, target_uncertainty, target_features = target_model(target_data, target_A_wave)
                    else:
                        target_pred, target_uncertainty, target_features = target_model(target_data)
                    
                    adapted_features, adaptation_loss = self.adapter(source_features, target_features)
                    
                    if hasattr(target_data, 'y') and target_data.y is not None:
                        if target_uncertainty is not None:
                            prediction_loss = self._compute_uncertainty_loss(target_pred, target_data.y, target_uncertainty)
                        else:
                            prediction_loss = nn.MSELoss()(target_pred, target_data.y)
                        loss = prediction_loss + 0.1 * adaptation_loss
                    else:
                        loss = adaptation_loss
                    
                    batch_size = target_data.num_graphs if hasattr(target_data, 'num_graphs') else 1
                    val_loss += loss.item() * batch_size
                    val_samples += batch_size
            
            avg_val_loss = val_loss / val_samples if val_samples > 0 else float('inf')
            
            self.adapter_scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stopping_counter = 0
                
                self._save_model(self.adapter, f'adapter_{source_city}_to_{target_city}_best.pth')
                self._save_model(target_model, f'target_model_{target_city}_best.pth')
            else:
                early_stopping_counter += 1
                
                if early_stopping_counter >= early_stopping_patience:
                    break
        
        self._load_model(self.adapter, f'adapter_{source_city}_to_{target_city}_best.pth')
        self._load_model(target_model, f'target_model_{target_city}_best.pth')
        
        return self.adapter, target_model
    
    def _plot_adaptation_metrics(self, metrics, target_city):
        plt.figure(figsize=(12, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(metrics['support_loss'], label='Support Loss')
        plt.plot(metrics['query_loss'], label='Query Loss')
        plt.yscale('log')
        plt.title(f'Few-Shot Adaptation Losses for {target_city}')
        plt.xlabel('Adaptation Step')
        plt.ylabel('Loss (log scale)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if 'rmse' in metrics and len(metrics['rmse']) > 0:
            steps = list(range(0, len(metrics['support_loss']), 10))
            if len(steps) < len(metrics['rmse']):
                steps.append(len(metrics['support_loss']) - 1)
            
            plt.subplot(2, 1, 2)
            plt.plot(steps, metrics['rmse'], 'o-', label='RMSE')
            plt.plot(steps, metrics['mae'], 's-', label='MAE')
            plt.title(f'Few-Shot Adaptation Performance for {target_city}')
            plt.xlabel('Adaptation Step')
            plt.ylabel('Error')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.log_dir, f'few_shot_adaptation_{target_city}.png')
        plt.savefig(save_path)
        plt.close()

    def _validate_model(self, model, val_loader):
        model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                data, A_wave = batch_data
                data = data.to(self.device)
                A_wave = A_wave.to(self.device)
                predictions, uncertainties, _ = model(data, A_wave)
                if uncertainties is not None:
                    loss = self._compute_uncertainty_loss(predictions, data.y, uncertainties)
                else:
                    loss = nn.MSELoss()(predictions, data.y)
                batch_size = data.num_graphs if hasattr(data, 'num_graphs') else 1
                val_loss += loss.item() * batch_size
                val_samples += batch_size
        
        return val_loss / val_samples if val_samples > 0 else float('inf')
    
    def _find_matching_key(self, available_keys, city_name):
        if city_name in available_keys:
            return city_name
        if city_name.lower() in available_keys:
            return city_name.lower()
        if city_name.replace('-', '_') in available_keys:
            return city_name.replace('-', '_')
        if city_name.replace('_', '-') in available_keys:
            return city_name.replace('_', '-')
        if '-' in city_name and city_name.split('-')[0] in available_keys:
            return city_name.split('-')[0]
        if '_' in city_name and city_name.split('_')[0] in available_keys:
            return city_name.split('_')[0]

        for key in available_keys:
            if city_name in key or key in city_name:
                return key

        if available_keys:
            return list(available_keys)[0]
        
        return None
    
    def _normalize_city_name(self, city_name):
        if isinstance(city_name, torch.Tensor):
            return str(city_name.item())
        if isinstance(city_name, str) and '/' in city_name:
            return city_name.split('/')[-1].split('.')[0]
        return city_name
    
    def _compute_uncertainty_loss(self, predictions, targets, uncertainties):
        uncertainties = torch.clamp(uncertainties, min=1e-6)
        squared_error = (predictions - targets) ** 2
        loss = 0.5 * (squared_error / uncertainties + torch.log(uncertainties))
        
        return torch.mean(loss)
    
    def _convert_to_pyg_if_needed(self, data, labels=None):
        from torch_geometric.data import Data, Batch
        
        if hasattr(data, 'x') and hasattr(data, 'edge_index'):
            return data
        if hasattr(data, 'batch'):
            return data
        if isinstance(data, torch.Tensor):
            pyg_data = Data(x=data)
            if labels is not None and isinstance(labels, torch.Tensor):
                pyg_data.y = labels
            return pyg_data
        
        if isinstance(data, list) and all(hasattr(d, 'x') for d in data):
            return Batch.from_data_list(data)
        return data
    
    def _calculate_metrics(self, truth, pred):
        truth = truth.cpu().contiguous()
        pred = pred.cpu().contiguous()
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
            
        if not math.isfinite(rmse):
            rmse = float('nan')
        if not math.isfinite(mae):
            mae = float('nan')
        if not math.isfinite(mape):
            mape = float('nan')
        
        return rmse, mae, mape
    
    def _save_model(self, model, filename):
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', filename)
        torch.save(model.state_dict(), checkpoint_path)
    
    def _load_model(self, model, filename):
        checkpoint_path = os.path.join(self.log_dir, 'checkpoints', filename)
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
    
    def plot_training_metrics(self, save_path=None):
        plt.figure(figsize=(15, 10))
        
        plot_idx = 1
        metrics_to_plot = [
            ('source_train_loss', 'source_val_loss', 'Source Model Training'),
            ('target_train_loss', 'target_val_loss', 'Target Model Training')
        ]
        
        for train_key, val_key, title in metrics_to_plot:
            if train_key in self.metrics and val_key in self.metrics:
                plt.subplot(2, 2, plot_idx)
                plt.plot(self.metrics[train_key], label='Train Loss')
                plt.plot(self.metrics[val_key], label='Validation Loss')
                plt.title(title)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plot_idx += 1
        
        if 'few_shot_rmse' in self.metrics:
            plt.subplot(2, 2, plot_idx)
            plt.bar(['RMSE', 'MAE'], [self.metrics['few_shot_rmse'], self.metrics['few_shot_mae']])
            plt.title('Few-Shot Adaptation Performance')
            plt.ylabel('Error')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.log_dir, 'training_metrics.png')
        
        plt.savefig(save_path)
        plt.close()

    patch_train_cross_city_adapter = patch_train_cross_city_adapter

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

def run_end_to_end_training(args):
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(args.log_dir, f"traffic_prediction_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    
    with open(args.config_file, 'r') as f:
        config = json.load(f)

    model_args = config['model_args']
    task_args = config['task_args']
    data_args = config['data_args']
    
    device = torch.device(args.device if torch.backends.mps.is_available() else "cpu")

    source_dataset = TrafficDataset(
        data_args=data_args,
        task_args=task_args,
        model_args=model_args,
        stage='source',
        test_data=args.target_city,
        add_target=False,
        cache_dir=args.cache_dir,
        use_weather=args.use_weather,
        use_time_features=args.use_time_features
    )
    
    target_dataset = TrafficDataset(
        data_args=data_args,
        task_args=task_args,
        model_args=model_args,
        stage='target',
        test_data=args.target_city,
        add_target=True,
        target_days=args.target_days,
        cache_dir=args.cache_dir,
        use_weather=args.use_weather,
        use_time_features=args.use_time_features
    )
    
    framework = TrafficPredictionFramework(
        model_args=model_args,
        task_args=task_args,
        source_dataset=source_dataset,
        target_dataset=target_dataset,
        device=device,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        scheduler_factor=float(args.scheduler_factor),
        scheduler_patience=int(args.scheduler_patience),
        clip_grad_norm=float(args.clip_grad_norm),
        log_dir=log_dir
    )
    
    results = {}
    
    if args.train_source:
        source_model = framework.train_source_model(
            batch_size=args.batch_size,
            epochs=args.source_epochs, 
            early_stopping_patience=args.source_patience,
            source_city=args.source_city,
            val_ratio=args.val_ratio
        )
        
        source_metrics = framework.evaluate_multi_horizon(
            model=source_model,
            dataset=source_dataset,
            city_name=args.source_city,
            horizons=[5, 15, 30, 60, 120]
        )
        
        results['source_metrics'] = source_metrics
    
    if args.train_adapter:
        adapter, target_model = framework.train_cross_city_adapter(
            batch_size=args.batch_size,
            epochs=args.adapter_epochs,
            early_stopping_patience=args.adapter_patience,
            source_city=args.source_city,
            target_city=args.target_city
        )
        
        target_metrics = framework.evaluate_multi_horizon(
            model=target_model,
            dataset=target_dataset,
            city_name=args.target_city,
            horizons=[5, 15, 30, 60, 120]
        )
        
        results['target_metrics'] = target_metrics
    
    if args.perform_few_shot:
        adapted_model, few_shot_metrics = framework.few_shot_adaptation(
            target_city=args.target_city,
            k_shot=args.k_shot,
            query_size=args.query_size
        )
        
        few_shot_eval_metrics = framework.evaluate_multi_horizon(
            model=adapted_model,
            dataset=target_dataset,
            city_name=args.target_city,
            horizons=[5, 15, 30, 60, 120]
        )
        
        results['few_shot_metrics'] = few_shot_metrics
        results['few_shot_eval_metrics'] = few_shot_eval_metrics
    
    framework.plot_training_metrics()
    
    with open(os.path.join(log_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spatio-Temporal Feature Fusion and Few-Shot Learning Framework for Traffic Flow Prediction")
    
    parser.add_argument('--config_file', type=str, default='./config.yaml', help='Path to configuration file')
    parser.add_argument('--source_city', type=str, default='./data/metr-la', help='Source city dataset name')
    parser.add_argument('--target_city', type=str, default='./data/pems-bay', help='Target city dataset name')
    parser.add_argument('--target_days', type=int, default=3, help='Number of days for target city data')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='Directory to cache processed datasets')
    parser.add_argument('--use_weather', action='store_true', help='Use weather data if available')
    parser.add_argument('--use_time_features', action='store_true', help='Use time features')
    
    parser.add_argument('--device', type=str, default='mps', help='Device to use')
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay for L2 regularization')
    parser.add_argument('--scheduler_factor', type=float, default=0.5, help='Factor to reduce learning rate')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='Patience for learning rate scheduler')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Maximum norm for gradient clipping')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of data to use for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory to save logs and checkpoints')
    
    parser.add_argument('--train_source', action='store_true', help='Train source model')
    parser.add_argument('--train_target', action='store_true', help='Train target model directly')
    parser.add_argument('--train_adapter', action='store_true', help='Train cross-city adapter')
    parser.add_argument('--perform_few_shot', action='store_true', help='Perform few-shot adaptation')
    parser.add_argument('--source_epochs', type=int, default=200, help='Maximum epochs for source model training')
    parser.add_argument('--source_patience', type=int, default=15, help='Early stopping patience for source model')
    parser.add_argument('--target_epochs', type=int, default=150, help='Maximum epochs for target model training')
    parser.add_argument('--target_patience', type=int, default=15, help='Early stopping patience for target model')
    parser.add_argument('--with_source_knowledge', action='store_true', help='Initialize target model with source model weights')
    parser.add_argument('--adapter_epochs', type=int, default=120, help='Maximum epochs for adapter training')
    parser.add_argument('--adapter_patience', type=int, default=10, help='Early stopping patience for adapter')
    
    parser.add_argument('--k_shot', type=int, default=5, help='K-shot for few-shot learning')
    parser.add_argument('--query_size', type=int, default=120, help='Query set size for few-shot evaluation')
    
    args = parser.parse_args()
    
    if not any([args.train_source, args.train_target, args.train_adapter, args.perform_few_shot]):
        args.train_source = True
        args.train_target = True
        args.train_adapter = True
        args.perform_few_shot = True
    
    run_end_to_end_training(args)