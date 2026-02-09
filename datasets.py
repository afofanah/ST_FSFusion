import torch
from torch_geometric.data import Data, Dataset, DataLoader
import numpy as np
import random
import sys 
import scipy.sparse as sp
import os
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected, add_self_loops

os.environ["TORCH_USE_WEIGHTS_ONLY_DEFAULT"] = "0"

_original_torch_load = torch.load

def _safe_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _safe_load

class BBDefinedError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo

class TrafficDataset(Dataset):
    def __init__(self, 
                 data_args, 
                 task_args, 
                 model_args, 
                 stage='source', 
                 test_data='metr-la', 
                 add_target=True, 
                 target_days=3,
                 time_granularity=5,
                 cache_dir=None,
                 use_weather=False,
                 use_time_features=True):
        super(TrafficDataset, self).__init__()
        
        if not isinstance(model_args, dict):
            raise TypeError(f"model_args must be a dictionary, but got {type(model_args)}.")
        
        self.data_args = data_args
        self.task_args = task_args
        self.model_args = model_args
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.stage = stage
        self.add_target = add_target
        self.test_data = test_data
        self.target_days = target_days
        self.time_granularity = time_granularity
        self.cache_dir = cache_dir
        self.use_weather = use_weather
        self.use_time_features = use_time_features
        
        if self.cache_dir and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
        self.time_horizons = np.array([5, 10, 15, 30, 60, 120, 180, 240, 1440])  
        self.A_list = {}  
        self.edge_index_list = {}
        self.edge_attr_list = {}  
        self.x_list = {}  
        self.y_list = {} 
        self.means_list, self.stds_list = {}, {} 
        self.data_objects = []
        self.node_features = {}  
        self.time_features = {}  
        
        self.data_list = data_args.get('data_keys', [])
        if not self.data_list:
            raise ValueError("No data keys provided in data_args")
            
        self.load_data(stage, test_data)
        if self.add_target and self.test_data not in self.data_list:
            self.data_list = np.append(self.data_list, self.test_data)

    def load_data(self, stage, test_data):
        for dataset_name in self.data_list:
            if dataset_name not in self.data_args:
                raise KeyError(f"{dataset_name} not found in data_args.")
            
            cache_file = None
            if self.cache_dir:
                cache_file = os.path.join(self.cache_dir, f"{dataset_name}_{stage}_{self.his_num}_{self.pred_num}.pt")
                if os.path.exists(cache_file):
                    cached_data = torch.load(cache_file, weights_only=False)
                    self.A_list[dataset_name] = cached_data['A']
                    self.edge_index_list[dataset_name] = cached_data['edge_index']
                    self.edge_attr_list[dataset_name] = cached_data.get('edge_attr', None)
                    self.x_list[dataset_name] = cached_data['x']
                    self.y_list[dataset_name] = cached_data['y']
                    self.means_list[dataset_name] = cached_data['means']
                    self.stds_list[dataset_name] = cached_data['stds']
                    
                    if 'node_features' in cached_data:
                        self.node_features[dataset_name] = cached_data['node_features']
                    
                    if 'time_features' in cached_data:
                        self.time_features[dataset_name] = cached_data['time_features']
                    
                    data_i = Data(
                        x=self.x_list[dataset_name], 
                        edge_index=self.edge_index_list[dataset_name],
                        edge_attr=self.edge_attr_list[dataset_name] if dataset_name in self.edge_attr_list else None,
                        y=self.y_list[dataset_name]
                    )
                    data_i.data_name = dataset_name
                    self.data_objects.append(data_i)
                    continue
    
            adjacency_matrix_path = self.data_args[dataset_name]['adjacency_matrix_path']
            A = np.load(adjacency_matrix_path)
            
            if A.shape[0] != A.shape[1]:
                raise ValueError(f"Adjacency matrix for {dataset_name} is not square: {A.shape}")
            
            from utils import get_normalized_adj
            A_normalized = get_normalized_adj(A)
            self.A_list[dataset_name] = torch.from_numpy(A_normalized).float()
            edge_index, edge_attr, _ = self.get_graph_structure(adjacency_matrix_path)
            self.edge_index_list[dataset_name] = edge_index
            
            if edge_attr is not None:
                self.edge_attr_list[dataset_name] = edge_attr
     
            dataset_path = self.data_args[dataset_name]['dataset_path']
            X = np.load(dataset_path)
            
            if X.ndim != 3:
                raise ValueError(f"Incorrect data shape for {dataset_name}: {X.shape}. Expected 3D.")
            X = X.transpose((1, 2, 0)).astype(np.float32)
            means = np.mean(X, axis=(0, 2))
            stds = np.std(X, axis=(0, 2))
            stds[stds < 1e-10] = 1.0  
            self.means_list[dataset_name] = means
            self.stds_list[dataset_name] = stds
            
            X_normalized = (X - means.reshape(1, -1, 1)) / stds.reshape(1, -1, 1)
            
            if stage == 'target':
                X_normalized = X_normalized[:, :, :288 * self.target_days]
            elif stage == 'target_maml':
                X_normalized = X_normalized[:, :, :288 * self.target_days]
            elif stage == 'test':
                X_normalized = X_normalized[:, :, int(X_normalized.shape[2] * 0.8):]
                
            if self.use_time_features:
                time_features = self.generate_time_features(X_normalized.shape[0], X_normalized.shape[2])
                self.time_features[dataset_name] = time_features
                
            from utils import generate_dataset
            x_inputs, y_outputs = generate_dataset(X_normalized, self.his_num, self.pred_num, means, stds)
            self.x_list[dataset_name] = torch.tensor(x_inputs, dtype=torch.float)
            self.y_list[dataset_name] = torch.tensor(y_outputs, dtype=torch.float)
            
            for horizon in self.time_horizons:
                time_step_count = min(horizon // self.time_granularity, y_outputs.shape[1])
                if time_step_count > 0:
                    horizon_key = f'{dataset_name}_{horizon}'
                    expanded_outputs = y_outputs[:, :time_step_count, :]
                    self.y_list[horizon_key] = torch.tensor(expanded_outputs, dtype=torch.float)
            
            data_i = Data(
                x=self.x_list[dataset_name], 
                edge_index=self.edge_index_list[dataset_name],
                edge_attr=self.edge_attr_list[dataset_name] if dataset_name in self.edge_attr_list else None,
                y=self.y_list[dataset_name]
            )
            data_i.data_name = dataset_name
            self.data_objects.append(data_i)
            
            if cache_file:
                cache_data = {
                    'A': self.A_list[dataset_name],
                    'edge_index': self.edge_index_list[dataset_name],
                    'x': self.x_list[dataset_name],
                    'y': self.y_list[dataset_name],
                    'means': self.means_list[dataset_name],
                    'stds': self.stds_list[dataset_name]
                }
                
                if dataset_name in self.edge_attr_list:
                    cache_data['edge_attr'] = self.edge_attr_list[dataset_name]
                
                if dataset_name in self.node_features:
                    cache_data['node_features'] = self.node_features[dataset_name]
                
                if dataset_name in self.time_features:
                    cache_data['time_features'] = self.time_features[dataset_name]
                
                torch.save(cache_data, cache_file)

    def get_graph_structure(self, matrix_path):
        matrix = np.load(matrix_path)
        source_nodes, target_nodes = np.where(matrix > 0)
        edge_index = torch.tensor(np.vstack((source_nodes, target_nodes)), dtype=torch.long)
        if not np.array_equal(matrix, matrix.T):
            edge_index = to_undirected(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=matrix.shape[0])
        edge_weights = matrix[source_nodes, target_nodes]
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).view(-1, 1)
        if edge_attr.size(0) < edge_index.size(1):
            self_loop_attr = torch.ones(edge_index.size(1) - edge_attr.size(0), 1, dtype=torch.float)
            edge_attr = torch.cat([edge_attr, self_loop_attr], dim=0)
        
        node_attr = None
        geo_path = matrix_path.replace('adj', 'geo').replace('adjacency', 'geo')
        if os.path.exists(geo_path):
            geo_data = np.load(geo_path)
            node_attr = torch.tensor(geo_data, dtype=torch.float)
        
        return edge_index, edge_attr, node_attr

    def generate_time_features(self, num_nodes, seq_length, start_hour=0):
        hours_in_day = 24
        days_in_week = 7
        time_features = np.zeros((seq_length, 2), dtype=np.float32)
        
        for i in range(seq_length):
            current_hour = (start_hour + i * self.time_granularity // 60) % hours_in_day
            current_day = ((start_hour + i * self.time_granularity // 60) // hours_in_day) % days_in_week
            time_features[i, 0] = current_hour / hours_in_day
            time_features[i, 1] = current_day / days_in_week
        
        return torch.tensor(time_features, dtype=torch.float)

    def get_maml_task_batch(self, task_num):
        spt_task_data, qry_task_data = [], []
        spt_task_A_wave, qry_task_A_wave = [], []
        valid_datasets = set(self.data_list) & set(self.x_list.keys()) & set(self.A_list.keys()) & set(self.edge_index_list.keys())
        if not valid_datasets:
            raise ValueError("No valid datasets found in all required lists.")

        select_dataset = random.choice(list(valid_datasets))
        batch_size = self.task_args.get('batch_size', 5)
        message_dim = self.model_args.get('message_dim', 2)
        total_samples = self.x_list[select_dataset].shape[0]
        for i in range(task_num * 2):
            permutation = torch.randperm(total_samples)
            indices = permutation[:batch_size]
            x_data = self.x_list[select_dataset][indices]
            y_data = self.y_list[select_dataset][indices]
            node_num = self.A_list[select_dataset].shape[0]
   
            if x_data.shape[-1] != message_dim:
                if x_data.shape[-1] > message_dim:
                    x_data = x_data[..., :message_dim]
                else:
                    padding = torch.zeros(*x_data.shape[:-1], message_dim - x_data.shape[-1], device=x_data.device)
                    x_data = torch.cat([x_data, padding], dim=-1)
        
            data_i = Data(
                node_num=node_num, 
                x=x_data, 
                y=y_data,
                edge_index=self.edge_index_list[select_dataset],
                edge_attr=self.edge_attr_list[select_dataset] if select_dataset in self.edge_attr_list else None
            )
            data_i.data_name = select_dataset
            A_wave = self.A_list[select_dataset].float()
            if i % 2 == 0:
                spt_task_data.append(data_i)
                spt_task_A_wave.append(A_wave)
            else:
                qry_task_data.append(data_i)
                qry_task_A_wave.append(A_wave)

        return spt_task_data, spt_task_A_wave, qry_task_data, qry_task_A_wave
    
    def get_few_shot_support_query(self, dataset_name=None, k_shot=5, query_size=100):
        if dataset_name is None:
            dataset_name = self.test_data
        
        if isinstance(dataset_name, str) and '/' in dataset_name:
            base_name = dataset_name.split('/')[-1]
            dataset_name = base_name
        
        if dataset_name not in self.x_list:
            raise ValueError(f"Dataset {dataset_name} not found in available datasets")

        total_samples = self.x_list[dataset_name].shape[0]
        if total_samples < k_shot + query_size:
            raise ValueError(f"Not enough samples in {dataset_name}. Need at least {k_shot + query_size}, but got {total_samples}")
        
        permutation = torch.randperm(total_samples)
        support_indices = permutation[:k_shot]
        query_indices = permutation[k_shot:k_shot + query_size]
        support_x = self.x_list[dataset_name][support_indices]
        support_y = self.y_list[dataset_name][support_indices]
        query_x = self.x_list[dataset_name][query_indices]
        query_y = self.y_list[dataset_name][query_indices]
        node_num = self.A_list[dataset_name].shape[0]
        
        support_data = []
        for i in range(k_shot):
            data_i = Data(
                node_num=node_num,
                x=support_x[i].unsqueeze(0),
                y=support_y[i].unsqueeze(0),
                edge_index=self.edge_index_list[dataset_name],
                edge_attr=self.edge_attr_list[dataset_name] if dataset_name in self.edge_attr_list else None
            )
            data_i.data_name = dataset_name
            support_data.append(data_i)

        query_data = Data(
            node_num=node_num,
            x=query_x,
            y=query_y,
            edge_index=self.edge_index_list[dataset_name],
            edge_attr=self.edge_attr_list[dataset_name] if dataset_name in self.edge_attr_list else None
        )
        query_data.data_name = dataset_name
        return support_data, support_y, query_data, query_y
    
    def get_cross_city_batch(self, source_dataset, target_dataset, batch_size=5):
        if source_dataset not in self.x_list:
            raise ValueError(f"Source dataset {source_dataset} not found")
        if target_dataset not in self.x_list:
            raise ValueError(f"Target dataset {target_dataset} not found")
        source_samples = min(batch_size, self.x_list[source_dataset].shape[0])
        source_indices = torch.randperm(self.x_list[source_dataset].shape[0])[:source_samples]
        source_x = self.x_list[source_dataset][source_indices]
        source_y = self.y_list[source_dataset][source_indices]
        
        target_samples = min(batch_size, self.x_list[target_dataset].shape[0])
        target_indices = torch.randperm(self.x_list[target_dataset].shape[0])[:target_samples]
        target_x = self.x_list[target_dataset][target_indices]
        target_y = self.y_list[target_dataset][target_indices]
        
        source_data = Data(
            node_num=self.A_list[source_dataset].shape[0],
            x=source_x,
            y=source_y,
            edge_index=self.edge_index_list[source_dataset],
            edge_attr=self.edge_attr_list[source_dataset] if source_dataset in self.edge_attr_list else None
        )
        source_data.data_name = source_dataset
        
        target_data = Data(
            node_num=self.A_list[target_dataset].shape[0],
            x=target_x,
            y=target_y,
            edge_index=self.edge_index_list[target_dataset],
            edge_attr=self.edge_attr_list[target_dataset] if target_dataset in self.edge_attr_list else None
        )
        target_data.data_name = target_dataset
        
        return source_data, target_data

    def __len__(self):
        if not self.data_list:
            return 0
        for dataset_name in self.data_list:
            if dataset_name in self.x_list:
                return len(self.x_list[dataset_name])
        return 0

    def __getitem__(self, idx):
        if not self.data_list:
            raise ValueError("No datasets loaded")
        
        for dataset_name in self.data_list:
            if dataset_name not in self.x_list or dataset_name not in self.y_list:
                continue
                
            if idx >= len(self.x_list[dataset_name]):
                continue
                
            x_data = self.x_list[dataset_name][idx]
            y_data = self.y_list[dataset_name][idx]
            edge_index = self.edge_index_list.get(dataset_name, None)
            edge_attr = self.edge_attr_list.get(dataset_name, None)
            A_wave = self.A_list.get(dataset_name, None)
            
            if edge_index is None or A_wave is None:
                continue
                
            data_i = Data(
                x=x_data,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y_data
            )
            data_i.data_name = dataset_name
            
            if dataset_name in self.time_features:
                time_idx = idx % len(self.time_features[dataset_name])
                data_i.time_features = self.time_features[dataset_name][time_idx:time_idx+self.his_num]
            
            return data_i, A_wave
        
        raise ValueError(f"Could not get item {idx} from any dataset")

    def get_prediction_for_horizon(self, horizon, idx, dataset_name=None):
        if dataset_name is None:
            if not self.data_list:
                raise ValueError("No datasets available")
            dataset_name = self.data_list[0]
        
        dataset_name = str(dataset_name).split('/')[-1].split('.')[0]
        
        horizon_key = f"{dataset_name}_{horizon}"
        if horizon_key in self.y_list and idx < len(self.y_list[horizon_key]):
            return self.y_list[horizon_key][idx]
        
        if dataset_name in self.y_list and idx < len(self.y_list[dataset_name]):
            time_step_count = min(horizon // self.time_granularity, self.y_list[dataset_name].shape[1])
            if time_step_count > 0:
                return self.y_list[dataset_name][idx, :time_step_count, :]
        
        for key in self.y_list.keys():
            if idx < len(self.y_list[key]):
                time_step_count = min(horizon // self.time_granularity, self.y_list[key].shape[1])
                if time_step_count > 0:
                    return self.y_list[key][idx, :time_step_count, :]
        
        raise ValueError(f"No valid data found for horizon {horizon} at index {idx}")

    def get_dataloader(self, batch_size=None, shuffle=True, dataset_name=None):
        if batch_size is None:
            batch_size = self.task_args.get('batch_size', 5)
            
        if dataset_name is None:
            dataset_name = self.data_list[0]
        if dataset_name not in self.x_list:
            raise ValueError(f"Dataset {dataset_name} not found")
        subset_indices = range(self.x_list[dataset_name].shape[0])
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  
            pin_memory=True,
            sampler=torch.utils.data.SubsetRandomSampler(subset_indices) if shuffle else None
        )
        
        return dataloader