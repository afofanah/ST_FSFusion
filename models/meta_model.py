import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy
import gc
import contextlib

class STMetaLearner(nn.Module):
    def __init__(self, model_args, task_args, device='cuda'):
        super(STMetaLearner, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.tp = model_args.get('tp', False)
        self.sp = model_args.get('sp', False)
        self.node_feature_dim = model_args['node_feature_dim']
        self.edge_feature_dim = model_args.get('edge_feature_dim', 0)
        original_message_dim = model_args.get('message_dim', self.node_feature_dim)
        self.num_heads = model_args.get('num_heads', 2)

        def calculate_embedding_dim(dim):
            base_dim = max(dim, 16)  
            return ((base_dim // self.num_heads) * self.num_heads)
            
        self.message_dim = calculate_embedding_dim(original_message_dim)
        self.hidden_dim = model_args.get('hidden_dim', self.message_dim)
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        self.meta_out = model_args['meta_dim']
        
        if self.tp:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.message_dim,
                nhead=self.num_heads,
                dim_feedforward=self.message_dim,  
                dropout=0.1,
                batch_first=True,
                activation='relu',  
                device=self.device
            )
            self.tp_learner = nn.TransformerEncoder(
                encoder_layer,
                num_layers=1
            ).to(self.device)

        if self.sp:
            self.sp_learner = GCNConv(
                in_channels=self.his_num * self.message_dim,
                out_channels=self.hidden_dim,
                improved=True,
                add_self_loops=True
            ).to(self.device)
        
        if self.tp and self.sp:
            projector_input_dim = self.message_dim + self.hidden_dim
        elif self.tp:
            projector_input_dim = self.message_dim
        elif self.sp:
            projector_input_dim = self.hidden_dim
        else:
            projector_input_dim = self.his_num * self.message_dim
            
        self.feature_projector = nn.Linear(
            projector_input_dim, 
            self.meta_out
        ).to(self.device)

        self._reset_parameters()
        
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, data, dim):
        data = data.to(self.device)

        if data.x.dim() == 3:
            batch_size, node_num, orig_message_dim = data.x.shape
            if orig_message_dim != self.message_dim:
                if orig_message_dim > self.message_dim:
                    data.x = data.x[:, :, :self.message_dim]
                else:
                    pad_size = self.message_dim - orig_message_dim
                    padding = torch.zeros(
                        batch_size, node_num, pad_size, 
                        device=data.x.device, 
                        dtype=data.x.dtype
                    )
                    data.x = torch.cat([data.x, padding], dim=2)
            data.x = data.x.unsqueeze(2).expand(-1, -1, self.his_num, -1)
        
        elif data.x.dim() == 4:
            batch_size, node_num, his_len, message_dim = data.x.shape
            if his_len > self.his_num:
                data.x = data.x[:, :, :self.his_num, :]
            if message_dim != self.message_dim:
                if message_dim > self.message_dim:
                    data.x = data.x[:, :, :, :self.message_dim]
                else:
                    pad_size = self.message_dim - message_dim
                    padding = torch.zeros(
                        batch_size, node_num, his_len, pad_size, 
                        device=data.x.device, 
                        dtype=data.x.dtype
                    )
                    data.x = torch.cat([data.x, padding], dim=3)
        
        else:
            raise ValueError(f"Input tensor must be 3D or 4D, got {data.x.dim()}D.")
            
        tp_output, sp_output = None, None

        if self.tp:
            batch_node_size = batch_size * node_num
            tp_input = data.x.reshape(batch_node_size, self.his_num, self.message_dim)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                tp_output = self.tp_learner(tp_input)[:, -1, :]
     
        if self.sp:
            sp_outputs = []
            for b in range(batch_size):
                batch_features = data.x[b].reshape(node_num, self.his_num * self.message_dim)
                if hasattr(data, 'edge_index'):
                    edge_index = data.edge_index.to(self.device)
                    valid_mask = (edge_index[0] < node_num) & (edge_index[1] < node_num)
                    filtered_edge_index = edge_index[:, valid_mask]
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        batch_sp_output = self.sp_learner(batch_features, filtered_edge_index)
                else:
                    with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                        batch_sp_output = self.sp_learner(batch_features)
                
                sp_outputs.append(batch_sp_output)
            sp_output = torch.stack(sp_outputs, dim=0).reshape(batch_size * node_num, -1)
        
        if tp_output is not None and sp_output is not None:
            tp_projection = F.relu(tp_output)
            sp_projection = F.relu(sp_output)
            mk_input = torch.cat([tp_projection, sp_projection], dim=1)
        elif tp_output is not None:
            mk_input = F.relu(tp_output)
        elif sp_output is not None:
            mk_input = F.relu(sp_output)
        else:
            batch_features = data.x.reshape(batch_size * node_num, self.his_num * self.message_dim)
            mk_input = F.relu(batch_features)
        
        meta_knowledge = self.feature_projector(mk_input)
        meta_knowledge = F.layer_norm(
            meta_knowledge, 
            normalized_shape=[meta_knowledge.size(-1)]
        )
        meta_knowledge = meta_knowledge.view(batch_size, node_num, self.meta_out)
        
        return meta_knowledge


class ResourceSafeModule(nn.Module):
    def __init__(self, device='cuda'):
        super(ResourceSafeModule, self).__init__()
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
    def to_device(self, tensor):
        if tensor is None:
            return None
        if 'CUDA out of memory' in str(e):
            if self.device == 'cuda':
                torch.cuda.empty_cache()
            return tensor.cpu()
        return tensor.to(self.device)

    @contextlib.contextmanager
    def autocast_context(self):
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available() and self.device == 'cuda'):
            yield
                
    def clean_memory(self):
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

   
class MetaTransformer(nn.Module):
    def __init__(self, model_args, task_args, input_dim=None, hidden_dim=None, output_dim=None, device='cuda'):
        super(MetaTransformer, self).__init__()
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.meta_dim = model_args.get('meta_dim', model_args.get('hidden_dim', 32))  
        self.input_dim = model_args.get('message_dim', 8) if input_dim is None else input_dim
        self.hidden_dim = model_args.get('hidden_dim', 16) if hidden_dim is None else hidden_dim
        self.output_dim = model_args.get('output_dim', 2) if output_dim is None else output_dim
        self.num_heads = model_args.get('num_heads', 2)
        self.num_layers = 1 
        self.dropout = model_args.get('dropout', 0.1)
       
        if self.hidden_dim % self.num_heads != 0:
            self.hidden_dim = (self.hidden_dim // self.num_heads) * self.num_heads
            
        self.build()
        self.to(self.device)

    def build(self):
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        ).to(self.device)

        self.meta_proj = nn.Linear(self.meta_dim, self.hidden_dim * 3, device=self.device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim,  
            dropout=self.dropout,
            activation='relu',  
            batch_first=True,
            device=self.device
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_dim, device=self.device)
        )

        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim, device=self.device)
        
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
 
        if hasattr(self.output_layer, 'bias') and self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, meta_knowledge, data, input=None):
        input = data if input is None else input
        input = input.to(self.device)
        meta_knowledge = meta_knowledge.to(self.device)
        
        if input.dim() == 4:
            batch_size, node_num, seq_len, features = input.shape
        elif input.dim() == 3:
            batch_size, node_num, features = input.shape
            seq_len = 1
            input = input.unsqueeze(2)
        else:
            raise ValueError(f"Invalid input shape: {input.shape}")
            
        input_reshaped = input.reshape(-1, seq_len, features)
    
        if features != self.input_dim:
            if features < self.input_dim:
                padding = torch.zeros(
                    input_reshaped.size(0), input_reshaped.size(1), 
                    self.input_dim - features, 
                    device=input_reshaped.device, 
                    dtype=input_reshaped.dtype
                )
                input_reshaped = torch.cat([input_reshaped, padding], dim=-1)
            else:
                input_reshaped = input_reshaped[..., :self.input_dim]
        
        processed_input = self.input_projection(input_reshaped)
        meta_knowledge_reshaped = meta_knowledge.reshape(-1, self.meta_dim)

        qkv = self.meta_proj(meta_knowledge_reshaped)
        meta_q, meta_k, meta_v = torch.chunk(qkv, 3, dim=-1)
        
        meta_q = meta_q.unsqueeze(1)
        meta_k = meta_k.unsqueeze(1)
        meta_v = meta_v.unsqueeze(1)

        scale = float(self.hidden_dim) ** -0.5
        attention_scores = torch.bmm(processed_input, meta_k.transpose(1, 2)) * scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        meta_attended = torch.bmm(attention_weights, meta_v)
        
        enhanced_input = processed_input + meta_attended
 
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  
            transformer_output = self.transformer_encoder(enhanced_input)
            output = self.output_layer(transformer_output)
            
        output = output.reshape(batch_size, node_num, seq_len, -1)
        return output


class CrossDomainFusion(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, fusion_dim, device='cuda'):
        super(CrossDomainFusion, self).__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim = fusion_dim
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.shared_dim = min(spatial_dim, temporal_dim, fusion_dim) // 4
    
        self.spatial_proj = nn.Linear(spatial_dim, self.shared_dim).to(self.device)
        self.temporal_proj = nn.Linear(temporal_dim, self.shared_dim).to(self.device)
    
        self.fusion_proj = nn.Linear(self.shared_dim * 2, fusion_dim).to(self.device)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
    def forward(self, spatial_features, temporal_features):
        spatial_features = spatial_features.to(self.device)
        temporal_features = temporal_features.to(self.device)
        batch_size = spatial_features.size(0)
        
        if spatial_features.dim() > 2:
            spatial_features = spatial_features.reshape(batch_size, -1)
        if temporal_features.dim() > 2:
            temporal_features = temporal_features.reshape(batch_size, -1)
            
        if spatial_features.size(1) != self.spatial_dim:
            if spatial_features.size(1) > self.spatial_dim:
                spatial_features = spatial_features[:, :self.spatial_dim]
            else:
                padding = torch.zeros(batch_size, self.spatial_dim - spatial_features.size(1), 
                                      device=self.device)
                spatial_features = torch.cat([spatial_features, padding], dim=1)           
                
        if temporal_features.size(1) != self.temporal_dim:
            if temporal_features.size(1) > self.temporal_dim:
                temporal_features = temporal_features[:, :self.temporal_dim]
            else:
                padding = torch.zeros(batch_size, self.temporal_dim - temporal_features.size(1), 
                                     device=self.device)
                temporal_features = torch.cat([temporal_features, padding], dim=1)
                
        spatial_features = spatial_features.to(self.spatial_proj.weight.dtype)
        temporal_features = temporal_features.to(self.temporal_proj.weight.dtype)
    
        spatial_proj = F.relu(self.spatial_proj(spatial_features))
        temporal_proj = F.relu(self.temporal_proj(temporal_features))
        
        fused_features = torch.cat([spatial_proj, temporal_proj], dim=-1)
        output = self.fusion_proj(fused_features)
        
        if spatial_proj.size(1) == output.size(1):
            output = output + spatial_proj
        if temporal_proj.size(1) == output.size(1):
            output = output + temporal_proj
            
        return output


class RobustTransformer(nn.Module):
    def __init__(self, model_args, task_args, device: str = 'cuda') -> None:
        super(RobustTransformer, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.build()
        self.to(self.device)

    def build(self):
        self.input_dim = self.model_args.get('input_dim', 16)  
        self.hidden_dim = self.model_args.get('hidden_dim', 32)
        self.num_heads = self.model_args.get('num_heads', 2)
        
        self.hidden_dim = ((self.hidden_dim + self.num_heads - 1) // self.num_heads) * self.num_heads
        self.num_layers = 1
        self.output_dim = self.task_args.get('output_dim', self.task_args.get('pred_num', 6))
        self.dropout = self.model_args.get('dropout', 0.1)
        self.activation = self.model_args.get('activation', 'relu')
        
        self.use_layer_norm = self.model_args.get('use_layer_norm', True)
        self.use_residual = self.model_args.get('use_residual', True)
        self.intermediate_factor = 1
        self.batch_first = self.model_args.get('batch_first', True)

        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if self.use_layer_norm else nn.Identity(),
            nn.Dropout(self.dropout)
        ).to(self.device)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * self.intermediate_factor,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=self.batch_first,
            device=self.device
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_dim) if self.use_layer_norm else None
        ).to(self.device)

        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)

        self._reset_parameters()

    def _reset_parameters(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(init_weights)

    def forward(self, data: torch.Tensor, A_wave: torch.Tensor = None) -> torch.Tensor:
        original_shape = data.shape
        device_data = data.to(self.device)
        
        if device_data.dim() == 4:
            batch_size, num_nodes, seq_len, input_dim = device_data.shape
            device_data = device_data.view(batch_size * num_nodes, seq_len, input_dim)
        elif device_data.dim() == 3:
            batch_size, seq_len, input_dim = device_data.shape
            num_nodes = None
        else:
            raise ValueError(f"Expected 3D or 4D input, got {device_data.dim()}D with shape {original_shape}")

        if input_dim != self.input_dim:
            if input_dim < self.input_dim:
                padding = torch.zeros(
                    *device_data.shape[:-1], self.input_dim - input_dim, 
                    device=self.device,
                    dtype=device_data.dtype
                )
                device_data = torch.cat([device_data, padding], dim=-1)
            else:
                device_data = device_data[..., :self.input_dim]

        data_proj = self.input_proj(device_data)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            if not self.batch_first:
                data_proj = data_proj.permute(1, 0, 2)
            transformer_output = self.transformer_encoder(data_proj)
            if not self.batch_first:
                transformer_output = transformer_output.permute(1, 0, 2)
            output = self.output_proj(transformer_output)
            
        if num_nodes is not None:
            output = output.view(batch_size, num_nodes, seq_len, -1)
            if not self.model_args.get('return_sequences', False):
                output = output[:, :, -1, :]
                
        return output
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(\n"
                f"  input_dim={self.input_dim}, hidden_dim={self.hidden_dim},\n"
                f"  num_heads={self.num_heads}, num_layers={self.num_layers},\n"
                f"  output_dim={self.output_dim}, dropout={self.dropout},\n"
                f"  intermediate_factor={self.intermediate_factor}\n"
                f")")


class TemporalEncoderWrapper(nn.Module):
    def __init__(self, model_args, task_args, device='cuda'):
        super(TemporalEncoderWrapper, self).__init__()
        if 'meta_dim' not in model_args:
            model_args['meta_dim'] = model_args.get('hidden_dim', 32)
            
        if model_args.get('use_meta_transformer', False):
            self.encoder = MetaTransformer(
                model_args=model_args, 
                task_args=task_args, 
                device=device
            )
            self.use_meta_transformer = True
        else:
            self.encoder = RobustTransformer(
                model_args=model_args, 
                task_args=task_args, 
                device=device
            )
            self.use_meta_transformer = False
        
        self.model_args = model_args
        self.device = device
    
    def forward(self, data, meta_knowledge=None):
        if self.use_meta_transformer:
            if meta_knowledge is None:
                raise ValueError("Meta knowledge is required when using OptimizedMetaTransformer")
            data = data.to(self.device)
            output = self.encoder(meta_knowledge, data)
        else:
            output = self.encoder(data)
        return output


class MetaCross_DomainFusion(nn.Module):
    def __init__(self, model_args, task_args, device='cuda'):
        super(MetaCross_DomainFusion, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.hidden_dim = model_args['hidden_dim']
        self.meta_dim = model_args['meta_dim']
        self.message_dim = model_args['message_dim']
        self.output_dim = model_args.get('output_dim', 1)
        self.his_num = task_args['his_num']
        self.pred_num = task_args['pred_num']
        
        self.gnn_hidden_dim = min(64, self.hidden_dim)
        self.trans_hidden_dim = min(32, self.hidden_dim)

        self.build_model_components()
        self.build_prediction_heads()
        self.reset_parameters()
        self.to(self.device)
        
    def build_model_components(self):
        self.mk_learner = STMetaLearner(
            self.model_args, 
            self.task_args, 
            device=self.device
        )
        
        self.spatial_encoder = nn.ModuleList([
            GCNConv(
                in_channels=self.message_dim * self.his_num,
                out_channels=self.gnn_hidden_dim),
            nn.LayerNorm(self.gnn_hidden_dim),
            nn.ReLU()
        ])
  
        temporal_model_args = {
            'meta_dim': self.meta_dim,
            'message_dim': self.message_dim,
            'hidden_dim': self.trans_hidden_dim,
            'num_heads': min(2, self.trans_hidden_dim // 16),
            'num_layers': 1,
            'dropout': 0.1,
            'use_meta_transformer': True
        }
        
        self.temporal_encoder = TemporalEncoderWrapper(
            model_args=temporal_model_args,
            task_args=self.task_args,
            device=self.device
        )
        
        self.cross_domain_fusion = CrossDomainFusion(
            spatial_dim=self.gnn_hidden_dim,
            temporal_dim=self.trans_hidden_dim,
            fusion_dim=self.hidden_dim,
            device=self.device
        )
        
        self.meta_attention = nn.Sequential(
            nn.Linear(self.meta_dim, self.hidden_dim, device=self.device),
            nn.Sigmoid()
        )
    
    def build_prediction_heads(self):
        self.flow_predictor = nn.Linear(self.hidden_dim, self.pred_num, device=self.device)
        
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.pred_num, device=self.device),
            nn.Softplus()
        )
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                    
    def forward(self, data, A_hat=None):
        input_data = data.x.to(self.device)
  
        if input_data.dim() == 3:
            batch_size, node_num, feature_dim = input_data.shape
            if feature_dim % self.message_dim == 0:
                his_len = feature_dim // self.message_dim
                input_data = input_data.reshape(batch_size, node_num, his_len, self.message_dim)
            else:
                input_data = input_data.unsqueeze(2)
                
        batch_size, node_num = input_data.shape[:2]
        
        meta_knowledge = self.mk_learner(data, dim=3)
        traffic_graph_structure = torch.eye(node_num, device=self.device)
        
        spatial_features = None
        if hasattr(data, 'edge_index'):
            edge_index = data.edge_index.to(self.device)
            if input_data.dim() == 4:
                spatial_input = input_data.reshape(batch_size, node_num, -1)
            else:
                spatial_input = input_data
            
            spatial_features = []
            for b in range(batch_size):
                batch_feats = spatial_input[b]
                if batch_feats.dim() == 2:
                    batch_feats = batch_feats.reshape(node_num, -1)
                elif batch_feats.dim() == 1:
                    batch_feats = batch_feats.unsqueeze(1)
                    
                if batch_feats.size(1) < self.message_dim * self.his_num:
                    pad_size = self.message_dim * self.his_num - batch_feats.size(1)
                    padding = torch.zeros(
                        batch_feats.size(0), pad_size, 
                        device=batch_feats.device, 
                        dtype=batch_feats.dtype
                    )
                    batch_feats = torch.cat([batch_feats, padding], dim=1)
                elif batch_feats.size(1) > self.message_dim * self.his_num:
                    batch_feats = batch_feats[:, :self.message_dim * self.his_num]
                    
                max_node_index = node_num - 1
                valid_mask = (edge_index[0] <= max_node_index) & (edge_index[1] <= max_node_index)
                filtered_edge_index = edge_index[:, valid_mask]
                
                x = batch_feats
                for layer in self.spatial_encoder:
                    if isinstance(layer, GCNConv):
                        x = layer(x, filtered_edge_index)
                    else:
                        x = layer(x)
                spatial_features.append(x)
            
            spatial_features = torch.stack(spatial_features)
        
        temporal_features = self.temporal_encoder(input_data, meta_knowledge)
        if temporal_features.dim() == 4:
            temporal_features = temporal_features[:, :, -1, :]
            
        if spatial_features is not None:
            fused_features = self.cross_domain_fusion(spatial_features, temporal_features)
            
            meta_global = torch.sigmoid(meta_knowledge.mean(dim=(1, 2), keepdim=True))
            fused_features = fused_features * meta_global
        else:
            fused_features = temporal_features
            
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            flow_prediction = self.flow_predictor(fused_features)
            uncertainty = self.uncertainty_predictor(fused_features)
            
        return flow_prediction, uncertainty, traffic_graph_structure
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(\n"
                f"  hidden_dim={self.hidden_dim}, meta_dim={self.meta_dim},\n"
                f"  message_dim={self.message_dim}, output_dim={self.output_dim},\n"
                f"  his_num={self.his_num}, pred_num={self.pred_num}\n"
                f")")


class TransformerBasedPredictor(nn.Module):
    def __init__(self, model_args, task_args, device='cuda'):
        super(TransformerBasedPredictor, self).__init__()
        self.model_args = model_args
        self.task_args = task_args
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.input_dim = model_args.get('input_dim', 16)
        self.hidden_dim = model_args.get('hidden_dim', 32)
        self.message_dim = model_args.get('message_dim', 16)
        self.output_dim = model_args.get('output_dim', task_args.get('pred_num', 6))
        self.his_num = task_args.get('his_num', 12)
        self.pred_num = task_args.get('pred_num', 6)
        
        self.feature_extractor = nn.Linear(self.input_dim, self.message_dim).to(self.device)
        
        self.temporal_transformer = RobustTransformer(
            model_args={
                'input_dim': self.message_dim,
                'hidden_dim': self.hidden_dim,
                'num_heads': 2,
                'num_layers': 1,
                'dropout': model_args.get('dropout', 0.1),
                'activation': 'relu',
                'batch_first': True,
                'return_sequences': False  
            },
            task_args={'output_dim': self.hidden_dim},
            device=self.device
        )
        
        self.spatial_embedder = nn.Linear(self.message_dim, self.hidden_dim).to(self.device) if model_args.get('sp', False) else None
        
        if self.spatial_embedder:
            self.fusion_layer = nn.Linear(self.hidden_dim * 2, self.hidden_dim).to(self.device)
        else:
            self.fusion_layer = nn.Identity()
        
        self.predictor = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        
        self.predict_uncertainty = model_args.get('predict_uncertainty', False)
        if self.predict_uncertainty:
            self.uncertainty_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.output_dim),
                nn.Softplus()  
            ).to(self.device)
        
        self.learn_graph = model_args.get('learn_graph', False)
        if self.learn_graph:
            self.graph_learner = nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device)
 
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, data, A_wave=None):
        if hasattr(data, 'x'):
            x = data.x
            if A_wave is None and hasattr(data, 'edge_index'):
                edge_index = data.edge_index
        else:
            x = data
            
        x = x.to(self.device)
        if A_wave is not None:
            A_wave = A_wave.to(self.device)
            
        if x.dim() == 4:
            batch_size, num_nodes, seq_len, _ = x.shape
        elif x.dim() == 3:
            batch_size, num_nodes, _ = x.shape
            x = x.unsqueeze(2)
            seq_len = 1
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")

        x = self.feature_extractor(x)

        node_features = x.reshape(batch_size * num_nodes, seq_len, -1)
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            temporal_features = self.temporal_transformer(node_features)
            temporal_features = temporal_features.view(batch_size, num_nodes, -1)
            
            if self.spatial_embedder is not None:
                spatial_input = x.mean(dim=2)
                spatial_features = self.spatial_embedder(spatial_input)
                combined_features = torch.cat([spatial_features, temporal_features], dim=-1)
                fused_features = self.fusion_layer(combined_features)
            else:
                fused_features = temporal_features
                
            learned_graph = None
            if self.learn_graph:
                node_embeddings = fused_features
                graph_list = []
                for b in range(batch_size):
                    batch_embeddings = node_embeddings[b]
                    similarity = torch.mm(batch_embeddings, batch_embeddings.transpose(0, 1))
                    graph = torch.sigmoid(similarity)
                    graph_list.append(graph)
                learned_graph = torch.stack(graph_list)

            predictions = self.predictor(fused_features)
            uncertainty = None
            if self.predict_uncertainty:
                uncertainty = self.uncertainty_predictor(fused_features)
                
        return predictions, uncertainty, learned_graph


def create_optimized_model(model_args, task_args, device='cuda'):
    model_type = model_args.get('type', 'MetaCross_DomainFusion')
    
    if model_type == 'TransformerBased':
        return TransformerBasedPredictor(model_args, task_args, device)
    elif model_type == 'RandomTransformer':
        return RobustTransformer(model_args, task_args, device)
    elif model_type == 'MetaTransfomer':
        return MetaTransformer(model_args, task_args, device)
    else:
        return MetaCross_DomainFusion(model_args, task_args, device)


class CrossCityAdapter(nn.Module):
    def __init__(self, source_dim, target_dim, hidden_dim=32, device='cuda'):
        super(CrossCityAdapter, self).__init__()
        self.source_dim = source_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.source_adapter = nn.Sequential(
            nn.Linear(source_dim, hidden_dim, device=self.device),
            nn.LayerNorm(hidden_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=self.device),
        )
        self.target_adapter = nn.Sequential(
            nn.Linear(target_dim, hidden_dim, device=self.device),
            nn.LayerNorm(hidden_dim, device=self.device),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, device=self.device),
        )
        
        self.city_features = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim, device=self.device),
            nn.LayerNorm(hidden_dim, device=self.device),
            nn.ReLU()
        )
        self.similarity_module = nn.Sequential(
            nn.Linear(hidden_dim, 1, device=self.device),
            nn.Sigmoid()
        )
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, source_features, target_features):
        batch_size = source_features.size(0)
        source_nodes = source_features.size(1)
        target_nodes = target_features.size(1)
        
        source_adapted = self.source_adapter(source_features)  
        target_adapted = self.target_adapter(target_features)  
    
        s_expand = source_adapted.unsqueeze(1)  
        t_expand = target_adapted.unsqueeze(2)  
 
        similarity_features = torch.cat([
            t_expand.expand(-1, -1, source_nodes, -1),
            s_expand.expand(-1, target_nodes, -1, -1)
        ], dim=-1)

        similarity_features = similarity_features.view(batch_size * target_nodes * source_nodes, -1)
        city_pattern_features = self.city_features(similarity_features)
        similarity_scores = self.similarity_module(city_pattern_features)
        similarity_scores = similarity_scores.view(batch_size, target_nodes, source_nodes)

        attn_weights = F.softmax(similarity_scores, dim=-1)
        transferred_knowledge = torch.bmm(attn_weights, source_adapted)
        gate = torch.sigmoid(target_adapted.mean(dim=-1, keepdim=True))
        enhanced_features = target_adapted * (1 - gate) + transferred_knowledge * gate
        if self.hidden_dim != self.target_dim:
            output_proj = nn.Linear(self.hidden_dim, self.target_dim, device=self.device)
            adapted_features = output_proj(enhanced_features)
        else:
            adapted_features = enhanced_features
        
        return adapted_features, similarity_scores
        
    def transfer_meta_knowledge(self, source_meta, target_meta):
        batch_size = source_meta.size(0)
        source_proj = nn.Linear(source_meta.size(-1), self.hidden_dim, device=self.device)
        target_proj = nn.Linear(target_meta.size(-1), self.hidden_dim, device=self.device)
        
        source_meta_proj = source_proj(source_meta)
        target_meta_proj = target_proj(target_meta)
        similarity = torch.bmm(
            target_meta_proj, 
            source_meta_proj.transpose(1, 2))
        
        attn_weights = F.softmax(similarity, dim=-1)
        
        transferred_meta = torch.bmm(attn_weights, source_meta)
        enhanced_meta = (transferred_meta + target_meta) / 2.0
        
        return enhanced_meta

class FewShotTrafficLearner:
    def __init__(self, base_model, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.base_model = base_model.to(self.device)
        
        self.proto_encoder = nn.Sequential(
            nn.Linear(base_model.hidden_dim, base_model.hidden_dim // 2),
            nn.LayerNorm(base_model.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(base_model.hidden_dim // 2, base_model.hidden_dim // 4)
        ).to(self.device)
        
        self.adaptation_layers = nn.ModuleList([
            nn.Linear(base_model.hidden_dim, base_model.hidden_dim),
            nn.Linear(base_model.hidden_dim // 2, base_model.hidden_dim // 2),
            nn.Linear(base_model.pred_num, base_model.pred_num)
        ]).to(self.device)
        self.temperature = nn.Parameter(torch.tensor(1.0).to(self.device))
        
    def create_support_prototypes(self, support_data, support_labels):
        with torch.no_grad():
            features_list = []
            for data in support_data:
                pred, _, _ = self.base_model(data)
                intermediate_features = self._get_intermediate_features(data)
                features_list.append(intermediate_features)

            all_features = torch.cat(features_list, dim=0)  
            proto_features = self.proto_encoder(all_features)  
            binned_labels = self._discretize_flow_values(support_labels)
            unique_bins = torch.unique(binned_labels)

            prototypes = {}
            for bin_id in unique_bins:
                bin_mask = (binned_labels == bin_id)
                bin_features = proto_features[bin_mask]

                if len(bin_features) > 0:
                    prototype = bin_features.mean(dim=0)
                    prototypes[bin_id.item()] = prototype
            
            return prototypes
    
    def _discretize_flow_values(self, flow_values, num_bins=10):
        flat_values = flow_values.reshape(-1)
        min_val, max_val = flat_values.min(), flat_values.max()
        bin_width = (max_val - min_val) / num_bins
        binned_values = torch.floor((flow_values - min_val) / bin_width).long()
        binned_values = torch.clamp(binned_values, 0, num_bins - 1)
        
        return binned_values
    
    def _get_intermediate_features(self, data):
        meta_knowledge = self.base_model.mk_learner(data, dim=3)
        spatial_features = None
        if hasattr(data, 'edge_index'):
            input_data = data.x.to(self.device)
            batch_size, node_num = input_data.shape[:2]
            edge_index = data.edge_index.to(self.device)
            if input_data.dim() == 4:
                spatial_input = input_data.reshape(batch_size, node_num, -1)
            else:
                spatial_input = input_data
            spatial_features = []
            for b in range(batch_size):
                batch_feats = spatial_input[b]
                x = batch_feats
                for layer in self.base_model.spatial_encoder:
                    if isinstance(layer, GCNConv):
                        x = layer(x, edge_index)
                    else:
                        x = layer(x)
                
                spatial_features.append(x)
            
            spatial_features = torch.stack(spatial_features)
        temporal_features = self.base_model.temporal_encoder(input_data)
        if temporal_features.dim() == 4:
            temporal_features = temporal_features[:, :, -1, :]

        if spatial_features is not None:
            fused_features = self.base_model.cross_domain_fusion(
                spatial_features, temporal_features
            )
        else:
            fused_features = temporal_features
        
        return fused_features
    
    def adapt_to_target(self, support_data, support_labels, num_adaptation_steps=10, lr=0.001):
        prototypes = self.create_support_prototypes(support_data, support_labels)
        adapted_model = copy.deepcopy(self.base_model)
        optimizer = torch.optim.Adam(self.adaptation_layers.parameters(), lr=lr)
        
        for step in range(num_adaptation_steps):
            total_loss = 0
            for i, (data, label) in enumerate(zip(support_data, support_labels)):
                features = self._get_intermediate_features(data)
                proto_features = self.proto_encoder(features)

                distances = {}
                for bin_id, prototype in prototypes.items():
                    dist = torch.sum((proto_features - prototype.unsqueeze(0))**2, dim=-1)
                    distances[bin_id] = dist
                
                logits = []
                bin_ids = []
                for bin_id, dist in distances.items():
                    logits.append(-dist / self.temperature)
                    bin_ids.append(bin_id)
                
                logits = torch.stack(logits, dim=-1)
                probs = F.softmax(logits, dim=-1)
                
                bin_values = torch.tensor(bin_ids, device=self.device).float()
                pred_bin = torch.sum(probs * bin_values.unsqueeze(0).unsqueeze(0), dim=-1)
            
                min_val = support_labels.min()
                max_val = support_labels.max()
                num_bins = len(bin_ids)
                bin_width = (max_val - min_val) / num_bins
                
                pred_flow = min_val + pred_bin * bin_width + bin_width / 2
                
                for layer in self.adaptation_layers:
                    pred_flow = layer(pred_flow)

                loss = F.mse_loss(pred_flow, label)
                total_loss += loss
            
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        for i, layer in enumerate(self.adaptation_layers):
            if i == 0:
                adapted_model.mk_learner.feature_projector.weight.data = layer.weight.data
            elif i == 1:
                adapted_model.cross_domain_fusion.fusion_proj[0].weight.data = layer.weight.data
            elif i == 2:
                adapted_model.flow_predictor[-1].weight.data = layer.weight.data
        
        return adapted_model
    
    def predict(self, adapted_model, query_data):
        with torch.no_grad():
            predictions, uncertainties, _ = adapted_model(query_data)
        
        return predictions, uncertainties