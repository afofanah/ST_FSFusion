"""
MCD-CKT: Meta Cross-Domain Fusion with Cross-City Knowledge Transfer
=====================================================================
Base: doc 8 (user-provided architecture), aligned to train_mcd.py / utils_mcd.py.

Pipeline contract
-----------------
Input:  PyG Data  —  x=(B, N, T, F),  edge_index=(2, E),  y=(B, pred_num, N)
Output: (flow_pred, uncertainty, graph)
        flow_pred   : (B, pred_num, N)   ← .transpose(1,2) applied
        uncertainty : (B, pred_num, N)
        graph       : (N, N)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import types
import copy
from typing import Dict, Optional


# ── Device helpers ──────────────────────────────────────────────────────────

def ensure_all_tensors_on_same_device(model, device=None):
    if device is None:
        devices = {}
        for param in model.parameters():
            dev = param.device
            devices[dev] = devices.get(dev, 0) + 1
        device = max(devices.items(), key=lambda x: x[1])[0]
    model = model.to(device)
    for module in model.modules():
        if hasattr(module, "device"):
            module.device = device
    return model, device


def process_input_data(data, device):
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Data):
        return data.to(device)
    if isinstance(data, (list, tuple)):
        return type(data)(process_input_data(i, device) for i in data)
    if isinstance(data, dict):
        return {k: process_input_data(v, device) for k, v in data.items()}
    return data


# ── STMetaLearner ────────────────────────────────────────────────────────────

class STMetaLearner(nn.Module):
    def __init__(self, model_args, task_args, device="cuda"):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tp = model_args.get("tp", False)
        self.sp = model_args.get("sp", False)
        self.node_feature_dim = model_args["node_feature_dim"]
        self.num_heads = model_args.get("num_heads", 2)
        self.message_dim = self._compat_dim(
            model_args.get("message_dim", self.node_feature_dim), self.num_heads)
        self.hidden_dim = model_args.get("hidden_dim", self.message_dim)
        self.his_num = task_args["his_num"]
        self.pred_num = task_args["pred_num"]
        self.meta_out = model_args["meta_dim"]
        self.max_chunk = 256
        self._build(); self._reset()

    def _compat_dim(self, d, h):
        return (max(d, 16) // h) * h

    def _build(self):
        if self.tp:
            enc = nn.TransformerEncoderLayer(
                d_model=self.message_dim, nhead=self.num_heads,
                dim_feedforward=self.message_dim * 2, dropout=0.1,
                batch_first=True, device=self.device)
            self.tp_learner = nn.TransformerEncoder(enc, num_layers=1)
        if self.sp:
            self.sp_learner = GCNConv(
                in_channels=self.his_num * self.message_dim,
                out_channels=self.hidden_dim,
                improved=True, add_self_loops=True).to(self.device)
        if self.tp and self.sp: p = self.message_dim + self.hidden_dim
        elif self.tp:            p = self.message_dim
        elif self.sp:            p = self.hidden_dim
        else:                    p = self.his_num * self.message_dim
        self.feature_projector = nn.Linear(p, self.meta_out).to(self.device)

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, data, dim=3):
        if isinstance(data, torch.Tensor):
            x, ei = data.to(self.device), None
        elif hasattr(data, "x"):
            x = data.x.to(self.device)
            ei = getattr(data, "edge_index", None)
            if ei is not None: ei = ei.to(self.device)
        else:
            raise ValueError(f"Unsupported: {type(data)}")

        B, N = x.shape[:2]
        xp = self._prep3(x, B, N) if x.dim() == 3 else self._prep4(x, B, N)
        tp = self._temporal(xp, B, N) if self.tp else None
        sp = self._spatial(xp, B, N, ei) if self.sp else None
        c  = self._combine(tp, sp, xp, B, N)
        m  = self.feature_projector(c)
        m  = F.layer_norm(m, [m.size(-1)])
        return m.view(B, N, self.meta_out)

    def _prep3(self, x, B, N):
        D = x.size(2)
        if D != self.message_dim:
            if D > self.message_dim: x = x[:, :, :self.message_dim]
            else:
                x = torch.cat([x, torch.zeros(B, N, self.message_dim - D,
                                               device=x.device, dtype=x.dtype)], 2)
        return x.unsqueeze(2).expand(-1, -1, self.his_num, -1)

    def _prep4(self, x, B, N):
        T, D = x.size(2), x.size(3)
        if T > self.his_num: x = x[:, :, :self.his_num, :]
        if D != self.message_dim:
            if D > self.message_dim: x = x[:, :, :, :self.message_dim]
            else:
                x = torch.cat([x, torch.zeros(B, N, T, self.message_dim - D,
                                               device=x.device, dtype=x.dtype)], 3)
        return x

    def _temporal(self, x, B, N):
        flat = x.reshape(B * N, self.his_num, self.message_dim)
        outs = []
        for i in range(0, B * N, self.max_chunk):
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outs.append(self.tp_learner(flat[i:i + self.max_chunk])[:, -1, :])
        return torch.cat(outs, 0)

    def _spatial(self, x, B, N, ei):
        outs = []
        for b in range(B):
            feats = x[b].reshape(N, self.his_num * self.message_dim)
            if ei is None:
                e2 = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            else:
                valid = (ei[0] < N) & (ei[1] < N)
                e2    = ei[:, valid]
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outs.append(self.sp_learner(feats, e2))
        return torch.stack(outs).reshape(B * N, -1)

    def _combine(self, tp, sp, x, B, N):
        if tp is not None and sp is not None: return torch.cat([F.relu(tp), F.relu(sp)], 1)
        if tp is not None: return F.relu(tp)
        if sp is not None: return F.relu(sp)
        return F.relu(x.reshape(B * N, self.his_num * self.message_dim))


# ── MetaTransformer ──────────────────────────────────────────────────────────

class MetaTransformer(nn.Module):
    def __init__(self, model_args, task_args, input_dim=None, hidden_dim=None,
                 output_dim=None, device="cuda"):
        super().__init__()
        self.device     = device if torch.cuda.is_available() else "cpu"
        self.meta_dim   = model_args.get("meta_dim", model_args.get("hidden_dim", 64))
        self.input_dim  = model_args.get("message_dim", 4) if input_dim  is None else input_dim
        self.hidden_dim = model_args.get("hidden_dim", 16) if hidden_dim is None else hidden_dim
        self.output_dim = model_args.get("output_dim", 2)  if output_dim is None else output_dim
        self.num_heads  = model_args.get("num_heads", 4)
        self.num_layers = model_args.get("num_layers", 2)
        self.dropout    = model_args.get("dropout", 0.1)
        if self.hidden_dim % self.num_heads != 0:
            self.hidden_dim = (self.hidden_dim // self.num_heads) * self.num_heads
        self._build(); self.to(self.device)

    def _build(self):
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.ReLU()).to(self.device)
        self.q_proj = nn.Linear(self.meta_dim, self.hidden_dim, device=self.device)
        self.k_proj = nn.Linear(self.meta_dim, self.hidden_dim, device=self.device)
        self.v_proj = nn.Linear(self.meta_dim, self.hidden_dim, device=self.device)
        enc = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 2, dropout=self.dropout,
            activation="gelu", batch_first=True, device=self.device)
        self.transformer_encoder = nn.TransformerEncoder(
            enc, num_layers=min(2, self.num_layers),
            norm=nn.LayerNorm(self.hidden_dim, device=self.device))
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_dim, device=self.device))
        for p in self.parameters():
            if p.dim() > 1: nn.init.xavier_uniform_(p)

    def forward(self, meta_knowledge, data, input=None):
        inp  = (data if input is None else input).to(self.device)
        meta = meta_knowledge.to(self.device)
        if inp.dim() == 4:
            B, N, T, F = inp.shape
        elif inp.dim() == 3:
            B, N, F = inp.shape; T = 1; inp = inp.unsqueeze(2)
        else: raise ValueError(f"Invalid shape: {inp.shape}")
        flat = inp.reshape(-1, T, F)
        if F != self.input_dim:
            if F < self.input_dim:
                flat = torch.cat([flat, torch.zeros(*flat.shape[:-1], self.input_dim - F,
                                                     device=flat.device, dtype=flat.dtype)], -1)
            else: flat = flat[..., :self.input_dim]
        proj = self.input_projection(flat)
        mf   = meta.reshape(-1, self.meta_dim)
        q = self.q_proj(mf).unsqueeze(1)
        k = self.k_proj(mf).unsqueeze(1)
        v = self.v_proj(mf).unsqueeze(1)
        aw  = F.softmax(torch.bmm(proj, k.transpose(1, 2)) * self.hidden_dim ** -0.5, -1)
        enh = proj + torch.bmm(aw, v)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            out = self.output_layer(self.transformer_encoder(enh))
        return out.reshape(B, N, T, -1)


# ── AdaptiveLayerNorm ────────────────────────────────────────────────────────

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int): normalized_shape = [normalized_shape]
        self.layer_norm = nn.LayerNorm(normalized_shape)
        self.norm_shape = normalized_shape

    def forward(self, x):
        if not isinstance(x, torch.Tensor): raise TypeError(f"Expected Tensor, got {type(x)}")
        if x.dim() > len(self.norm_shape) + 1:
            x = x.view(-1, *x.shape[-len(self.norm_shape):])
        if x.dim() == 2 and len(self.norm_shape) == 1 and x.size(1) != self.norm_shape[0]:
            x = x.t()
        return self.layer_norm(x)


# ── CrossDomainFusion ────────────────────────────────────────────────────────

class CrossDomainFusion(nn.Module):
    def __init__(self, spatial_dim, temporal_dim, fusion_dim, device="cuda"):
        super().__init__()
        self.spatial_dim  = spatial_dim
        self.temporal_dim = temporal_dim
        self.fusion_dim   = fusion_dim
        self.device       = device if torch.cuda.is_available() else "cpu"
        self.shared_dim   = min(spatial_dim, temporal_dim, fusion_dim) // 2
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim,  self.shared_dim), nn.LayerNorm(self.shared_dim), nn.ReLU()).to(self.device)
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_dim, self.shared_dim), nn.LayerNorm(self.shared_dim), nn.ReLU()).to(self.device)
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=self.shared_dim, num_heads=1, batch_first=True, device=self.device)
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.shared_dim * 2, fusion_dim), nn.LayerNorm(fusion_dim)).to(self.device)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        self.to(self.device)

    def _match(self, f, target, B):
        if f.size(1) > target: return f[:, :target]
        return torch.cat([f, torch.zeros(B, target - f.size(1), device=self.device)], 1)

    def forward(self, sp, tp):
        sp, tp = sp.to(self.device), tp.to(self.device)
        B = sp.size(0)
        if sp.dim() > 2: sp = sp.reshape(B, -1)
        if tp.dim() > 2: tp = tp.reshape(B, -1)
        if sp.size(1) != self.spatial_dim:  sp = self._match(sp, self.spatial_dim, B)
        if tp.size(1) != self.temporal_dim: tp = self._match(tp, self.temporal_dim, B)
        sp = sp.to(self.spatial_proj[0].weight.dtype)
        tp = tp.to(self.temporal_proj[0].weight.dtype)
        sp_p = self.spatial_proj(sp); tp_p = self.temporal_proj(tp)
        a, _ = self.fusion_attention(sp_p.unsqueeze(1), tp_p.unsqueeze(1), tp_p.unsqueeze(1))
        out  = self.fusion_proj(torch.cat([sp_p, a.squeeze(1)], -1))
        if sp_p.size(1) == out.size(1): out = out + sp_p
        if tp_p.size(1) == out.size(1): out = out + tp_p
        return out


# ── RandomTransformer ────────────────────────────────────────────────────────

class RandomTransformer(nn.Module):
    def __init__(self, model_args, task_args, device="cuda"):
        super().__init__()
        self.model_args = model_args; self.task_args = task_args
        self.device = device if torch.cuda.is_available() else "cpu"
        self._build(); self.to(self.device)

    def _build(self):
        self.input_dim  = self.model_args.get("input_dim", 2)
        self.hidden_dim = self.model_args.get("hidden_dim", 16)
        self.num_heads  = self.model_args.get("num_heads", 2)
        self.hidden_dim = ((self.hidden_dim + self.num_heads - 1) // self.num_heads) * self.num_heads
        self.num_layers = self.model_args.get("num_layers", 1)
        self.output_dim = self.task_args.get("output_dim", self.task_args.get("pred_num", 6))
        self.dropout    = self.model_args.get("dropout", 0.1)
        act             = self.model_args.get("activation", "relu")
        use_ln          = self.model_args.get("use_layer_norm", True)
        self.batch_first= self.model_args.get("batch_first", True)
        inter           = self.model_args.get("intermediate_factor", 1)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim) if use_ln else nn.Identity(),
            nn.Dropout(self.dropout)).to(self.device)
        enc = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * inter, dropout=self.dropout,
            activation=act, batch_first=self.batch_first, device=self.device)
        self.transformer_encoder = nn.TransformerEncoder(
            enc, num_layers=self.num_layers,
            norm=nn.LayerNorm(self.hidden_dim) if use_ln else None).to(self.device)
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim).to(self.device)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, data, A_wave=None):
        x = (data.x if hasattr(data, "x") else data).to(self.device)
        if x.dim() == 4:
            B, N, T, F = x.shape; x = x.view(B * N, T, F)
        elif x.dim() == 3:
            B = N = None; F = x.shape[2]
        else: raise ValueError(f"Expected 3D or 4D, got {x.dim()}D")
        if F != self.input_dim:
            if F < self.input_dim:
                x = torch.cat([x, torch.zeros(*x.shape[:-1], self.input_dim - F,
                                               device=self.device, dtype=x.dtype)], -1)
            else: x = x[..., :self.input_dim]
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            out = self.output_proj(self.transformer_encoder(self.input_proj(x)))
        if B is not None:
            out = out.view(B, N, -1 if self.model_args.get("return_sequences") else out.shape[1], -1)
            if not self.model_args.get("return_sequences"): out = out[:, :, -1, :]
        return out


# ── TemporalEncoderWrapper ───────────────────────────────────────────────────

class TemporalEncoderWrapper(nn.Module):
    def __init__(self, model_args, task_args, device="cuda"):
        super().__init__()
        self.device = device if torch.cuda.is_available() else "cpu"
        if "meta_dim" not in model_args:
            model_args["meta_dim"] = model_args.get("hidden_dim", 32)
        if model_args.get("use_meta_transformer", False):
            self.encoder = MetaTransformer(model_args, task_args, device=self.device)
            self.use_meta = True
        else:
            self.encoder = RandomTransformer(model_args, task_args, device=self.device)
            self.use_meta = False
        self.model_args = model_args; self.to(self.device)

    def forward(self, data, meta_knowledge=None):
        if data is None: raise ValueError("Input cannot be None")
        x = data.to(self.device) if isinstance(data, torch.Tensor) else data.x.to(self.device)
        if self.use_meta:
            if meta_knowledge is None: raise ValueError("meta_knowledge required")
            return self.encoder(meta_knowledge.to(self.device), x)
        return self.encoder(x)


# ── MetaCrossDomainFusion ────────────────────────────────────────────────────

class MetaCrossDomainFusion(nn.Module):
    """
    Pipeline contract
    -----------------
    Input  : PyG Data  x=(B,N,T,F)  edge_index=(2,E)  y=(B,pred_num,N)
    Output : flow_pred (B,pred_num,N), uncertainty (B,pred_num,N), graph (N,N)
    """
    def __init__(self, model_args, task_args, device="cuda"):
        super().__init__()
        self.device           = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_args       = model_args
        self.task_args        = task_args
        self.hidden_dim       = model_args["hidden_dim"]
        self.meta_dim         = model_args["meta_dim"]
        self.message_dim      = model_args["message_dim"]
        self.output_dim       = model_args.get("output_dim", 1)
        self.his_num          = task_args["his_num"]
        self.pred_num         = task_args["pred_num"]
        self.gnn_hidden_dim   = min(128, self.hidden_dim)
        self.trans_hidden_dim = min(64,  self.hidden_dim)
        self._build_model(); self._build_heads(); self._reset(); self.to(self.device)

    def _build_model(self):
        self.mk_learner = STMetaLearner(self.model_args, self.task_args,
                                        device=str(self.device))
        self.spatial_encoder = nn.ModuleList([
            GCNConv(in_channels=self.message_dim * self.his_num,
                    out_channels=self.gnn_hidden_dim),
            AdaptiveLayerNorm(self.gnn_hidden_dim), nn.ReLU()])
        self.temporal_encoder = TemporalEncoderWrapper(
            model_args={
                "meta_dim":              self.meta_dim,
                "message_dim":           self.message_dim,
                "hidden_dim":            self.trans_hidden_dim,
                "num_heads":             min(4, max(1, self.trans_hidden_dim // 16)),
                "num_layers":            2,
                "dropout":               0.1,
                "use_meta_transformer":  True,
            },
            task_args=self.task_args, device=str(self.device))
        self.cross_domain_fusion = CrossDomainFusion(
            self.gnn_hidden_dim, self.trans_hidden_dim,
            self.hidden_dim, device=str(self.device))
        self.meta_attention = nn.Sequential(
            nn.Linear(self.meta_dim, self.hidden_dim, device=self.device), nn.Sigmoid())

    def _build_heads(self):
        self.flow_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2, device=self.device),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, self.pred_num, device=self.device))
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.pred_num, device=self.device),
            nn.Softplus())

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, data, A_hat=None):
        if isinstance(data, torch.Tensor):
            x, ei = data.to(self.device), None
        elif hasattr(data, "x"):
            x  = data.x.to(self.device)
            ei = getattr(data, "edge_index", None)
            if ei is not None: ei = ei.to(self.device)
        else: raise ValueError("Input must be tensor or PyG Data")

        B, N = x.shape[:2]

        # Normalise to (B, N, T, F)
        if x.dim() == 3:
            F = x.shape[2]
            input_data = (x.reshape(B, N, F // self.message_dim, self.message_dim)
                          if F % self.message_dim == 0 else x.unsqueeze(2))
        elif x.dim() == 4:
            input_data = x
        else: raise ValueError(f"Expected 3D or 4D, got {x.dim()}D")

        data_obj = data if not isinstance(data, torch.Tensor) else Data(x=x)
        data_obj = data_obj.to(self.device)

        meta   = self.mk_learner(data_obj)            # (B, N, meta_dim)
        graph  = torch.eye(N, device=self.device)

        # Spatial branch
        sp_feat = None
        if ei is not None:
            si = input_data.reshape(B, N, -1)
            ex = self.message_dim * self.his_num
            if si.shape[2] > ex: si = si[:, :, :ex]
            elif si.shape[2] < ex:
                si = torch.cat([si, torch.zeros(B, N, ex - si.shape[2],
                                                 device=self.device, dtype=si.dtype)], 2)
            valid = (ei[0] <= N - 1) & (ei[1] <= N - 1)
            ei_v  = ei[:, valid]
            pieces = []
            for b in range(B):
                h = si[b]
                for layer in self.spatial_encoder:
                    h = layer(h, ei_v) if isinstance(layer, GCNConv) else layer(h)
                pieces.append(h)
            sp_feat = torch.stack(pieces)             # (B, N, gnn_hidden)

        # Temporal branch
        tp_feat = self.temporal_encoder(input_data.to(self.device), meta)
        if tp_feat.dim() == 4: tp_feat = tp_feat[:, :, -1, :]  # (B, N, trans_hidden)

        # Fusion
        if sp_feat is not None:
            fused = self.cross_domain_fusion(sp_feat.to(self.device), tp_feat)
            gate  = torch.sigmoid(meta.mean(dim=(1, 2), keepdim=True)).to(self.device)
            fused = fused * gate
        else:
            fused = tp_feat                           # (B, N, hidden_dim)

        # Predict then TRANSPOSE → (B, pred_num, N) matching targets
        flow = self.flow_predictor(fused).transpose(1, 2).contiguous()
        unc  = self.uncertainty_predictor(fused).transpose(1, 2).contiguous()
        return flow, unc, graph

    def force_cpu(self):
        self.to("cpu")
        for p in self.parameters():
            p.data = p.data.cpu()
            if p.grad is not None: p.grad.data = p.grad.data.cpu()
        return self


# ── TransformerBasedPredictor ────────────────────────────────────────────────

class TransformerBasedPredictor(nn.Module):
    def __init__(self, model_args, task_args, device="cuda"):
        super().__init__()
        self.device      = device if torch.cuda.is_available() else "cpu"
        self.input_dim   = model_args.get("input_dim", 128)
        self.hidden_dim  = model_args.get("hidden_dim", 128)
        self.message_dim = model_args.get("message_dim", 8)
        self.pred_num    = task_args.get("pred_num", 6)
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.input_dim, self.message_dim),
            nn.LayerNorm(self.message_dim), nn.ReLU())
        self.temporal_transformer = RandomTransformer(
            {"input_dim": self.message_dim, "hidden_dim": self.hidden_dim,
             "num_heads": model_args.get("num_heads", 4),
             "num_layers": model_args.get("num_layers", 2),
             "dropout": model_args.get("dropout", 0.1), "batch_first": True},
            {"output_dim": self.hidden_dim}, device=self.device)
        sp = model_args.get("sp", False)
        if sp:
            self.spatial_embedder = nn.Sequential(
                nn.Linear(self.message_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim), nn.ReLU())
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * (2 if sp else 1), self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), nn.ReLU(), nn.Dropout(0.1))
        self.predictor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.pred_num))
        if model_args.get("predict_uncertainty"):
            self.uncertainty_predictor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2), nn.ReLU(),
                nn.Linear(self.hidden_dim // 2, self.pred_num), nn.Softplus())
        if model_args.get("learn_graph"):
            self.graph_learner = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        self.to(device)

    def forward(self, data, A_wave=None):
        x = (data.x if hasattr(data, "x") else data)
        x = (x.unsqueeze(2) if x.dim() == 3 else x).to(self.device)
        x = self.feature_extractor(x)
        B, N = x.shape[:2]
        tf = self.temporal_transformer(x.reshape(B * N, *x.shape[2:])).reshape(B, N, -1)
        fts = torch.cat([self.spatial_embedder(x.mean(2)), tf], -1)               if hasattr(self, "spatial_embedder") else tf
        fused = self.fusion_layer(fts)
        lg = None
        if hasattr(self, "graph_learner"):
            e = self.graph_learner(fused)
            lg = torch.sigmoid(F.cosine_similarity(e.unsqueeze(1), e.unsqueeze(0), dim=-1))
        preds = self.predictor(fused).transpose(1, 2).contiguous()
        unc   = self.uncertainty_predictor(fused).transpose(1, 2).contiguous()                 if hasattr(self, "uncertainty_predictor") else None
        return preds, unc, lg


# ── CrossCityAdapter ─────────────────────────────────────────────────────────

class CrossCityAdapter(nn.Module):
    def __init__(self, source_dim, target_dim, hidden_dim=128, device="cuda"):
        super().__init__()
        self.source_dim = source_dim; self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.device = torch.device("cpu" if not torch.cuda.is_available() else device)
        self.source_adapter = nn.Sequential(
            nn.Linear(source_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.target_adapter = nn.Sequential(
            nn.Linear(target_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        self.city_features = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.similarity_module = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.output_proj = nn.Linear(hidden_dim, target_dim) if hidden_dim != target_dim else None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        self.to(self.device)

    def forward(self, source, target):
        dev = next(self.parameters()).device
        if not isinstance(source, torch.Tensor): source = torch.tensor(source)
        if not isinstance(target, torch.Tensor): target = torch.tensor(target)
        if source.dim() == 2: source = source.unsqueeze(0)
        if target.dim() == 2: target = target.unsqueeze(0)
        if source.numel() == 0 or target.numel() == 0:
            raise ValueError("Empty input tensors")
        source, target = source.to(dev), target.to(dev)
        if source.size(0) != target.size(0):
            if source.size(0) == 1:   source = source.expand(target.size(0), -1, -1)
            elif target.size(0) == 1: target = target.expand(source.size(0), -1, -1)
            else:
                m = min(source.size(0), target.size(0))
                source, target = source[:m], target[:m]
        if source.size(2) != self.source_dim:
            source = nn.Linear(source.size(2), self.source_dim).to(dev)(source)
        if target.size(2) != self.target_dim:
            target = nn.Linear(target.size(2), self.target_dim).to(dev)(target)
        sa = self.source_adapter(source); ta = self.target_adapter(target)
        sim = torch.bmm(ta, sa.transpose(1, 2)) / self.hidden_dim ** 0.5
        w   = F.softmax(sim, -1)
        tr  = torch.bmm(w, sa)
        g   = torch.sigmoid(ta.mean(-1, keepdim=True))
        enh = ta * (1 - g) + tr * g
        if self.output_proj is not None: enh = self.output_proj(enh)
        return enh, w

    def transfer_meta_knowledge(self, source_meta, target_meta):
        dev = next(self.parameters()).device
        if source_meta.dim() == 2: source_meta = source_meta.unsqueeze(0)
        if target_meta.dim() == 2: target_meta = target_meta.unsqueeze(0)
        source_meta, target_meta = source_meta.to(dev), target_meta.to(dev)
        sp = nn.Linear(source_meta.size(-1), self.hidden_dim).to(dev)
        tp = nn.Linear(target_meta.size(-1), self.hidden_dim).to(dev)
        sim = torch.bmm(tp(target_meta), sp(source_meta).transpose(1, 2)) / self.hidden_dim ** 0.5
        w   = F.softmax(sim, -1)
        return (torch.bmm(w, source_meta) + target_meta) / 2.0


# ── FewShotTrafficLearner ────────────────────────────────────────────────────

class FewShotTrafficLearner:
    """Prototype-based few-shot adaptation (hard traffic-state binning, FSL section)."""
    def __init__(self, base_model, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.base_model = base_model.to(self.device)
        hd = base_model.hidden_dim
        self.proto_encoder = nn.Sequential(
            nn.Linear(hd, hd // 2), nn.LayerNorm(hd // 2), nn.ReLU(),
            nn.Linear(hd // 2, hd // 4)).to(self.device)
        self.adaptation_layers = nn.ModuleList([
            nn.Linear(hd, hd), nn.Linear(hd // 2, hd // 2),
            nn.Linear(base_model.pred_num, base_model.pred_num)]).to(self.device)
        self.temperature = nn.Parameter(torch.tensor(1.0, device=self.device))

    def _discretize(self, vals, num_bins=10):
        flat = vals.reshape(-1); lo, hi = flat.min(), flat.max()
        b = torch.floor((vals - lo) / ((hi - lo) / num_bins)).long()
        return torch.clamp(b, 0, num_bins - 1)

    def _intermediate(self, data):
        meta = self.base_model.mk_learner(data)
        x    = data.x.to(self.device); B, N = x.shape[:2]
        ei   = getattr(data, "edge_index", None)
        if ei is not None: ei = ei.to(self.device)
        sp_feat = None
        if ei is not None:
            si = x.reshape(B, N, -1)
            ex = self.base_model.message_dim * self.base_model.his_num
            if si.shape[2] < ex:
                si = torch.cat([si, torch.zeros(B, N, ex - si.shape[2],
                                                 device=self.device, dtype=si.dtype)], 2)
            elif si.shape[2] > ex: si = si[:, :, :ex]
            valid = (ei[0] < N) & (ei[1] < N); ei_v = ei[:, valid]
            pieces = []
            for b in range(B):
                h = si[b]
                for layer in self.base_model.spatial_encoder:
                    h = layer(h, ei_v) if isinstance(layer, GCNConv) else layer(h)
                pieces.append(h)
            sp_feat = torch.stack(pieces)
        tp_feat = self.base_model.temporal_encoder(x, meta)
        if tp_feat.dim() == 4: tp_feat = tp_feat[:, :, -1, :]
        if sp_feat is not None:
            fused = self.base_model.cross_domain_fusion(sp_feat, tp_feat)
            fused = fused * torch.sigmoid(meta.mean(dim=(1, 2), keepdim=True))
        else: fused = tp_feat
        return fused

    def create_support_prototypes(self, support_data, support_labels):
        with torch.no_grad():
            feats = torch.cat([self._intermediate(d) for d in support_data], 0)
            pf    = self.proto_encoder(feats)
            bins  = self._discretize(support_labels)
            return {b.item(): pf[bins == b].mean(0)
                    for b in torch.unique(bins) if (bins == b).any()}

    def adapt_to_target(self, support_data, support_labels, num_steps=10, lr=0.001):
        protos = self.create_support_prototypes(support_data, support_labels)
        adapted = copy.deepcopy(self.base_model)
        opt = torch.optim.Adam(self.adaptation_layers.parameters(), lr=lr)
        bin_ids = sorted(protos.keys())
        lo, hi  = support_labels.min(), support_labels.max()
        bw      = (hi - lo) / len(bin_ids)
        bv      = torch.tensor(bin_ids, device=self.device, dtype=torch.float)
        for _ in range(num_steps):
            for data, label in zip(support_data, support_labels):
                pf = self.proto_encoder(self._intermediate(data))
                lg = torch.stack([-((pf - protos[b].unsqueeze(0)) ** 2).sum(-1) / self.temperature
                                   for b in bin_ids], -1)
                pr = F.softmax(lg, -1)
                pred = lo + (pr * bv.unsqueeze(0).unsqueeze(0)).sum(-1) * bw + bw / 2
                for layer in self.adaptation_layers: pred = layer(pred)
                opt.zero_grad(); F.mse_loss(pred, label).backward(); opt.step()
        for i, layer in enumerate(self.adaptation_layers):
            if i == 0 and hasattr(adapted.mk_learner, "feature_projector"):
                adapted.mk_learner.feature_projector.weight.data = layer.weight.data
            elif i == 2:
                adapted.flow_predictor[-1].weight.data = layer.weight.data
        return adapted

    def predict(self, adapted_model, query_data):
        with torch.no_grad():
            return adapted_model(query_data)[:2]


# ── Factory ──────────────────────────────────────────────────────────────────

def create_model(model_args, task_args, device="cpu"):
    mtype = model_args.get("type", "MetaCrossDomainFusion")
    if mtype == "TransformerBased":      m = TransformerBasedPredictor(model_args, task_args, device)
    elif mtype == "RandomTransformer":   m = RandomTransformer(model_args, task_args, device)
    elif mtype in ("MetaTransformer", "MetaTransfomer"):
                                         m = MetaTransformer(model_args, task_args, device=device)
    else:                                m = MetaCrossDomainFusion(model_args, task_args, device)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    return m.to(dev)


def fix_existing_model(model, device=None):
    model, device = ensure_all_tensors_on_same_device(model, device)
    orig = model.forward
    def fixed_forward(self, data, A_hat=None):
        return orig(process_input_data(data, device), process_input_data(A_hat, device))
    model.forward = types.MethodType(fixed_forward, model)
    return model