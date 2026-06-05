import os
import math
import time
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional

from utils import (enhanced_metric_calculation, result_print,
                       EarlyStopping, MetricsTracker, fmt_time)


def _denorm(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return arr * std + mean


def _default_mb() -> int:
    return 2048 if torch.cuda.is_available() else 4096


class MCDTrainer:
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model          = model.to(device)
        self.config         = config
        self.device         = device
        self.model_batch    = int(config.get('training', {}).get('model_batch_size', 8))

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=float(config['training'].get('learning_rate', 1e-4)),
            weight_decay=float(config['training'].get('weight_decay', 1e-4)),
            betas=(0.9, 0.999),
        )

        sc = config['training'].get('scheduler', {})
        if sc.get('type', 'plateau') == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='min',
                factor=float(sc.get('factor', 0.5)),
                patience=int(sc.get('patience', 5)),
                min_lr=float(sc.get('min_lr', 1e-6)),
            )
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=int(sc.get('T_0', 50)),
                T_mult=int(sc.get('T_mult', 2)),
                eta_min=float(sc.get('eta_min', 1e-6)),
            )

        self.gradient_clip      = float(config['training'].get('gradient_clip', 1.0))
        self.loss_history       = defaultdict(list)
        self.metrics_tracker    = MetricsTracker()
        self.scaler             = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
        self._denorm_confirmed  = False
        # Paper Eq. 4 — nuclear norm weight for low-rank meta-knowledge
        model_cfg = config.get('model', config.get('mcd_ckt', {}))
        self.nuclear_norm_weight = float(model_cfg.get('nuclear_norm_weight', 0.001))

    def _compute_loss(self, flow_pred: torch.Tensor, uncertainty: torch.Tensor,
                       targets: torch.Tensor,
                       meta_knowledge: torch.Tensor = None) -> torch.Tensor:
        if flow_pred.shape != targets.shape:
            if flow_pred.dim() == 3 and targets.dim() == 3:
                if flow_pred.shape[1] == targets.shape[2] and flow_pred.shape[2] == targets.shape[1]:
                    flow_pred = flow_pred.transpose(1, 2)

        mse = F.mse_loss(flow_pred, targets)

        if uncertainty is not None:
            uncertainty = uncertainty.clamp(min=1e-3)
            if uncertainty.shape != targets.shape and uncertainty.shape[1] == targets.shape[2]:
                uncertainty = uncertainty.transpose(1, 2)
            precision = 1.0 / (uncertainty + 1e-6)
            aleatoric = torch.mean(precision * (flow_pred - targets)**2
                                   + torch.log(uncertainty + 1e-6))
            loss = mse + 0.1 * aleatoric
        else:
            loss = mse

        # Paper Eq. 4 — nuclear norm regularisation on meta-knowledge matrix
        if meta_knowledge is not None and self.nuclear_norm_weight > 0:
            from models.meta_model import STMetaLearner
            nuc = STMetaLearner.nuclear_norm_loss(meta_knowledge)
            loss = loss + self.nuclear_norm_weight * nuc

        loss = torch.max(loss, mse * 0.1)
        if not torch.isfinite(loss):
            loss = mse
        return loss

    def _forward(self, data):
        data = data.to(self.device)
        if self.scaler:
            with torch.cuda.amp.autocast():
                out = self.model(data)
        else:
            out = self.model(data)
        # Extract meta-knowledge for nuclear norm loss (paper Eq. 4)
        try:
            with torch.no_grad():
                self._last_meta = self.model.mk_learner(data)
        except Exception:
            self._last_meta = None
        return out

    def train_epoch(self, dataloader, epoch: int, epochs: int) -> Dict[str, float]:
        self.model.train()
        total_loss  = 0.0
        num_batches = 0
        total_b     = len(dataloader) if hasattr(dataloader, '__len__') else None
        batch_times = []

        for idx, batch in enumerate(dataloader):
            t0 = time.time()
            batch = batch.to(self.device)
            B     = batch.x.shape[0]

            self.optimizer.zero_grad()
            ep_loss = 0.0

            for cs in range(0, B, self.model_batch):
                ce = min(cs + self.model_batch, B)
                sub = batch.__class__(
                    x          = batch.x[cs:ce],
                    y          = batch.y[cs:ce],
                    edge_index = batch.edge_index,
                    num_nodes  = batch.num_nodes,
                )
                w = (ce - cs) / B

                flow_pred, uncertainty, _ = self._forward(sub)
                loss = self._compute_loss(flow_pred, uncertainty, sub.y,
                                          meta_knowledge=getattr(self, '_last_meta', None)) * w

                if not loss.requires_grad:
                    continue
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                ep_loss += loss.item()

            if self.gradient_clip > 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)

            if self.scaler:
                self.scaler.step(self.optimizer); self.scaler.update()
            else:
                self.optimizer.step()

            total_loss  += ep_loss; num_batches += 1
            batch_times.append(time.time() - t0)
            avg_t = np.mean(batch_times[-20:])
            eta   = avg_t * ((total_b or num_batches+1) - num_batches)

            if total_b:
                pct = (idx+1) / total_b
                bar = '█' * int(30*pct) + '░' * (30 - int(30*pct))
                print(f"\r  Epoch {epoch+1}/{epochs} [{bar}] {idx+1}/{total_b} "
                      f"loss={ep_loss:.4f} {avg_t:.2f}s/b ETA:{eta/60:.1f}m",
                      end='', flush=True)

        print()
        avg = total_loss / max(num_batches, 1)
        self.loss_history['total_loss'].append(avg)
        return {'total_loss': avg}

    @torch.no_grad()
    def evaluate(self, dataloader,
                 denorm_mean: float = None, denorm_std: float = None) -> Dict:
        # Auto-detect denorm params
        if denorm_mean is None or denorm_std is None:
            ds = dataloader.dataset
            denorm_mean = getattr(ds, 'raw_mean', None)
            denorm_std  = getattr(ds, 'raw_std',  None)

        self.model.eval()
        all_preds, all_targets        = [], []
        all_uncertainty, all_graphs   = [], []
        total_loss  = 0.0; num_batches = 0

        for batch in dataloader:
            batch = batch.to(self.device)
            B     = batch.x.shape[0]
            batch_preds = []; batch_loss = 0.0

            for cs in range(0, B, self.model_batch):
                ce = min(cs + self.model_batch, B)
                sub = batch.__class__(
                    x          = batch.x[cs:ce],
                    y          = batch.y[cs:ce],
                    edge_index = batch.edge_index,
                    num_nodes  = batch.num_nodes,
                )
                w = (ce - cs) / B

                if self.scaler:
                    with torch.cuda.amp.autocast():
                        flow_pred, uncertainty, graph = self.model(sub)
                else:
                    flow_pred, uncertainty, graph = self.model(sub)

                loss = self._compute_loss(flow_pred, uncertainty, sub.y)
                batch_loss += loss.item() * w

                # Align to (B, pred_num, N)
                fp = flow_pred
                if fp.shape != sub.y.shape and fp.dim() == 3:
                    if fp.shape[1] == sub.y.shape[2] and fp.shape[2] == sub.y.shape[1]:
                        fp = fp.transpose(1, 2)
                batch_preds.append(torch.nan_to_num(fp, nan=0.0).cpu().numpy())

                if uncertainty is not None:
                    all_uncertainty.append(uncertainty.cpu().numpy())
                if graph is not None and len(all_graphs) == 0:
                    all_graphs.append(graph.detach().cpu().numpy())

            total_loss  += batch_loss; num_batches += 1
            all_preds.append(np.concatenate(batch_preds, axis=0))
            all_targets.append(batch.y.cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)
        preds_np = np.concatenate(all_preds,   axis=0)   # (N, pred_num, nodes)
        tgts_np  = np.concatenate(all_targets, axis=0)   # (N, pred_num, nodes)

        if denorm_mean is not None and denorm_std is not None:
            preds_np = _denorm(preds_np, denorm_mean, denorm_std)
            tgts_np  = _denorm(tgts_np,  denorm_mean, denorm_std)
            if not self._denorm_confirmed:
                print(f"  [Denorm] mean={denorm_mean:.3f} std={denorm_std:.3f} "
                      f"→ target range [{tgts_np.min():.1f}, {tgts_np.max():.1f}]")
                self._denorm_confirmed = True

        times   = preds_np.shape[1]
        metrics = enhanced_metric_calculation(pred=preds_np, y=tgts_np, times=times)

        unc_arr   = (np.concatenate(all_uncertainty, axis=0).mean(axis=(0,-1))
                     if all_uncertainty else None)
        graph_arr = all_graphs[0] if all_graphs else None

        return {
            'loss':        avg_loss,
            'metrics':     metrics,
            'predictions': preds_np,
            'targets':     tgts_np,
            'uncertainty': unc_arr,
            'graph':       graph_arr,
        }

    def train(self, train_dl, val_dl, epochs: int, save_dir: str = None) -> Dict:
        print(f"\nTraining {epochs} epochs | "
              f"Params: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,} | "
              f"Device: {self.device} | model_bs: {self.model_batch}")
        sys.stdout.flush()

        es_cfg = self.config['training'].get('early_stopping', {})
        es = EarlyStopping(
            patience=int(es_cfg.get('patience', 30)),
            min_delta=float(es_cfg.get('min_delta', 1e-4)),
            restore_best=bool(es_cfg.get('restore_best_weights', True)),
            min_epochs=int(es_cfg.get('min_epochs', 0)),
        )

        best_val = float('inf'); best_mae = best_rmse = best_mape = float('inf')
        ep_times  = []; t0_total = time.time()
        last_real_val = float('inf')

        HEADER = (f"\n  {'Epoch':>5} | {'Train':>8} | {'Val':>8} | "
                  f"{'MAE':>8} | {'RMSE':>8} | {'MAPE%':>7} | {'LR':>9} | {'Time':>6} | {'ETA':>6}")
        SEP    = "  " + "-" * 86

        for epoch in range(epochs):
            t0     = time.time()
            do_val = ((epoch+1) % 5 == 0 or epoch == 0 or epoch == epochs-1)

            train_res = self.train_epoch(train_dl, epoch, epochs)

            if do_val:
                val_res       = self.evaluate(val_dl)
                _vl           = val_res['loss']
                last_real_val = _vl if np.isfinite(_vl) else last_real_val
            else:
                val_res = {'loss': train_res['total_loss'], 'metrics': {}}

            lr   = self.optimizer.param_groups[0]['lr']
            ep_t = time.time() - t0; ep_times.append(ep_t)
            eta  = np.mean(ep_times[-10:]) * (epochs - epoch - 1)

            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(last_real_val)
                else:
                    self.scheduler.step()

            self.metrics_tracker.update(train_res, 'train')
            self.metrics_tracker.update({'total_loss': val_res['loss']}, 'val')

            if do_val and val_res.get('metrics') and val_res['metrics'].get('MAE') is not None:
                mae  = float(np.mean(val_res['metrics']['MAE'][:3]))
                rmse = float(np.mean(val_res['metrics']['RMSE'][:3]))
                mape = float(np.mean(val_res['metrics']['MAPE'][:3]) * 100)
                if mae  < best_mae:  best_mae  = mae
                if rmse < best_rmse: best_rmse = rmse
                if mape < best_mape: best_mape = mape
                print(HEADER, flush=True); print(SEP, flush=True)
                print(f"  {epoch+1:5d} | {train_res['total_loss']:8.4f} | {val_res['loss']:8.4f} | "
                      f"{mae:8.4f} | {rmse:8.4f} | {mape:7.2f} | {lr:9.2e} | "
                      f"{ep_t:5.1f}s | {eta/60:5.1f}m", flush=True)
            else:
                print(f"  Epoch {epoch+1:3d} | loss={train_res['total_loss']:.4f} | "
                      f"lr={lr:.2e} | {ep_t:.1f}s | ETA:{eta/60:.1f}m", flush=True)

            if val_res['loss'] < best_val:
                best_val = val_res['loss']
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'optimizer_state_dict': self.optimizer.state_dict(),
                                'epoch': epoch, 'loss': best_val,
                                'mae': best_mae, 'config': self.config},
                               f"{save_dir}/best_model.pth")

            if es(val_res['loss'], self.model, epoch):
                print(f"\nEarly stopping at epoch {epoch+1}", flush=True)
                break

        total_t = time.time() - t0_total
        final   = self.evaluate(val_dl)
        print(f"\n{'='*60}")
        print(f"TRAINING DONE | {fmt_time(total_t)} | "
              f"Best val={best_val:.4f} MAE={best_mae:.4f} RMSE={best_rmse:.4f} MAPE={best_mape:.2f}%")
        print(f"{'='*60}\n", flush=True)

        return {
            'training_time':     total_t,
            'best_val_loss':     best_val,
            'loss_history':      dict(self.loss_history),
            'final_val_results': final,
            'total_epochs':      epoch + 1,
            'epoch_times':       ep_times,
            'best_metrics':      {'mae': best_mae, 'rmse': best_rmse, 'mape': best_mape},
            'epoch_times':       ep_times,
        }

    def finetune(self, target_dl, test_dl, epochs: int, save_dir: str = None) -> Dict:
        print(f"\nFine-tuning {epochs} epochs | model_bs: {self.model_batch}")

        # Paper §4.4 — use FewShotTrafficLearner for hybrid loss + proximity constraint
        model_cfg = self.config.get('model', self.config.get('mcd_ckt', {}))
        _lambda1  = float(model_cfg.get('lambda1', 0.1))
        _lambda2  = float(model_cfg.get('lambda2', 0.01))
        _epsilon  = float(model_cfg.get('epsilon',  0.1))
        try:
            from models.meta_model import FewShotTrafficLearner
            self._fsl = FewShotTrafficLearner(self.model, device=str(self.device),
                                              lambda1=_lambda1, lambda2=_lambda2)
            _theta_0 = {n: p.clone().detach() for n, p in self.model.named_parameters()}
        except Exception:
            self._fsl = None; _theta_0 = {}

        ft_opt = optim.Adam(self.model.parameters(),
                            lr=float(self.config['training'].get('target_lr', 5e-5)),
                            weight_decay=1e-4)
        ft_sch = optim.lr_scheduler.ReduceLROnPlateau(ft_opt, mode='min', factor=0.7, patience=15)
        es     = EarlyStopping(patience=50, min_delta=1e-5, min_epochs=epochs // 2)

        best_loss = best_mae = best_rmse = best_mape = float('inf')
        ep_times  = []

        FT_HEADER = (f"\n  {'Epoch':>5} | {'Loss':>10} | {'MAE':>8} | "
                     f"{'RMSE':>8} | {'MAPE%':>7} | {'LR':>9} | {'Time':>6} | {'ETA':>6}")
        FT_SEP    = "  " + "-" * 76

        for epoch in range(epochs):
            t0       = time.time(); self.model.train()
            ep_loss  = 0.0; n_batches = 0

            for batch in target_dl:
                batch = batch.to(self.device); B = batch.x.shape[0]
                ft_opt.zero_grad(); accum = 0.0

                for cs in range(0, B, self.model_batch):
                    ce = min(cs + self.model_batch, B)
                    sub = batch.__class__(
                        x=batch.x[cs:ce], y=batch.y[cs:ce],
                        edge_index=batch.edge_index, num_nodes=batch.num_nodes)
                    w = (ce - cs) / B
                    flow_pred, uncertainty, _ = self._forward(sub)
                    meta = getattr(self, '_last_meta', None)
                    loss = self._compute_loss(flow_pred, uncertainty, sub.y, meta) * w
                    if self.scaler:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    accum += loss.item()

                if self.scaler:
                    self.scaler.unscale_(ft_opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                if self.scaler:
                    self.scaler.step(ft_opt); self.scaler.update()
                else:
                    ft_opt.step()
                # Paper Eq. 16 — proximity constraint ||θ-θ₀|| ≤ ε
                if _theta_0 and _epsilon > 0:
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if name in _theta_0:
                                diff = param.data - _theta_0[name].to(param.device)
                                nrm  = diff.norm()
                                if nrm > _epsilon:
                                    param.data = _theta_0[name].to(param.device) + diff * (_epsilon / nrm)
                ep_loss += accum; n_batches += 1

            avg_loss = ep_loss / max(n_batches, 1)
            ft_sch.step(avg_loss)
            ep_t = time.time() - t0; ep_times.append(ep_t)
            eta  = np.mean(ep_times[-10:]) * (epochs - epoch - 1)
            lr   = ft_opt.param_groups[0]['lr']

            do_eval = ((epoch+1) % 5 == 0 or epoch == 0 or epoch == epochs-1)
            if do_eval:
                ev = self.evaluate(test_dl)
                if ev['metrics'].get('MAE') is not None:
                    mae  = float(np.mean(ev['metrics']['MAE'][:3]))
                    rmse = float(np.mean(ev['metrics']['RMSE'][:3]))
                    mape = float(np.mean(ev['metrics']['MAPE'][:3]) * 100)
                    if mae  < best_mae:  best_mae  = mae
                    if rmse < best_rmse: best_rmse = rmse
                    if mape < best_mape: best_mape = mape
                    print(FT_HEADER, flush=True); print(FT_SEP, flush=True)
                    print(f"  {epoch+1:5d} | {avg_loss:10.4f} | {mae:8.4f} | "
                          f"{rmse:8.4f} | {mape:7.2f} | {lr:9.2e} | "
                          f"{ep_t:5.1f}s | {eta/60:5.1f}m", flush=True)
                else:
                    print(f"  Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                          f"lr={lr:.2e} | {ep_t:.1f}s | ETA:{eta/60:.1f}m", flush=True)
            else:
                print(f"  Epoch {epoch+1:3d} | loss={avg_loss:.4f} | "
                      f"lr={lr:.2e} | {ep_t:.1f}s | ETA:{eta/60:.1f}m", flush=True)

            if avg_loss < best_loss:
                best_loss = avg_loss
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save({'model_state_dict': self.model.state_dict(),
                                'epoch': epoch, 'loss': avg_loss, 'mae': best_mae,
                                'stage': 'finetuned'}, f"{save_dir}/finetuned_model.pth")

            if es(avg_loss, self.model, epoch):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        print(f"\n{'='*60}")
        print(f"FINE-TUNE DONE | Avg {np.mean(ep_times):.1f}s/epoch | "
              f"Best loss={best_loss:.4f} MAE={best_mae:.4f} RMSE={best_rmse:.4f} MAPE={best_mape:.2f}%")
        print(f"{'='*60}\n")
        return {'best_finetune_loss': best_loss, 'finetune_epochs': epoch+1,
                'best_mae': best_mae, 'best_rmse': best_rmse, 'best_mape': best_mape}