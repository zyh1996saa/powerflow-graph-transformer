from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    class SummaryWriter:  # type: ignore[override]
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass

from pf_physics_losses import BusTypeDeltaLoss, MaskedMSELoss, physics_residual_loss, reconstruct_h_from_delta
from pf_topology_utils import (
    FeatureStandardizer,
    StandardizationBundle,
    create_bus_type_target_mask,
    ensure_dir,
    move_batch_to_device,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    eta_min: float = 1e-5
    amp_enable: bool = True


@dataclass
class LossConfig:
    use_bus_type_aware_loss: bool = True
    pq_loss_weight: float = 1.0
    pv_loss_weight: float = 1.0
    slack_loss_weight: float = 1.0
    phy_loss_weight: float = 0.10


@dataclass
class CheckpointConfig:
    save_every_epochs: int = 5
    keep_last_n_epoch_ckpts: int = 5
    resume_checkpoint: Optional[str] = None


@dataclass
class ModelConfig:
    node_feat_dim: int = 6
    edge_feat_dim: int = 4
    output_dim: int = 6
    d_model: int = 192
    num_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.10
    edge_threshold: float = 1e-8
    dynamic_depth_sampling: bool = True
    max_num_nodes: int = 2048


@dataclass
class StageRunConfig:
    num_epochs: int = 220
    do_final_test: bool = True
    print_grad_norm: bool = False


class PFDeltaTrainer:
    def __init__(
        self,
        model: nn.Module,
        bus_type: torch.Tensor,
        standardizers: Optional[StandardizationBundle],
        device: str,
        output_dir: Path,
        base_mva: float,
        optimization_cfg: OptimizationConfig,
        loss_cfg: LossConfig,
        checkpoint_cfg: CheckpointConfig,
        model_cfg: ModelConfig,
        stats_path: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.bus_type = bus_type.to(self.device)
        self.standardizers = standardizers
        self.input_standardizer: Optional[FeatureStandardizer] = None if standardizers is None else standardizers.h_dc
        self.delta_standardizer: Optional[FeatureStandardizer] = None if standardizers is None else standardizers.delta
        self.base_mva = float(base_mva)
        self.output_dir = output_dir
        ensure_dir(self.output_dir)
        self.stats_path = stats_path

        self.optimization_cfg = optimization_cfg
        self.loss_cfg = loss_cfg
        self.checkpoint_cfg = checkpoint_cfg
        self.model_cfg = model_cfg

        self.use_amp = bool(optimization_cfg.amp_enable and self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        self.writer = SummaryWriter(str(self.output_dir / "tb"))
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        self.global_step = 0
        self.best_val = {"delta": float("inf")}
        self.history = {"delta": {"train_loss": [], "val_loss": [], "train_rmse": [], "val_rmse": []}}
        self._epoch_ckpt_paths: List[Path] = []
        self.delta_criterion = BusTypeDeltaLoss(
            pq_weight=loss_cfg.pq_loss_weight,
            pv_weight=loss_cfg.pv_loss_weight,
            slack_weight=loss_cfg.slack_loss_weight,
        )
        self.masked_mse = MaskedMSELoss()

    def close(self) -> None:
        self.writer.close()

    def reset_optimizer_scheduler(self, total_epochs: int) -> None:
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.optimization_cfg.learning_rate,
            weight_decay=self.optimization_cfg.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, total_epochs),
            eta_min=self.optimization_cfg.eta_min,
        )
        self._epoch_ckpt_paths = []

    def normalize_h_dc(self, H_dc_phys: torch.Tensor, state_valid_mask: torch.Tensor) -> torch.Tensor:
        if self.input_standardizer is None:
            return H_dc_phys
        return self.input_standardizer.normalize(H_dc_phys, state_valid_mask=state_valid_mask)

    def normalize_delta(self, delta_phys: torch.Tensor, state_valid_mask: torch.Tensor) -> torch.Tensor:
        if self.delta_standardizer is None:
            return delta_phys
        return self.delta_standardizer.normalize(delta_phys, state_valid_mask=state_valid_mask)

    def denormalize_delta(self, delta_norm: torch.Tensor, state_valid_mask: torch.Tensor) -> torch.Tensor:
        if self.delta_standardizer is None:
            return delta_norm
        return self.delta_standardizer.denormalize(delta_norm, state_valid_mask=state_valid_mask)

    def forward_batch(
        self,
        H_dc_norm: torch.Tensor,
        Y: torch.Tensor,
        node_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(H_dc_norm, Y, node_valid_mask=node_valid_mask, feature_visible_mask=None)

    def _atomic_torch_save(self, obj: Dict[str, Any], path: Path) -> None:
        ensure_dir(path.parent)
        with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

    def _checkpoint_payload(self, stage: str, epoch: int) -> Dict[str, Any]:
        return {
            "stage": stage,
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": None if self.optimizer is None else self.optimizer.state_dict(),
            "scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "best_val": self.best_val,
            "history": self.history,
            "bus_type": self.bus_type.detach().cpu(),
            "model_config": asdict(self.model_cfg),
            "optimization_config": asdict(self.optimization_cfg),
            "loss_config": asdict(self.loss_cfg),
            "base_mva": self.base_mva,
            "stats_path": self.stats_path,
            "target_formulation": "H_pred = H_dc + delta_on_bus_type_targets",
            "timestamp": datetime.now().isoformat(),
        }

    def save_checkpoint(self, path: Path, stage: str, epoch: int) -> None:
        self._atomic_torch_save(self._checkpoint_payload(stage=stage, epoch=epoch), path)

    def save_periodic_checkpoint(self, stage: str, epoch: int) -> None:
        if self.checkpoint_cfg.save_every_epochs > 0 and (epoch + 1) % self.checkpoint_cfg.save_every_epochs == 0:
            ckpt_path = self.output_dir / f"ckpt_{stage}_epoch_{epoch + 1:04d}.pt"
            self.save_checkpoint(ckpt_path, stage=stage, epoch=epoch)
            self._epoch_ckpt_paths.append(ckpt_path)
            keep_n = self.checkpoint_cfg.keep_last_n_epoch_ckpts
            if keep_n > 0 and len(self._epoch_ckpt_paths) > keep_n:
                old = self._epoch_ckpt_paths.pop(0)
                if old.exists():
                    old.unlink()
        self.save_checkpoint(self.output_dir / f"ckpt_{stage}_last.pt", stage=stage, epoch=epoch)

    def save_best_checkpoint(self, stage: str, epoch: int) -> None:
        self.save_checkpoint(self.output_dir / f"ckpt_{stage}_best.pt", stage=stage, epoch=epoch)

    def load_checkpoint(
        self,
        path: str,
        load_optimizer: bool = True,
        load_scheduler: bool = True,
        load_scaler: bool = True,
    ) -> Dict[str, Any]:
        payload = torch.load(Path(path), map_location=self.device)
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        if load_optimizer and self.optimizer is not None and payload.get("optimizer_state_dict") is not None:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if load_scheduler and self.scheduler is not None and payload.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(payload["scheduler_state_dict"])
        if load_scaler and self.use_amp and payload.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(payload["scaler_state_dict"])
        if isinstance(payload.get("best_val"), dict):
            self.best_val.update(payload["best_val"])
        if isinstance(payload.get("history"), dict):
            self.history = payload["history"]
        self.global_step = int(payload.get("global_step", 0))
        return payload

    def _backward_and_step(self, loss: torch.Tensor) -> float:
        if self.optimizer is None:
            raise RuntimeError("optimizer has not been initialized")
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optimization_cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optimization_cfg.grad_clip)
            self.optimizer.step()
        return float(grad_norm)

    def _compute_total_loss(
        self,
        pred_delta_norm: torch.Tensor,
        target_delta_norm: torch.Tensor,
        H_dc_phys: torch.Tensor,
        Y: torch.Tensor,
        state_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        stats: Dict[str, float] = {}
        if self.loss_cfg.use_bus_type_aware_loss:
            sup_loss, group_stats = self.delta_criterion(
                pred_delta_norm,
                target_delta_norm,
                self.bus_type,
                state_valid_mask=state_valid_mask,
            )
            stats.update(group_stats)
        else:
            target_mask = create_bus_type_target_mask(state_valid_mask=state_valid_mask, bus_type=self.bus_type)
            sup_loss = self.masked_mse(pred_delta_norm, target_delta_norm, target_mask)
        stats["supervised_delta_mse"] = float(sup_loss.detach().item())

        total_loss = sup_loss
        phy_weight = float(self.loss_cfg.phy_loss_weight)
        if phy_weight > 0:
            pred_delta_phys = self.denormalize_delta(pred_delta_norm, state_valid_mask=state_valid_mask)
            H_pred_phys = reconstruct_h_from_delta(
                H_dc_phys=H_dc_phys,
                delta_phys=pred_delta_phys,
                bus_type=self.bus_type,
                state_valid_mask=state_valid_mask,
            )
            phy_loss, phy_stats = physics_residual_loss(
                H_pred_phys,
                Y,
                state_valid_mask=state_valid_mask,
                base_mva=self.base_mva,
            )
            total_loss = total_loss + phy_weight * phy_loss
            stats.update(phy_stats)
            stats["phy_weight"] = phy_weight

        stats["loss"] = float(total_loss.detach().item())
        stats["rmse"] = math.sqrt(max(float(sup_loss.detach().item()), 0.0))
        return total_loss, stats

    def _write_epoch_scalars(self, stage: str, epoch: int, scalar_dict: Dict[str, float]) -> None:
        for key, value in scalar_dict.items():
            self.writer.add_scalar(f"{stage}/{key}", float(value), epoch + 1)

    def _run_one_epoch(self, data_loader, stage: str, training: bool, print_grad_norm: bool = False) -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        total: Dict[str, float] = {}
        n_batches = 0
        iterator = tqdm(data_loader, desc=f"[{stage}] {'train' if training else 'eval'}", leave=False)

        for batch in iterator:
            H_true, H_dc, delta, Y, node_valid_mask, state_valid_mask = move_batch_to_device(batch, self.device)
            del H_true
            H_dc_norm = self.normalize_h_dc(H_dc, state_valid_mask=state_valid_mask)
            target_delta_norm = self.normalize_delta(delta, state_valid_mask=state_valid_mask)

            if training:
                if self.optimizer is None:
                    raise RuntimeError("optimizer has not been initialized")
                self.optimizer.zero_grad(set_to_none=True)

            grad_norm = None
            with torch.set_grad_enabled(training):
                with autocast(enabled=self.use_amp):
                    pred_delta_norm = self.forward_batch(
                        H_dc_norm,
                        Y,
                        node_valid_mask=node_valid_mask,
                    )
                    loss, metrics = self._compute_total_loss(
                        pred_delta_norm,
                        target_delta_norm,
                        H_dc,
                        Y,
                        state_valid_mask=state_valid_mask,
                    )
                if training:
                    grad_norm = self._backward_and_step(loss)
                    self.global_step += 1

            if print_grad_norm and grad_norm is not None:
                metrics["grad_norm"] = grad_norm

            n_batches += 1
            for key, value in metrics.items():
                total[key] = total.get(key, 0.0) + float(value)
            iterator.set_postfix(loss=f"{metrics['loss']:.5f}", rmse=f"{metrics['rmse']:.5f}")

        if n_batches == 0:
            return {"loss": float("nan"), "rmse": float("nan")}
        return {k: v / n_batches for k, v in total.items()}

    def run(self, train_loader, val_loader, num_epochs: int, start_epoch: int = 0, print_grad_norm: bool = False) -> Dict[str, float]:
        self.reset_optimizer_scheduler(total_epochs=num_epochs)
        if self.checkpoint_cfg.resume_checkpoint:
            self.load_checkpoint(self.checkpoint_cfg.resume_checkpoint, load_optimizer=False, load_scheduler=False, load_scaler=False)
        if start_epoch > 0 and self.scheduler is not None:
            self.scheduler.last_epoch = start_epoch - 1
        stage = "delta"
        for epoch in range(start_epoch, num_epochs):
            train_metrics = self._run_one_epoch(train_loader, stage=stage, training=True, print_grad_norm=print_grad_norm)
            val_metrics = self._run_one_epoch(val_loader, stage=stage, training=False, print_grad_norm=False)
            self.history[stage]["train_loss"].append(train_metrics["loss"])
            self.history[stage]["val_loss"].append(val_metrics["loss"])
            self.history[stage]["train_rmse"].append(train_metrics["rmse"])
            self.history[stage]["val_rmse"].append(val_metrics["rmse"])
            self._write_epoch_scalars(stage, epoch, {f"train_{k}": v for k, v in train_metrics.items()})
            self._write_epoch_scalars(stage, epoch, {f"val_{k}": v for k, v in val_metrics.items()})
            LOGGER.info(
                "[Delta] Epoch %d/%d: train_loss=%.6f, train_rmse=%.6f, val_loss=%.6f, val_rmse=%.6f",
                epoch + 1,
                num_epochs,
                train_metrics["loss"],
                train_metrics["rmse"],
                val_metrics["loss"],
                val_metrics["rmse"],
            )
            self.save_periodic_checkpoint(stage, epoch)
            if val_metrics["loss"] < self.best_val[stage]:
                self.best_val[stage] = val_metrics["loss"]
                self.save_best_checkpoint(stage, epoch)
            if self.scheduler is not None:
                self.scheduler.step()
        return {"best_val": self.best_val[stage]}

    @torch.no_grad()
    def evaluate(self, data_loader) -> Dict[str, float]:
        return self._run_one_epoch(data_loader, stage="delta", training=False, print_grad_norm=False)

    def save_history_json(self) -> None:
        with open(self.output_dir / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)


# Backward-compatible alias.
PFTrainer = PFDeltaTrainer
