from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from pf_topology_utils import (
    BUS_TYPE_PQ,
    BUS_TYPE_PV,
    BUS_TYPE_SLACK,
    IDX_PD,
    IDX_PG,
    IDX_QD,
    IDX_QG,
    IDX_VA,
    IDX_VM,
    create_bus_type_target_mask,
)


class MaskedMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, recon_mask: torch.Tensor) -> torch.Tensor:
        err2 = ((pred - target) ** 2) * recon_mask.to(pred.dtype)
        denom = recon_mask.to(pred.dtype).sum().clamp_min(self.eps)
        return err2.sum() / denom


class BusTypePowerFlowLoss(nn.Module):
    def __init__(self, pq_weight: float = 1.0, pv_weight: float = 1.0, slack_weight: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.pq_weight = float(pq_weight)
        self.pv_weight = float(pv_weight)
        self.slack_weight = float(slack_weight)
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        bus_type: torch.Tensor,
        state_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            state_valid_mask = state_valid_mask.unsqueeze(0)
        bsz, num_nodes, _ = pred.shape
        bt = bus_type[:num_nodes]
        total_loss = pred.new_tensor(0.0)
        total_weight = pred.new_tensor(0.0)
        stats: Dict[str, float] = {}

        def group_loss(name: str, node_mask_1d: torch.Tensor, cols: List[int], weight: float) -> None:
            nonlocal total_loss, total_weight, stats
            if weight <= 0:
                stats[f"{name}_mse"] = 0.0
                stats[f"{name}_rmse"] = 0.0
                return
            group_mask = state_valid_mask & node_mask_1d.view(1, num_nodes)
            expanded = group_mask.unsqueeze(-1).expand(bsz, num_nodes, len(cols))
            denom = expanded.to(pred.dtype).sum().clamp_min(self.eps)
            mse = ((((pred[:, :, cols] - target[:, :, cols]) ** 2) * expanded.to(pred.dtype)).sum() / denom)
            total_loss = total_loss + weight * mse
            total_weight = total_weight + weight
            mse_value = float(mse.detach().item())
            stats[f"{name}_mse"] = mse_value
            stats[f"{name}_rmse"] = math.sqrt(max(mse_value, 0.0))

        group_loss("pq", bt == BUS_TYPE_PQ, [IDX_VM, IDX_VA], self.pq_weight)
        group_loss("pv", bt == BUS_TYPE_PV, [IDX_QG, IDX_VA], self.pv_weight)
        group_loss("slack", bt == BUS_TYPE_SLACK, [IDX_PG, IDX_QG], self.slack_weight)
        total_loss = total_loss / total_weight.clamp_min(self.eps)
        return total_loss, stats


def compute_complex_power_from_voltage(Y: torch.Tensor, vm: torch.Tensor, va_degree: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    va_rad = torch.deg2rad(va_degree)
    v_complex = vm.to(torch.complex64) * torch.exp(1j * va_rad.to(torch.complex64))
    current = torch.einsum("bij,bj->bi", Y.to(torch.complex64), v_complex)
    s_inj = v_complex * torch.conj(current)
    return s_inj.real.float(), s_inj.imag.float()


def physics_residual_loss(
    H_pred_phys: torch.Tensor,
    Y: torch.Tensor,
    state_valid_mask: torch.Tensor,
    base_mva: float,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if base_mva <= 0:
        raise ValueError(f"base_mva must be positive, got {base_mva}")

    pd = H_pred_phys[:, :, IDX_PD]
    qd = H_pred_phys[:, :, IDX_QD]
    pg = H_pred_phys[:, :, IDX_PG]
    qg = H_pred_phys[:, :, IDX_QG]
    vm = H_pred_phys[:, :, IDX_VM].clamp_min(1e-5)
    va = H_pred_phys[:, :, IDX_VA]

    p_spec_pu = (pg - pd) / float(base_mva)
    q_spec_pu = (qg - qd) / float(base_mva)
    p_calc_pu, q_calc_pu = compute_complex_power_from_voltage(Y, vm, va)

    mask = state_valid_mask.to(H_pred_phys.dtype)
    p_res = (p_spec_pu - p_calc_pu) * mask
    q_res = (q_spec_pu - q_calc_pu) * mask
    denom = mask.sum().clamp_min(eps)
    loss = (p_res.square().sum() + q_res.square().sum()) / (2.0 * denom)

    stats = {
        "phy_mse": float(loss.detach().item()),
        "phy_p_rmse_pu": float(torch.sqrt(p_res.square().sum() / denom).detach().item()),
        "phy_q_rmse_pu": float(torch.sqrt(q_res.square().sum() / denom).detach().item()),
        "base_mva": float(base_mva),
        "num_state_valid": int(state_valid_mask.sum().item()),
    }
    return loss, stats


def finetune_supervised_mse(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    bus_type: torch.Tensor,
    state_valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_mask = create_bus_type_target_mask(state_valid_mask=state_valid_mask, bus_type=bus_type, feat_dim=pred_norm.shape[-1])
    criterion = MaskedMSELoss(eps=eps)
    loss = criterion(pred_norm, target_norm, target_mask)
    stats = {
        "supervised_mse": float(loss.detach().item()),
        "rmse": math.sqrt(max(float(loss.detach().item()), 0.0)),
    }
    return loss, stats
