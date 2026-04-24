from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

IDX_PD = 0
IDX_QD = 1
IDX_PG = 2
IDX_QG = 3
IDX_VM = 4
IDX_VA = 5
FEATURE_NAMES = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]

PQ_TARGET_COLS = [IDX_VM, IDX_VA]
PV_TARGET_COLS = [IDX_QG, IDX_VA]
SLACK_TARGET_COLS = [IDX_PG, IDX_QG]

BUS_TYPE_PQ = 0
BUS_TYPE_PV = 1
BUS_TYPE_SLACK = 2
BUS_TYPE_NAMES = {
    BUS_TYPE_PQ: "PQ",
    BUS_TYPE_PV: "PV",
    BUS_TYPE_SLACK: "SLACK",
}

LOGGER = logging.getLogger(__name__)

ALIAS_MAP = {
    "gen": ["gen", "gens"],
    "ext_grid": ["ext_grid", "ext_grids"],
    "line": ["line", "lines"],
    "trafo": ["trafo", "trafos"],
    "switch": ["switch", "switches"],
    "impedance": ["impedance", "impedances"],
    "load": ["load", "loads"],
}


def resolve_dataset_dirs(data_dir: str) -> Tuple[Path, Optional[Path]]:
    root = Path(data_dir)
    train_dir = root / "train"
    test_dir = root / "test"
    if train_dir.exists() and test_dir.exists():
        return train_dir, test_dir
    return root, None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_records(metadata: Dict[str, Any], base_key: str) -> List[Dict[str, Any]]:
    for key in ALIAS_MAP.get(base_key, [base_key]):
        value = metadata.get(key, None)
        if isinstance(value, list):
            return value
    return []


def load_network_metadata(data_dir: str) -> Dict[str, Any]:
    train_dir, _ = resolve_dataset_dirs(data_dir)
    candidate_paths = [train_dir / "network_metadata.json", Path(data_dir) / "network_metadata.json"]
    for path in candidate_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"未找到 network_metadata.json，已检查: {[str(p) for p in candidate_paths]}")


def get_sorted_bus_ids(network_metadata: Dict[str, Any]) -> List[int]:
    buses_dict = network_metadata.get("buses", {})
    if not isinstance(buses_dict, dict) or not buses_dict:
        raise ValueError("network_metadata.json 中缺少 buses 字段")
    return sorted(int(k) for k in buses_dict.keys())


def get_network_base_mva(network_metadata: Dict[str, Any], default: float = 100.0) -> float:
    network_info = network_metadata.get("network_info", {})
    if isinstance(network_info, dict):
        for key in ["sn_mva", "base_mva", "sn_MVA", "baseMVA"]:
            if key in network_info:
                try:
                    value = float(network_info[key])
                    if value > 0:
                        return value
                except Exception:
                    pass
    for key in ["sn_mva", "base_mva", "sn_MVA", "baseMVA"]:
        if key in network_metadata:
            try:
                value = float(network_metadata[key])
                if value > 0:
                    return value
            except Exception:
                pass
    return float(default)


def build_bus_type_vector(network_metadata: Dict[str, Any], device: Optional[torch.device] = None) -> torch.Tensor:
    bus_ids = get_sorted_bus_ids(network_metadata)
    bus_type = torch.full((len(bus_ids),), BUS_TYPE_PQ, dtype=torch.long)
    id_to_pos = {bus_id: idx for idx, bus_id in enumerate(bus_ids)}

    for item in get_records(network_metadata, "gen"):
        if isinstance(item, dict) and "bus" in item:
            bus = int(item["bus"])
            if bus in id_to_pos:
                bus_type[id_to_pos[bus]] = BUS_TYPE_PV

    for item in get_records(network_metadata, "ext_grid"):
        if isinstance(item, dict) and "bus" in item:
            bus = int(item["bus"])
            if bus in id_to_pos:
                bus_type[id_to_pos[bus]] = BUS_TYPE_SLACK

    if device is not None:
        bus_type = bus_type.to(device)
    return bus_type


class NodeFeatureStandardizer:
    def __init__(self, mean: torch.Tensor, std_safe: torch.Tensor, zero_std_mask: torch.Tensor):
        self.mean = mean.float()
        self.std_safe = std_safe.float()
        self.zero_std_mask = zero_std_mask.bool()

    @classmethod
    def from_npz(cls, path: str, device: torch.device) -> "NodeFeatureStandardizer":
        payload = np.load(Path(path), allow_pickle=True)
        mean = torch.as_tensor(payload["mean"], dtype=torch.float32, device=device)
        if "std_safe" in payload:
            std_safe = torch.as_tensor(payload["std_safe"], dtype=torch.float32, device=device)
        else:
            std = torch.as_tensor(payload["std"], dtype=torch.float32, device=device)
            std_safe = std.clone()
            std_safe[std_safe <= 1e-8] = 1.0
        zero_std_mask = torch.as_tensor(payload.get("zero_std_mask", std_safe <= 1e-8), device=device)
        return cls(mean=mean, std_safe=std_safe, zero_std_mask=zero_std_mask)

    def _fit_shape(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.mean.dim() == 1:
            mean = self.mean.view(*([1] * (H.dim() - 1)), -1)
            std = self.std_safe.view(*([1] * (H.dim() - 1)), -1)
            return mean, std
        if self.mean.dim() == 2 and self.mean.shape[0] == 1:
            mean = self.mean.view(*([1] * (H.dim() - 1)), -1)
            std = self.std_safe.view(*([1] * (H.dim() - 1)), -1)
            return mean, std
        if H.dim() == 2:
            n, f = H.shape
            return self.mean[:n, :f], self.std_safe[:n, :f]
        _, n, f = H.shape
        return self.mean[:n, :f].unsqueeze(0), self.std_safe[:n, :f].unsqueeze(0)

    def normalize(self, H: torch.Tensor, state_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mean, std_safe = self._fit_shape(H)
        Hn = (H - mean) / std_safe
        if state_valid_mask is not None:
            Hn = torch.where(state_valid_mask.unsqueeze(-1), Hn, torch.zeros_like(Hn))
        return Hn

    def denormalize(self, H: torch.Tensor, state_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        mean, std_safe = self._fit_shape(H)
        x = H * std_safe + mean
        if state_valid_mask is not None:
            x = torch.where(state_valid_mask.unsqueeze(-1), x, torch.zeros_like(x))
        return x


def create_bus_type_target_mask(state_valid_mask: torch.Tensor, bus_type: torch.Tensor, feat_dim: int = 6) -> torch.Tensor:
    if state_valid_mask.dim() != 2:
        raise ValueError(f"state_valid_mask 期望形状为 (B,N)，实际为 {tuple(state_valid_mask.shape)}")
    bsz, num_nodes = state_valid_mask.shape
    bt = bus_type[:num_nodes]
    mask = torch.zeros((bsz, num_nodes, feat_dim), dtype=torch.bool, device=state_valid_mask.device)
    pq = bt == BUS_TYPE_PQ
    pv = bt == BUS_TYPE_PV
    sl = bt == BUS_TYPE_SLACK
    mask[:, pq, IDX_VM] = True
    mask[:, pq, IDX_VA] = True
    mask[:, pv, IDX_QG] = True
    mask[:, pv, IDX_VA] = True
    mask[:, sl, IDX_PG] = True
    mask[:, sl, IDX_QG] = True
    mask &= state_valid_mask.unsqueeze(-1)
    return mask


def create_input_feature_mask_for_finetune(
    node_valid_mask: torch.Tensor,
    state_valid_mask: torch.Tensor,
    bus_type: torch.Tensor,
    feat_dim: int = 6,
) -> torch.Tensor:
    visible = torch.ones((node_valid_mask.shape[0], node_valid_mask.shape[1], feat_dim), dtype=torch.bool, device=node_valid_mask.device)
    target_mask = create_bus_type_target_mask(state_valid_mask=state_valid_mask, bus_type=bus_type, feat_dim=feat_dim)
    visible &= (~target_mask)
    visible &= node_valid_mask.unsqueeze(-1)
    return visible


def create_structured_pretrain_feature_mask(H: torch.Tensor, bus_type: torch.Tensor, state_valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    keep_mask = torch.ones_like(H, dtype=torch.bool)
    num_nodes = H.shape[-2]
    bt = bus_type[:num_nodes]
    pq = bt == BUS_TYPE_PQ
    pv = bt == BUS_TYPE_PV
    sl = bt == BUS_TYPE_SLACK
    if H.dim() == 2:
        keep_mask[pq, IDX_VM] = False
        keep_mask[pq, IDX_VA] = False
        keep_mask[pv, IDX_QG] = False
        keep_mask[pv, IDX_VA] = False
        keep_mask[sl, IDX_PG] = False
        keep_mask[sl, IDX_QG] = False
        if state_valid_mask is not None:
            keep_mask = torch.where(state_valid_mask.unsqueeze(-1), keep_mask, torch.ones_like(keep_mask))
        return keep_mask
    keep_mask[:, pq, IDX_VM] = False
    keep_mask[:, pq, IDX_VA] = False
    keep_mask[:, pv, IDX_QG] = False
    keep_mask[:, pv, IDX_VA] = False
    keep_mask[:, sl, IDX_PG] = False
    keep_mask[:, sl, IDX_QG] = False
    if state_valid_mask is not None:
        keep_mask = torch.where(state_valid_mask.unsqueeze(-1), keep_mask, torch.ones_like(keep_mask))
    return keep_mask


def move_batch_to_device(batch: Dict[str, Any], device: torch.device):
    H = batch["H"].to(device, non_blocking=True)
    Y = batch["Y"].to(device, non_blocking=True)
    node_valid_mask = batch["node_valid_mask"].to(device, non_blocking=True)
    state_valid_mask = batch.get("state_valid_mask", batch["node_valid_mask"]).to(device, non_blocking=True)
    return H, Y, node_valid_mask, state_valid_mask


def compute_complex_power_from_voltage(Y: torch.Tensor, vm: torch.Tensor, va_degree: torch.Tensor):
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
):
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
