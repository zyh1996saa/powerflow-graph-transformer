# In[]
from __future__ import annotations

"""
Full physical evaluation for a trained DC-delta power-flow model.

This script evaluates three comparisons in original physical units:
    1) model_vs_ac: H_pred / derived physical quantities versus AC truth
    2) dc_vs_ac:    DC baseline / derived physical quantities versus AC truth
    3) model_vs_dc: H_pred / derived physical quantities versus DC baseline

The model formulation follows the training code:
    input  = normalized H_dc
    output = normalized delta
    H_pred = H_dc + denormalized_delta on bus-type target columns only

Main output files:
    - metrics_summary.csv / metrics_summary.json
    - selected_node_comparison_all.csv
    - selected_branch_comparison_all.csv
    - selected_node_comparison_<split>_sample_<id>.csv
    - selected_branch_comparison_<split>_sample_<id>.csv
    - selected_sample_summary.csv
    - eval_run_info.json

Notes on branch flow calculation:
    The dataset stores Ybus, not pandapower res_line/res_trafo tables. Therefore branch
    P/Q is computed from each active off-diagonal Ybus entry as a series-branch flow:
        y_ij = -Ybus[i, j]
        I_ij = (V_i - V_j) * y_ij
        S_ij = V_i * conj(I_ij) * base_mva
    This is suitable for comparing AC/DC/model states under the same Ybus topology.
    It does not separately reconstruct line charging/shunt/tap-specific pandapower
    result-table conventions unless those effects are recoverable from the off-diagonal
    term itself.
"""

import importlib.util
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def _ensure_canonical_module(module_name: str) -> None:
    if module_name in sys.modules:
        return
    try:
        if importlib.util.find_spec(module_name) is not None:
            return
    except Exception:
        pass
    candidates = sorted(SCRIPT_DIR.glob(f"{module_name}*.py"))
    if not candidates:
        return
    spec = importlib.util.spec_from_file_location(module_name, candidates[0])
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


for _module_name in [
    "pf_topology_utils",
    "pf_topology_encoder",
    "pf_powerflow_model",
    "pf_physics_losses",
    "pf_data_loader",
]:
    _ensure_canonical_module(_module_name)

from pf_data_loader import create_dataloaders  # noqa: E402
from pf_physics_losses import reconstruct_h_from_delta  # noqa: E402
from pf_powerflow_model import HybridGTForPowerFlow  # noqa: E402
from pf_topology_utils import (  # noqa: E402
    BUS_TYPE_NAMES,
    FEATURE_NAMES,
    IDX_PD,
    IDX_PG,
    IDX_QD,
    IDX_QG,
    IDX_VA,
    IDX_VM,
    StandardizationBundle,
    build_branch_catalog,
    build_bus_type_vector,
    create_bus_type_target_mask,
    get_network_base_mva,
    get_sorted_bus_ids,
    load_network_metadata,
)


# ============================================================
# 手动配置区（不要 argparse）
# ============================================================
DATA_DIR = "/data2/zyh/case39_samples_dc_delta"
OUTPUT_DIR = "./pf_eval_outputs_dc_delta_full_physics"

CHECKPOINT_PATH = ""
AUTO_DISCOVER_CHECKPOINT = True
CHECKPOINT_SEARCH_ROOT = "./logs/20260507_165445_pf_dc_delta"
CHECKPOINT_CANDIDATE_FILENAMES = [
    "ckpt_delta_best.pt",
    "ckpt_delta_last.pt",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
USE_AMP_FOR_INFERENCE = True

BATCH_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
PAD_TO_MAX = False
Y_AS_DENSE = True
CACHE_METADATA = True
CACHE_ARRAYS_IN_MEMORY = False
SHUFFLE_TRAIN_FOR_EVAL = False

ENABLE_STANDARDIZATION = True
STANDARDIZATION_STATS_PATH = str(Path(DATA_DIR) / "train_hdc_delta_stats_global_6.npz")
REQUIRE_STANDARDIZATION_STATS = True

# 节点原始 H 特征是否也输出统计。H 特征为 [Pd, Qd, Pg, Qg, Vm, Va]。
# 若只关注用户提到的节点注入、节点电压/相角、支路潮流，可保持 True，便于同时检查 Pg/Qg 等模型目标列。
EVALUATE_NODE_H_FEATURES = True
EVALUATE_H_FEATURES_BUS_TYPE_TARGET_ONLY = False
EVALUATED_H_FEATURE_INDICES = [IDX_PD, IDX_QD, IDX_PG, IDX_QG, IDX_VM, IDX_VA]

# 节点注入功率：
#   spec: Pg-Pd, Qg-Qd，直接来自 H/H_dc/H_pred 的有功/无功注入设定。
#   calc: base_mva * V * conj(YV)，由电压状态和 Ybus 反算的网络注入。
EVALUATE_NODE_SPEC_INJECTION = True
EVALUATE_NODE_CALC_INJECTION = True

# 支路潮流由 Ybus off-diagonal 的 series admittance 计算。
EVALUATE_BRANCH_FLOW = True
BRANCH_Y_ABS_EPS = 1e-10

# 百分比误差的分母阈值。参考值绝对值 <= 此阈值时，不计入 MAPE/max_APE，避免零值附近的无意义爆炸。
PERCENT_DENOM_EPS = 1e-8

# 比较项。默认包含用户通常关心的 model/DC 相对 AC，以及 model 相对 DC 的差异。
COMPARISONS = [
    ("model_vs_ac", "model", "ac"),
    ("dc_vs_ac", "dc", "ac"),
    ("model_vs_dc", "model", "dc"),
]

SELECTED_TRAIN_SAMPLE_IDS: List[int] = []
SELECTED_TEST_SAMPLE_IDS: List[int] = []
AUTO_SELECT_NUM_SAMPLES_PER_SPLIT = 5
AUTO_SELECT_SAMPLE_STRATEGY = "random"  # first / random / evenly_spaced
RANDOM_SEED_FOR_SAMPLE_SELECTION = 42

SAVE_SELECTED_NODE_COMPARISON = True
SAVE_SELECTED_BRANCH_COMPARISON = True
SAVE_SELECTED_SAMPLE_SUMMARY = True
PRINT_SELECTED_DETAIL_TO_CONSOLE = True

FALLBACK_MODEL_CONFIG = {
    "node_feat_dim": 6,
    "edge_feat_dim": 4,
    "output_dim": 6,
    "d_model": 192,
    "num_layers": 4,
    "num_heads": 8,
    "mlp_ratio": 4.0,
    "dropout": 0.10,
    "edge_threshold": 1e-8,
    "dynamic_depth_sampling": True,
    "max_num_nodes": 2048,
}

LOG_LEVEL = logging.INFO

PHYSICAL_UNITS = {
    "Pd": "MW",
    "Qd": "Mvar",
    "Pg": "MW",
    "Qg": "Mvar",
    "Vm": "p.u.",
    "Va": "degree",
    "P_inj_spec": "MW",
    "Q_inj_spec": "Mvar",
    "P_inj_calc": "MW",
    "Q_inj_calc": "Mvar",
    "P_ij": "MW",
    "Q_ij": "Mvar",
    "P_ji": "MW",
    "Q_ji": "Mvar",
    "P_loss_ij_ji": "MW",
    "Q_loss_ij_ji": "Mvar",
}

LOGGER = logging.getLogger("evaluate_pf_model_dc_delta_full_physics")


@dataclass
class MetricAccumulator:
    count: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    signed_sum: float = 0.0
    max_abs: float = 0.0
    percent_count: int = 0
    ape_sum_percent: float = 0.0
    max_ape_percent: float = 0.0

    def update(self, candidate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> None:
        candidate64 = candidate.detach().to(dtype=torch.float64)
        reference64 = reference.detach().to(dtype=torch.float64)
        mask_bool = mask.detach().to(dtype=torch.bool)
        finite_mask = torch.isfinite(candidate64) & torch.isfinite(reference64)
        mask_bool = mask_bool & finite_mask
        if not bool(mask_bool.any().item()):
            return

        err = (candidate64 - reference64)[mask_bool]
        ref = reference64[mask_bool]
        abs_err = err.abs()
        self.count += int(err.numel())
        self.abs_sum += float(abs_err.sum().item())
        self.sq_sum += float((err ** 2).sum().item())
        self.signed_sum += float(err.sum().item())
        self.max_abs = max(self.max_abs, float(abs_err.max().item()))

        ref_abs = ref.abs()
        pct_mask = ref_abs > float(PERCENT_DENOM_EPS)
        if bool(pct_mask.any().item()):
            ape_percent = abs_err[pct_mask] / ref_abs[pct_mask] * 100.0
            self.percent_count += int(ape_percent.numel())
            self.ape_sum_percent += float(ape_percent.sum().item())
            self.max_ape_percent = max(self.max_ape_percent, float(ape_percent.max().item()))

    def as_row(self, split: str, comparison: str, category: str, quantity: str, unit: str) -> Dict[str, Any]:
        if self.count <= 0:
            return {
                "split": split,
                "comparison": comparison,
                "category": category,
                "quantity": quantity,
                "unit": unit,
                "count": 0,
                "mae": math.nan,
                "rmse": math.nan,
                "mean_signed_error": math.nan,
                "max_abs_error": math.nan,
                "percent_count": 0,
                "mape_percent": math.nan,
                "max_ape_percent": math.nan,
                "percent_denom_eps": PERCENT_DENOM_EPS,
            }
        return {
            "split": split,
            "comparison": comparison,
            "category": category,
            "quantity": quantity,
            "unit": unit,
            "count": self.count,
            "mae": self.abs_sum / self.count,
            "rmse": math.sqrt(max(self.sq_sum / self.count, 0.0)),
            "mean_signed_error": self.signed_sum / self.count,
            "max_abs_error": self.max_abs,
            "percent_count": self.percent_count,
            "mape_percent": (self.ape_sum_percent / self.percent_count) if self.percent_count > 0 else math.nan,
            "max_ape_percent": self.max_ape_percent if self.percent_count > 0 else math.nan,
            "percent_denom_eps": PERCENT_DENOM_EPS,
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_checkpoint_path() -> Path:
    if CHECKPOINT_PATH.strip():
        path = Path(CHECKPOINT_PATH).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"指定的 checkpoint 不存在: {path}")
        return path

    if not AUTO_DISCOVER_CHECKPOINT:
        raise ValueError("CHECKPOINT_PATH 为空，且 AUTO_DISCOVER_CHECKPOINT=False")

    root = Path(CHECKPOINT_SEARCH_ROOT).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"checkpoint 搜索根目录不存在: {root}")

    candidates: List[Path] = []
    for filename in CHECKPOINT_CANDIDATE_FILENAMES:
        candidates.extend(root.rglob(filename))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"未在 {root} 下找到候选 checkpoint 文件: {CHECKPOINT_CANDIDATE_FILENAMES}")

    priority = {name: idx for idx, name in enumerate(CHECKPOINT_CANDIDATE_FILENAMES)}
    candidates.sort(key=lambda p: (p.stat().st_mtime, -priority.get(p.name, 999)), reverse=True)
    return candidates[0]


def load_checkpoint_payload(path: Path, device: torch.device) -> Dict[str, Any]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if "model_state_dict" not in payload:
        raise KeyError(f"checkpoint 缺少 model_state_dict: {path}")
    return payload


def sanitize_model_config(config: Dict[str, Any], num_buses: int) -> Dict[str, Any]:
    allowed = set(FALLBACK_MODEL_CONFIG.keys())
    merged = dict(FALLBACK_MODEL_CONFIG)
    for key, value in dict(config).items():
        if key in allowed:
            merged[key] = value
    merged["max_num_nodes"] = max(int(merged.get("max_num_nodes", 0)), int(num_buses))
    return merged


def build_model_from_checkpoint(
    checkpoint_payload: Dict[str, Any],
    network_metadata: Dict[str, Any],
    bus_type: torch.Tensor,
    device: torch.device,
) -> HybridGTForPowerFlow:
    model_config = sanitize_model_config(
        checkpoint_payload.get("model_config", FALLBACK_MODEL_CONFIG),
        num_buses=len(bus_type),
    )
    model = HybridGTForPowerFlow(
        node_feat_dim=int(model_config["node_feat_dim"]),
        edge_feat_dim=int(model_config["edge_feat_dim"]),
        output_dim=int(model_config["output_dim"]),
        d_model=int(model_config["d_model"]),
        num_layers=int(model_config["num_layers"]),
        num_heads=int(model_config["num_heads"]),
        mlp_ratio=float(model_config["mlp_ratio"]),
        dropout=float(model_config["dropout"]),
        edge_threshold=float(model_config["edge_threshold"]),
        max_num_nodes=int(model_config["max_num_nodes"]),
        network_metadata=network_metadata,
        dynamic_depth_sampling=bool(model_config.get("dynamic_depth_sampling", True)),
    )
    model.load_state_dict(checkpoint_payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def build_standardizers(device: torch.device, checkpoint_payload: Dict[str, Any]) -> Optional[StandardizationBundle]:
    if not ENABLE_STANDARDIZATION:
        return None
    candidate_paths: List[Path] = []
    ckpt_stats = checkpoint_payload.get("stats_path", None)
    if isinstance(ckpt_stats, str) and ckpt_stats.strip():
        candidate_paths.append(Path(ckpt_stats))
    candidate_paths.append(Path(STANDARDIZATION_STATS_PATH))

    # 兼容 checkpoint 保存了绝对路径，但迁移机器后数据根目录发生变化的情况。
    for p in list(candidate_paths):
        if p.name:
            candidate_paths.append(Path(DATA_DIR) / p.name)

    seen: set[str] = set()
    unique_paths: List[Path] = []
    for path in candidate_paths:
        key = str(path.expanduser())
        if key not in seen:
            seen.add(key)
            unique_paths.append(path.expanduser())

    for path in unique_paths:
        if path.exists():
            LOGGER.info("使用标准化统计文件: %s", path)
            return StandardizationBundle.from_npz(str(path), device=device)
    if REQUIRE_STANDARDIZATION_STATS:
        raise FileNotFoundError(f"未找到标准化文件，已检查: {[str(p) for p in unique_paths]}")
    return None


def maybe_normalize_h_dc(
    H_dc: torch.Tensor,
    state_valid_mask: torch.Tensor,
    standardizers: Optional[StandardizationBundle],
) -> torch.Tensor:
    if standardizers is None:
        return H_dc
    return standardizers.h_dc.normalize(H_dc, state_valid_mask=state_valid_mask)


def maybe_denormalize_delta(
    delta_norm: torch.Tensor,
    state_valid_mask: torch.Tensor,
    standardizers: Optional[StandardizationBundle],
) -> torch.Tensor:
    if standardizers is None:
        return delta_norm
    return standardizers.delta.denormalize(delta_norm, state_valid_mask=state_valid_mask)


def get_underlying_dataset(dataset: Dataset) -> Dataset:
    current = dataset
    while isinstance(current, Subset):
        current = current.dataset
    return current


def dataset_position_to_sample_idx(dataset: Dataset, position: int) -> int:
    if isinstance(dataset, Subset):
        return dataset_position_to_sample_idx(dataset.dataset, int(dataset.indices[position]))
    indices = getattr(dataset, "indices", None)
    if indices is not None:
        return int(indices[position])
    return int(position)


def sample_indices_from_loader(loader: Optional[DataLoader]) -> List[int]:
    if loader is None:
        return []
    dataset = loader.dataset
    return [dataset_position_to_sample_idx(dataset, i) for i in range(len(dataset))]


def choose_sample_ids(available: Sequence[int], explicit_ids: Sequence[int]) -> List[int]:
    available_sorted = sorted(int(x) for x in available)
    available_set = set(available_sorted)
    if explicit_ids:
        selected = [int(x) for x in explicit_ids if int(x) in available_set]
        missing = [int(x) for x in explicit_ids if int(x) not in available_set]
        if missing:
            LOGGER.warning("指定样本不在当前 split 中，已忽略: %s", missing)
        return selected

    n = min(int(AUTO_SELECT_NUM_SAMPLES_PER_SPLIT), len(available_sorted))
    if n <= 0:
        return []
    if AUTO_SELECT_SAMPLE_STRATEGY == "first":
        return available_sorted[:n]
    if AUTO_SELECT_SAMPLE_STRATEGY == "random":
        rng = np.random.default_rng(RANDOM_SEED_FOR_SAMPLE_SELECTION)
        chosen = rng.choice(np.asarray(available_sorted), size=n, replace=False)
        return sorted(int(x) for x in chosen.tolist())
    if AUTO_SELECT_SAMPLE_STRATEGY == "evenly_spaced":
        if n == 1:
            return [available_sorted[0]]
        positions = np.linspace(0, len(available_sorted) - 1, n)
        return [available_sorted[int(round(pos))] for pos in positions.tolist()]
    raise ValueError(f"未知 AUTO_SELECT_SAMPLE_STRATEGY: {AUTO_SELECT_SAMPLE_STRATEGY}")


def build_h_feature_eval_mask(
    state_valid_mask: torch.Tensor,
    bus_type: torch.Tensor,
    feat_dim: int,
    evaluated_feature_indices: Sequence[int],
) -> torch.Tensor:
    if EVALUATE_H_FEATURES_BUS_TYPE_TARGET_ONLY:
        mask = create_bus_type_target_mask(
            state_valid_mask=state_valid_mask,
            bus_type=bus_type,
            feat_dim=feat_dim,
        )
    else:
        mask = state_valid_mask.unsqueeze(-1).expand(-1, -1, feat_dim).clone()
    feature_keep = torch.zeros((feat_dim,), dtype=torch.bool, device=state_valid_mask.device)
    feature_keep[list(evaluated_feature_indices)] = True
    mask &= feature_keep.view(1, 1, feat_dim)
    return mask


def complex_voltage_from_h(H: torch.Tensor) -> torch.Tensor:
    vm = H[:, :, IDX_VM].clamp_min(1e-8)
    va_rad = torch.deg2rad(H[:, :, IDX_VA])
    return vm.to(torch.complex64) * torch.exp(1j * va_rad.to(torch.complex64))


def compute_node_spec_injection(H: torch.Tensor) -> Dict[str, torch.Tensor]:
    return {
        "P_inj_spec": H[:, :, IDX_PG] - H[:, :, IDX_PD],
        "Q_inj_spec": H[:, :, IDX_QG] - H[:, :, IDX_QD],
    }


def compute_node_calc_injection_from_y(H: torch.Tensor, Y: torch.Tensor, base_mva: float) -> Dict[str, torch.Tensor]:
    V = complex_voltage_from_h(H)
    current = torch.einsum("bij,bj->bi", Y.to(torch.complex64), V)
    S_pu = V * torch.conj(current)
    S = S_pu * float(base_mva)
    return {
        "P_inj_calc": S.real.float(),
        "Q_inj_calc": S.imag.float(),
    }


def prepare_branch_tensors(branch_catalog: Sequence[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    if not branch_catalog:
        return {
            "u_pos": torch.empty((0,), dtype=torch.long, device=device),
            "v_pos": torch.empty((0,), dtype=torch.long, device=device),
            "records": [],
        }
    u_pos = torch.as_tensor([int(rec["u_pos"]) for rec in branch_catalog], dtype=torch.long, device=device)
    v_pos = torch.as_tensor([int(rec["v_pos"]) for rec in branch_catalog], dtype=torch.long, device=device)
    return {"u_pos": u_pos, "v_pos": v_pos, "records": list(branch_catalog)}


def compute_branch_flows_from_y(
    H: torch.Tensor,
    Y: torch.Tensor,
    state_valid_mask: torch.Tensor,
    branch_data: Dict[str, Any],
    base_mva: float,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    u_pos = branch_data["u_pos"]
    v_pos = branch_data["v_pos"]
    num_branches = int(u_pos.numel())
    batch_size = int(H.shape[0])
    if num_branches == 0:
        empty = torch.empty((batch_size, 0), dtype=H.dtype, device=H.device)
        return {
            "P_ij": empty,
            "Q_ij": empty,
            "P_ji": empty,
            "Q_ji": empty,
            "P_loss_ij_ji": empty,
            "Q_loss_ij_ji": empty,
        }, torch.zeros((batch_size, 0), dtype=torch.bool, device=H.device)

    V = complex_voltage_from_h(H)
    Yc = Y.to(torch.complex64)
    y_ij = -Yc[:, u_pos, v_pos]
    V_i = V[:, u_pos]
    V_j = V[:, v_pos]
    I_ij = (V_i - V_j) * y_ij
    I_ji = (V_j - V_i) * y_ij
    S_ij = V_i * torch.conj(I_ij) * float(base_mva)
    S_ji = V_j * torch.conj(I_ji) * float(base_mva)

    valid = (
        state_valid_mask[:, u_pos]
        & state_valid_mask[:, v_pos]
        & torch.isfinite(y_ij.real)
        & torch.isfinite(y_ij.imag)
        & (torch.abs(y_ij) > float(BRANCH_Y_ABS_EPS))
    )
    flows = {
        "P_ij": S_ij.real.float(),
        "Q_ij": S_ij.imag.float(),
        "P_ji": S_ji.real.float(),
        "Q_ji": S_ji.imag.float(),
        "P_loss_ij_ji": (S_ij.real + S_ji.real).float(),
        "Q_loss_ij_ji": (S_ij.imag + S_ji.imag).float(),
    }
    return flows, valid


def make_quantity_dict(H: torch.Tensor, Y: torch.Tensor, state_valid_mask: torch.Tensor, branch_data: Dict[str, Any], base_mva: float) -> Dict[str, Dict[str, Any]]:
    quantities: Dict[str, Dict[str, Any]] = {}

    if EVALUATE_NODE_H_FEATURES:
        quantities["node_h_feature"] = {
            "values": {name: H[:, :, idx] for idx, name in enumerate(FEATURE_NAMES) if idx in set(EVALUATED_H_FEATURE_INDICES)},
            "mask": build_h_feature_eval_mask(
                state_valid_mask=state_valid_mask,
                bus_type=make_quantity_dict.bus_type_device,
                feat_dim=H.shape[-1],
                evaluated_feature_indices=EVALUATED_H_FEATURE_INDICES,
            ),
            "per_quantity_mask": True,
        }

    if EVALUATE_NODE_SPEC_INJECTION:
        quantities["node_spec_injection"] = {
            "values": compute_node_spec_injection(H),
            "mask": state_valid_mask,
            "per_quantity_mask": False,
        }

    if EVALUATE_NODE_CALC_INJECTION:
        quantities["node_calc_injection"] = {
            "values": compute_node_calc_injection_from_y(H, Y, base_mva=base_mva),
            "mask": state_valid_mask,
            "per_quantity_mask": False,
        }

    if EVALUATE_BRANCH_FLOW:
        branch_flows, branch_mask = compute_branch_flows_from_y(
            H=H,
            Y=Y,
            state_valid_mask=state_valid_mask,
            branch_data=branch_data,
            base_mva=base_mva,
        )
        quantities["branch_flow"] = {
            "values": branch_flows,
            "mask": branch_mask,
            "per_quantity_mask": False,
        }

    return quantities


# 函数属性用于避免在每个调用中重复传 bus_type；main 中会设置。
make_quantity_dict.bus_type_device = torch.empty((0,), dtype=torch.long)  # type: ignore[attr-defined]


def get_mask_for_quantity(category_payload: Dict[str, Any], quantity: str) -> torch.Tensor:
    mask = category_payload["mask"]
    if bool(category_payload.get("per_quantity_mask", False)):
        quantity_to_idx = {name: idx for idx, name in enumerate(FEATURE_NAMES)}
        return mask[:, :, quantity_to_idx[quantity]]
    return mask


def init_metric_accumulators() -> Dict[Tuple[str, str, str], MetricAccumulator]:
    acc: Dict[Tuple[str, str, str], MetricAccumulator] = {}
    node_h_names = [name for idx, name in enumerate(FEATURE_NAMES) if idx in set(EVALUATED_H_FEATURE_INDICES)]
    categories: List[Tuple[str, Iterable[str]]] = []
    if EVALUATE_NODE_H_FEATURES:
        categories.append(("node_h_feature", node_h_names))
    if EVALUATE_NODE_SPEC_INJECTION:
        categories.append(("node_spec_injection", ["P_inj_spec", "Q_inj_spec"]))
    if EVALUATE_NODE_CALC_INJECTION:
        categories.append(("node_calc_injection", ["P_inj_calc", "Q_inj_calc"]))
    if EVALUATE_BRANCH_FLOW:
        categories.append(("branch_flow", ["P_ij", "Q_ij", "P_ji", "Q_ji", "P_loss_ij_ji", "Q_loss_ij_ji"]))

    for comparison, _, _ in COMPARISONS:
        for category, names in categories:
            for quantity in names:
                acc[(comparison, category, quantity)] = MetricAccumulator()
    return acc


def update_metric_accumulators(
    acc: Dict[Tuple[str, str, str], MetricAccumulator],
    quantity_by_case: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    for comparison, candidate_case, reference_case in COMPARISONS:
        candidate_categories = quantity_by_case[candidate_case]
        reference_categories = quantity_by_case[reference_case]
        for category, cand_payload in candidate_categories.items():
            ref_payload = reference_categories[category]
            for quantity, cand_values in cand_payload["values"].items():
                if quantity not in ref_payload["values"]:
                    continue
                ref_values = ref_payload["values"][quantity]
                cand_mask = get_mask_for_quantity(cand_payload, quantity)
                ref_mask = get_mask_for_quantity(ref_payload, quantity)
                mask = cand_mask & ref_mask
                acc[(comparison, category, quantity)].update(cand_values, ref_values, mask)


def metric_rows_from_accumulators(split_name: str, acc: Dict[Tuple[str, str, str], MetricAccumulator]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in sorted(acc.keys()):
        comparison, category, quantity = key
        rows.append(acc[key].as_row(
            split=split_name,
            comparison=comparison,
            category=category,
            quantity=quantity,
            unit=PHYSICAL_UNITS.get(quantity, ""),
        ))
    return rows


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else math.nan
    except Exception:
        return math.nan


def build_selected_node_rows(
    split: str,
    sample_idx: int,
    H_ac: torch.Tensor,
    H_dc: torch.Tensor,
    H_model: torch.Tensor,
    Y: torch.Tensor,
    state_valid_mask: torch.Tensor,
    node_valid_mask: torch.Tensor,
    bus_ids: Sequence[int],
    bus_type_np: np.ndarray,
    base_mva: float,
) -> List[Dict[str, Any]]:
    H_case = {
        "ac": H_ac.unsqueeze(0),
        "dc": H_dc.unsqueeze(0),
        "model": H_model.unsqueeze(0),
    }
    Y1 = Y.unsqueeze(0)
    state1 = state_valid_mask.unsqueeze(0)

    values: Dict[str, Dict[str, torch.Tensor]] = {}
    for case_name, H_case_tensor in H_case.items():
        case_values: Dict[str, torch.Tensor] = {}
        for idx, name in enumerate(FEATURE_NAMES):
            case_values[name] = H_case_tensor[0, :, idx]
        for k, v in compute_node_spec_injection(H_case_tensor).items():
            case_values[k] = v[0]
        for k, v in compute_node_calc_injection_from_y(H_case_tensor, Y1, base_mva=base_mva).items():
            case_values[k] = v[0]
        values[case_name] = case_values

    quantities = []
    if EVALUATE_NODE_H_FEATURES:
        quantities.extend([name for idx, name in enumerate(FEATURE_NAMES) if idx in set(EVALUATED_H_FEATURE_INDICES)])
    if EVALUATE_NODE_SPEC_INJECTION:
        quantities.extend(["P_inj_spec", "Q_inj_spec"])
    if EVALUATE_NODE_CALC_INJECTION:
        quantities.extend(["P_inj_calc", "Q_inj_calc"])

    rows: List[Dict[str, Any]] = []
    state_np = state_valid_mask.detach().cpu().numpy().astype(bool)
    node_np = node_valid_mask.detach().cpu().numpy().astype(bool)
    for bus_pos in range(int(H_ac.shape[0])):
        bus_id = int(bus_ids[bus_pos]) if bus_pos < len(bus_ids) else int(bus_pos)
        bus_type_id = int(bus_type_np[bus_pos]) if bus_pos < len(bus_type_np) else -1
        for quantity in quantities:
            ac_value = _safe_float(values["ac"][quantity][bus_pos].detach().cpu().item())
            dc_value = _safe_float(values["dc"][quantity][bus_pos].detach().cpu().item())
            model_value = _safe_float(values["model"][quantity][bus_pos].detach().cpu().item())
            rows.append({
                "split": split,
                "sample_idx": int(sample_idx),
                "bus_pos": int(bus_pos),
                "bus_id": bus_id,
                "bus_type_id": bus_type_id,
                "bus_type": BUS_TYPE_NAMES.get(bus_type_id, "UNKNOWN"),
                "node_valid": bool(node_np[bus_pos]),
                "state_valid": bool(state_np[bus_pos]),
                "quantity": quantity,
                "unit": PHYSICAL_UNITS.get(quantity, ""),
                "ac": ac_value,
                "dc": dc_value,
                "model": model_value,
                "model_minus_ac": model_value - ac_value,
                "dc_minus_ac": dc_value - ac_value,
                "model_minus_dc": model_value - dc_value,
                "model_ape_vs_ac_percent": abs(model_value - ac_value) / abs(ac_value) * 100.0 if abs(ac_value) > PERCENT_DENOM_EPS else math.nan,
                "dc_ape_vs_ac_percent": abs(dc_value - ac_value) / abs(ac_value) * 100.0 if abs(ac_value) > PERCENT_DENOM_EPS else math.nan,
                "model_ape_vs_dc_percent": abs(model_value - dc_value) / abs(dc_value) * 100.0 if abs(dc_value) > PERCENT_DENOM_EPS else math.nan,
            })
    return rows


def build_selected_branch_rows(
    split: str,
    sample_idx: int,
    H_ac: torch.Tensor,
    H_dc: torch.Tensor,
    H_model: torch.Tensor,
    Y: torch.Tensor,
    state_valid_mask: torch.Tensor,
    branch_data: Dict[str, Any],
    base_mva: float,
) -> List[Dict[str, Any]]:
    if not EVALUATE_BRANCH_FLOW:
        return []
    branch_records = branch_data["records"]
    H_case = {
        "ac": H_ac.unsqueeze(0),
        "dc": H_dc.unsqueeze(0),
        "model": H_model.unsqueeze(0),
    }
    Y1 = Y.unsqueeze(0)
    state1 = state_valid_mask.unsqueeze(0)
    values: Dict[str, Dict[str, torch.Tensor]] = {}
    masks: Dict[str, torch.Tensor] = {}
    for case_name, H_case_tensor in H_case.items():
        flows, mask = compute_branch_flows_from_y(
            H=H_case_tensor,
            Y=Y1,
            state_valid_mask=state1,
            branch_data=branch_data,
            base_mva=base_mva,
        )
        values[case_name] = {k: v[0] for k, v in flows.items()}
        masks[case_name] = mask[0]

    quantities = ["P_ij", "Q_ij", "P_ji", "Q_ji", "P_loss_ij_ji", "Q_loss_ij_ji"]
    rows: List[Dict[str, Any]] = []
    for branch_idx, rec in enumerate(branch_records):
        active = bool((masks["ac"][branch_idx] & masks["dc"][branch_idx] & masks["model"][branch_idx]).detach().cpu().item())
        for quantity in quantities:
            ac_value = _safe_float(values["ac"][quantity][branch_idx].detach().cpu().item())
            dc_value = _safe_float(values["dc"][quantity][branch_idx].detach().cpu().item())
            model_value = _safe_float(values["model"][quantity][branch_idx].detach().cpu().item())
            rows.append({
                "split": split,
                "sample_idx": int(sample_idx),
                "branch_idx": int(branch_idx),
                "branch_type": str(rec.get("type_name", "")),
                "from_bus": int(rec.get("u_bus", -1)),
                "to_bus": int(rec.get("v_bus", -1)),
                "from_pos": int(rec.get("u_pos", -1)),
                "to_pos": int(rec.get("v_pos", -1)),
                "active_by_ybus": active,
                "quantity": quantity,
                "unit": PHYSICAL_UNITS.get(quantity, ""),
                "ac": ac_value,
                "dc": dc_value,
                "model": model_value,
                "model_minus_ac": model_value - ac_value,
                "dc_minus_ac": dc_value - ac_value,
                "model_minus_dc": model_value - dc_value,
                "model_ape_vs_ac_percent": abs(model_value - ac_value) / abs(ac_value) * 100.0 if abs(ac_value) > PERCENT_DENOM_EPS else math.nan,
                "dc_ape_vs_ac_percent": abs(dc_value - ac_value) / abs(ac_value) * 100.0 if abs(ac_value) > PERCENT_DENOM_EPS else math.nan,
                "model_ape_vs_dc_percent": abs(model_value - dc_value) / abs(dc_value) * 100.0 if abs(dc_value) > PERCENT_DENOM_EPS else math.nan,
            })
    return rows


def print_selected_tables(split: str, sample_idx: int, node_rows: List[Dict[str, Any]], branch_rows: List[Dict[str, Any]]) -> None:
    if not PRINT_SELECTED_DETAIL_TO_CONSOLE:
        return
    if node_rows:
        node_df = pd.DataFrame(node_rows)
        LOGGER.info("\n[%s sample=%s] 节点物理量对比:\n%s", split, sample_idx, node_df.to_string(index=False))
    if branch_rows:
        branch_df = pd.DataFrame(branch_rows)
        LOGGER.info("\n[%s sample=%s] 支路物理量对比:\n%s", split, sample_idx, branch_df.to_string(index=False))


def summarize_selected_rows(node_rows: List[Dict[str, Any]], branch_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for category_name, rows, valid_col in [
        ("node", node_rows, "state_valid"),
        ("branch", branch_rows, "active_by_ybus"),
    ]:
        if not rows:
            continue
        df = pd.DataFrame(rows)
        for (split, sample_idx, quantity), group in df.groupby(["split", "sample_idx", "quantity"]):
            if valid_col in group.columns:
                group = group[group[valid_col] == True]  # noqa: E712
            if len(group) == 0:
                continue
            out = {
                "split": split,
                "sample_idx": int(sample_idx),
                "category": category_name,
                "quantity": quantity,
                "unit": str(group["unit"].iloc[0]) if "unit" in group.columns else "",
                "count": int(len(group)),
            }
            for prefix in ["model_minus_ac", "dc_minus_ac", "model_minus_dc"]:
                err = group[prefix].to_numpy(dtype=float)
                err = err[np.isfinite(err)]
                if err.size == 0:
                    continue
                out[f"{prefix}_mae"] = float(np.mean(np.abs(err)))
                out[f"{prefix}_rmse"] = float(np.sqrt(np.mean(err ** 2)))
                out[f"{prefix}_max_abs"] = float(np.max(np.abs(err)))
            summary.append(out)
    return summary


@torch.no_grad()
def evaluate_split(
    split_name: str,
    loader: DataLoader,
    model: HybridGTForPowerFlow,
    standardizers: Optional[StandardizationBundle],
    bus_type: torch.Tensor,
    bus_ids: Sequence[int],
    branch_data: Dict[str, Any],
    selected_sample_ids: Sequence[int],
    output_dir: Path,
    device: torch.device,
    base_mva: float,
) -> Dict[str, Any]:
    acc = init_metric_accumulators()
    selected_set = set(int(x) for x in selected_sample_ids)
    selected_node_rows_all: List[Dict[str, Any]] = []
    selected_branch_rows_all: List[Dict[str, Any]] = []
    bus_type_np = bus_type.detach().cpu().numpy()
    bus_type_device = bus_type.to(device)

    iterator = tqdm(loader, desc=f"Eval {split_name}", leave=False)
    for batch in iterator:
        H_ac = batch["H"].to(device, non_blocking=True)
        H_dc = batch["H_dc"].to(device, non_blocking=True)
        Y = batch["Y"].to(device, non_blocking=True)
        node_valid_mask = batch["node_valid_mask"].to(device, non_blocking=True)
        state_valid_mask = batch["state_valid_mask"].to(device, non_blocking=True)
        sample_ids = [int(x) for x in batch["sample_idx"]]

        H_dc_norm = maybe_normalize_h_dc(H_dc, state_valid_mask, standardizers)
        with autocast(enabled=(USE_AMP_FOR_INFERENCE and device.type == "cuda")):
            delta_pred_norm = model(H_dc_norm, Y, node_valid_mask=node_valid_mask, feature_visible_mask=None)
        delta_pred = maybe_denormalize_delta(delta_pred_norm, state_valid_mask, standardizers)
        H_model = reconstruct_h_from_delta(
            H_dc_phys=H_dc,
            delta_phys=delta_pred,
            bus_type=bus_type_device,
            state_valid_mask=state_valid_mask,
        )

        make_quantity_dict.bus_type_device = bus_type_device  # type: ignore[attr-defined]
        quantity_by_case = {
            "ac": make_quantity_dict(H_ac, Y, state_valid_mask, branch_data, base_mva),
            "dc": make_quantity_dict(H_dc, Y, state_valid_mask, branch_data, base_mva),
            "model": make_quantity_dict(H_model, Y, state_valid_mask, branch_data, base_mva),
        }
        update_metric_accumulators(acc, quantity_by_case)

        if SAVE_SELECTED_NODE_COMPARISON or SAVE_SELECTED_BRANCH_COMPARISON:
            for local_i, sample_idx in enumerate(sample_ids):
                if sample_idx not in selected_set:
                    continue
                node_rows = build_selected_node_rows(
                    split=split_name,
                    sample_idx=sample_idx,
                    H_ac=H_ac[local_i],
                    H_dc=H_dc[local_i],
                    H_model=H_model[local_i],
                    Y=Y[local_i],
                    state_valid_mask=state_valid_mask[local_i],
                    node_valid_mask=node_valid_mask[local_i],
                    bus_ids=bus_ids,
                    bus_type_np=bus_type_np,
                    base_mva=base_mva,
                )
                branch_rows = build_selected_branch_rows(
                    split=split_name,
                    sample_idx=sample_idx,
                    H_ac=H_ac[local_i],
                    H_dc=H_dc[local_i],
                    H_model=H_model[local_i],
                    Y=Y[local_i],
                    state_valid_mask=state_valid_mask[local_i],
                    branch_data=branch_data,
                    base_mva=base_mva,
                )
                selected_node_rows_all.extend(node_rows)
                selected_branch_rows_all.extend(branch_rows)

                if SAVE_SELECTED_NODE_COMPARISON and node_rows:
                    pd.DataFrame(node_rows).to_csv(
                        output_dir / f"selected_node_comparison_{split_name}_sample_{sample_idx}.csv",
                        index=False,
                    )
                if SAVE_SELECTED_BRANCH_COMPARISON and branch_rows:
                    pd.DataFrame(branch_rows).to_csv(
                        output_dir / f"selected_branch_comparison_{split_name}_sample_{sample_idx}.csv",
                        index=False,
                    )
                print_selected_tables(split_name, sample_idx, node_rows, branch_rows)

    return {
        "metrics_rows": metric_rows_from_accumulators(split_name, acc),
        "selected_node_rows": selected_node_rows_all,
        "selected_branch_rows": selected_branch_rows_all,
    }


def log_dataset_info(train_loader: DataLoader, test_loader: Optional[DataLoader]) -> None:
    LOGGER.info("train dataset length: %d", len(train_loader.dataset))
    if test_loader is not None:
        LOGGER.info("test dataset length: %d", len(test_loader.dataset))
    base_train = get_underlying_dataset(train_loader.dataset)
    LOGGER.info("underlying train dataset: %s", type(base_train).__name__)


def main() -> None:
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    set_seed(SEED)
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)

    device = torch.device(DEVICE)
    checkpoint_path = resolve_checkpoint_path()
    LOGGER.info("使用 checkpoint: %s", checkpoint_path)
    checkpoint_payload = load_checkpoint_payload(checkpoint_path, device)

    network_metadata = load_network_metadata(DATA_DIR)
    base_mva = float(checkpoint_payload.get("base_mva", get_network_base_mva(network_metadata)))
    if base_mva <= 0:
        raise ValueError(f"base_mva 必须为正数，实际为 {base_mva}")
    LOGGER.info("base_mva=%.6f", base_mva)

    bus_type = build_bus_type_vector(network_metadata)
    bus_ids = get_sorted_bus_ids(network_metadata)
    branch_catalog = build_branch_catalog(network_metadata)
    branch_data = prepare_branch_tensors(branch_catalog, device=device)
    LOGGER.info("num_buses=%d, num_candidate_branches=%d", len(bus_ids), len(branch_catalog))

    model = build_model_from_checkpoint(checkpoint_payload, network_metadata, bus_type, device)
    standardizers = build_standardizers(device, checkpoint_payload)

    train_loader, _, test_loader = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        pad_to_max=PAD_TO_MAX,
        device=None,
        num_workers=NUM_WORKERS,
        seed=SEED,
        pin_memory=PIN_MEMORY,
        y_as_dense=Y_AS_DENSE,
        shuffle_train=SHUFFLE_TRAIN_FOR_EVAL,
        cache_metadata=CACHE_METADATA,
        cache_arrays_in_memory=CACHE_ARRAYS_IN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )
    log_dataset_info(train_loader, test_loader)

    train_available = sample_indices_from_loader(train_loader)
    test_available = sample_indices_from_loader(test_loader) if test_loader is not None else []
    selected_train_ids = choose_sample_ids(train_available, SELECTED_TRAIN_SAMPLE_IDS)
    selected_test_ids = choose_sample_ids(test_available, SELECTED_TEST_SAMPLE_IDS)
    LOGGER.info("selected train samples: %s", selected_train_ids)
    LOGGER.info("selected test samples: %s", selected_test_ids)

    all_metric_rows: List[Dict[str, Any]] = []
    all_selected_node_rows: List[Dict[str, Any]] = []
    all_selected_branch_rows: List[Dict[str, Any]] = []

    train_result = evaluate_split(
        split_name="train",
        loader=train_loader,
        model=model,
        standardizers=standardizers,
        bus_type=bus_type,
        bus_ids=bus_ids,
        branch_data=branch_data,
        selected_sample_ids=selected_train_ids,
        output_dir=output_dir,
        device=device,
        base_mva=base_mva,
    )
    all_metric_rows.extend(train_result["metrics_rows"])
    all_selected_node_rows.extend(train_result["selected_node_rows"])
    all_selected_branch_rows.extend(train_result["selected_branch_rows"])

    if test_loader is not None:
        test_result = evaluate_split(
            split_name="test",
            loader=test_loader,
            model=model,
            standardizers=standardizers,
            bus_type=bus_type,
            bus_ids=bus_ids,
            branch_data=branch_data,
            selected_sample_ids=selected_test_ids,
            output_dir=output_dir,
            device=device,
            base_mva=base_mva,
        )
        all_metric_rows.extend(test_result["metrics_rows"])
        all_selected_node_rows.extend(test_result["selected_node_rows"])
        all_selected_branch_rows.extend(test_result["selected_branch_rows"])

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_csv = output_dir / "metrics_summary.csv"
    metrics_json = output_dir / "metrics_summary.json"
    metrics_df.to_csv(metrics_csv, index=False)
    write_json(metrics_json, all_metric_rows)

    if SAVE_SELECTED_NODE_COMPARISON and all_selected_node_rows:
        pd.DataFrame(all_selected_node_rows).to_csv(output_dir / "selected_node_comparison_all.csv", index=False)
    if SAVE_SELECTED_BRANCH_COMPARISON and all_selected_branch_rows:
        pd.DataFrame(all_selected_branch_rows).to_csv(output_dir / "selected_branch_comparison_all.csv", index=False)
    if SAVE_SELECTED_SAMPLE_SUMMARY and (all_selected_node_rows or all_selected_branch_rows):
        summary_rows = summarize_selected_rows(all_selected_node_rows, all_selected_branch_rows)
        pd.DataFrame(summary_rows).to_csv(output_dir / "selected_sample_summary.csv", index=False)

    run_info = {
        "checkpoint_path": str(checkpoint_path),
        "data_dir": DATA_DIR,
        "output_dir": str(output_dir),
        "base_mva": base_mva,
        "selected_train_ids": selected_train_ids,
        "selected_test_ids": selected_test_ids,
        "comparisons": COMPARISONS,
        "evaluate_node_h_features": EVALUATE_NODE_H_FEATURES,
        "evaluated_h_feature_indices": EVALUATED_H_FEATURE_INDICES,
        "evaluate_h_features_bus_type_target_only": EVALUATE_H_FEATURES_BUS_TYPE_TARGET_ONLY,
        "evaluate_node_spec_injection": EVALUATE_NODE_SPEC_INJECTION,
        "evaluate_node_calc_injection": EVALUATE_NODE_CALC_INJECTION,
        "evaluate_branch_flow": EVALUATE_BRANCH_FLOW,
        "branch_flow_formula": "y_ij=-Ybus[i,j]; I_ij=(V_i-V_j)*y_ij; S_ij=V_i*conj(I_ij)*base_mva",
        "branch_y_abs_eps": BRANCH_Y_ABS_EPS,
        "percent_denom_eps": PERCENT_DENOM_EPS,
        "formulation": "H_pred = H_dc + denormalized_delta_pred on bus-type target columns",
    }
    write_json(output_dir / "eval_run_info.json", run_info)

    LOGGER.info("评估完成: %s", output_dir)
    LOGGER.info("metrics: %s", metrics_csv)
    LOGGER.info("selected node rows: %s", output_dir / "selected_node_comparison_all.csv")
    LOGGER.info("selected branch rows: %s", output_dir / "selected_branch_comparison_all.csv")


if __name__ == "__main__":
    main()
