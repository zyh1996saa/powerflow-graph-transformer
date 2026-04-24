# In[]
from __future__ import annotations

"""
Evaluate a trained Hybrid Graph Transformer power-flow model in original physical units.

This script intentionally uses a manual configuration block instead of argparse.
Place this file in the same project directory as:
    pf_data_loader.py
    pf_powerflow_model.py
    pf_topology_utils.py

Important evaluation convention:
    The model head outputs six features, but finetune supervision is bus-type aware.
    Only the following columns are treated as model-predicted physical targets:
        PQ    -> Vm, Va
        PV    -> Qg, Va
        SLACK -> Pg, Qg
    For all non-target columns at a bus, the reported effective prediction is copied
    from the given/true H value. Raw model outputs can optionally be saved in
    diagnostic columns, but they are not used as physical predictions or metrics.

Main outputs:
    1) metrics_summary.csv/json
       Split-level MAE / RMSE / max absolute error for each evaluated physical quantity.
       Metrics are computed only on bus-type target columns by default.
    2) selected_node_comparison_<split>_sample_<id>.csv
       Bus-wise true/effective-prediction/error comparison for selected samples.
    3) selected_node_comparison_all.csv
       Combined bus-wise comparison table for all selected samples.
    4) selected_sample_summary.csv
       Per-selected-sample summary statistics.
"""

import json
import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

# Ensure local project modules are importable when this script is launched by path.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pf_data_loader import create_dataloaders  # noqa: E402
from pf_powerflow_model import HybridGTForPowerFlow  # noqa: E402
from pf_topology_utils import (  # noqa: E402
    BUS_TYPE_NAMES,
    BUS_TYPE_PQ,
    BUS_TYPE_PV,
    BUS_TYPE_SLACK,
    FEATURE_NAMES,
    IDX_PG,
    IDX_QG,
    IDX_VA,
    IDX_VM,
    NodeFeatureStandardizer,
    build_bus_type_vector,
    create_bus_type_target_mask,
    create_input_feature_mask_for_finetune,
    get_network_base_mva,
    get_sorted_bus_ids,
    load_network_metadata,
)


# ============================================================
# Manual configuration block. Do not use argparse.
# ============================================================
DATA_DIR = "/data2/zyh/case39_samples"
OUTPUT_DIR = "./pf_eval_outputs_physical"

# Set this explicitly when possible, for example:
CHECKPOINT_PATH = "./logs/20260423_170301_pf_modular_gpt0421/ckpt_finetune_best.pt"
CHECKPOINT_PATH = ""
AUTO_DISCOVER_CHECKPOINT = True
CHECKPOINT_SEARCH_ROOT = "./logs"
CHECKPOINT_CANDIDATE_FILENAMES = [
    "ckpt_finetune_best.pt",
    "ckpt_finetune_last.pt",
    "ckpt_pretrain_best.pt",
    "ckpt_pretrain_last.pt",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
USE_AMP_FOR_INFERENCE = True

BATCH_SIZE = 256
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

# These must match the training script if you want the train split to be exactly
# the subset used by training.
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
PAD_TO_MAX = False
Y_AS_DENSE = True
CACHE_METADATA = True
CACHE_ARRAYS_IN_MEMORY = False
SHUFFLE_TRAIN_FOR_EVAL = False

# Normalization. If the model was trained with normalized H, keep this enabled.
ENABLE_NODE_FEATURE_STANDARDIZATION = True
STANDARDIZATION_STATS_PATH = str(Path(DATA_DIR) / "train_h_stats_global_6_gpt0421_modular.npz")
REQUIRE_STANDARDIZATION_STATS = True

# Finetune evaluation should mirror training: target features are hidden from input.
ZERO_TARGET_FIELDS_IN_INPUT = True

# For power-flow supervised finetuning, only these four quantities are true solved
# targets in the current formulation. Pd/Qd are load inputs and are not evaluated
# by default.
EVALUATED_FEATURE_INDICES = [IDX_PG, IDX_QG, IDX_VM, IDX_VA]
EVALUATE_BUS_TYPE_TARGET_ONLY = True

# Sample-level detailed comparison settings.
SELECTED_TRAIN_SAMPLE_IDS: List[int] = []
SELECTED_TEST_SAMPLE_IDS: List[int] = []
AUTO_SELECT_NUM_SAMPLES_PER_SPLIT = 5
AUTO_SELECT_SAMPLE_STRATEGY = "evenly_spaced"  # first / random / evenly_spaced
RANDOM_SEED_FOR_SAMPLE_SELECTION = 42

SAVE_SELECTED_NODE_COMPARISON = True
SAVE_SELECTED_SAMPLE_SUMMARY = True

# Keep diagnostic raw model outputs in detailed sample CSVs.
# The effective *_pred columns still follow the bus-type-aware convention above.
INCLUDE_RAW_MODEL_OUTPUT_IN_DETAIL = True

# Fallback model configuration. Checkpoint model_config overrides these values.
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


# ============================================================
# Constants and small helpers
# ============================================================
PHYSICAL_UNITS = {
    "Pd": "MW",
    "Qd": "Mvar",
    "Pg": "MW",
    "Qg": "Mvar",
    "Vm": "p.u.",
    "Va": "degree",
}

TARGET_FEATURES_BY_BUS_TYPE = {
    BUS_TYPE_PQ: [IDX_VM, IDX_VA],
    BUS_TYPE_PV: [IDX_QG, IDX_VA],
    BUS_TYPE_SLACK: [IDX_PG, IDX_QG],
}

LOGGER = logging.getLogger("evaluate_pf_model_physical")


@dataclass
class FeatureAccumulator:
    count: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    signed_sum: float = 0.0
    max_abs: float = 0.0

    def update(self, err: torch.Tensor) -> None:
        if err.numel() == 0:
            return
        err64 = err.detach().to(dtype=torch.float64)
        abs_err = err64.abs()
        self.count += int(err64.numel())
        self.abs_sum += float(abs_err.sum().item())
        self.sq_sum += float((err64 ** 2).sum().item())
        self.signed_sum += float(err64.sum().item())
        self.max_abs = max(self.max_abs, float(abs_err.max().item()))

    def as_row(self, split: str, feature_idx: int) -> Dict[str, Any]:
        feature_name = FEATURE_NAMES[feature_idx]
        if self.count <= 0:
            return {
                "split": split,
                "feature": feature_name,
                "unit": PHYSICAL_UNITS.get(feature_name, ""),
                "count": 0,
                "mae": math.nan,
                "rmse": math.nan,
                "mean_signed_error": math.nan,
                "max_abs_error": math.nan,
            }
        return {
            "split": split,
            "feature": feature_name,
            "unit": PHYSICAL_UNITS.get(feature_name, ""),
            "count": self.count,
            "mae": self.abs_sum / self.count,
            "rmse": math.sqrt(max(self.sq_sum / self.count, 0.0)),
            "mean_signed_error": self.signed_sum / self.count,
            "max_abs_error": self.max_abs,
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
        raise FileNotFoundError(
            f"未在 {root} 下找到候选 checkpoint 文件: {CHECKPOINT_CANDIDATE_FILENAMES}"
        )

    # Prefer newer run directories and better checkpoint names.
    priority = {name: i for i, name in enumerate(CHECKPOINT_CANDIDATE_FILENAMES)}
    candidates.sort(key=lambda p: (priority.get(p.name, 999), -p.stat().st_mtime))
    chosen = candidates[0]
    LOGGER.info("自动选择 checkpoint: %s", chosen)
    return chosen


def load_checkpoint_payload(path: Path, device: torch.device) -> Dict[str, Any]:
    payload = torch.load(path, map_location=device)
    if isinstance(payload, dict):
        return payload
    raise TypeError(f"checkpoint payload 类型异常: {type(payload)}")


def sanitize_model_config(config: Dict[str, Any], num_buses: int) -> Dict[str, Any]:
    clean = dict(FALLBACK_MODEL_CONFIG)
    for key in list(clean.keys()):
        if key in config:
            clean[key] = config[key]
    clean["max_num_nodes"] = max(int(clean.get("max_num_nodes", 2048)), int(num_buses))
    return clean


def build_model_from_checkpoint(
    checkpoint_payload: Dict[str, Any],
    network_metadata: Dict[str, Any],
    num_buses: int,
    device: torch.device,
) -> HybridGTForPowerFlow:
    ckpt_model_config = checkpoint_payload.get("model_config", {})
    if not isinstance(ckpt_model_config, dict):
        ckpt_model_config = {}
    model_config = sanitize_model_config(ckpt_model_config, num_buses=num_buses)
    LOGGER.info("使用模型配置: %s", json.dumps(model_config, ensure_ascii=False))

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
        dynamic_depth_sampling=bool(model_config["dynamic_depth_sampling"]),
    )

    state_dict = checkpoint_payload.get("model_state_dict", checkpoint_payload)
    if not isinstance(state_dict, dict):
        raise TypeError("checkpoint 中未找到有效的 model_state_dict")

    # Support DataParallel / DistributedDataParallel checkpoints.
    if any(str(k).startswith("module.") for k in state_dict.keys()):
        state_dict = {str(k).replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "模型权重加载存在不匹配。\n"
            f"missing keys: {missing}\n"
            f"unexpected keys: {unexpected}\n"
            "通常是 MODEL_CFG 与训练时不一致，或 checkpoint 不是该模型的 finetune/pretrain checkpoint。"
        )
    model.to(device)
    model.eval()
    return model


def build_standardizer(device: torch.device) -> Optional[NodeFeatureStandardizer]:
    if not ENABLE_NODE_FEATURE_STANDARDIZATION:
        LOGGER.warning("未启用标准化反变换。若模型训练使用了归一化，物理量纲评估会错误。")
        return None
    stats_path = Path(STANDARDIZATION_STATS_PATH).expanduser()
    if not stats_path.exists():
        message = f"标准化统计文件不存在: {stats_path}"
        if REQUIRE_STANDARDIZATION_STATS:
            raise FileNotFoundError(message)
        LOGGER.warning("%s；将按未标准化模型评估。", message)
        return None
    LOGGER.info("加载标准化统计: %s", stats_path)
    return NodeFeatureStandardizer.from_npz(str(stats_path), device=device)


def maybe_normalize(
    H_raw: torch.Tensor,
    state_valid_mask: torch.Tensor,
    standardizer: Optional[NodeFeatureStandardizer],
) -> torch.Tensor:
    if standardizer is None:
        return H_raw
    return standardizer.normalize(H_raw, state_valid_mask=state_valid_mask)


def maybe_denormalize(
    H_norm: torch.Tensor,
    state_valid_mask: torch.Tensor,
    standardizer: Optional[NodeFeatureStandardizer],
) -> torch.Tensor:
    if standardizer is None:
        return H_norm
    return standardizer.denormalize(H_norm, state_valid_mask=state_valid_mask)


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


def sample_indices_from_loader(loader: DataLoader) -> List[int]:
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


def build_eval_mask(
    state_valid_mask: torch.Tensor,
    bus_type: torch.Tensor,
    feat_dim: int,
    evaluated_feature_indices: Sequence[int],
) -> torch.Tensor:
    if EVALUATE_BUS_TYPE_TARGET_ONLY:
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


def create_eval_input_feature_mask(
    node_valid_mask: torch.Tensor,
    state_valid_mask: torch.Tensor,
    bus_type: torch.Tensor,
    feat_dim: int,
) -> Optional[torch.Tensor]:
    if not ZERO_TARGET_FIELDS_IN_INPUT:
        return None
    return create_input_feature_mask_for_finetune(
        node_valid_mask=node_valid_mask,
        state_valid_mask=state_valid_mask,
        bus_type=bus_type,
        feat_dim=feat_dim,
    )


def feature_target_flag(bus_type_id: int, feature_idx: int) -> bool:
    return int(feature_idx) in TARGET_FEATURES_BY_BUS_TYPE.get(int(bus_type_id), [])


def build_node_rows_for_selected_sample(
    split: str,
    sample_idx: int,
    pred_effective_phys: torch.Tensor,
    pred_raw_phys: torch.Tensor,
    target_phys: torch.Tensor,
    state_valid_mask: torch.Tensor,
    node_valid_mask: torch.Tensor,
    bus_ids: Sequence[int],
    bus_type_np: np.ndarray,
) -> List[Dict[str, Any]]:
    pred_eff_np = pred_effective_phys.detach().cpu().numpy()
    pred_raw_np = pred_raw_phys.detach().cpu().numpy()
    target_np = target_phys.detach().cpu().numpy()
    state_valid_np = state_valid_mask.detach().cpu().numpy().astype(bool)
    node_valid_np = node_valid_mask.detach().cpu().numpy().astype(bool)

    rows: List[Dict[str, Any]] = []
    num_nodes = int(target_np.shape[0])
    for bus_pos in range(num_nodes):
        bus_id = int(bus_ids[bus_pos]) if bus_pos < len(bus_ids) else int(bus_pos)
        bus_type_id = int(bus_type_np[bus_pos]) if bus_pos < len(bus_type_np) else -1
        row: Dict[str, Any] = {
            "split": split,
            "sample_idx": int(sample_idx),
            "bus_pos": int(bus_pos),
            "bus_id": bus_id,
            "bus_type_id": bus_type_id,
            "bus_type": BUS_TYPE_NAMES.get(bus_type_id, "UNKNOWN"),
            "node_valid": bool(node_valid_np[bus_pos]),
            "state_valid": bool(state_valid_np[bus_pos]),
        }
        for feat_idx in EVALUATED_FEATURE_INDICES:
            feat_name = FEATURE_NAMES[feat_idx]
            is_target = feature_target_flag(bus_type_id, feat_idx)
            true_val = float(target_np[bus_pos, feat_idx])
            pred_val = float(pred_eff_np[bus_pos, feat_idx])
            err = pred_val - true_val

            row[f"{feat_name}_true"] = true_val
            row[f"{feat_name}_pred"] = pred_val
            row[f"{feat_name}_err"] = err
            row[f"{feat_name}_abs_err"] = abs(err)
            row[f"{feat_name}_unit"] = PHYSICAL_UNITS.get(feat_name, "")
            row[f"{feat_name}_is_bus_type_target"] = bool(is_target)
            row[f"{feat_name}_pred_source"] = "model_target" if is_target else "given_input"

            if INCLUDE_RAW_MODEL_OUTPUT_IN_DETAIL:
                raw_val = float(pred_raw_np[bus_pos, feat_idx])
                row[f"{feat_name}_model_raw"] = raw_val
                row[f"{feat_name}_model_raw_err"] = raw_val - true_val

        rows.append(row)
    return rows


def summarize_selected_sample_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    out: List[Dict[str, Any]] = []
    for (split, sample_idx), group in df.groupby(["split", "sample_idx"], sort=True):
        summary: Dict[str, Any] = {"split": split, "sample_idx": int(sample_idx)}
        for feat_idx in EVALUATED_FEATURE_INDICES:
            feat_name = FEATURE_NAMES[feat_idx]
            target_col = f"{feat_name}_is_bus_type_target"
            err_col = f"{feat_name}_abs_err"
            valid = group["state_valid"].astype(bool)
            if EVALUATE_BUS_TYPE_TARGET_ONLY and target_col in group.columns:
                valid = valid & group[target_col].astype(bool)
            values = group.loc[valid, err_col].astype(float)
            summary[f"{feat_name}_count"] = int(values.shape[0])
            summary[f"{feat_name}_mae"] = float(values.mean()) if not values.empty else math.nan
            summary[f"{feat_name}_max_abs_error"] = float(values.max()) if not values.empty else math.nan
        out.append(summary)
    return out


@torch.no_grad()
def evaluate_split(
    split_name: str,
    loader: DataLoader,
    model: HybridGTForPowerFlow,
    standardizer: Optional[NodeFeatureStandardizer],
    bus_type: torch.Tensor,
    bus_ids: Sequence[int],
    selected_sample_ids: Set[int],
    output_dir: Path,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    model.eval()
    accumulators = {
        feat_idx: FeatureAccumulator()
        for feat_idx in EVALUATED_FEATURE_INDICES
    }
    selected_rows: List[Dict[str, Any]] = []
    bus_type_np = bus_type.detach().cpu().numpy().astype(np.int64)

    iterator = tqdm(loader, desc=f"Evaluate {split_name}", leave=True)
    for batch in iterator:
        H_raw = batch["H"].to(device, non_blocking=True)
        Y = batch["Y"].to(device, non_blocking=True)
        node_valid_mask = batch["node_valid_mask"].to(device, non_blocking=True)
        state_valid_mask = batch.get("state_valid_mask", batch["node_valid_mask"]).to(device, non_blocking=True)
        sample_idx_list = [int(x) for x in batch["sample_idx"]]

        H_norm = maybe_normalize(H_raw, state_valid_mask=state_valid_mask, standardizer=standardizer)
        feature_visible_mask = create_eval_input_feature_mask(
            node_valid_mask=node_valid_mask,
            state_valid_mask=state_valid_mask,
            bus_type=bus_type,
            feat_dim=H_norm.shape[-1],
        )

        with autocast(enabled=bool(USE_AMP_FOR_INFERENCE and device.type == "cuda")):
            pred_norm = model(
                H_norm,
                Y,
                node_valid_mask=node_valid_mask,
                feature_visible_mask=feature_visible_mask,
            )

        pred_raw_phys = maybe_denormalize(pred_norm.float(), state_valid_mask=state_valid_mask, standardizer=standardizer)
        target_phys = H_raw.float()

        eval_mask = build_eval_mask(
            state_valid_mask=state_valid_mask,
            bus_type=bus_type,
            feat_dim=H_raw.shape[-1],
            evaluated_feature_indices=EVALUATED_FEATURE_INDICES,
        )

        # Effective physical prediction: model output only replaces bus-type target fields.
        # Non-target fields are known/given quantities and are copied from the input/target H.
        pred_effective_phys = target_phys.clone()
        pred_effective_phys[eval_mask] = pred_raw_phys[eval_mask]
        err_phys = pred_effective_phys - target_phys

        for feat_idx in EVALUATED_FEATURE_INDICES:
            mask_f = eval_mask[:, :, feat_idx]
            if bool(mask_f.any().item()):
                accumulators[feat_idx].update(err_phys[:, :, feat_idx][mask_f])

        if SAVE_SELECTED_NODE_COMPARISON and selected_sample_ids:
            for b, sample_idx in enumerate(sample_idx_list):
                if sample_idx not in selected_sample_ids:
                    continue
                rows = build_node_rows_for_selected_sample(
                    split=split_name,
                    sample_idx=sample_idx,
                    pred_effective_phys=pred_effective_phys[b],
                    pred_raw_phys=pred_raw_phys[b],
                    target_phys=target_phys[b],
                    state_valid_mask=state_valid_mask[b],
                    node_valid_mask=node_valid_mask[b],
                    bus_ids=bus_ids,
                    bus_type_np=bus_type_np,
                )
                selected_rows.extend(rows)
                sample_df = pd.DataFrame(rows)
                sample_df.to_csv(
                    output_dir / f"selected_node_comparison_{split_name}_sample_{sample_idx}.csv",
                    index=False,
                )

    metric_rows = [
        accumulators[feat_idx].as_row(split=split_name, feature_idx=feat_idx)
        for feat_idx in EVALUATED_FEATURE_INDICES
    ]
    return metric_rows, selected_rows


def write_json(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def log_dataset_info(train_loader: DataLoader, test_loader: Optional[DataLoader]) -> None:
    train_ids = sample_indices_from_loader(train_loader)
    LOGGER.info(
        "训练评估 loader: batches=%d, samples=%d, sample_idx范围=[%s, %s]",
        len(train_loader),
        len(train_ids),
        min(train_ids) if train_ids else "NA",
        max(train_ids) if train_ids else "NA",
    )
    if test_loader is not None:
        test_ids = sample_indices_from_loader(test_loader)
        LOGGER.info(
            "测试评估 loader: batches=%d, samples=%d, sample_idx范围=[%s, %s]",
            len(test_loader),
            len(test_ids),
            min(test_ids) if test_ids else "NA",
            max(test_ids) if test_ids else "NA",
        )


def main() -> None:
    logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(SEED)

    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)
    device = torch.device(DEVICE)

    checkpoint_path = resolve_checkpoint_path()
    checkpoint_payload = load_checkpoint_payload(checkpoint_path, device=device)

    network_metadata = load_network_metadata(DATA_DIR)
    bus_ids = get_sorted_bus_ids(network_metadata)
    bus_type = build_bus_type_vector(network_metadata, device=device)
    base_mva = float(checkpoint_payload.get("base_mva", get_network_base_mva(network_metadata)))
    LOGGER.info("DATA_DIR=%s", DATA_DIR)
    LOGGER.info("checkpoint=%s", checkpoint_path)
    LOGGER.info("network buses=%d, base_mva=%.6f", len(bus_ids), base_mva)

    model = build_model_from_checkpoint(
        checkpoint_payload=checkpoint_payload,
        network_metadata=network_metadata,
        num_buses=len(bus_ids),
        device=device,
    )
    standardizer = build_standardizer(device=device)

    train_loader, val_loader, test_loader = create_dataloaders(
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
    del val_loader  # This script reports train and test only.
    if test_loader is None:
        raise RuntimeError("create_dataloaders 未返回 test_loader，请检查 DATA_DIR 布局或数据划分设置。")

    log_dataset_info(train_loader=train_loader, test_loader=test_loader)

    train_available_ids = sample_indices_from_loader(train_loader)
    test_available_ids = sample_indices_from_loader(test_loader)
    selected_train_ids = choose_sample_ids(train_available_ids, SELECTED_TRAIN_SAMPLE_IDS)
    selected_test_ids = choose_sample_ids(test_available_ids, SELECTED_TEST_SAMPLE_IDS)
    LOGGER.info("选中的训练样本: %s", selected_train_ids)
    LOGGER.info("选中的测试样本: %s", selected_test_ids)

    all_metric_rows: List[Dict[str, Any]] = []
    all_selected_rows: List[Dict[str, Any]] = []

    train_metric_rows, train_selected_rows = evaluate_split(
        split_name="train",
        loader=train_loader,
        model=model,
        standardizer=standardizer,
        bus_type=bus_type,
        bus_ids=bus_ids,
        selected_sample_ids=set(selected_train_ids),
        output_dir=output_dir,
        device=device,
    )
    all_metric_rows.extend(train_metric_rows)
    all_selected_rows.extend(train_selected_rows)

    test_metric_rows, test_selected_rows = evaluate_split(
        split_name="test",
        loader=test_loader,
        model=model,
        standardizer=standardizer,
        bus_type=bus_type,
        bus_ids=bus_ids,
        selected_sample_ids=set(selected_test_ids),
        output_dir=output_dir,
        device=device,
    )
    all_metric_rows.extend(test_metric_rows)
    all_selected_rows.extend(test_selected_rows)

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_csv_path = output_dir / "metrics_summary.csv"
    metrics_json_path = output_dir / "metrics_summary.json"
    metrics_df.to_csv(metrics_csv_path, index=False)
    write_json(metrics_json_path, all_metric_rows)

    if SAVE_SELECTED_NODE_COMPARISON and all_selected_rows:
        selected_df = pd.DataFrame(all_selected_rows)
        selected_df.to_csv(output_dir / "selected_node_comparison_all.csv", index=False)
        if SAVE_SELECTED_SAMPLE_SUMMARY:
            sample_summary_rows = summarize_selected_sample_rows(all_selected_rows)
            pd.DataFrame(sample_summary_rows).to_csv(output_dir / "selected_sample_summary.csv", index=False)
            write_json(output_dir / "selected_sample_summary.json", sample_summary_rows)

    run_info = {
        "data_dir": str(DATA_DIR),
        "checkpoint_path": str(checkpoint_path),
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "base_mva": base_mva,
        "evaluated_features": [FEATURE_NAMES[i] for i in EVALUATED_FEATURE_INDICES],
        "evaluate_bus_type_target_only": bool(EVALUATE_BUS_TYPE_TARGET_ONLY),
        "zero_target_fields_in_input": bool(ZERO_TARGET_FIELDS_IN_INPUT),
        "standardization_enabled": bool(standardizer is not None),
        "standardization_stats_path": STANDARDIZATION_STATS_PATH if standardizer is not None else None,
        "target_features_by_bus_type": {
            "PQ": [FEATURE_NAMES[i] for i in TARGET_FEATURES_BY_BUS_TYPE[BUS_TYPE_PQ]],
            "PV": [FEATURE_NAMES[i] for i in TARGET_FEATURES_BY_BUS_TYPE[BUS_TYPE_PV]],
            "SLACK": [FEATURE_NAMES[i] for i in TARGET_FEATURES_BY_BUS_TYPE[BUS_TYPE_SLACK]],
        },
        "detail_pred_column_convention": "*_pred is effective prediction: raw model output on bus-type target fields; given input/true value on non-target fields.",
        "include_raw_model_output_in_detail": bool(INCLUDE_RAW_MODEL_OUTPUT_IN_DETAIL),
        "train_split": TRAIN_SPLIT,
        "val_split": VAL_SPLIT,
        "batch_size": BATCH_SIZE,
        "selected_train_sample_ids": selected_train_ids,
        "selected_test_sample_ids": selected_test_ids,
    }
    write_json(output_dir / "run_info.json", run_info)

    print("\n===== 物理量纲评估汇总 =====")
    print(metrics_df.to_string(index=False))
    print(f"\n评估输出目录: {output_dir.resolve()}")
    print(f"汇总指标: {metrics_csv_path.resolve()}")
    if SAVE_SELECTED_NODE_COMPARISON:
        print(f"逐节点对比: {(output_dir / 'selected_node_comparison_all.csv').resolve()}")


if __name__ == "__main__":
    main()
