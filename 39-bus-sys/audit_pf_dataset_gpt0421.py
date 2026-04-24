# In[]
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


# ============================================================
# 手动配置区（不要 argparse）
# ============================================================
DATA_ROOT = "/data2/zyh/case39_samples"
OUTPUT_DIR = "./dataset_audit_outputs_gpt0421"
CHECK_SPLITS = True                    # 若 DATA_ROOT/train 与 DATA_ROOT/test 都存在，则分别检查
MAX_SAMPLES_PER_SPLIT: Optional[int] = None   # None 表示全量检查
SAVE_PER_SAMPLE_CSV = True
SAVE_FEATURE_SUMMARY_CSV = True
SAVE_TOPK_BAD_SAMPLES_CSV = True
SAVE_HISTOGRAMS = True
TOPK_BAD_SAMPLES = 50

Y_SYMMETRY_ATOL = 1e-8                 # 检查 Y 是否接近复对称 Y == Y.T
PHYSICS_RESIDUAL_WARN_PU = 1e-4        # 仅用于标记可疑样本，不影响主统计
NONZERO_TOL = 1e-10

FEATURE_NAMES = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]
IDX_PD = 0
IDX_QD = 1
IDX_PG = 2
IDX_QG = 3
IDX_VM = 4
IDX_VA = 5

BUS_TYPE_PQ = 0
BUS_TYPE_PV = 1
BUS_TYPE_SLACK = 2
BUS_TYPE_NAMES = {
    BUS_TYPE_PQ: "PQ",
    BUS_TYPE_PV: "PV",
    BUS_TYPE_SLACK: "SLACK",
}


# ============================================================
# 基础 I/O
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def resolve_dataset_dirs(data_root: str, check_splits: bool = True) -> List[Path]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")
    train_dir = root / "train"
    test_dir = root / "test"
    if check_splits and train_dir.exists() and test_dir.exists():
        return [train_dir, test_dir]
    return [root]


def discover_sample_indices(data_dir: Path) -> List[int]:
    indices: List[int] = []
    for fname in os.listdir(data_dir):
        if not (fname.startswith("metadata_") and fname.endswith(".json")):
            continue
        try:
            idx = int(fname[len("metadata_"):-len(".json")])
        except Exception:
            continue
        if (data_dir / f"H_{idx}.npy").exists() and (data_dir / f"Y_{idx}.npz").exists():
            indices.append(idx)
    return sorted(indices)


# ============================================================
# network_metadata / bus type
# ============================================================
def load_network_metadata(data_dir: Path) -> Dict[str, Any]:
    candidates = [data_dir / "network_metadata.json", data_dir.parent / "network_metadata.json"]
    for path in candidates:
        if path.exists():
            meta = safe_load_json(path)
            if isinstance(meta, dict):
                return meta
    raise FileNotFoundError(f"未找到 network_metadata.json，已检查: {[str(p) for p in candidates]}")


def get_records(metadata: Dict[str, Any], keys: Sequence[str]) -> List[Dict[str, Any]]:
    for key in keys:
        value = metadata.get(key, None)
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, dict):
                    out.append(item)
            return out
    return []


def get_sorted_bus_ids(network_metadata: Dict[str, Any]) -> List[int]:
    buses = network_metadata.get("buses", None)
    if not isinstance(buses, dict) or not buses:
        raise ValueError("network_metadata 缺少 buses 字段")
    return sorted(int(k) for k in buses.keys())


def get_base_mva(network_metadata: Dict[str, Any], default: float = 100.0) -> float:
    network_info = network_metadata.get("network_info", {})
    if isinstance(network_info, dict):
        for key in ["sn_mva", "base_mva", "sn_MVA", "baseMVA"]:
            value = network_info.get(key, None)
            if value is None:
                continue
            try:
                value = float(value)
                if value > 0:
                    return value
            except Exception:
                pass
    for key in ["sn_mva", "base_mva", "sn_MVA", "baseMVA"]:
        value = network_metadata.get(key, None)
        if value is None:
            continue
        try:
            value = float(value)
            if value > 0:
                return value
        except Exception:
            pass
    return float(default)


def build_bus_type_vector(network_metadata: Dict[str, Any]) -> np.ndarray:
    bus_ids = get_sorted_bus_ids(network_metadata)
    id_to_pos = {bus_id: i for i, bus_id in enumerate(bus_ids)}
    bus_type = np.full((len(bus_ids),), BUS_TYPE_PQ, dtype=np.int64)
    for item in get_records(network_metadata, ["gen", "gens"]):
        if "bus" in item:
            bus = int(item["bus"])
            if bus in id_to_pos:
                bus_type[id_to_pos[bus]] = BUS_TYPE_PV
    for item in get_records(network_metadata, ["ext_grid", "ext_grids"]):
        if "bus" in item:
            bus = int(item["bus"])
            if bus in id_to_pos:
                bus_type[id_to_pos[bus]] = BUS_TYPE_SLACK
    return bus_type


# ============================================================
# 物理量 / 掩码
# ============================================================
def infer_state_valid_mask(H: np.ndarray, metadata: Optional[Dict[str, Any]]) -> np.ndarray:
    num_nodes = int(H.shape[0])
    if isinstance(metadata, dict):
        for key in ["state_valid_mask", "pf_valid_mask", "connected_bus_mask", "solver_bus_mask"]:
            value = metadata.get(key, None)
            if isinstance(value, list) and len(value) == num_nodes:
                try:
                    return np.asarray(value, dtype=bool)
                except Exception:
                    pass
    vm = H[:, IDX_VM] if H.shape[1] > IDX_VM else np.zeros((num_nodes,), dtype=np.float64)
    va = H[:, IDX_VA] if H.shape[1] > IDX_VA else np.zeros((num_nodes,), dtype=np.float64)
    return np.isfinite(vm) & np.isfinite(va) & (np.abs(vm) > 0.0)


def build_target_mask(state_valid_mask: np.ndarray, bus_type: np.ndarray) -> np.ndarray:
    num_nodes = int(state_valid_mask.shape[0])
    mask = np.zeros((num_nodes, 6), dtype=bool)
    bt = bus_type[:num_nodes]
    pq = bt == BUS_TYPE_PQ
    pv = bt == BUS_TYPE_PV
    sl = bt == BUS_TYPE_SLACK
    mask[pq, IDX_VM] = True
    mask[pq, IDX_VA] = True
    mask[pv, IDX_QG] = True
    mask[pv, IDX_VA] = True
    mask[sl, IDX_PG] = True
    mask[sl, IDX_QG] = True
    mask &= state_valid_mask[:, None]
    return mask


# ============================================================
# 潮流残差
# ============================================================
def compute_complex_power_from_voltage_np(Y: np.ndarray, vm: np.ndarray, va_degree: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    va_rad = np.deg2rad(va_degree.astype(np.float64))
    v_complex = vm.astype(np.float64) * np.exp(1j * va_rad)
    current = Y.astype(np.complex128) @ v_complex.astype(np.complex128)
    s_inj = v_complex * np.conj(current)
    return s_inj.real.astype(np.float64), s_inj.imag.astype(np.float64)


def evaluate_physics_residual(H: np.ndarray, Y: np.ndarray, state_valid_mask: np.ndarray, base_mva: float) -> Dict[str, Any]:
    if base_mva <= 0:
        raise ValueError(f"base_mva 必须为正，当前为 {base_mva}")

    pd = H[:, IDX_PD]
    qd = H[:, IDX_QD]
    pg = H[:, IDX_PG]
    qg = H[:, IDX_QG]
    vm = np.clip(H[:, IDX_VM], 1e-5, None)
    va = H[:, IDX_VA]

    p_spec_pu = (pg - pd) / float(base_mva)
    q_spec_pu = (qg - qd) / float(base_mva)
    p_calc_pu, q_calc_pu = compute_complex_power_from_voltage_np(Y, vm, va)

    mask = state_valid_mask.astype(bool)
    if int(mask.sum()) <= 0:
        return {
            "num_state_valid": 0,
            "phy_mse": float("nan"),
            "phy_p_rmse_pu": float("nan"),
            "phy_q_rmse_pu": float("nan"),
            "phy_p_mae_pu": float("nan"),
            "phy_q_mae_pu": float("nan"),
            "phy_p_max_abs_pu": float("nan"),
            "phy_q_max_abs_pu": float("nan"),
        }

    p_res = p_spec_pu[mask] - p_calc_pu[mask]
    q_res = q_spec_pu[mask] - q_calc_pu[mask]
    phy_mse = (np.mean(p_res ** 2) + np.mean(q_res ** 2)) / 2.0
    return {
        "num_state_valid": int(mask.sum()),
        "phy_mse": float(phy_mse),
        "phy_p_rmse_pu": float(np.sqrt(np.mean(p_res ** 2))),
        "phy_q_rmse_pu": float(np.sqrt(np.mean(q_res ** 2))),
        "phy_p_mae_pu": float(np.mean(np.abs(p_res))),
        "phy_q_mae_pu": float(np.mean(np.abs(q_res))),
        "phy_p_max_abs_pu": float(np.max(np.abs(p_res))),
        "phy_q_max_abs_pu": float(np.max(np.abs(q_res))),
    }


# ============================================================
# 单样本检查
# ============================================================
def check_one_sample(sample_idx: int, data_dir: Path, bus_type: np.ndarray, base_mva: float) -> Dict[str, Any]:
    h_path = data_dir / f"H_{sample_idx}.npy"
    y_path = data_dir / f"Y_{sample_idx}.npz"
    m_path = data_dir / f"metadata_{sample_idx}.json"

    metadata = safe_load_json(m_path)
    H = np.asarray(np.load(h_path), dtype=np.float64)
    Y = load_npz(y_path).toarray().astype(np.complex128)

    problems: List[str] = []
    if H.ndim != 2:
        problems.append(f"H 不是二维矩阵，shape={tuple(H.shape)}")
    if Y.ndim != 2:
        problems.append(f"Y 不是二维矩阵，shape={tuple(Y.shape)}")
    if H.ndim == 2 and H.shape[1] != 6:
        problems.append(f"H 的特征维不是 6，shape={tuple(H.shape)}")
    if Y.ndim == 2 and Y.shape[0] != Y.shape[1]:
        problems.append(f"Y 不是方阵，shape={tuple(Y.shape)}")
    if H.ndim == 2 and Y.ndim == 2 and H.shape[0] != Y.shape[0]:
        problems.append(f"H 节点数与 Y 维度不匹配: H={H.shape}, Y={Y.shape}")

    meta_h_shape = metadata.get("h_shape") if isinstance(metadata, dict) else None
    meta_y_shape = metadata.get("y_shape") if isinstance(metadata, dict) else None
    if isinstance(meta_h_shape, list) and list(H.shape) != list(meta_h_shape):
        problems.append(f"metadata.h_shape={meta_h_shape} 与实际 H.shape={list(H.shape)} 不一致")
    if isinstance(meta_y_shape, list) and list(Y.shape) != list(meta_y_shape):
        problems.append(f"metadata.y_shape={meta_y_shape} 与实际 Y.shape={list(Y.shape)} 不一致")

    state_valid_mask = infer_state_valid_mask(H, metadata)
    if state_valid_mask.shape[0] != H.shape[0]:
        problems.append("state_valid_mask 长度与 H 节点数不一致")
        state_valid_mask = np.ones((H.shape[0],), dtype=bool)

    active_bus_ids = metadata.get("active_bus_ids", []) if isinstance(metadata, dict) else []
    isolated_bus_ids = metadata.get("isolated_bus_ids", []) if isinstance(metadata, dict) else []
    if isinstance(active_bus_ids, list) and len(active_bus_ids) > 0:
        if len(active_bus_ids) != int(state_valid_mask.sum()):
            problems.append(
                f"active_bus_ids 数量({len(active_bus_ids)}) 与 state_valid_mask 求和({int(state_valid_mask.sum())}) 不一致"
            )
    if isinstance(isolated_bus_ids, list) and len(isolated_bus_ids) != int((~state_valid_mask).sum()):
        problems.append(
            f"isolated_bus_ids 数量({len(isolated_bus_ids)}) 与失效节点数({int((~state_valid_mask).sum())}) 不一致"
        )

    y_symmetry_max_abs = float(np.max(np.abs(Y - Y.T))) if Y.size > 0 else 0.0
    if y_symmetry_max_abs > Y_SYMMETRY_ATOL:
        problems.append(f"Y 与 Y.T 不够接近，max|Y-Y.T|={y_symmetry_max_abs:.3e}")

    valid_rows = np.any(np.abs(Y) > NONZERO_TOL, axis=1)
    valid_cols = np.any(np.abs(Y) > NONZERO_TOL, axis=0)
    y_nonzero_mask = valid_rows | valid_cols
    if np.any(y_nonzero_mask != state_valid_mask):
        problems.append("Y 的非零行列支持与 state_valid_mask 不完全一致")

    physics = evaluate_physics_residual(H=H, Y=Y, state_valid_mask=state_valid_mask, base_mva=base_mva)
    suspicious_physics = False
    phy_rmse = max(physics["phy_p_rmse_pu"], physics["phy_q_rmse_pu"])
    if np.isfinite(phy_rmse) and phy_rmse > PHYSICS_RESIDUAL_WARN_PU:
        suspicious_physics = True
        problems.append(f"潮流残差偏大，max(P/Q RMSE)={phy_rmse:.3e} p.u.")

    target_mask = build_target_mask(state_valid_mask=state_valid_mask, bus_type=bus_type[: H.shape[0]])
    target_values = H[target_mask]
    row: Dict[str, Any] = {
        "sample_idx": int(sample_idx),
        "h_shape": list(H.shape),
        "y_shape": list(Y.shape),
        "state_valid_count": int(state_valid_mask.sum()),
        "isolated_count": int((~state_valid_mask).sum()),
        "target_count": int(target_mask.sum()),
        "num_outages": int(metadata.get("num_outages", 0)) if isinstance(metadata, dict) else 0,
        "y_nnz_metadata": int(metadata.get("y_nnz", 0)) if isinstance(metadata, dict) else -1,
        "y_nnz_actual": int(np.count_nonzero(np.abs(Y) > NONZERO_TOL)),
        "y_sparsity_metadata": float(metadata.get("y_sparsity", np.nan)) if isinstance(metadata, dict) else float("nan"),
        "vm_min": float(np.min(H[:, IDX_VM])) if H.size > 0 else float("nan"),
        "vm_max": float(np.max(H[:, IDX_VM])) if H.size > 0 else float("nan"),
        "va_min_deg": float(np.min(H[:, IDX_VA])) if H.size > 0 else float("nan"),
        "va_max_deg": float(np.max(H[:, IDX_VA])) if H.size > 0 else float("nan"),
        "pg_sum_mw": float(np.sum(H[:, IDX_PG])),
        "pd_sum_mw": float(np.sum(H[:, IDX_PD])),
        "qg_sum_mvar": float(np.sum(H[:, IDX_QG])),
        "qd_sum_mvar": float(np.sum(H[:, IDX_QD])),
        "net_p_injection_mw": float(np.sum(H[:, IDX_PG] - H[:, IDX_PD])),
        "net_q_injection_mvar": float(np.sum(H[:, IDX_QG] - H[:, IDX_QD])),
        "target_abs_mean": float(np.mean(np.abs(target_values))) if target_values.size > 0 else float("nan"),
        "y_symmetry_max_abs": y_symmetry_max_abs,
        "physics_suspicious": bool(suspicious_physics),
        "num_problems": int(len(problems)),
        "problems": " | ".join(problems),
        **physics,
    }
    return row


# ============================================================
# 分布统计
# ============================================================
def summarize_feature_distribution(
    rows: List[Dict[str, Any]],
    sample_buffers: List[Tuple[np.ndarray, np.ndarray]],
    bus_type: np.ndarray,
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    # 全体 state-valid 节点统计
    all_h_valid: List[np.ndarray] = []
    all_h_all: List[np.ndarray] = []
    by_bus_type_valid: Dict[str, List[np.ndarray]] = {"PQ": [], "PV": [], "SLACK": []}

    for H, state_valid_mask in sample_buffers:
        all_h_all.append(H)
        all_h_valid.append(H[state_valid_mask])
        bt = bus_type[: H.shape[0]]
        for bt_value, bt_name in BUS_TYPE_NAMES.items():
            mask = state_valid_mask & (bt == bt_value)
            if np.any(mask):
                by_bus_type_valid[bt_name].append(H[mask])

    def append_group(group_name: str, arrays: List[np.ndarray]) -> None:
        if not arrays:
            return
        X = np.concatenate(arrays, axis=0)
        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            v = X[:, feat_idx]
            finite = np.isfinite(v)
            if not np.any(finite):
                continue
            vf = v[finite]
            records.append({
                "group": group_name,
                "feature": feat_name,
                "count": int(vf.size),
                "mean": float(np.mean(vf)),
                "std": float(np.std(vf)),
                "min": float(np.min(vf)),
                "p01": float(np.quantile(vf, 0.01)),
                "p05": float(np.quantile(vf, 0.05)),
                "p50": float(np.quantile(vf, 0.50)),
                "p95": float(np.quantile(vf, 0.95)),
                "p99": float(np.quantile(vf, 0.99)),
                "max": float(np.max(vf)),
            })

    append_group("all_nodes", all_h_all)
    append_group("state_valid_nodes", all_h_valid)
    for bt_name, arrays in by_bus_type_valid.items():
        append_group(f"state_valid_{bt_name}", arrays)

    return pd.DataFrame(records)


def save_histograms(sample_buffers: List[Tuple[np.ndarray, np.ndarray]], output_dir: Path) -> None:
    if plt is None:
        return
    valid_arrays = [H[state_valid_mask] for H, state_valid_mask in sample_buffers if np.any(state_valid_mask)]
    if not valid_arrays:
        return
    X = np.concatenate(valid_arrays, axis=0)
    ensure_dir(output_dir)
    for feat_idx, feat_name in enumerate(FEATURE_NAMES):
        values = X[:, feat_idx]
        values = values[np.isfinite(values)]
        if values.size <= 0:
            continue
        fig = plt.figure(figsize=(6.0, 4.0))
        plt.hist(values, bins=80)
        plt.title(f"{feat_name} on state-valid nodes")
        plt.xlabel(feat_name)
        plt.ylabel("count")
        plt.tight_layout()
        fig.savefig(output_dir / f"hist_{feat_name}.png", dpi=150)
        plt.close(fig)


# ============================================================
# split 级审计
# ============================================================
def audit_one_split(data_dir: Path, output_dir: Path) -> Dict[str, Any]:
    ensure_dir(output_dir)
    network_metadata = load_network_metadata(data_dir)
    base_mva = get_base_mva(network_metadata)
    bus_ids = get_sorted_bus_ids(network_metadata)
    bus_type = build_bus_type_vector(network_metadata)

    sample_indices = discover_sample_indices(data_dir)
    if MAX_SAMPLES_PER_SPLIT is not None:
        sample_indices = sample_indices[: int(MAX_SAMPLES_PER_SPLIT)]

    rows: List[Dict[str, Any]] = []
    sample_buffers: List[Tuple[np.ndarray, np.ndarray]] = []

    for sample_idx in sample_indices:
        row = check_one_sample(sample_idx=sample_idx, data_dir=data_dir, bus_type=bus_type, base_mva=base_mva)
        rows.append(row)

        H = np.asarray(np.load(data_dir / f"H_{sample_idx}.npy"), dtype=np.float64)
        metadata = safe_load_json(data_dir / f"metadata_{sample_idx}.json")
        state_valid_mask = infer_state_valid_mask(H, metadata)
        sample_buffers.append((H, state_valid_mask))

    df = pd.DataFrame(rows)
    feature_df = summarize_feature_distribution(rows=rows, sample_buffers=sample_buffers, bus_type=bus_type)

    if SAVE_PER_SAMPLE_CSV and not df.empty:
        df.to_csv(output_dir / "sample_checks.csv", index=False)
    if SAVE_FEATURE_SUMMARY_CSV and not feature_df.empty:
        feature_df.to_csv(output_dir / "feature_summary.csv", index=False)
    if SAVE_TOPK_BAD_SAMPLES_CSV and not df.empty:
        bad_df = df.sort_values(by=["num_problems", "phy_mse"], ascending=[False, False]).head(TOPK_BAD_SAMPLES)
        bad_df.to_csv(output_dir / "topk_bad_samples.csv", index=False)
    if SAVE_HISTOGRAMS:
        save_histograms(sample_buffers=sample_buffers, output_dir=output_dir / "histograms")

    summary = {
        "split_dir": str(data_dir),
        "num_samples_checked": int(len(rows)),
        "base_mva": float(base_mva),
        "num_buses_from_metadata": int(len(bus_ids)),
        "num_problem_samples": int((df["num_problems"] > 0).sum()) if not df.empty else 0,
        "num_physics_suspicious_samples": int(df["physics_suspicious"].sum()) if not df.empty else 0,
        "max_phy_p_rmse_pu": float(df["phy_p_rmse_pu"].max()) if not df.empty else float("nan"),
        "max_phy_q_rmse_pu": float(df["phy_q_rmse_pu"].max()) if not df.empty else float("nan"),
        "median_phy_p_rmse_pu": float(df["phy_p_rmse_pu"].median()) if not df.empty else float("nan"),
        "median_phy_q_rmse_pu": float(df["phy_q_rmse_pu"].median()) if not df.empty else float("nan"),
        "num_unique_h_shapes": int(df["h_shape"].astype(str).nunique()) if not df.empty else 0,
        "num_unique_y_shapes": int(df["y_shape"].astype(str).nunique()) if not df.empty else 0,
        "num_unique_state_valid_count": int(df["state_valid_count"].nunique()) if not df.empty else 0,
        "top_problem_sample_ids": df.sort_values(by=["num_problems", "phy_mse"], ascending=[False, False])["sample_idx"].head(20).tolist() if not df.empty else [],
    }

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main() -> None:
    output_root = Path(OUTPUT_DIR)
    ensure_dir(output_root)

    dataset_dirs = resolve_dataset_dirs(DATA_ROOT, check_splits=CHECK_SPLITS)
    all_summaries: List[Dict[str, Any]] = []

    for data_dir in dataset_dirs:
        split_name = data_dir.name
        split_output_dir = output_root / split_name
        print(f"\n===== 开始检查: {data_dir} =====")
        summary = audit_one_split(data_dir=data_dir, output_dir=split_output_dir)
        all_summaries.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(output_root / "all_summaries.json", "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print(f"\n检查完成，输出目录: {output_root.resolve()}")


if __name__ == "__main__":
    main()

# %%
