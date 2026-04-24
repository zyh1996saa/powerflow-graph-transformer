# In[]
from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

import pandapower as pp
import pandapower.networks as nw


# ============================================================
# 手动配置区（不要 argparse）
# ============================================================
DATA_ROOT = "/data2/zyh/case39_samples"
SPLIT = "train"                        # 可选: "train" / "test" / ""(表示 DATA_ROOT 本身)
OUTPUT_DIR = "./sample_replay_outputs_gpt0421"

SAMPLE_IDS: List[int] = []              # 为空时自动选择 AUTO_SELECT_NUM 个样本
AUTO_SELECT_NUM = 5
AUTO_SELECT_STRATEGY = "random"         # 可选: first / evenly_spaced / random
RANDOM_SEED = 42

CHECK_Y_MATRIX = True
CHECK_H_MATRIX = True
SAVE_BUSWISE_DETAIL_CSV = True
SAVE_SAMPLE_SUMMARY_CSV = True
SAVE_Y_DIFF_NPY = False                 # 如需保存每个样本 Y 差值矩阵，可打开
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
# 通用工具
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


def resolve_dataset_dir(data_root: str, split: str) -> Path:
    root = Path(data_root)
    if split and split.strip():
        candidate = root / split
        if not candidate.exists():
            raise FileNotFoundError(f"split 目录不存在: {candidate}")
        return candidate
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")
    return root


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


def load_network_metadata(data_dir: Path) -> Dict[str, Any]:
    candidates = [data_dir / "network_metadata.json", data_dir.parent / "network_metadata.json"]
    for path in candidates:
        if path.exists():
            meta = safe_load_json(path)
            if isinstance(meta, dict):
                return meta
    raise FileNotFoundError(f"未找到 network_metadata.json，已检查: {[str(p) for p in candidates]}")


def get_sorted_bus_ids(network_metadata: Dict[str, Any]) -> List[int]:
    buses = network_metadata.get("buses", None)
    if not isinstance(buses, dict) or not buses:
        raise ValueError("network_metadata 缺少 buses 字段")
    return sorted(int(k) for k in buses.keys())


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


def choose_sample_ids(all_indices: List[int]) -> List[int]:
    if SAMPLE_IDS:
        return [int(x) for x in SAMPLE_IDS]
    if len(all_indices) <= AUTO_SELECT_NUM:
        return list(all_indices)
    if AUTO_SELECT_STRATEGY == "first":
        return all_indices[:AUTO_SELECT_NUM]
    if AUTO_SELECT_STRATEGY == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        chosen = rng.choice(np.asarray(all_indices), size=AUTO_SELECT_NUM, replace=False)
        return sorted(int(x) for x in chosen.tolist())
    if AUTO_SELECT_STRATEGY == "evenly_spaced":
        pos = np.linspace(0, len(all_indices) - 1, AUTO_SELECT_NUM)
        chosen = sorted({all_indices[int(round(p))] for p in pos.tolist()})
        return chosen
    raise ValueError(f"未知 AUTO_SELECT_STRATEGY: {AUTO_SELECT_STRATEGY}")


# ============================================================
# 与生成脚本对齐的工具
# ============================================================
def prepare_network(net):
    if "gen" in net and len(net.gen) > 0:
        if "q_mvar" not in net.gen.columns:
            net.gen["q_mvar"] = 0.0
        net.gen["q_mvar"] = net.gen["q_mvar"].fillna(0.0)
        net.gen["p_mw"] = net.gen["p_mw"].fillna(0.0)
    if "load" in net and len(net.load) > 0:
        if "q_mvar" not in net.load.columns:
            net.load["q_mvar"] = 0.0
        net.load["q_mvar"] = net.load["q_mvar"].fillna(0.0)
        net.load["p_mw"] = net.load["p_mw"].fillna(0.0)
    return net


def _safe_lookup_internal_bus_idx(bus_lookup, bus_id: int) -> int:
    if bus_lookup is None:
        return -1
    if isinstance(bus_lookup, dict):
        value = bus_lookup.get(bus_id, -1)
        try:
            return int(value)
        except Exception:
            return -1
    try:
        if 0 <= int(bus_id) < len(bus_lookup):
            return int(bus_lookup[int(bus_id)])
    except Exception:
        pass
    return -1


def _get_result_value(result_table, idx: int, key: str) -> float:
    if result_table is None or len(result_table) <= 0:
        return 0.0
    if idx not in result_table.index:
        return 0.0
    try:
        return float(result_table.loc[idx].get(key, 0.0))
    except Exception:
        return 0.0


def get_network_matrices(net) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    pp.runpp(net, check=False, check_connectivity=True)

    bus_ids = sorted(int(x) for x in net.bus.index.tolist())
    bus_map = {bus_id: pos for pos, bus_id in enumerate(bus_ids)}
    n_buses = len(bus_ids)
    H = np.zeros((n_buses, 6), dtype=np.float64)

    for _, load in net.load.iterrows():
        i = bus_map[int(load.bus)]
        H[i, IDX_PD] += float(load.get("p_mw", 0.0))
        H[i, IDX_QD] += float(load.get("q_mvar", 0.0))

    res_gen = getattr(net, "res_gen", None)
    for gen_idx, gen in net.gen.iterrows():
        i = bus_map[int(gen.bus)]
        H[i, IDX_PG] += _get_result_value(res_gen, int(gen_idx), "p_mw")
        H[i, IDX_QG] += _get_result_value(res_gen, int(gen_idx), "q_mvar")

    if len(net.ext_grid) > 0 and hasattr(net, "res_ext_grid") and len(net.res_ext_grid) > 0:
        for ext_idx, ext_grid in net.ext_grid.iterrows():
            i = bus_map[int(ext_grid["bus"])]
            H[i, IDX_PG] += _get_result_value(net.res_ext_grid, int(ext_idx), "p_mw")
            H[i, IDX_QG] += _get_result_value(net.res_ext_grid, int(ext_idx), "q_mvar")

    bus_lookup = getattr(net, "_pd2ppc_lookups", {}).get("bus", None)
    Y_internal = np.asarray(net._ppc["internal"]["Ybus"].todense(), dtype=np.complex128)
    Y = np.zeros((n_buses, n_buses), dtype=np.complex128)
    state_valid_mask = np.zeros((n_buses,), dtype=bool)
    internal_index_by_bus_id: Dict[int, int] = {}
    isolated_bus_ids: List[int] = []

    for bus_id in bus_ids:
        pos = bus_map[bus_id]
        internal_idx = _safe_lookup_internal_bus_idx(bus_lookup, bus_id)
        internal_index_by_bus_id[bus_id] = internal_idx
        has_internal = 0 <= internal_idx < Y_internal.shape[0]
        res_row = net.res_bus.loc[bus_id] if bus_id in net.res_bus.index else None
        vm = float(res_row.get("vm_pu", np.nan)) if res_row is not None else np.nan
        va = float(res_row.get("va_degree", np.nan)) if res_row is not None else np.nan
        finite_state = np.isfinite(vm) and np.isfinite(va)
        if has_internal and finite_state:
            state_valid_mask[pos] = True
            H[pos, IDX_VM] = vm
            H[pos, IDX_VA] = va
        else:
            H[pos, IDX_VM] = 0.0
            H[pos, IDX_VA] = 0.0
            isolated_bus_ids.append(int(bus_id))

    valid_positions = np.flatnonzero(state_valid_mask)
    for i_pos in valid_positions.tolist():
        bus_i = bus_ids[i_pos]
        int_i = internal_index_by_bus_id[bus_i]
        for j_pos in valid_positions.tolist():
            bus_j = bus_ids[j_pos]
            int_j = internal_index_by_bus_id[bus_j]
            Y[i_pos, j_pos] = Y_internal[int_i, int_j]

    aux = {
        "bus_ids_sorted": [int(x) for x in bus_ids],
        "state_valid_mask": state_valid_mask.astype(bool).tolist(),
        "isolated_bus_ids": isolated_bus_ids,
    }
    return H, Y, aux


def apply_sample_metadata_to_case39(sample_metadata: Dict[str, Any]) -> Any:
    net = nw.case39()
    net = prepare_network(net)

    # 1) 拓扑扰动
    outaged_indices = [int(x) for x in sample_metadata.get("outaged_line_original_indices", [])]
    if outaged_indices:
        valid_drop_indices = [idx for idx in outaged_indices if idx in net.line.index]
        if valid_drop_indices:
            net.line.drop(valid_drop_indices, inplace=True)
            net.line.reset_index(drop=True, inplace=True)

    # 2) 负荷恢复
    final_loads_p = sample_metadata.get("final_loads_p", None)
    final_loads_q = sample_metadata.get("final_loads_q", None)
    load_factors = sample_metadata.get("load_factors", None)

    if isinstance(final_loads_p, list) and len(final_loads_p) == len(net.load):
        net.load["p_mw"] = np.asarray(final_loads_p, dtype=np.float64)
    elif isinstance(load_factors, list) and len(load_factors) == len(net.load):
        base_p = net.load["p_mw"].to_numpy(dtype=np.float64)
        net.load["p_mw"] = base_p * np.asarray(load_factors, dtype=np.float64)

    if isinstance(final_loads_q, list) and len(final_loads_q) == len(net.load):
        net.load["q_mvar"] = np.asarray(final_loads_q, dtype=np.float64)
    elif isinstance(load_factors, list) and len(load_factors) == len(net.load):
        base_q = net.load["q_mvar"].to_numpy(dtype=np.float64)
        net.load["q_mvar"] = base_q * np.asarray(load_factors, dtype=np.float64)

    # 3) 发电机输入恢复
    gen_scale_factor = float(sample_metadata.get("gen_scale_factor", 1.0))
    if len(net.gen) > 0:
        base_gen_p = net.gen["p_mw"].to_numpy(dtype=np.float64)
        base_gen_q = net.gen["q_mvar"].to_numpy(dtype=np.float64)
        net.gen["p_mw"] = base_gen_p * gen_scale_factor
        net.gen["q_mvar"] = base_gen_q * gen_scale_factor

    return net


# ============================================================
# 对比
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
    vm = H[:, IDX_VM]
    va = H[:, IDX_VA]
    return np.isfinite(vm) & np.isfinite(va) & (np.abs(vm) > 0.0)


def compare_matrices(saved: np.ndarray, replay: np.ndarray) -> Dict[str, float]:
    diff = replay - saved
    abs_diff = np.abs(diff)
    return {
        "max_abs_err": float(np.max(abs_diff)) if abs_diff.size > 0 else 0.0,
        "mae": float(np.mean(abs_diff)) if abs_diff.size > 0 else 0.0,
        "rmse": float(np.sqrt(np.mean(abs_diff ** 2))) if abs_diff.size > 0 else 0.0,
    }


def compare_sample(sample_idx: int, data_dir: Path, network_metadata: Dict[str, Any], bus_type: np.ndarray, output_dir: Path) -> Dict[str, Any]:
    h_path = data_dir / f"H_{sample_idx}.npy"
    y_path = data_dir / f"Y_{sample_idx}.npz"
    m_path = data_dir / f"metadata_{sample_idx}.json"

    saved_H = np.asarray(np.load(h_path), dtype=np.float64)
    saved_Y = load_npz(y_path).toarray().astype(np.complex128)
    sample_metadata = safe_load_json(m_path)
    if not isinstance(sample_metadata, dict):
        raise RuntimeError(f"metadata 读取失败: {m_path}")

    replay_net = apply_sample_metadata_to_case39(sample_metadata)
    replay_net = prepare_network(replay_net)
    pp.runpp(replay_net, check=False, check_connectivity=True)
    if not replay_net.converged:
        raise RuntimeError(f"样本 {sample_idx} 重算潮流未收敛")

    replay_H, replay_Y, aux = get_network_matrices(replay_net)
    saved_state_valid = infer_state_valid_mask(saved_H, sample_metadata)
    replay_state_valid = np.asarray(aux["state_valid_mask"], dtype=bool)

    H_stats = compare_matrices(saved=saved_H, replay=replay_H)
    Y_real_stats = compare_matrices(saved=saved_Y.real, replay=replay_Y.real)
    Y_imag_stats = compare_matrices(saved=saved_Y.imag, replay=replay_Y.imag)
    Y_abs_stats = compare_matrices(saved=np.abs(saved_Y), replay=np.abs(replay_Y))

    per_bus_rows: List[Dict[str, Any]] = []
    bus_ids = get_sorted_bus_ids(network_metadata)
    for i in range(saved_H.shape[0]):
        row = {
            "sample_idx": int(sample_idx),
            "bus_pos": int(i),
            "bus_id": int(bus_ids[i]) if i < len(bus_ids) else i,
            "bus_type": BUS_TYPE_NAMES[int(bus_type[i])] if i < len(bus_type) else "UNK",
            "saved_state_valid": bool(saved_state_valid[i]),
            "replay_state_valid": bool(replay_state_valid[i]),
            "state_valid_match": bool(saved_state_valid[i] == replay_state_valid[i]),
        }
        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            sval = float(saved_H[i, feat_idx])
            rval = float(replay_H[i, feat_idx])
            row[f"{feat_name}_saved"] = sval
            row[f"{feat_name}_replay"] = rval
            row[f"{feat_name}_abs_err"] = float(abs(rval - sval))
        per_bus_rows.append(row)

    per_bus_df = pd.DataFrame(per_bus_rows)
    if SAVE_BUSWISE_DETAIL_CSV:
        per_bus_df.to_csv(output_dir / f"sample_{sample_idx}_buswise_detail.csv", index=False)

    if CHECK_Y_MATRIX and SAVE_Y_DIFF_NPY:
        np.save(output_dir / f"sample_{sample_idx}_Y_diff_real.npy", (replay_Y.real - saved_Y.real).astype(np.float64))
        np.save(output_dir / f"sample_{sample_idx}_Y_diff_imag.npy", (replay_Y.imag - saved_Y.imag).astype(np.float64))

    err_cols = [f"{name}_abs_err" for name in FEATURE_NAMES]
    worst_row = per_bus_df.loc[per_bus_df[err_cols].max(axis=1).idxmax()] if not per_bus_df.empty else None

    summary = {
        "sample_idx": int(sample_idx),
        "num_outages": int(sample_metadata.get("num_outages", 0)),
        "metadata_num_retries": int(sample_metadata.get("num_retries", 0)),
        "metadata_gen_scale_factor": float(sample_metadata.get("gen_scale_factor", 1.0)),
        "saved_state_valid_count": int(saved_state_valid.sum()),
        "replay_state_valid_count": int(replay_state_valid.sum()),
        "state_valid_match_all": bool(np.all(saved_state_valid == replay_state_valid)),
        "outaged_line_indices": [int(x) for x in sample_metadata.get("outaged_line_original_indices", [])],
        "H_max_abs_err": H_stats["max_abs_err"],
        "H_mae": H_stats["mae"],
        "H_rmse": H_stats["rmse"],
        "Y_real_max_abs_err": Y_real_stats["max_abs_err"],
        "Y_real_mae": Y_real_stats["mae"],
        "Y_real_rmse": Y_real_stats["rmse"],
        "Y_imag_max_abs_err": Y_imag_stats["max_abs_err"],
        "Y_imag_mae": Y_imag_stats["mae"],
        "Y_imag_rmse": Y_imag_stats["rmse"],
        "Y_abs_max_abs_err": Y_abs_stats["max_abs_err"],
        "Y_abs_mae": Y_abs_stats["mae"],
        "Y_abs_rmse": Y_abs_stats["rmse"],
        "worst_bus_pos": int(worst_row["bus_pos"]) if worst_row is not None else -1,
        "worst_bus_id": int(worst_row["bus_id"]) if worst_row is not None else -1,
        "worst_bus_type": str(worst_row["bus_type"]) if worst_row is not None else "UNK",
        "worst_feature_max_abs_err": float(per_bus_df[err_cols].to_numpy().max()) if not per_bus_df.empty else 0.0,
    }
    for feat_name in FEATURE_NAMES:
        col = f"{feat_name}_abs_err"
        summary[f"{feat_name}_mae"] = float(per_bus_df[col].mean()) if not per_bus_df.empty else 0.0
        summary[f"{feat_name}_max_abs_err"] = float(per_bus_df[col].max()) if not per_bus_df.empty else 0.0

    with open(output_dir / f"sample_{sample_idx}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


# ============================================================
# 主程序
# ============================================================
def main() -> None:
    data_dir = resolve_dataset_dir(DATA_ROOT, SPLIT)
    output_dir = Path(OUTPUT_DIR)
    ensure_dir(output_dir)

    network_metadata = load_network_metadata(data_dir)
    network_name = str(network_metadata.get("network_info", {}).get("name", ""))
    if "39" not in network_name:
        raise RuntimeError(
            "当前脚本按 IEEE 39-bus / pandapower.networks.case39() 重建样本。"
            f"检测到 network name={network_name!r}，若后续切换到其它系统，需要补充更完整的 branch/trafo/switch 元数据并改写重建逻辑。"
        )

    all_indices = discover_sample_indices(data_dir)
    selected_ids = choose_sample_ids(all_indices)
    bus_type = build_bus_type_vector(network_metadata)

    print(f"数据目录: {data_dir}")
    print(f"选中的样本编号: {selected_ids}")

    rows: List[Dict[str, Any]] = []
    for sample_idx in selected_ids:
        print(f"\n===== 重放样本 {sample_idx} =====")
        sample_output_dir = output_dir / f"sample_{sample_idx}"
        ensure_dir(sample_output_dir)
        summary = compare_sample(
            sample_idx=sample_idx,
            data_dir=data_dir,
            network_metadata=network_metadata,
            bus_type=bus_type,
            output_dir=sample_output_dir,
        )
        rows.append(summary)
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    df = pd.DataFrame(rows)
    if SAVE_SAMPLE_SUMMARY_CSV and not df.empty:
        df.to_csv(output_dir / "sample_replay_summary.csv", index=False)
    with open(output_dir / "sample_replay_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"\n样本重放完成，输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

# %%
