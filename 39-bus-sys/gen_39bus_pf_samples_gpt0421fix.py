from __future__ import annotations

import json
import os
import random
import shutil
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandapower as pp
import pandapower.networks as nw
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm


# ============================================================
# 手动配置区（不要 argparse）
# ============================================================
DATAPATH = "/data2/zyh"
DATASET_ROOT = os.path.join(DATAPATH, "case39_samples")
TRAIN_SUBDIR = "train"
TEST_SUBDIR = "test"
NETWORK_METADATA_FILE = "network_metadata.json"

TRAIN_NUM_SUCCESS_SAMPLES = 2048 * 32
TEST_NUM_SUCCESS_SAMPLES = 2048
NUM_WORKERS = min(16, cpu_count())

MAX_OUTAGES = 3
MAX_FACTOR = 1.5
MIN_FACTOR = 0.5
SCALE = 0.95
MAX_RETRIES = 10

AVOID_SLACK_LINES = True
CLEAR_DATASET_ROOT_BEFORE_GENERATION = False
SAVE_NETWORK_METADATA_TO_ROOT = True
SAVE_NETWORK_METADATA_TO_SPLITS = True

GENERATOR_VERSION = "gpt0421fix_train_test_layout_and_pv_qg_from_res_gen"


# ============================================================
# 基础工具
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_split_dir(split_dir: Path) -> None:
    if not split_dir.exists():
        return
    for p in split_dir.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


def prepare_dataset_layout(dataset_root: str, clear_root: bool = False) -> Tuple[Path, Path, Path]:
    root = Path(dataset_root)
    train_dir = root / TRAIN_SUBDIR
    test_dir = root / TEST_SUBDIR

    ensure_dir(root)
    ensure_dir(train_dir)
    ensure_dir(test_dir)

    if clear_root:
        clear_split_dir(train_dir)
        clear_split_dir(test_dir)

    return root, train_dir, test_dir


# ============================================================
# 网络与元数据
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


def save_network_metadata(net, output_dir: str) -> Dict[str, Any]:
    metadata = {
        "network_info": {
            "name": "IEEE 39-Bus Test Case",
            "num_buses": int(len(net.bus)),
            "num_lines": int(len(net.line)),
            "num_loads": int(len(net.load)),
            "num_gens": int(len(net.gen)),
            "sn_mva": float(getattr(net, "sn_mva", 100.0)),
        },
        "buses": {
            int(bus_idx): {
                "name": str(net.bus.loc[bus_idx, "name"]) if "name" in net.bus.columns else f"Bus_{bus_idx}",
                "vn_kv": float(net.bus.loc[bus_idx, "vn_kv"]) if "vn_kv" in net.bus.columns else 1.0,
            }
            for bus_idx in net.bus.index
        },
        "lines": [
            {
                "line_idx": int(idx),
                "from_bus": int(line["from_bus"]),
                "to_bus": int(line["to_bus"]),
                "r_ohm_per_km": float(line.get("r_ohm_per_km", 0.0)),
                "x_ohm_per_km": float(line.get("x_ohm_per_km", 0.0)),
                "length_km": float(line.get("length_km", 1.0)),
                "in_service": bool(line.get("in_service", True)),
            }
            for idx, line in net.line.iterrows()
        ],
        "gens": [
            {
                "gen_idx": int(idx),
                "bus": int(gen["bus"]),
                "p_mw": float(gen.get("p_mw", 0.0)),
                "q_mvar": float(gen.get("q_mvar", 0.0)),
                "vm_pu": float(gen.get("vm_pu", 1.0)),
                "in_service": bool(gen.get("in_service", True)),
            }
            for idx, gen in net.gen.iterrows()
        ],
        "ext_grids": [
            {
                "ext_grid_idx": int(idx),
                "bus": int(ext_grid["bus"]),
                "vm_pu": float(ext_grid.get("vm_pu", 1.0)),
                "in_service": bool(ext_grid.get("in_service", True)),
            }
            for idx, ext_grid in net.ext_grid.iterrows()
        ],
        "loads": [
            {
                "load_idx": int(idx),
                "bus": int(load["bus"]),
                "p_mw": float(load.get("p_mw", 0.0)),
                "q_mvar": float(load.get("q_mvar", 0.0)),
                "in_service": bool(load.get("in_service", True)),
            }
            for idx, load in net.load.iterrows()
        ],
    }
    metadata["line"] = metadata["lines"]
    metadata["gen"] = metadata["gens"]
    metadata["ext_grid"] = metadata["ext_grids"]
    metadata["load"] = metadata["loads"]
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, NETWORK_METADATA_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


# ============================================================
# 样本矩阵构造
# ============================================================
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
        H[i, 0] += float(load.get("p_mw", 0.0))
        H[i, 1] += float(load.get("q_mvar", 0.0))

    # 关键修复：普通发电机使用潮流结果表 net.res_gen，而不是输入表 net.gen。
    # 这样 PV 节点的 Qg 才是求解器实际算出的无功出力。
    res_gen = getattr(net, "res_gen", None)
    for gen_idx, gen in net.gen.iterrows():
        i = bus_map[int(gen.bus)]
        H[i, 2] += _get_result_value(res_gen, int(gen_idx), "p_mw")
        H[i, 3] += _get_result_value(res_gen, int(gen_idx), "q_mvar")

    if len(net.ext_grid) > 0 and hasattr(net, "res_ext_grid") and len(net.res_ext_grid) > 0:
        for ext_idx, ext_grid in net.ext_grid.iterrows():
            slack_bus = int(ext_grid["bus"])
            i = bus_map[slack_bus]
            H[i, 2] += _get_result_value(net.res_ext_grid, int(ext_idx), "p_mw")
            H[i, 3] += _get_result_value(net.res_ext_grid, int(ext_idx), "q_mvar")

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
            H[pos, 4] = vm
            H[pos, 5] = va
        else:
            H[pos, 4] = 0.0
            H[pos, 5] = 0.0
            isolated_bus_ids.append(int(bus_id))

    valid_positions = np.flatnonzero(state_valid_mask)
    valid_bus_ids = [bus_ids[pos] for pos in valid_positions.tolist()]
    for i_pos in valid_positions.tolist():
        bus_i = bus_ids[i_pos]
        int_i = internal_index_by_bus_id[bus_i]
        for j_pos in valid_positions.tolist():
            bus_j = bus_ids[j_pos]
            int_j = internal_index_by_bus_id[bus_j]
            Y[i_pos, j_pos] = Y_internal[int_i, int_j]

    aux = {
        "bus_ids_sorted": [int(x) for x in bus_ids],
        "internal_index_by_bus_id": {str(k): int(v) for k, v in internal_index_by_bus_id.items()},
        "state_valid_mask": state_valid_mask.astype(bool).tolist(),
        "isolated_bus_ids": [int(x) for x in isolated_bus_ids],
        "active_bus_ids": [int(x) for x in valid_bus_ids],
        "y_internal_shape": [int(x) for x in Y_internal.shape],
        "y_external_shape": [int(x) for x in Y.shape],
    }
    return H, Y, aux


# ============================================================
# 拓扑扰动与并行采样
# ============================================================
def generate_nk_topology_delete(net, max_outages: int = 3, avoid_slack_lines: bool = True) -> List[int]:
    n_lines = len(net.line)
    available_indices = list(range(n_lines))
    if avoid_slack_lines and len(net.ext_grid) > 0:
        slack_buses = set(int(x) for x in net.ext_grid["bus"].tolist())
        available_indices = [
            idx for idx in available_indices
            if (int(net.line.iloc[idx].from_bus) not in slack_buses) and (int(net.line.iloc[idx].to_bus) not in slack_buses)
        ]
    if not available_indices:
        return []
    num_outages = random.randint(1, min(max_outages, len(available_indices)))
    return sorted(random.sample(available_indices, num_outages), reverse=True)


def apply_nk_topology(net, outaged_indices: List[int]):
    for idx in outaged_indices:
        if idx < len(net.line):
            net.line.drop(net.line.index[idx], inplace=True)
    net.line.reset_index(drop=True, inplace=True)
    return net


BASE_NET = None
ORIGINAL_PD = None
ORIGINAL_QD = None
ORIGINAL_LINES = None
MAX_OUTAGES_VAR = None
MAX_FACTOR_VAR = None
MIN_FACTOR_VAR = None
SCALE_VAR = None
MAX_RETRIES_VAR = None
AVOID_SLACK_LINES_VAR = None


def init_worker(
    base_net,
    original_pd,
    original_qd,
    original_lines,
    max_outages,
    max_factor,
    min_factor,
    scale,
    max_retries,
    avoid_slack_lines,
):
    global BASE_NET, ORIGINAL_PD, ORIGINAL_QD, ORIGINAL_LINES
    global MAX_OUTAGES_VAR, MAX_FACTOR_VAR, MIN_FACTOR_VAR, SCALE_VAR, MAX_RETRIES_VAR, AVOID_SLACK_LINES_VAR
    BASE_NET = base_net
    ORIGINAL_PD = original_pd
    ORIGINAL_QD = original_qd
    ORIGINAL_LINES = original_lines
    MAX_OUTAGES_VAR = max_outages
    MAX_FACTOR_VAR = max_factor
    MIN_FACTOR_VAR = min_factor
    SCALE_VAR = scale
    MAX_RETRIES_VAR = max_retries
    AVOID_SLACK_LINES_VAR = avoid_slack_lines


def build_one_sample(sample_idx: int) -> Optional[Dict[str, Any]]:
    seed = (sample_idx * 2654435761) % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    try:
        net = deepcopy(BASE_NET)
        net = prepare_network(net)
        outaged_indices = generate_nk_topology_delete(
            net,
            max_outages=MAX_OUTAGES_VAR,
            avoid_slack_lines=AVOID_SLACK_LINES_VAR,
        )
        outaged_line_pairs = [ORIGINAL_LINES[idx] for idx in outaged_indices]
        net = apply_nk_topology(net, outaged_indices)

        n_load = len(net.load)
        load_factors = np.random.uniform(MIN_FACTOR_VAR, MAX_FACTOR_VAR, n_load)
        current_load_p = ORIGINAL_PD * load_factors
        current_load_q = ORIGINAL_QD * load_factors
        net.load["p_mw"] = current_load_p
        net.load["q_mvar"] = current_load_q

        original_gen_p = net.gen["p_mw"].values.copy()
        original_gen_q = net.gen["q_mvar"].values.copy()
        total_orig_gen_p = max(float(original_gen_p.sum()), 1e-6)
        converged = False
        gen_scale_factor = 1.0
        num_retries = 0
        for retry in range(MAX_RETRIES_VAR):
            total_load_p = float(current_load_p.sum())
            gen_scale_factor = total_load_p / total_orig_gen_p
            net.gen["p_mw"] = original_gen_p * gen_scale_factor
            net.gen["q_mvar"] = original_gen_q * gen_scale_factor
            try:
                pp.runpp(net, check=False, check_connectivity=True)
                if net.converged:
                    converged = True
                    break
            except Exception:
                pass
            num_retries = retry + 1
            current_load_p *= SCALE_VAR
            current_load_q *= SCALE_VAR
            net.load["p_mw"] = current_load_p
            net.load["q_mvar"] = current_load_q
        if not converged:
            return None

        H, Y, aux = get_network_matrices(net)
        sparse_y = csr_matrix(Y)
        sample_metadata = {
            "sample_idx": int(sample_idx),
            "generation_seed": int(seed),
            "outaged_line_original_indices": [int(x) for x in outaged_indices],
            "outaged_line_pairs": [[int(a), int(b)] for a, b in outaged_line_pairs],
            "num_outages": int(len(outaged_indices)),
            "load_factors": load_factors.tolist(),
            "original_loads_p": ORIGINAL_PD.tolist(),
            "original_loads_q": ORIGINAL_QD.tolist(),
            "final_loads_p": current_load_p.tolist(),
            "final_loads_q": current_load_q.tolist(),
            "gen_scale_factor": float(gen_scale_factor),
            "num_retries": int(num_retries),
            "converged": bool(converged),
            "h_shape": list(H.shape),
            "y_shape": list(Y.shape),
            "y_nnz": int(sparse_y.nnz),
            "y_sparsity": float(sparse_y.nnz / (Y.shape[0] * Y.shape[1])),
            "generator_version": GENERATOR_VERSION,
            **aux,
        }
        return {
            "sample_idx": int(sample_idx),
            "H": H,
            "Y": Y,
            "metadata": sample_metadata,
        }
    except Exception:
        return None


# ============================================================
# 主进程写盘：保证恰好写入指定数量的成功样本
# ============================================================
def write_sample_package(sample_pkg: Dict[str, Any], output_dir: Path) -> None:
    sample_idx = int(sample_pkg["sample_idx"])
    H = np.asarray(sample_pkg["H"])
    Y = np.asarray(sample_pkg["Y"])
    metadata = dict(sample_pkg["metadata"])

    ensure_dir(output_dir)
    np.save(output_dir / f"H_{sample_idx}.npy", H)
    save_npz(output_dir / f"Y_{sample_idx}.npz", csr_matrix(Y))
    with open(output_dir / f"metadata_{sample_idx}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def generate_dataset_success_count(
    num_success_samples: int,
    num_workers: int,
    max_outages: int,
    max_factor: float,
    min_factor: float,
    scale: float,
    output_dir: str,
    net,
    original_pd,
    original_qd,
    original_lines,
    desc_prefix: str,
    start_idx: int = 0,
    max_retries: int = 10,
    avoid_slack_lines: bool = True,
) -> List[Dict[str, Any]]:
    output_path = Path(output_dir)
    ensure_dir(output_path)

    success = 0
    sample_idx = start_idx
    results: List[Dict[str, Any]] = []

    with Pool(
        processes=num_workers,
        initializer=init_worker,
        initargs=(
            net,
            original_pd,
            original_qd,
            original_lines,
            max_outages,
            max_factor,
            min_factor,
            scale,
            max_retries,
            avoid_slack_lines,
        ),
    ) as pool:
        with tqdm(total=num_success_samples, desc=desc_prefix) as pbar:
            while success < num_success_samples:
                batch_ids = list(range(sample_idx, sample_idx + max(1, num_workers * 4)))
                sample_idx += len(batch_ids)
                sample_pkgs = pool.map(build_one_sample, batch_ids)
                for item in sample_pkgs:
                    if item is None:
                        continue
                    write_sample_package(item, output_path)
                    results.append(item["metadata"])
                    success += 1
                    pbar.update(1)
                    if success >= num_success_samples:
                        break

    return results


# ============================================================
# 主程序
# ============================================================
def main() -> None:
    root_dir, train_dir, test_dir = prepare_dataset_layout(
        DATASET_ROOT,
        clear_root=CLEAR_DATASET_ROOT_BEFORE_GENERATION,
    )

    net = nw.case39()
    net = prepare_network(net)
    original_pd = net.load["p_mw"].to_numpy()
    original_qd = net.load["q_mvar"].to_numpy()
    original_lines = [(int(line["from_bus"]), int(line["to_bus"])) for _, line in net.line.iterrows()]

    if SAVE_NETWORK_METADATA_TO_ROOT:
        save_network_metadata(net, str(root_dir))
    if SAVE_NETWORK_METADATA_TO_SPLITS:
        save_network_metadata(net, str(train_dir))
        save_network_metadata(net, str(test_dir))

    generate_dataset_success_count(
        num_success_samples=TRAIN_NUM_SUCCESS_SAMPLES,
        num_workers=NUM_WORKERS,
        max_outages=MAX_OUTAGES,
        max_factor=MAX_FACTOR,
        min_factor=MIN_FACTOR,
        scale=SCALE,
        output_dir=str(train_dir),
        net=net,
        original_pd=original_pd,
        original_qd=original_qd,
        original_lines=original_lines,
        desc_prefix="训练集",
        start_idx=0,
        max_retries=MAX_RETRIES,
        avoid_slack_lines=AVOID_SLACK_LINES,
    )

    generate_dataset_success_count(
        num_success_samples=TEST_NUM_SUCCESS_SAMPLES,
        num_workers=NUM_WORKERS,
        max_outages=MAX_OUTAGES,
        max_factor=MAX_FACTOR,
        min_factor=MIN_FACTOR,
        scale=SCALE,
        output_dir=str(test_dir),
        net=net,
        original_pd=original_pd,
        original_qd=original_qd,
        original_lines=original_lines,
        desc_prefix="测试集",
        start_idx=TRAIN_NUM_SUCCESS_SAMPLES,
        max_retries=MAX_RETRIES,
        avoid_slack_lines=AVOID_SLACK_LINES,
    )

    print(f"数据集已生成到: {root_dir}")
    print(f"训练集目录: {train_dir}")
    print(f"测试集目录: {test_dir}")


if __name__ == "__main__":
    main()
