# In[]
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import load_npz

# ============================================================
# 手动配置区（不要 argparse）
# ============================================================
DATA_ROOT = "/data2/zyh/case39_samples"   # 可改成你的数据目录
OUTPUT_DIR = "./y_shape_diagnosis_outputs_gpt0419"
CHECK_TRAIN_TEST_SUBDIRS = True           # 若 DATA_ROOT 下有 train/test，则一并检查
VERIFY_ACTUAL_ARRAY_SHAPES = True         # True: 真正读取 H/Y 文件 shape；False: 仅依赖 metadata
MAX_SHOW_PER_GROUP = 20


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_dataset_dirs(data_root: str, check_train_test_subdirs: bool = True) -> List[Path]:
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"数据目录不存在: {root}")

    dataset_dirs: List[Path] = []
    train_dir = root / "train"
    test_dir = root / "test"

    if check_train_test_subdirs and train_dir.exists() and test_dir.exists():
        dataset_dirs.extend([train_dir, test_dir])
    else:
        dataset_dirs.append(root)
    return dataset_dirs


def discover_indices(data_dir: Path) -> Dict[int, Dict[str, Path]]:
    """
    收集目录中所有可能的样本索引，不要求 H/Y/metadata 三件套一定同时存在。
    """
    by_idx: Dict[int, Dict[str, Path]] = {}

    def add_path(idx: int, kind: str, p: Path) -> None:
        if idx not in by_idx:
            by_idx[idx] = {}
        by_idx[idx][kind] = p

    for fname in os.listdir(data_dir):
        p = data_dir / fname
        if fname.startswith("H_") and fname.endswith(".npy"):
            try:
                idx = int(fname[2:-4])
                add_path(idx, "H", p)
            except Exception:
                pass
        elif fname.startswith("Y_") and fname.endswith(".npz"):
            try:
                idx = int(fname[2:-4])
                add_path(idx, "Y", p)
            except Exception:
                pass
        elif fname.startswith("metadata_") and fname.endswith(".json"):
            try:
                idx = int(fname[len("metadata_"):-5])
                add_path(idx, "metadata", p)
            except Exception:
                pass
    return dict(sorted(by_idx.items(), key=lambda kv: kv[0]))


def safe_load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_actual_shapes(h_path: Optional[Path], y_path: Optional[Path]) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[str]]:
    err = None
    h_shape = None
    y_shape = None

    if h_path is not None and h_path.exists():
        try:
            h = np.load(h_path, mmap_mode="r")
            h_shape = list(h.shape)
        except Exception as e:
            err = f"H读取失败: {repr(e)}"

    if y_path is not None and y_path.exists():
        try:
            y = load_npz(y_path)
            y_shape = list(y.shape)
        except Exception as e:
            msg = f"Y读取失败: {repr(e)}"
            err = msg if err is None else err + " | " + msg

    return h_shape, y_shape, err


def normalize_shape(x) -> Optional[List[int]]:
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        out = []
        try:
            for v in x:
                out.append(int(v))
            return out
        except Exception:
            return None
    return None


def infer_sample_problem(
    row: Dict,
    expected_num_buses: Optional[int],
) -> List[str]:
    reasons: List[str] = []

    has_h = row["has_H"]
    has_y = row["has_Y"]
    has_m = row["has_metadata"]

    if not has_h or not has_y or not has_m:
        reasons.append("样本三件套不完整(H/Y/metadata缺失)")

    h_shape_meta = row["metadata_h_shape"]
    y_shape_meta = row["metadata_y_shape"]
    h_shape_act = row["actual_h_shape"]
    y_shape_act = row["actual_y_shape"]

    if has_m and row["metadata_parse_error"]:
        reasons.append("metadata 无法解析")
    if row["array_read_error"]:
        reasons.append("H/Y 文件读取异常")

    if h_shape_meta is not None and h_shape_act is not None and h_shape_meta != h_shape_act:
        reasons.append("metadata.h_shape 与实际 H.shape 不一致")
    if y_shape_meta is not None and y_shape_act is not None and y_shape_meta != y_shape_act:
        reasons.append("metadata.y_shape 与实际 Y.shape 不一致")

    if h_shape_act is not None and len(h_shape_act) != 2:
        reasons.append("H 不是二维矩阵")
    if y_shape_act is not None and len(y_shape_act) != 2:
        reasons.append("Y 不是二维矩阵")

    if y_shape_act is not None and len(y_shape_act) == 2 and y_shape_act[0] != y_shape_act[1]:
        reasons.append("Y 不是方阵")

    if h_shape_act is not None and y_shape_act is not None and len(h_shape_act) == 2 and len(y_shape_act) == 2:
        if h_shape_act[0] != y_shape_act[0]:
            reasons.append("H 节点数与 Y 尺寸不匹配")

    if expected_num_buses is not None:
        if h_shape_act is not None and len(h_shape_act) >= 1 and h_shape_act[0] != expected_num_buses:
            reasons.append(f"H 节点数 != network_metadata.num_buses({expected_num_buses})")
        if y_shape_act is not None and len(y_shape_act) == 2 and y_shape_act[0] != expected_num_buses:
            reasons.append(f"Y 尺寸 != network_metadata.num_buses({expected_num_buses})")

    if y_shape_act is not None and h_shape_act is not None and h_shape_act[0] == y_shape_act[0]:
        if expected_num_buses is not None and y_shape_act[0] != expected_num_buses:
            reasons.append("样本内部自洽，但与当前网络元数据规模不一致；疑似混入其它系统样本")

    if not reasons and y_shape_act is None and y_shape_meta is None:
        reasons.append("无法确定 Y 形状")
    return reasons


def load_network_num_buses(data_dir: Path) -> Optional[int]:
    candidates = [data_dir / "network_metadata.json", data_dir.parent / "network_metadata.json"]
    for p in candidates:
        if p.exists():
            meta = safe_load_json(p)
            if not isinstance(meta, dict):
                continue
            network_info = meta.get("network_info", {})
            if isinstance(network_info, dict) and "num_buses" in network_info:
                try:
                    return int(network_info["num_buses"])
                except Exception:
                    pass
            buses = meta.get("buses", {})
            if isinstance(buses, dict) and len(buses) > 0:
                return len(buses)
    return None


def inspect_one_dir(data_dir: Path) -> Dict:
    idx_map = discover_indices(data_dir)
    expected_num_buses = load_network_num_buses(data_dir)

    rows: List[Dict] = []
    y_shape_counter: Dict[str, int] = {}
    h_shape_counter: Dict[str, int] = {}
    complete_count = 0

    for idx, files in idx_map.items():
        h_path = files.get("H")
        y_path = files.get("Y")
        m_path = files.get("metadata")

        has_h = h_path is not None and h_path.exists()
        has_y = y_path is not None and y_path.exists()
        has_m = m_path is not None and m_path.exists()

        metadata = safe_load_json(m_path) if has_m else None
        metadata_parse_error = has_m and metadata is None

        metadata_h_shape = None
        metadata_y_shape = None
        if isinstance(metadata, dict):
            metadata_h_shape = normalize_shape(metadata.get("h_shape"))
            metadata_y_shape = normalize_shape(metadata.get("y_shape"))

        if VERIFY_ACTUAL_ARRAY_SHAPES:
            actual_h_shape, actual_y_shape, array_read_error = read_actual_shapes(h_path, y_path)
        else:
            actual_h_shape, actual_y_shape, array_read_error = None, None, None

        if has_h and has_y and has_m:
            complete_count += 1

        if actual_h_shape is not None:
            h_shape_counter[str(tuple(actual_h_shape))] = h_shape_counter.get(str(tuple(actual_h_shape)), 0) + 1
        elif metadata_h_shape is not None:
            h_shape_counter[str(tuple(metadata_h_shape))] = h_shape_counter.get(str(tuple(metadata_h_shape)), 0) + 1

        if actual_y_shape is not None:
            y_shape_counter[str(tuple(actual_y_shape))] = y_shape_counter.get(str(tuple(actual_y_shape)), 0) + 1
        elif metadata_y_shape is not None:
            y_shape_counter[str(tuple(metadata_y_shape))] = y_shape_counter.get(str(tuple(metadata_y_shape)), 0) + 1

        row = {
            "sample_idx": idx,
            "dir": str(data_dir),
            "has_H": has_h,
            "has_Y": has_y,
            "has_metadata": has_m,
            "metadata_parse_error": bool(metadata_parse_error),
            "array_read_error": array_read_error,
            "metadata_h_shape": metadata_h_shape,
            "metadata_y_shape": metadata_y_shape,
            "actual_h_shape": actual_h_shape,
            "actual_y_shape": actual_y_shape,
            "network_num_buses": expected_num_buses,
        }
        row["problems"] = infer_sample_problem(row, expected_num_buses)
        rows.append(row)

    unique_y_shapes = sorted(y_shape_counter.items(), key=lambda kv: (-kv[1], kv[0]))
    unique_h_shapes = sorted(h_shape_counter.items(), key=lambda kv: (-kv[1], kv[0]))

    majority_y_shape = unique_y_shapes[0][0] if unique_y_shapes else None
    majority_h_shape = unique_h_shapes[0][0] if unique_h_shapes else None

    abnormal_rows: List[Dict] = []
    for row in rows:
        actual_y = row["actual_y_shape"]
        actual_h = row["actual_h_shape"]

        if actual_y is not None:
            y_key = str(tuple(actual_y))
        else:
            y_key = str(tuple(row["metadata_y_shape"])) if row["metadata_y_shape"] is not None else None

        if actual_h is not None:
            h_key = str(tuple(actual_h))
        else:
            h_key = str(tuple(row["metadata_h_shape"])) if row["metadata_h_shape"] is not None else None

        if row["problems"]:
            abnormal_rows.append(row)
            continue

        if majority_y_shape is not None and y_key is not None and y_key != majority_y_shape:
            row["problems"].append(f"Y 形状偏离多数形状 {majority_y_shape}")
            abnormal_rows.append(row)
            continue

        if majority_h_shape is not None and h_key is not None and h_key != majority_h_shape:
            row["problems"].append(f"H 形状偏离多数形状 {majority_h_shape}")
            abnormal_rows.append(row)
            continue

    diagnosis: List[str] = []
    if len(y_shape_counter) <= 1:
        diagnosis.append("该目录下 Y 形状一致；若 DataLoader 仍报错，更可能是 train/val/test 混用了不同目录或运行时读了别的路径。")
    else:
        diagnosis.append("该目录下存在多种 Y 形状，说明不是单纯拓扑数值变化，而是样本规模混杂或文件不一致。")
        if expected_num_buses is not None:
            diagnosis.append(f"当前 network_metadata 指示 num_buses={expected_num_buses}。凡是 Y.shape[0] != {expected_num_buses} 的样本，都高度可疑。")
        diagnosis.append("若异常样本内部 H 与 Y 尺寸彼此匹配，但与多数样本不同，通常表示目录中混入了其它系统的数据。")
        diagnosis.append("若 metadata.y_shape 与实际 Y.shape 不一致，通常表示旧 metadata 残留、文件覆盖不完整或索引复用。")
        diagnosis.append("若缺少 H/Y/metadata 任一文件，说明目录清理不彻底或生成中断。")

    return {
        "data_dir": str(data_dir),
        "network_num_buses": expected_num_buses,
        "num_discovered_indices": len(idx_map),
        "num_complete_samples": complete_count,
        "unique_h_shapes": unique_h_shapes,
        "unique_y_shapes": unique_y_shapes,
        "majority_h_shape": majority_h_shape,
        "majority_y_shape": majority_y_shape,
        "abnormal_rows": abnormal_rows,
        "all_rows": rows,
        "diagnosis": diagnosis,
    }


def save_json(path: Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_csv(path: Path, rows: List[Dict]) -> None:
    import csv

    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx", "dir", "problems"])
        return

    keys = [
        "sample_idx",
        "dir",
        "has_H",
        "has_Y",
        "has_metadata",
        "metadata_parse_error",
        "array_read_error",
        "metadata_h_shape",
        "metadata_y_shape",
        "actual_h_shape",
        "actual_y_shape",
        "network_num_buses",
        "problems",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            row2 = dict(row)
            row2["problems"] = " | ".join(row2.get("problems", []))
            writer.writerow(row2)


def print_report(report: Dict) -> None:
    print("=" * 90)
    print(f"[目录] {report['data_dir']}")
    print(f"network_num_buses: {report['network_num_buses']}")
    print(f"发现索引数: {report['num_discovered_indices']}")
    print(f"完整样本数: {report['num_complete_samples']}")
    print(f"多数 H 形状: {report['majority_h_shape']}")
    print(f"多数 Y 形状: {report['majority_y_shape']}")
    print(f"唯一 H 形状分布: {report['unique_h_shapes']}")
    print(f"唯一 Y 形状分布: {report['unique_y_shapes']}")

    print("\n诊断结论：")
    for line in report["diagnosis"]:
        print(f"  - {line}")

    abnormal_rows = report["abnormal_rows"]
    print(f"\n异常样本数: {len(abnormal_rows)}")
    for row in abnormal_rows[:MAX_SHOW_PER_GROUP]:
        print(
            f"  sample_idx={row['sample_idx']}, "
            f"H={row['actual_h_shape'] or row['metadata_h_shape']}, "
            f"Y={row['actual_y_shape'] or row['metadata_y_shape']}, "
            f"problems={row['problems']}"
        )
    if len(abnormal_rows) > MAX_SHOW_PER_GROUP:
        print(f"  ... 其余 {len(abnormal_rows) - MAX_SHOW_PER_GROUP} 个异常样本见 CSV/JSON 报告")


def main() -> None:
    out_dir = Path(OUTPUT_DIR)
    ensure_dir(out_dir)

    dataset_dirs = find_dataset_dirs(DATA_ROOT, CHECK_TRAIN_TEST_SUBDIRS)
    all_reports = []

    for data_dir in dataset_dirs:
        report = inspect_one_dir(data_dir)
        all_reports.append(report)
        print_report(report)

        short_name = data_dir.name
        save_json(out_dir / f"y_shape_diagnosis_{short_name}_gpt0419.json", report)
        save_csv(out_dir / f"y_shape_abnormal_samples_{short_name}_gpt0419.csv", report["abnormal_rows"])

    merged = {
        "data_root": DATA_ROOT,
        "dataset_dirs": [str(p) for p in dataset_dirs],
        "reports": all_reports,
    }
    save_json(out_dir / "y_shape_diagnosis_merged_gpt0419.json", merged)

    print("\n输出文件：")
    print(f"  - {out_dir / 'y_shape_diagnosis_merged_gpt0419.json'}")
    for data_dir in dataset_dirs:
        short_name = data_dir.name
        print(f"  - {out_dir / f'y_shape_diagnosis_{short_name}_gpt0419.json'}")
        print(f"  - {out_dir / f'y_shape_abnormal_samples_{short_name}_gpt0419.csv'}")


if __name__ == "__main__":
    main()

# %%
