# In[]
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pf_data_loader import create_dataloaders


# ============================================================
# 手动配置区（不要 argparse）
# ============================================================
DATA_DIR = "/data2/zyh/case39_samples"
OUTPUT_STATS_PATH = str(Path(DATA_DIR) / "train_h_stats_global_6_gpt0421_modular.npz")

BATCH_SIZE = 128
NUM_WORKERS = 0
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
PAD_TO_MAX = True
Y_AS_DENSE = True
SHUFFLE_TRAIN = False
SEED = 42
STD_EPS = 1e-8
FEATURE_NAMES = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("compute_train_h_stats_modular")


def main() -> None:
    LOGGER.info("开始统计训练集全局逐特征 mean/std（使用 state_valid_mask 排除孤岛/失效节点）")
    train_loader, _, _ = create_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        pad_to_max=PAD_TO_MAX,
        device=None,
        num_workers=NUM_WORKERS,
        seed=SEED,
        y_as_dense=Y_AS_DENSE,
        shuffle_train=SHUFFLE_TRAIN,
    )

    sum_h = None
    sumsq_h = None
    count_h = None
    total_samples = 0
    total_state_valid_nodes = 0

    for batch in tqdm(train_loader, desc="Collect train H stats"):
        H = batch["H"].to(torch.float64)
        state_valid_mask = batch["state_valid_mask"].to(H.device)
        if H.dim() != 3:
            raise RuntimeError(f"期望 H 为 3 维张量 (B,N,F)，实际得到 shape={tuple(H.shape)}")
        mask3 = state_valid_mask.unsqueeze(-1).to(H.dtype)
        batch_sum = (H * mask3).sum(dim=(0, 1))
        batch_sumsq = ((H ** 2) * mask3).sum(dim=(0, 1))
        batch_count = mask3.sum(dim=(0, 1))
        if sum_h is None:
            sum_h = batch_sum.clone()
            sumsq_h = batch_sumsq.clone()
            count_h = batch_count.clone()
        else:
            sum_h += batch_sum
            sumsq_h += batch_sumsq
            count_h += batch_count
        total_samples += H.shape[0]
        total_state_valid_nodes += int(state_valid_mask.sum().item())

    count_safe = count_h.clamp_min(1.0)
    mean_h = sum_h / count_safe
    var_h = torch.clamp(sumsq_h / count_safe - mean_h ** 2, min=0.0)
    std_h = torch.sqrt(var_h)
    zero_std_mask = std_h <= STD_EPS
    std_safe = std_h.clone()
    std_safe[zero_std_mask] = 1.0

    output_path = Path(OUTPUT_STATS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean=mean_h.cpu().numpy().astype(np.float32),
        std=std_h.cpu().numpy().astype(np.float32),
        std_safe=std_safe.cpu().numpy().astype(np.float32),
        zero_std_mask=zero_std_mask.cpu().numpy().astype(np.bool_),
        count=count_h.cpu().numpy().astype(np.int64),
        feature_names=np.array(FEATURE_NAMES, dtype=object),
        total_samples=np.array([total_samples], dtype=np.int64),
        total_state_valid_nodes=np.array([total_state_valid_nodes], dtype=np.int64),
        train_split=np.array([TRAIN_SPLIT], dtype=np.float32),
        val_split=np.array([VAL_SPLIT], dtype=np.float32),
        seed=np.array([SEED], dtype=np.int64),
        stats_mode=np.array(["global_feature_state_valid_only"], dtype=object),
    )

    summary = {
        "output_path": str(output_path),
        "shape": list(mean_h.shape),
        "total_samples": total_samples,
        "total_state_valid_nodes": total_state_valid_nodes,
        "zero_std_count": int(zero_std_mask.sum().item()),
        "feature_names": FEATURE_NAMES,
        "stats_mode": "global_feature_state_valid_only",
    }
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    LOGGER.info("统计完成: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

# %%
