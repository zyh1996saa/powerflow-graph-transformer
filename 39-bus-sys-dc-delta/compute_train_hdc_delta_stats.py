# In[]
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pf_data_loader import create_dataloaders
from pf_topology_utils import FEATURE_NAMES, build_bus_type_vector, create_bus_type_target_mask, load_network_metadata


# ============================================================
# 手动配置区（不要 argparse）
# ============================================================
DATA_DIR = "/data2/zyh/case39_samples_dc_delta"
OUTPUT_STATS_PATH = str(Path(DATA_DIR) / "train_hdc_delta_stats_global_6.npz")

BATCH_SIZE = 128
NUM_WORKERS = 0
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
PAD_TO_MAX = True
Y_AS_DENSE = True
SHUFFLE_TRAIN = False
SEED = 42
STD_EPS = 1e-8

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("compute_train_hdc_delta_stats")


def _finalize_stats(sum_x: torch.Tensor, sumsq_x: torch.Tensor, count_x: torch.Tensor):
    count_safe = count_x.clamp_min(1.0)
    mean = sum_x / count_safe
    var = torch.clamp(sumsq_x / count_safe - mean ** 2, min=0.0)
    std = torch.sqrt(var)
    zero_std_mask = (std <= STD_EPS) | (count_x <= 0)
    std_safe = std.clone()
    std_safe[zero_std_mask] = 1.0
    mean = torch.where(count_x > 0, mean, torch.zeros_like(mean))
    return mean, std, std_safe, zero_std_mask


def main() -> None:
    LOGGER.info("开始统计训练集 H_dc 输入 mean/std 与 target delta mean/std")
    network_metadata = load_network_metadata(DATA_DIR)
    bus_type = build_bus_type_vector(network_metadata)

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

    h_sum = None
    h_sumsq = None
    h_count = None
    d_sum = None
    d_sumsq = None
    d_count = None
    total_samples = 0
    total_state_valid_nodes = 0

    for batch in tqdm(train_loader, desc="Collect train H_dc/delta stats"):
        H_dc = batch["H_dc"].to(torch.float64)
        delta = batch["delta"].to(torch.float64)
        state_valid_mask = batch["state_valid_mask"].to(H_dc.device)
        if H_dc.dim() != 3:
            raise RuntimeError(f"期望 H_dc 为 3 维张量 (B,N,F)，实际得到 shape={tuple(H_dc.shape)}")

        h_mask3 = state_valid_mask.unsqueeze(-1).to(H_dc.dtype)
        h_batch_sum = (H_dc * h_mask3).sum(dim=(0, 1))
        h_batch_sumsq = ((H_dc ** 2) * h_mask3).sum(dim=(0, 1))
        h_batch_count = h_mask3.sum(dim=(0, 1)).expand(H_dc.shape[-1])

        target_mask = create_bus_type_target_mask(
            state_valid_mask=state_valid_mask,
            bus_type=bus_type.to(state_valid_mask.device),
            feat_dim=delta.shape[-1],
        ).to(delta.dtype)
        d_batch_sum = (delta * target_mask).sum(dim=(0, 1))
        d_batch_sumsq = ((delta ** 2) * target_mask).sum(dim=(0, 1))
        d_batch_count = target_mask.sum(dim=(0, 1))

        if h_sum is None:
            h_sum = h_batch_sum.clone()
            h_sumsq = h_batch_sumsq.clone()
            h_count = h_batch_count.clone()
            d_sum = d_batch_sum.clone()
            d_sumsq = d_batch_sumsq.clone()
            d_count = d_batch_count.clone()
        else:
            h_sum += h_batch_sum
            h_sumsq += h_batch_sumsq
            h_count += h_batch_count
            d_sum += d_batch_sum
            d_sumsq += d_batch_sumsq
            d_count += d_batch_count

        total_samples += H_dc.shape[0]
        total_state_valid_nodes += int(state_valid_mask.sum().item())

    H_mean, H_std, H_std_safe, H_zero_std_mask = _finalize_stats(h_sum, h_sumsq, h_count)
    delta_mean, delta_std, delta_std_safe, delta_zero_std_mask = _finalize_stats(d_sum, d_sumsq, d_count)

    output_path = Path(OUTPUT_STATS_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        H_mean=H_mean.cpu().numpy().astype(np.float32),
        H_std=H_std.cpu().numpy().astype(np.float32),
        H_std_safe=H_std_safe.cpu().numpy().astype(np.float32),
        H_zero_std_mask=H_zero_std_mask.cpu().numpy().astype(bool),
        delta_mean=delta_mean.cpu().numpy().astype(np.float32),
        delta_std=delta_std.cpu().numpy().astype(np.float32),
        delta_std_safe=delta_std_safe.cpu().numpy().astype(np.float32),
        delta_zero_std_mask=delta_zero_std_mask.cpu().numpy().astype(bool),
        H_count=h_count.cpu().numpy().astype(np.float64),
        delta_count=d_count.cpu().numpy().astype(np.float64),
        feature_names=np.asarray(FEATURE_NAMES),
        total_samples=np.asarray(total_samples),
        total_state_valid_nodes=np.asarray(total_state_valid_nodes),
        convention=np.asarray("input=H_dc; target=H_ac-H_dc on bus-type target columns"),
    )

    summary = {
        "output": str(output_path),
        "feature_names": FEATURE_NAMES,
        "H_mean": H_mean.cpu().tolist(),
        "H_std_safe": H_std_safe.cpu().tolist(),
        "delta_mean": delta_mean.cpu().tolist(),
        "delta_std_safe": delta_std_safe.cpu().tolist(),
        "H_count": h_count.cpu().tolist(),
        "delta_count": d_count.cpu().tolist(),
        "total_samples": int(total_samples),
        "total_state_valid_nodes": int(total_state_valid_nodes),
    }
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    LOGGER.info("统计完成，已写入: %s", output_path)


if __name__ == "__main__":
    main()
