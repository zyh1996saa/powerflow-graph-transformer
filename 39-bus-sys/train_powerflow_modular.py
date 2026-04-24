# In[]
from __future__ import annotations

import json
import logging
import os
import random
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from pf_data_loader import create_dataloaders
from pf_powerflow_model import HybridGTForPowerFlow
from pf_topology_utils import NodeFeatureStandardizer, build_bus_type_vector, ensure_dir, get_network_base_mva, load_network_metadata
from pf_trainer import (
    CheckpointConfig,
    LossConfig,
    ModelConfig,
    OptimizationConfig,
    PFTrainer,
    StageRunConfig,
)


# ============================================================
# 手动配置区（不要 argparse）
# ============================================================
DATA_DIR = "/data2/zyh/case39_samples"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 256
NUM_WORKERS = 4
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
PAD_TO_MAX = False
Y_AS_DENSE = True
SHUFFLE_TRAIN = True
CACHE_METADATA = True
CACHE_ARRAYS_IN_MEMORY = False
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 4

ENABLE_NODE_FEATURE_STANDARDIZATION = True
STANDARDIZATION_STATS_PATH = str(Path(DATA_DIR) / "train_h_stats_global_6_gpt0421_modular.npz")

MODEL_CFG = ModelConfig(
    node_feat_dim=6,
    edge_feat_dim=4,
    output_dim=6,
    d_model=192,
    num_layers=4,
    num_heads=8,
    mlp_ratio=4.0,
    dropout=0.10,
    edge_threshold=1e-8,
    dynamic_depth_sampling=True,
)

OPTIMIZATION_CFG = OptimizationConfig(
    learning_rate=3e-4,
    weight_decay=1e-5,
    grad_clip=1.0,
    eta_min=1e-5,
    amp_enable=True,
)

LOSS_CFG = LossConfig(
    use_bus_type_aware_loss=True,
    pq_loss_weight=1.0,
    pv_loss_weight=1.0,
    slack_loss_weight=1.0,
    pretrain_phy_loss_weight=0.02,
    finetune_phy_loss_weight=0.10,
    zero_target_fields_in_input=True,
    mask_rate_feature=0.08,
    use_structured_pretrain_mask=True,
)

RUN_CFG = StageRunConfig(
    pretrain=True,
    num_pretrain_epochs=120,
    num_finetune_epochs=220,
    do_final_test=True,
    print_grad_norm=False,
)

RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S") + "_pf_modular_gpt0421"
OUTPUT_ROOT = Path("./logs") / RUN_NAME

CHECKPOINT_CFG = CheckpointConfig(
    save_every_epochs=5,
    keep_last_n_epoch_ckpts=5,
    resume_checkpoint=None,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("train_powerflow_modular")


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


def log_runtime_environment(device: str) -> None:
    info = {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    }
    LOGGER.info("运行环境: %s", json.dumps(info, ensure_ascii=False))


def build_model(network_metadata, bus_type: torch.Tensor) -> HybridGTForPowerFlow:
    model_cfg = MODEL_CFG
    return HybridGTForPowerFlow(
        node_feat_dim=model_cfg.node_feat_dim,
        edge_feat_dim=model_cfg.edge_feat_dim,
        output_dim=model_cfg.output_dim,
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        mlp_ratio=model_cfg.mlp_ratio,
        dropout=model_cfg.dropout,
        edge_threshold=model_cfg.edge_threshold,
        max_num_nodes=max(len(bus_type), model_cfg.max_num_nodes),
        network_metadata=network_metadata,
        dynamic_depth_sampling=model_cfg.dynamic_depth_sampling,
    )


def build_standardizer(device: str) -> Optional[NodeFeatureStandardizer]:
    if not ENABLE_NODE_FEATURE_STANDARDIZATION:
        return None
    stats_path = Path(STANDARDIZATION_STATS_PATH)
    if not stats_path.exists():
        raise FileNotFoundError(f"标准化统计文件不存在: {stats_path}")
    return NodeFeatureStandardizer.from_npz(str(stats_path), device=torch.device(device))


def main() -> None:
    set_seed(SEED)
    log_runtime_environment(DEVICE)
    ensure_dir(OUTPUT_ROOT)

    network_metadata = load_network_metadata(DATA_DIR)
    base_mva = get_network_base_mva(network_metadata)
    LOGGER.info("使用 network base MVA: %.6f", base_mva)

    bus_type = build_bus_type_vector(network_metadata)
    model = build_model(network_metadata, bus_type)
    standardizer = build_standardizer(DEVICE)

    trainer = PFTrainer(
        model=model,
        bus_type=bus_type,
        standardizer=standardizer,
        device=DEVICE,
        output_dir=OUTPUT_ROOT,
        base_mva=base_mva,
        optimization_cfg=OPTIMIZATION_CFG,
        loss_cfg=LOSS_CFG,
        checkpoint_cfg=CHECKPOINT_CFG,
        model_cfg=MODEL_CFG,
    )

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
        shuffle_train=SHUFFLE_TRAIN,
        cache_metadata=CACHE_METADATA,
        cache_arrays_in_memory=CACHE_ARRAYS_IN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    try:
        if RUN_CFG.pretrain:
            trainer.run_pretrain(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=RUN_CFG.num_pretrain_epochs,
                start_epoch=0,
                print_grad_norm=RUN_CFG.print_grad_norm,
            )
        trainer.run_finetune(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=RUN_CFG.num_finetune_epochs,
            start_epoch=0,
            print_grad_norm=RUN_CFG.print_grad_norm,
        )
        trainer.save_history_json()

        if RUN_CFG.do_final_test and test_loader is not None:
            metrics = trainer.evaluate(test_loader, stage="finetune")
            LOGGER.info("[Final Test] %s", json.dumps(metrics, ensure_ascii=False))
            with open(OUTPUT_ROOT / "final_test_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
