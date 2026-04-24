# In[]
from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ============================================================
# 手动配置区（不要 argparse）
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = "/data2/zyh/case39_samples"
STANDARDIZATION_STATS_PATH = str(Path(DATA_DIR) / "train_h_stats_global_6_gpt0419fix2.npz")
OUTPUT_DIR = SCRIPT_DIR / "audit_pf_training_flow_outputs"
OUTPUT_JSON_PATH = OUTPUT_DIR / "audit_summary.json"
OUTPUT_TXT_PATH = OUTPUT_DIR / "audit_summary.txt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

BATCH_SIZE = 8
NUM_WORKERS = 0
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
PAD_TO_MAX = False
Y_AS_DENSE = True
SHUFFLE_TRAIN = False
CACHE_METADATA = True
CACHE_ARRAYS_IN_MEMORY = False
PIN_MEMORY = False
PERSISTENT_WORKERS = False
PREFETCH_FACTOR = None
NUM_DATA_BATCHES_TO_CHECK = 3
RUN_MODEL_FORWARD = True
RUN_SINGLE_TRAIN_STEP = True

# 阈值
ATOL_ZERO = 1e-8
ATOL_ROUNDTRIP = 1e-6
PHY_MSE_GOOD = 1e-8
PHY_MSE_WARN = 1e-4

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("audit_pf_training_flow")


# ============================================================
# 动态导入：兼容附件文件名带 (1)/(3)/(5) 的情况
# ============================================================
MODULE_LOAD_ORDER = [
    "pf_topology_utils",
    "pf_data_loader",
    "pf_physics_losses",
    "pf_topology_encoder",
    "pf_powerflow_model",
    "pf_trainer",
]


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



def find_module_file(module_name: str) -> Path:
    exact = SCRIPT_DIR / f"{module_name}.py"
    if exact.exists():
        return exact
    candidates = sorted(
        SCRIPT_DIR.glob(f"{module_name}*.py"),
        key=lambda p: (0 if p.name == f"{module_name}.py" else 1, len(p.name), p.name),
    )
    if not candidates:
        raise FileNotFoundError(f"未找到模块文件: {module_name}*.py")
    return candidates[0]



def load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {module_name} 构建 import spec: {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



def load_project_modules() -> Tuple[Dict[str, Any], Dict[str, str]]:
    modules: Dict[str, Any] = {}
    module_paths: Dict[str, str] = {}
    for module_name in MODULE_LOAD_ORDER:
        file_path = find_module_file(module_name)
        modules[module_name] = load_module_from_path(module_name, file_path)
        module_paths[module_name] = str(file_path)
        LOGGER.info("已加载模块 %s <- %s", module_name, file_path.name)
    return modules, module_paths


# ============================================================
# 统计与格式化
# ============================================================

def to_float(x: Any) -> float:
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().item())
    return float(x)



def tensor_max_abs(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.detach().abs().max().cpu().item())



def safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))



def finite_ratio(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 1.0
    return float(torch.isfinite(x).float().mean().item())



def classify_phy_mse(v: float) -> str:
    if not math.isfinite(v):
        return "nan"
    if v <= PHY_MSE_GOOD:
        return "very_small"
    if v <= PHY_MSE_WARN:
        return "acceptable_but_nonzero"
    return "large"



def save_text_report(path: Path, report: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("电力潮流训练流程审计报告")
    lines.append("=" * 64)
    lines.append(f"device: {report['config']['device']}")
    lines.append(f"data_dir: {report['config']['data_dir']}")
    lines.append("")

    lines.append("[1] 静态接口检查")
    static_checks = report.get("static_checks", {})
    for key, value in static_checks.items():
        lines.append(f"- {key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("")

    dynamic = report.get("dynamic_checks", {})
    lines.append("[2] 动态数据检查")
    lines.append(f"- status: {dynamic.get('status')}")
    if dynamic.get("status") != "ok":
        lines.append(f"- reason: {dynamic.get('reason')}")
    else:
        lines.append(f"- num_batches_checked: {dynamic.get('num_batches_checked')}")
        lines.append(f"- roundtrip_max_abs_mean: {dynamic.get('roundtrip_max_abs_mean')}")
        lines.append(f"- gt_bus_type_loss_mean: {dynamic.get('gt_bus_type_loss_mean')}")
        lines.append(f"- gt_phy_mse_mean: {dynamic.get('gt_phy_mse_mean')}")
        lines.append(f"- gt_phy_level: {dynamic.get('gt_phy_level')}")
        lines.append(f"- gt_total_finetune_loss_mean: {dynamic.get('gt_total_finetune_loss_mean')}")
        lines.append(f"- finetune_visible_target_overlap_total: {dynamic.get('finetune_visible_target_overlap_total')}")
        lines.append(f"- train_step: {json.dumps(dynamic.get('single_train_step', {}), ensure_ascii=False)}")
        if dynamic.get("warnings"):
            lines.append("- warnings:")
            for w in dynamic["warnings"]:
                lines.append(f"  * {w}")
    lines.append("")

    lines.append("[3] 结论")
    for item in report.get("conclusions", []):
        lines.append(f"- {item}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 静态检查
# ============================================================

def build_static_checks(mods: Dict[str, Any]) -> Dict[str, Any]:
    topo_utils = mods["pf_topology_utils"]
    trainer_mod = mods["pf_trainer"]
    phy_mod = mods["pf_physics_losses"]

    checks: Dict[str, Any] = {}

    move_sig = inspect.signature(topo_utils.move_batch_to_device)
    phy_sig = inspect.signature(phy_mod.physics_residual_loss)
    finetune_mask_sig = inspect.signature(topo_utils.create_input_feature_mask_for_finetune)
    standardize_sig = inspect.signature(topo_utils.NodeFeatureStandardizer.normalize)
    denorm_sig = inspect.signature(topo_utils.NodeFeatureStandardizer.denormalize)

    checks["move_batch_to_device_signature"] = str(move_sig)
    checks["physics_residual_loss_signature"] = str(phy_sig)
    checks["create_input_feature_mask_for_finetune_signature"] = str(finetune_mask_sig)
    checks["standardizer_normalize_signature"] = str(standardize_sig)
    checks["standardizer_denormalize_signature"] = str(denorm_sig)

    run_src = inspect.getsource(trainer_mod.PFTrainer._run_one_epoch)
    total_loss_src = inspect.getsource(trainer_mod.PFTrainer._compute_total_loss)
    pretrain_src = inspect.getsource(trainer_mod.PFTrainer._build_pretrain_masks)

    checks["run_one_epoch_uses_state_valid_mask"] = "state_valid_mask=state_valid_mask" in run_src
    checks["run_one_epoch_uses_feature_visible_mask"] = "feature_visible_mask=feature_visible_mask" in run_src
    checks["compute_total_loss_denormalizes_before_phy"] = "maybe_denormalize_H(pred_norm" in total_loss_src
    checks["compute_total_loss_uses_state_valid_mask_for_phy"] = "state_valid_mask=state_valid_mask" in total_loss_src
    checks["pretrain_masks_use_state_valid_mask"] = "valid_for_mask = state_valid_mask" in pretrain_src

    return checks


# ============================================================
# 动态检查
# ============================================================

def build_trainer_and_runtime(mods: Dict[str, Any], module_paths: Dict[str, str]):
    topo_utils = mods["pf_topology_utils"]
    model_mod = mods["pf_powerflow_model"]
    trainer_mod = mods["pf_trainer"]

    network_metadata = topo_utils.load_network_metadata(DATA_DIR)
    bus_type = topo_utils.build_bus_type_vector(network_metadata)
    base_mva = topo_utils.get_network_base_mva(network_metadata)

    standardizer = None
    if Path(STANDARDIZATION_STATS_PATH).exists():
        standardizer = topo_utils.NodeFeatureStandardizer.from_npz(STANDARDIZATION_STATS_PATH, device=torch.device(DEVICE))

    model = model_mod.HybridGTForPowerFlow(
        node_feat_dim=6,
        edge_feat_dim=4,
        output_dim=6,
        d_model=192,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.10,
        edge_threshold=1e-8,
        max_num_nodes=max(2048, int(bus_type.numel())),
        network_metadata=network_metadata,
        dynamic_depth_sampling=False,
    )

    optimization_cfg = trainer_mod.OptimizationConfig(
        learning_rate=3e-4,
        weight_decay=1e-5,
        grad_clip=1.0,
        eta_min=1e-5,
        amp_enable=True,
    )
    loss_cfg = trainer_mod.LossConfig(
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
    checkpoint_cfg = trainer_mod.CheckpointConfig(
        save_every_epochs=5,
        keep_last_n_epoch_ckpts=2,
        resume_checkpoint=None,
    )
    model_cfg = trainer_mod.ModelConfig(
        node_feat_dim=6,
        edge_feat_dim=4,
        output_dim=6,
        d_model=192,
        num_layers=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.10,
        edge_threshold=1e-8,
        dynamic_depth_sampling=False,
        max_num_nodes=max(2048, int(bus_type.numel())),
    )

    trainer = trainer_mod.PFTrainer(
        model=model,
        bus_type=bus_type,
        standardizer=standardizer,
        device=DEVICE,
        output_dir=OUTPUT_DIR / "tmp_trainer",
        base_mva=base_mva,
        optimization_cfg=optimization_cfg,
        loss_cfg=loss_cfg,
        checkpoint_cfg=checkpoint_cfg,
        model_cfg=model_cfg,
    )

    runtime_info = {
        "module_paths": module_paths,
        "bus_type_len": int(bus_type.numel()),
        "base_mva": float(base_mva),
        "standardization_enabled": standardizer is not None,
        "standardization_stats_path": STANDARDIZATION_STATS_PATH,
    }
    return network_metadata, bus_type, base_mva, standardizer, trainer, runtime_info



def parameter_checksum(model: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for p in model.parameters():
            total += float(p.detach().float().abs().sum().cpu().item())
    return total



def run_single_train_step_audit(
    trainer,
    topo_utils,
    H_raw: torch.Tensor,
    Y: torch.Tensor,
    node_valid_mask: torch.Tensor,
    state_valid_mask: torch.Tensor,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {"status": "skipped"}
    try:
        trainer.reset_optimizer_scheduler(total_epochs=1)
        trainer.model.train()

        H_norm = trainer.maybe_normalize_H(H_raw, state_valid_mask=state_valid_mask)
        feature_visible_mask = topo_utils.create_input_feature_mask_for_finetune(
            node_valid_mask=node_valid_mask,
            state_valid_mask=state_valid_mask,
            bus_type=trainer.bus_type,
            feat_dim=H_norm.shape[-1],
        )

        before = parameter_checksum(trainer.model)
        trainer.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=trainer.use_amp):
            pred_norm = trainer.forward_batch(
                H_norm,
                Y,
                node_valid_mask=node_valid_mask,
                feature_visible_mask=feature_visible_mask,
            )
            loss, metrics = trainer._compute_total_loss(
                pred_norm=pred_norm,
                target_norm=H_norm,
                Y=Y,
                state_valid_mask=state_valid_mask,
                stage="finetune",
                recon_mask=None,
            )
        grad_norm = trainer._backward_and_step(loss)
        after = parameter_checksum(trainer.model)
        result = {
            "status": "ok",
            "loss": float(loss.detach().item()),
            "rmse": float(metrics.get("rmse", float("nan"))),
            "phy_mse": float(metrics.get("phy_mse", float("nan"))),
            "grad_norm": float(grad_norm),
            "param_checksum_before": before,
            "param_checksum_after": after,
            "param_checksum_delta": after - before,
        }
    except Exception as exc:
        result = {
            "status": "error",
            "reason": repr(exc),
        }
    return result



def run_dynamic_checks(mods: Dict[str, Any], module_paths: Dict[str, str]) -> Dict[str, Any]:
    topo_utils = mods["pf_topology_utils"]
    data_loader_mod = mods["pf_data_loader"]
    phy_mod = mods["pf_physics_losses"]

    if not Path(DATA_DIR).exists():
        return {
            "status": "skipped",
            "reason": f"数据目录不存在: {DATA_DIR}",
        }

    network_metadata, bus_type, base_mva, standardizer, trainer, runtime_info = build_trainer_and_runtime(mods, module_paths)
    report: Dict[str, Any] = {
        "status": "ok",
        "runtime_info": runtime_info,
        "warnings": [],
        "per_batch": [],
    }

    train_loader, _, _ = data_loader_mod.create_dataloaders(
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

    roundtrip_errors: List[float] = []
    gt_masked_mse_values: List[float] = []
    gt_bus_type_loss_values: List[float] = []
    gt_phy_values: List[float] = []
    gt_phy_roundtrip_values: List[float] = []
    gt_total_finetune_values: List[float] = []
    gt_total_pretrain_values: List[float] = []
    overlap_counts: List[int] = []
    first_train_step_done = False
    single_train_step_result: Dict[str, Any] = {"status": "skipped"}

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= NUM_DATA_BATCHES_TO_CHECK:
            break

        H_raw, Y, node_valid_mask, state_valid_mask = topo_utils.move_batch_to_device(batch, torch.device(DEVICE))
        H_raw = H_raw.float()
        Y = Y.to(torch.complex64)
        node_valid_mask = node_valid_mask.bool()
        state_valid_mask = state_valid_mask.bool()
        bus_type_batch = bus_type.to(H_raw.device)

        batch_report: Dict[str, Any] = {
            "batch_idx": batch_idx,
            "sample_idx": [int(x) for x in batch["sample_idx"]],
            "shape_H": list(H_raw.shape),
            "shape_Y": list(Y.shape),
            "node_valid_subset_check": bool((~state_valid_mask | node_valid_mask).all().item()),
            "finite_ratio_H": finite_ratio(H_raw),
            "finite_ratio_Y_real": finite_ratio(Y.real),
            "finite_ratio_Y_imag": finite_ratio(Y.imag),
            "num_node_valid": int(node_valid_mask.sum().item()),
            "num_state_valid": int(state_valid_mask.sum().item()),
        }

        H_norm = trainer.maybe_normalize_H(H_raw, state_valid_mask=state_valid_mask)
        H_target_norm = H_norm.clone()

        target_mask = topo_utils.create_bus_type_target_mask(
            state_valid_mask=state_valid_mask,
            bus_type=bus_type_batch,
            feat_dim=H_norm.shape[-1],
        )
        finetune_visible_mask = topo_utils.create_input_feature_mask_for_finetune(
            node_valid_mask=node_valid_mask,
            state_valid_mask=state_valid_mask,
            bus_type=bus_type_batch,
            feat_dim=H_norm.shape[-1],
        )
        overlap = int((finetune_visible_mask & target_mask).sum().item())
        overlap_counts.append(overlap)
        batch_report["finetune_visible_target_overlap"] = overlap

        if standardizer is not None:
            H_roundtrip = trainer.maybe_denormalize_H(H_norm, state_valid_mask=state_valid_mask)
            valid3 = state_valid_mask.unsqueeze(-1).expand_as(H_raw)
            max_roundtrip = tensor_max_abs((H_roundtrip - H_raw)[valid3]) if valid3.any() else 0.0
        else:
            H_roundtrip = H_norm
            max_roundtrip = 0.0
        roundtrip_errors.append(max_roundtrip)
        batch_report["roundtrip_max_abs"] = max_roundtrip

        masked_mse = phy_mod.MaskedMSELoss()(H_target_norm, H_target_norm, target_mask)
        bus_loss, bus_stats = phy_mod.BusTypePowerFlowLoss()(H_target_norm, H_target_norm, bus_type_batch, state_valid_mask)
        gt_masked_mse_values.append(float(masked_mse.detach().item()))
        gt_bus_type_loss_values.append(float(bus_loss.detach().item()))
        batch_report["gt_masked_mse"] = float(masked_mse.detach().item())
        batch_report["gt_bus_type_loss"] = float(bus_loss.detach().item())
        batch_report["gt_bus_type_stats"] = {k: float(v) for k, v in bus_stats.items()}

        phy_loss_gt, phy_stats_gt = phy_mod.physics_residual_loss(
            H_raw,
            Y,
            state_valid_mask=state_valid_mask,
            base_mva=base_mva,
        )
        gt_phy_values.append(float(phy_loss_gt.detach().item()))
        batch_report["gt_phy_mse"] = float(phy_loss_gt.detach().item())
        batch_report["gt_phy_stats"] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in phy_stats_gt.items()}

        phy_loss_roundtrip, _ = phy_mod.physics_residual_loss(
            H_roundtrip,
            Y,
            state_valid_mask=state_valid_mask,
            base_mva=base_mva,
        )
        gt_phy_roundtrip_values.append(float(phy_loss_roundtrip.detach().item()))
        batch_report["gt_phy_mse_after_roundtrip"] = float(phy_loss_roundtrip.detach().item())

        total_finetune_gt, stats_finetune_gt = trainer._compute_total_loss(
            pred_norm=H_target_norm,
            target_norm=H_target_norm,
            Y=Y,
            state_valid_mask=state_valid_mask,
            stage="finetune",
            recon_mask=None,
        )
        gt_total_finetune_values.append(float(total_finetune_gt.detach().item()))
        batch_report["gt_total_finetune_loss"] = float(total_finetune_gt.detach().item())
        batch_report["gt_total_finetune_stats"] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in stats_finetune_gt.items()}

        pretrain_visible_mask, recon_mask = trainer._build_pretrain_masks(
            H=H_norm,
            node_valid_mask=node_valid_mask,
            state_valid_mask=state_valid_mask,
            training=False,
        )
        total_pretrain_gt, stats_pretrain_gt = trainer._compute_total_loss(
            pred_norm=H_target_norm,
            target_norm=H_target_norm,
            Y=Y,
            state_valid_mask=state_valid_mask,
            stage="pretrain",
            recon_mask=recon_mask,
        )
        gt_total_pretrain_values.append(float(total_pretrain_gt.detach().item()))
        batch_report["pretrain_recon_count"] = int(recon_mask.sum().item())
        batch_report["pretrain_visible_count"] = int(pretrain_visible_mask.sum().item())
        batch_report["gt_total_pretrain_loss"] = float(total_pretrain_gt.detach().item())
        batch_report["gt_total_pretrain_stats"] = {k: float(v) if isinstance(v, (int, float)) else v for k, v in stats_pretrain_gt.items()}

        if RUN_MODEL_FORWARD:
            with torch.no_grad():
                trainer.model.eval()
                pred_norm = trainer.forward_batch(
                    H_norm,
                    Y,
                    node_valid_mask=node_valid_mask,
                    feature_visible_mask=finetune_visible_mask,
                )
                batch_report["model_forward_ok"] = True
                batch_report["pred_norm_shape"] = list(pred_norm.shape)
                batch_report["pred_norm_finite_ratio"] = finite_ratio(pred_norm)
        else:
            batch_report["model_forward_ok"] = False

        if RUN_SINGLE_TRAIN_STEP and (not first_train_step_done):
            single_train_step_result = run_single_train_step_audit(
                trainer=trainer,
                topo_utils=topo_utils,
                H_raw=H_raw,
                Y=Y,
                node_valid_mask=node_valid_mask,
                state_valid_mask=state_valid_mask,
            )
            first_train_step_done = True

        report["per_batch"].append(batch_report)

    trainer.close()

    report["num_batches_checked"] = len(report["per_batch"])
    report["roundtrip_max_abs_mean"] = safe_mean(roundtrip_errors)
    report["gt_masked_mse_mean"] = safe_mean(gt_masked_mse_values)
    report["gt_bus_type_loss_mean"] = safe_mean(gt_bus_type_loss_values)
    report["gt_phy_mse_mean"] = safe_mean(gt_phy_values)
    report["gt_phy_mse_roundtrip_mean"] = safe_mean(gt_phy_roundtrip_values)
    report["gt_total_finetune_loss_mean"] = safe_mean(gt_total_finetune_values)
    report["gt_total_pretrain_loss_mean"] = safe_mean(gt_total_pretrain_values)
    report["gt_phy_level"] = classify_phy_mse(report["gt_phy_mse_mean"])
    report["finetune_visible_target_overlap_total"] = int(sum(overlap_counts))
    report["single_train_step"] = single_train_step_result

    if report["roundtrip_max_abs_mean"] > ATOL_ROUNDTRIP:
        report["warnings"].append(
            f"标准化/反标准化 roundtrip 误差偏大: {report['roundtrip_max_abs_mean']:.3e}"
        )
    if report["gt_bus_type_loss_mean"] > ATOL_ZERO:
        report["warnings"].append(
            f"GT 代入 bus-type supervised loss 后不接近 0: {report['gt_bus_type_loss_mean']:.3e}"
        )
    if report["finetune_visible_target_overlap_total"] > 0:
        report["warnings"].append(
            f"finetune 输入可见掩码与 target 掩码存在重叠，总数={report['finetune_visible_target_overlap_total']}"
        )
    if report["gt_phy_mse_mean"] > PHY_MSE_WARN:
        report["warnings"].append(
            f"GT 物理残差偏大: phy_mse_mean={report['gt_phy_mse_mean']:.3e}，需重点核查 H/Y/base_mva/符号约定"
        )
    elif report["gt_phy_mse_mean"] > PHY_MSE_GOOD:
        report["warnings"].append(
            f"GT 物理残差非零但仍可能可接受: phy_mse_mean={report['gt_phy_mse_mean']:.3e}"
        )
    if report["single_train_step"].get("status") != "ok":
        report["warnings"].append(
            f"单步训练检查未通过: {report['single_train_step']}"
        )
    return report


# ============================================================
# 主流程
# ============================================================

def build_conclusions(report: Dict[str, Any]) -> List[str]:
    conclusions: List[str] = []
    static_checks = report.get("static_checks", {})
    dynamic = report.get("dynamic_checks", {})

    if static_checks.get("run_one_epoch_uses_state_valid_mask"):
        conclusions.append("当前训练主循环已显式使用 state_valid_mask 做标准化、目标构造与物理损失约束。")
    else:
        conclusions.append("当前训练主循环没有明确体现 state_valid_mask 的统一使用，需要优先修复。")

    if static_checks.get("compute_total_loss_denormalizes_before_phy"):
        conclusions.append("当前总损失计算在物理损失前执行了反标准化，这一链路是正确的。")
    else:
        conclusions.append("当前总损失计算未明确在物理损失前反标准化，物理项数值可能失真。")

    if dynamic.get("status") == "ok":
        gt_sup = dynamic.get("gt_bus_type_loss_mean", float("nan"))
        gt_phy = dynamic.get("gt_phy_mse_mean", float("nan"))
        rt = dynamic.get("roundtrip_max_abs_mean", float("nan"))
        if math.isfinite(gt_sup) and gt_sup <= ATOL_ZERO:
            conclusions.append("GT 直接代入 supervised loss 基本为 0，监督损失公式本身没有明显错误。")
        else:
            conclusions.append("GT 直接代入 supervised loss 仍非零，说明 mask 或 bus-type target 逻辑存在问题。")
        if math.isfinite(rt) and rt <= ATOL_ROUNDTRIP:
            conclusions.append("标准化/反标准化 roundtrip 误差很小，标准化实现本身基本正确。")
        else:
            conclusions.append("标准化/反标准化 roundtrip 误差偏大，需要检查统计文件和遮罩逻辑。")
        if math.isfinite(gt_phy) and gt_phy <= PHY_MSE_GOOD:
            conclusions.append("GT 代入物理损失几乎为 0，H/Y/base_mva 基本一致。")
        elif math.isfinite(gt_phy) and gt_phy <= PHY_MSE_WARN:
            conclusions.append("GT 代入物理损失是小量但非零，这通常指向数值误差或数据生成与损失定义之间存在轻微偏差。")
        else:
            conclusions.append("GT 代入物理损失偏大，应优先排查数据生成阶段的 Ybus、注入功率、base_mva 和孤岛节点处理。")
        step = dynamic.get("single_train_step", {})
        if step.get("status") == "ok":
            conclusions.append("实际单步训练可以前向、反向并完成参数更新，训练 step 主干链路是可执行的。")
        else:
            conclusions.append("实际单步训练未通过，需先修复训练 step 的可执行性，再讨论模型拟合能力。")
    else:
        conclusions.append("当前环境缺少数据目录，动态数值审计被跳过；脚本已保留，放到数据环境即可直接运行。")

    return conclusions



def main() -> None:
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "config": {
            "script_dir": str(SCRIPT_DIR),
            "data_dir": DATA_DIR,
            "device": DEVICE,
            "stats_path": STANDARDIZATION_STATS_PATH,
            "batch_size": BATCH_SIZE,
            "num_data_batches_to_check": NUM_DATA_BATCHES_TO_CHECK,
            "run_model_forward": RUN_MODEL_FORWARD,
            "run_single_train_step": RUN_SINGLE_TRAIN_STEP,
        }
    }

    modules, module_paths = load_project_modules()
    report["static_checks"] = build_static_checks(modules)
    report["dynamic_checks"] = run_dynamic_checks(modules, module_paths)
    report["conclusions"] = build_conclusions(report)

    OUTPUT_JSON_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    save_text_report(OUTPUT_TXT_PATH, report)

    LOGGER.info("审计完成，JSON: %s", OUTPUT_JSON_PATH)
    LOGGER.info("审计完成，TXT : %s", OUTPUT_TXT_PATH)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

# %%
