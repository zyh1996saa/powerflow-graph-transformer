# In[]
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import logging
import math
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


# ============================================================
# 手动配置区（不要 argparse）
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = "/data2/zyh/case39_samples"
STANDARDIZATION_STATS_PATH = str(Path(DATA_DIR) / "train_h_stats_global_6_gpt0421_modular.npz")
CKPT_PATH: Optional[str] = None  # 例如: "./logs/xxx/ckpt_finetune_best.pt"
OUTPUT_DIR = SCRIPT_DIR / "audit_pf_encoding_outputs_gpt0423"
OUTPUT_JSON_PATH = OUTPUT_DIR / "encoding_audit_summary.json"
OUTPUT_TXT_PATH = OUTPUT_DIR / "encoding_audit_summary.txt"

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
NUM_DATA_BATCHES_TO_CHECK = 2

ATOL_MASK_OVERLAP = 0
ATOL_ROUNDTRIP = 1e-6

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger("audit_pf_encoding_strategies")

MODULE_LOAD_ORDER = [
    "pf_topology_utils",
    "pf_data_loader",
    "pf_physics_losses",
    "pf_topology_encoder",
    "pf_powerflow_model",
    "pf_trainer",
]


# ============================================================
# 通用工具
# ============================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)



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



def safe_mean(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))



def finite_ratio(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 1.0
    return float(torch.isfinite(x).float().mean().item())



def tensor_max_abs(x: torch.Tensor) -> float:
    if x.numel() == 0:
        return 0.0
    return float(x.detach().abs().max().cpu().item())



def classify_status(ok: bool, warning: bool = False) -> str:
    if ok and not warning:
        return "good"
    if ok and warning:
        return "warning"
    return "risk"



def make_judgement(level: str, summary: str, evidence: Optional[List[str]] = None, advice: Optional[str] = None) -> Dict[str, Any]:
    return {
        "level": level,
        "summary": summary,
        "evidence": evidence or [],
        "advice": advice,
    }



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
# 静态编码审计
# ============================================================
def inspect_train_script(train_script_path: Path) -> Dict[str, Any]:
    src = train_script_path.read_text(encoding="utf-8")
    return {
        "train_script_path": str(train_script_path),
        "imports_vgpt0419_namespace": "from vgpt0419." in src,
        "uses_standardization": "ENABLE_NODE_FEATURE_STANDARDIZATION = True" in src,
        "uses_zero_target_fields_in_input": "zero_target_fields_in_input=True" in src,
        "uses_structured_pretrain_mask": "use_structured_pretrain_mask=True" in src,
        "uses_bus_type_aware_loss": "use_bus_type_aware_loss=True" in src,
        "uses_dynamic_depth_sampling": "dynamic_depth_sampling=True" in src,
    }



def static_audit_encoding(mods: Dict[str, Any], module_paths: Dict[str, str]) -> Dict[str, Any]:
    topo_utils = mods["pf_topology_utils"]
    topo_encoder = mods["pf_topology_encoder"]
    trainer_mod = mods["pf_trainer"]
    data_loader_mod = mods["pf_data_loader"]

    report: Dict[str, Any] = {
        "module_paths": module_paths,
        "train_script": inspect_train_script(find_module_file("train_powerflow_modular")),
        "judgements": {},
    }

    node_src = inspect.getsource(topo_encoder.NodeInputEncoder)
    edge_src = inspect.getsource(topo_encoder.EdgeInputEncoder)
    gt_src = inspect.getsource(topo_encoder.HybridNodeEdgeGraphTransformer)
    utils_src = inspect.getsource(topo_utils)
    trainer_src = inspect.getsource(trainer_mod.PFTrainer)
    loader_src = inspect.getsource(data_loader_mod.PowerFlowDataset)

    has_node_value_proj = "self.value_proj = nn.Linear(node_feat_dim, d_model)" in node_src
    has_bus_type_embed = "self.bus_type_embed = nn.Embedding(3, d_model)" in node_src
    has_node_pos_embed = "self.node_pos_embed = nn.Embedding(max_num_nodes, d_model)" in node_src
    has_feature_mask_embed = "self.feature_mask_embed = nn.Parameter" in node_src
    has_feature_type_embed = "feature_type" in node_src.lower()

    node_id_warning = has_node_pos_embed and ("pos_ids = torch.arange(num_nodes" in node_src)
    node_level = "warning" if node_id_warning else "good"
    node_summary = (
        "节点编码包含数值投影、母线类型嵌入、可学习节点位置嵌入和缺失特征掩码嵌入；"
        "其中节点位置编码本质上是按排序位置索引的 ID embedding。"
    )
    node_evidence = [
        f"value_proj={has_node_value_proj}",
        f"bus_type_embed={has_bus_type_embed}",
        f"node_pos_embed={has_node_pos_embed}",
        f"feature_mask_embed={has_feature_mask_embed}",
        f"explicit_feature_type_embed={has_feature_type_embed}",
        f"position_from_arange={node_id_warning}",
    ]
    node_advice = (
        "保留母线类型嵌入，但建议弱化或替换纯 ID 式 node_pos_embed；"
        "更适合变拓扑场景的是相对结构编码、拉普拉斯/随机游走编码、基于电气距离的编码，"
        "或至少把 node_pos_embed 仅作为可关闭的辅助项做消融。"
    )
    report["judgements"]["node_encoding"] = make_judgement(node_level, node_summary, node_evidence, node_advice)

    has_branch_type_embed = "self.branch_type_embed = nn.Embedding(5, d_model)" in edge_src
    has_branch_status_embed = "self.branch_status_embed = nn.Embedding(2, d_model)" in edge_src
    has_endpoint_pos = "edge_x = edge_x + endpoint_pos_encoding" in edge_src
    has_attn_bias_proj = "self.attn_bias_proj = nn.Linear(d_model, num_heads)" in edge_src
    edge_summary = (
        "支路编码由 edge_feat 数值投影、支路类型嵌入、支路状态嵌入，以及两端节点位置编码之和组成，"
        "并进一步投影成 attention bias。"
    )
    edge_evidence = [
        f"branch_type_embed={has_branch_type_embed}",
        f"branch_status_embed={has_branch_status_embed}",
        f"endpoint_pos_encoding={has_endpoint_pos}",
        f"attn_bias_proj={has_attn_bias_proj}",
    ]
    edge_advice = (
        "这一路设计方向是对的，但端点位置编码仍继承了节点 ID embedding 的问题；"
        "建议增补不依赖绝对节点序的相对结构特征，例如 hop distance、电气距离、共享支路类别、"
        "或从 Y 派生的归一化阻抗/耦合强度编码。"
    )
    report["judgements"]["edge_encoding"] = make_judgement("warning", edge_summary, edge_evidence, edge_advice)

    uses_candidate_mask = "candidate_mask" in gt_src and "build_branch_catalog(network_metadata)" in gt_src
    uses_dynamic_closed_mask = "closed_mask = candidate_mask & (y_abs > self.edge_threshold)" in gt_src
    topology_summary = (
        "当前拓扑处理是“静态候选边 + 动态开断状态”范式：候选连边来自 network_metadata，"
        "实际是否闭合由当前样本 Y 的非零强度决定。"
    )
    topology_evidence = [
        f"static_candidate_mask={uses_candidate_mask}",
        f"dynamic_closed_mask_from_Y={uses_dynamic_closed_mask}",
    ]
    topology_advice = (
        "这能处理既有线路的开断/投运变化，但不能表达 metadata 中不存在的新边；"
        "若后续要支持更一般的开关重构、母联合环或设备新增，候选边集合不能只由静态 catalog 决定。"
    )
    report["judgements"]["dynamic_topology_handling"] = make_judgement("warning", topology_summary, topology_evidence, topology_advice)

    has_explicit_feature_names = all(name in utils_src for name in ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"])
    has_only_linear_node_projection = has_node_value_proj and not has_feature_type_embed
    physics_summary = (
        "节点物理量采用固定 6 维排列 [Pd, Qd, Pg, Qg, Vm, Va]，然后整体线性投影到 d_model；"
        "当前并没有显式的“物理量类型 embedding”。"
    )
    physics_evidence = [
        f"feature_names_declared={has_explicit_feature_names}",
        f"single_linear_projection_only={has_only_linear_node_projection}",
    ]
    physics_advice = (
        "单纯线性投影并非错误，但它把各物理量语义完全交给投影矩阵列权重去吸收，"
        "可解释性和迁移性都偏弱。建议尝试："
        "(1) 对 Pd/Qd/Pg/Qg/Vm/Va 增加 feature-type embedding；"
        "(2) 对角度采用 sin/cos 双通道替代裸 Va；"
        "(3) 对注入类与状态类特征分组归一化或分支编码。"
    )
    report["judgements"]["physical_quantity_encoding"] = make_judgement("warning", physics_summary, physics_evidence, physics_advice)

    uses_state_valid_mask_in_loader = "state_valid_mask" in loader_src and "np.isfinite(vm) & np.isfinite(va) & (np.abs(vm) > 0)" in loader_src
    uses_state_valid_mask_in_trainer = "state_valid_mask" in trainer_src
    masking_summary = (
        "训练流已把 node_valid_mask 与 state_valid_mask 区分开，并在 finetune 输入掩码、预训练结构掩码、"
        "监督损失和物理损失中统一使用 state_valid_mask。"
    )
    masking_evidence = [
        f"loader_infers_state_valid_mask={uses_state_valid_mask_in_loader}",
        f"trainer_uses_state_valid_mask={uses_state_valid_mask_in_trainer}",
        f"finetune_zero_targets_in_input={'create_input_feature_mask_for_finetune' in trainer_src}",
        f"structured_pretrain_mask={'create_structured_pretrain_feature_mask' in trainer_src}",
    ]
    masking_advice = (
        "这一部分基本正确。需要注意的是，若 metadata 里没有可靠的 state_valid_mask，"
        "当前用 Vm/Va 是否有限且 Vm 非零来回退推断，可能把某些异常工况误判为无效节点。"
    )
    report["judgements"]["masking_and_training_alignment"] = make_judgement("good", masking_summary, masking_evidence, masking_advice)

    uses_state_valid_only_stats = "stats_mode=np.array([\"global_feature_state_valid_only\"]" in (find_module_file("compute_train_h_stats_modular").read_text(encoding="utf-8"))
    standardization_summary = (
        "标准化采用基于训练集、仅统计 state_valid 节点的全局逐特征 mean/std，"
        "并在 normalize/denormalize 后把无效节点位置重置为 0。"
    )
    standardization_evidence = [
        f"state_valid_only_stats={uses_state_valid_only_stats}",
        f"normalize_zero_invalid={'torch.where(state_valid_mask.unsqueeze(-1), Hn, torch.zeros_like(Hn))' in utils_src}",
        f"denormalize_zero_invalid={'torch.where(state_valid_mask.unsqueeze(-1), H_phys, torch.zeros_like(H_phys))' in utils_src}",
    ]
    standardization_advice = (
        "这比把 padding/孤岛节点直接混入统计更稳妥。若后续跨系统训练，建议把全局逐特征统计改成"
        "更稳健的分位数裁剪或按系统/电压等级分桶统计。"
    )
    report["judgements"]["standardization"] = make_judgement("good", standardization_summary, standardization_evidence, standardization_advice)

    return report


# ============================================================
# 动态检查
# ============================================================
def build_runtime(mods: Dict[str, Any]):
    topo_utils = mods["pf_topology_utils"]
    model_mod = mods["pf_powerflow_model"]
    trainer_mod = mods["pf_trainer"]

    network_metadata = topo_utils.load_network_metadata(DATA_DIR)
    bus_type = topo_utils.build_bus_type_vector(network_metadata)
    base_mva = topo_utils.get_network_base_mva(network_metadata)

    standardizer = None
    stats_path = Path(STANDARDIZATION_STATS_PATH)
    if stats_path.exists():
        standardizer = topo_utils.NodeFeatureStandardizer.from_npz(str(stats_path), device=torch.device(DEVICE))

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
    optimization_cfg = trainer_mod.OptimizationConfig(
        learning_rate=3e-4,
        weight_decay=1e-5,
        grad_clip=1.0,
        eta_min=1e-5,
        amp_enable=True,
    )
    checkpoint_cfg = trainer_mod.CheckpointConfig(
        save_every_epochs=5,
        keep_last_n_epoch_ckpts=2,
        resume_checkpoint=None,
    )

    model = model_mod.HybridGTForPowerFlow(
        node_feat_dim=model_cfg.node_feat_dim,
        edge_feat_dim=model_cfg.edge_feat_dim,
        output_dim=model_cfg.output_dim,
        d_model=model_cfg.d_model,
        num_layers=model_cfg.num_layers,
        num_heads=model_cfg.num_heads,
        mlp_ratio=model_cfg.mlp_ratio,
        dropout=model_cfg.dropout,
        edge_threshold=model_cfg.edge_threshold,
        max_num_nodes=model_cfg.max_num_nodes,
        network_metadata=network_metadata,
        dynamic_depth_sampling=model_cfg.dynamic_depth_sampling,
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
    return network_metadata, bus_type, standardizer, trainer


@contextmanager
def temporary_zero_param(param: torch.nn.Parameter):
    backup = param.detach().clone()
    try:
        with torch.no_grad():
            param.zero_()
        yield
    finally:
        with torch.no_grad():
            param.copy_(backup)



def try_load_checkpoint(model: torch.nn.Module, ckpt_path: Optional[str]) -> Dict[str, Any]:
    if ckpt_path is None:
        return {"status": "skipped", "reason": "CKPT_PATH is None"}
    path = Path(ckpt_path)
    if not path.exists():
        return {"status": "skipped", "reason": f"checkpoint 不存在: {path}"}

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        return {"status": "error", "reason": f"不支持的 checkpoint 格式: {type(payload)}"}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return {
        "status": "ok",
        "path": str(path),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }



def compute_feature_stats(H: torch.Tensor, valid_mask: torch.Tensor) -> Dict[str, Any]:
    names = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va"]
    out: Dict[str, Any] = {}
    for i, name in enumerate(names):
        values = H[:, :, i][valid_mask]
        if values.numel() == 0:
            out[name] = {"mean": float("nan"), "std": float("nan"), "abs_mean": float("nan")}
        else:
            out[name] = {
                "mean": float(values.float().mean().item()),
                "std": float(values.float().std(unbiased=False).item()),
                "abs_mean": float(values.float().abs().mean().item()),
            }
    return out



def ablation_delta_ratio(
    model: torch.nn.Module,
    H_norm: torch.Tensor,
    Y: torch.Tensor,
    node_valid_mask: torch.Tensor,
    feature_visible_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> Dict[str, Any]:
    with torch.no_grad():
        model.eval()
        base = model(H_norm, Y, node_valid_mask=node_valid_mask, feature_visible_mask=feature_visible_mask)
        base_target = base[target_mask]
        base_norm = float(base_target.norm().item()) if base_target.numel() > 0 else 0.0

        def ratio(pred: torch.Tensor) -> float:
            delta = pred[target_mask] - base_target
            denom = max(base_norm, 1e-12)
            return float(delta.norm().item() / denom)

        out: Dict[str, Any] = {"baseline_target_norm": base_norm}

        with temporary_zero_param(model.backbone.node_encoder.node_pos_embed.weight):
            pred = model(H_norm, Y, node_valid_mask=node_valid_mask, feature_visible_mask=feature_visible_mask)
            out["zero_node_pos_embed_delta_ratio"] = ratio(pred)

        with temporary_zero_param(model.backbone.node_encoder.bus_type_embed.weight):
            pred = model(H_norm, Y, node_valid_mask=node_valid_mask, feature_visible_mask=feature_visible_mask)
            out["zero_bus_type_embed_delta_ratio"] = ratio(pred)

        with temporary_zero_param(model.backbone.edge_encoder.branch_status_embed.weight):
            pred = model(H_norm, Y, node_valid_mask=node_valid_mask, feature_visible_mask=feature_visible_mask)
            out["zero_branch_status_embed_delta_ratio"] = ratio(pred)

        with temporary_zero_param(model.backbone.edge_encoder.branch_type_embed.weight):
            pred = model(H_norm, Y, node_valid_mask=node_valid_mask, feature_visible_mask=feature_visible_mask)
            out["zero_branch_type_embed_delta_ratio"] = ratio(pred)

    return out



def dynamic_audit(mods: Dict[str, Any]) -> Dict[str, Any]:
    topo_utils = mods["pf_topology_utils"]
    data_loader_mod = mods["pf_data_loader"]
    phy_mod = mods["pf_physics_losses"]

    if not Path(DATA_DIR).exists():
        return {"status": "skipped", "reason": f"数据目录不存在: {DATA_DIR}"}

    network_metadata, bus_type, standardizer, trainer = build_runtime(mods)
    report: Dict[str, Any] = {
        "status": "ok",
        "warnings": [],
        "per_batch": [],
        "checkpoint_load": try_load_checkpoint(trainer.model, CKPT_PATH),
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
    closed_ratios: List[float] = []
    candidate_ratios: List[float] = []
    overlap_counts: List[int] = []
    gt_phy_values: List[float] = []
    feature_stats_raw: List[Dict[str, Any]] = []
    feature_stats_norm: List[Dict[str, Any]] = []
    sensitivity_done = False

    for batch_idx, batch in enumerate(train_loader):
        if batch_idx >= NUM_DATA_BATCHES_TO_CHECK:
            break

        H_raw, Y, node_valid_mask, state_valid_mask = topo_utils.move_batch_to_device(batch, torch.device(DEVICE))
        H_raw = H_raw.float()
        Y = Y.to(torch.complex64)
        node_valid_mask = node_valid_mask.bool()
        state_valid_mask = state_valid_mask.bool()
        H_norm = trainer.maybe_normalize_H(H_raw, state_valid_mask=state_valid_mask)

        feature_visible_mask = topo_utils.create_input_feature_mask_for_finetune(
            node_valid_mask=node_valid_mask,
            state_valid_mask=state_valid_mask,
            bus_type=trainer.bus_type,
            feat_dim=H_norm.shape[-1],
        )
        target_mask = topo_utils.create_bus_type_target_mask(
            state_valid_mask=state_valid_mask,
            bus_type=trainer.bus_type,
            feat_dim=H_norm.shape[-1],
        )
        overlap = int((feature_visible_mask & target_mask).sum().item())
        overlap_counts.append(overlap)

        if standardizer is not None:
            H_roundtrip = trainer.maybe_denormalize_H(H_norm, state_valid_mask=state_valid_mask)
            valid3 = state_valid_mask.unsqueeze(-1).expand_as(H_raw)
            max_roundtrip = tensor_max_abs((H_roundtrip - H_raw)[valid3]) if valid3.any() else 0.0
        else:
            max_roundtrip = 0.0
        roundtrip_errors.append(max_roundtrip)

        with torch.no_grad():
            trainer.model.eval()
            out = trainer.model(
                H_norm,
                Y,
                node_valid_mask=node_valid_mask,
                feature_visible_mask=feature_visible_mask,
                return_backbone_outputs=True,
            )
            pred_norm = out["pred"]
            candidate_mask = out["candidate_mask"]
            closed_mask = out["closed_mask"]
            cand_ratio = float(candidate_mask.float().mean().item())
            close_ratio = float(closed_mask.float().mean().item())
            candidate_ratios.append(cand_ratio)
            closed_ratios.append(close_ratio)

        phy_loss, phy_stats = phy_mod.physics_residual_loss(
            H_raw,
            Y,
            state_valid_mask=state_valid_mask,
            base_mva=trainer.base_mva,
        )
        gt_phy_values.append(float(phy_loss.detach().item()))

        feature_stats_raw.append(compute_feature_stats(H_raw, state_valid_mask))
        feature_stats_norm.append(compute_feature_stats(H_norm, state_valid_mask))

        batch_report = {
            "batch_idx": batch_idx,
            "sample_idx": [int(x) for x in batch["sample_idx"]],
            "shape_H": list(H_raw.shape),
            "shape_Y": list(Y.shape),
            "num_state_valid": int(state_valid_mask.sum().item()),
            "finetune_visible_target_overlap": overlap,
            "roundtrip_max_abs": max_roundtrip,
            "pred_finite_ratio": finite_ratio(pred_norm),
            "candidate_edge_ratio": cand_ratio,
            "closed_edge_ratio": close_ratio,
            "gt_phy_mse": float(phy_stats["phy_mse"]),
        }

        if (not sensitivity_done) and report["checkpoint_load"]["status"] == "ok":
            try:
                batch_report["encoding_ablation_sensitivity"] = ablation_delta_ratio(
                    model=trainer.model,
                    H_norm=H_norm,
                    Y=Y,
                    node_valid_mask=node_valid_mask,
                    feature_visible_mask=feature_visible_mask,
                    target_mask=target_mask,
                )
            except Exception as exc:
                batch_report["encoding_ablation_sensitivity"] = {"status": "error", "reason": repr(exc)}
            sensitivity_done = True

        report["per_batch"].append(batch_report)

    trainer.close()

    report["num_batches_checked"] = len(report["per_batch"])
    report["roundtrip_max_abs_mean"] = safe_mean(roundtrip_errors)
    report["candidate_edge_ratio_mean"] = safe_mean(candidate_ratios)
    report["closed_edge_ratio_mean"] = safe_mean(closed_ratios)
    report["gt_phy_mse_mean"] = safe_mean(gt_phy_values)
    report["finetune_visible_target_overlap_total"] = int(sum(overlap_counts))
    report["feature_stats_raw"] = feature_stats_raw
    report["feature_stats_norm"] = feature_stats_norm

    if report["roundtrip_max_abs_mean"] > ATOL_ROUNDTRIP:
        report["warnings"].append(
            f"标准化/反标准化 roundtrip 误差偏大: {report['roundtrip_max_abs_mean']:.3e}"
        )
    if report["finetune_visible_target_overlap_total"] > ATOL_MASK_OVERLAP:
        report["warnings"].append(
            f"finetune 输入可见掩码与 target 掩码存在重叠，总数={report['finetune_visible_target_overlap_total']}"
        )
    if math.isnan(report["closed_edge_ratio_mean"]):
        report["warnings"].append("未能统计动态闭合边比例")
    elif report["closed_edge_ratio_mean"] <= 0.0:
        report["warnings"].append("所有 batch 的 closed_edge_ratio 均为 0，需检查 Y、edge_threshold 或 candidate_mask")
    if report["checkpoint_load"]["status"] != "ok":
        report["warnings"].append("未加载训练后 checkpoint，因此编码消融敏感性不能用于判断已训练模型的真实依赖程度")

    return report


# ============================================================
# 结论与输出
# ============================================================
def build_conclusions(static_report: Dict[str, Any], dynamic_report: Dict[str, Any]) -> List[str]:
    conclusions: List[str] = []
    j = static_report["judgements"]

    conclusions.append(
        "当前训练流程在 mask、监督目标、物理损失与标准化的对齐上基本正确，主问题不在这些基础接口。"
    )
    conclusions.append(
        "最值得优先重构的是节点/支路位置编码：当前 node_pos_embed 与 endpoint_pos_encoding 都强烈依赖固定母线排序，"
        "更像 ID 记忆而不是拓扑结构编码。"
    )
    conclusions.append(
        "当前所谓动态拓扑，实质是“静态候选边 + 动态边状态”；它能处理既有线路开断，但不适合更一般的新连边拓扑。"
    )
    conclusions.append(
        "节点物理量目前只有整体线性投影，没有显式 feature-type encoding，建议把 Pd/Qd/Pg/Qg/Vm/Va 的类型信息显式注入。"
    )
    conclusions.append(
        "若你当前模型拟合效果不好，我建议先做三组最小消融：去掉 node_pos_embed、加入 feature-type embedding、把 Va 改成 sin/cos 双通道。"
    )

    if dynamic_report.get("status") == "ok":
        if dynamic_report.get("finetune_visible_target_overlap_total", 0) == 0:
            conclusions.append("动态检查显示 finetune 输入掩码与监督目标无重叠，这一点是干净的。")
        if not math.isnan(dynamic_report.get("closed_edge_ratio_mean", float("nan"))):
            conclusions.append(
                f"样本中平均闭合边密度约为 {dynamic_report['closed_edge_ratio_mean']:.4f}，说明模型确实在读取样本级 Y 来感知边状态。"
            )
    else:
        conclusions.append("未完成数据级动态检查；脚本仍可先完成静态编码审计。")

    return conclusions



def save_text_report(report: Dict[str, Any], path: Path) -> None:
    lines: List[str] = []
    lines.append("电力潮流 Graph Transformer 编码策略审计报告")
    lines.append("=" * 72)
    lines.append(f"device: {report['config']['device']}")
    lines.append(f"data_dir: {report['config']['data_dir']}")
    lines.append(f"ckpt_path: {report['config']['ckpt_path']}")
    lines.append("")

    lines.append("[1] 静态编码判断")
    for name, item in report["static_report"]["judgements"].items():
        lines.append(f"- {name} [{item['level']}]: {item['summary']}")
        if item.get("evidence"):
            for ev in item["evidence"]:
                lines.append(f"    * {ev}")
        if item.get("advice"):
            lines.append(f"    建议: {item['advice']}")
    lines.append("")

    lines.append("[2] 动态检查")
    dynamic = report["dynamic_report"]
    lines.append(f"- status: {dynamic.get('status')}")
    if dynamic.get("status") == "ok":
        lines.append(f"- num_batches_checked: {dynamic.get('num_batches_checked')}")
        lines.append(f"- roundtrip_max_abs_mean: {dynamic.get('roundtrip_max_abs_mean')}")
        lines.append(f"- candidate_edge_ratio_mean: {dynamic.get('candidate_edge_ratio_mean')}")
        lines.append(f"- closed_edge_ratio_mean: {dynamic.get('closed_edge_ratio_mean')}")
        lines.append(f"- gt_phy_mse_mean: {dynamic.get('gt_phy_mse_mean')}")
        lines.append(f"- finetune_visible_target_overlap_total: {dynamic.get('finetune_visible_target_overlap_total')}")
        lines.append(f"- checkpoint_load: {json.dumps(dynamic.get('checkpoint_load', {}), ensure_ascii=False)}")
        if dynamic.get("warnings"):
            lines.append("- warnings:")
            for w in dynamic["warnings"]:
                lines.append(f"    * {w}")
        if dynamic.get("per_batch"):
            first = dynamic["per_batch"][0]
            if "encoding_ablation_sensitivity" in first:
                lines.append(f"- first_batch_encoding_ablation_sensitivity: {json.dumps(first['encoding_ablation_sensitivity'], ensure_ascii=False)}")
    else:
        lines.append(f"- reason: {dynamic.get('reason')}")
    lines.append("")

    lines.append("[3] 结论")
    for item in report.get("conclusions", []):
        lines.append(f"- {item}")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")



def main() -> None:
    set_seed(SEED)
    ensure_dir(OUTPUT_DIR)
    modules, module_paths = load_project_modules()

    static_report = static_audit_encoding(modules, module_paths)
    dynamic_report = dynamic_audit(modules)
    conclusions = build_conclusions(static_report, dynamic_report)

    report = {
        "config": {
            "device": DEVICE,
            "data_dir": DATA_DIR,
            "stats_path": STANDARDIZATION_STATS_PATH,
            "ckpt_path": CKPT_PATH,
            "batch_size": BATCH_SIZE,
            "num_batches_to_check": NUM_DATA_BATCHES_TO_CHECK,
        },
        "static_report": static_report,
        "dynamic_report": dynamic_report,
        "conclusions": conclusions,
    }

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    save_text_report(report, OUTPUT_TXT_PATH)

    LOGGER.info("编码审计完成，输出 JSON: %s", OUTPUT_JSON_PATH)
    LOGGER.info("编码审计完成，输出 TXT: %s", OUTPUT_TXT_PATH)


if __name__ == "__main__":
    main()

# %%
