from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

from pf_topology_utils import build_branch_catalog, build_bus_type_vector, get_sorted_bus_ids

TensorLikeY = Union[torch.Tensor, np.ndarray, sp.spmatrix]


def y_to_dense_complex(Y: TensorLikeY, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(Y, torch.Tensor):
        if Y.is_sparse:
            Y = Y.to_dense()
        if not torch.is_complex(Y):
            raise TypeError("Y must be complex-valued.")
        return Y.to(device=device) if device is not None else Y
    if sp.issparse(Y):
        arr = Y.toarray()
    else:
        arr = np.asarray(Y)
    if not np.iscomplexobj(arr):
        raise TypeError("Y must be complex-valued.")
    return torch.as_tensor(arr, dtype=torch.complex64, device=device)


@dataclass
class StaticTopologyBuffers:
    candidate_mask: torch.Tensor
    branch_type_matrix: torch.Tensor
    bus_type_ids: torch.Tensor


class NodeInputEncoder(nn.Module):
    def __init__(self, node_feat_dim: int, d_model: int, max_num_nodes: int, dropout: float = 0.0):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.d_model = d_model
        self.max_num_nodes = max_num_nodes
        self.value_proj = nn.Linear(node_feat_dim, d_model)
        self.feature_mask_embed = nn.Parameter(torch.empty(node_feat_dim, d_model))
        self.bus_type_embed = nn.Embedding(3, d_model)
        self.node_pos_embed = nn.Embedding(max_num_nodes, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.feature_mask_embed, std=0.02)
        nn.init.normal_(self.bus_type_embed.weight, std=0.02)
        nn.init.normal_(self.node_pos_embed.weight, std=0.02)

    def forward(
        self,
        H: torch.Tensor,
        bus_type_ids: torch.Tensor,
        feature_visible_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, num_nodes, feat_dim = H.shape
        if feat_dim != self.node_feat_dim:
            raise ValueError(f"H last dim must be {self.node_feat_dim}, got {feat_dim}")
        if num_nodes > self.max_num_nodes:
            raise ValueError(f"num_nodes={num_nodes} exceeds max_num_nodes={self.max_num_nodes}")

        if feature_visible_mask is None:
            feature_visible_mask = torch.ones_like(H, dtype=torch.bool, device=H.device)
        else:
            feature_visible_mask = feature_visible_mask.to(device=H.device, dtype=torch.bool)
            if feature_visible_mask.shape != H.shape:
                raise ValueError(f"feature_visible_mask shape {tuple(feature_visible_mask.shape)} != H shape {tuple(H.shape)}")

        vis = feature_visible_mask.to(H.dtype)
        H_visible = H * vis
        mask_bias = torch.matmul((1.0 - vis), self.feature_mask_embed)

        pos_ids = torch.arange(num_nodes, dtype=torch.long, device=H.device)
        node_x = self.value_proj(H_visible)
        node_x = node_x + mask_bias
        node_x = node_x + self.bus_type_embed(bus_type_ids[:num_nodes]).unsqueeze(0)
        node_x = node_x + self.node_pos_embed(pos_ids).unsqueeze(0)
        return self.dropout(self.out_norm(node_x))


class EdgeInputEncoder(nn.Module):
    def __init__(self, edge_feat_dim: int, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.edge_feat_dim = edge_feat_dim
        self.value_proj = nn.Linear(edge_feat_dim, d_model)
        self.branch_type_embed = nn.Embedding(5, d_model)
        self.branch_status_embed = nn.Embedding(2, d_model)
        self.attn_bias_proj = nn.Linear(d_model, num_heads)
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.normal_(self.branch_type_embed.weight, std=0.02)
        nn.init.normal_(self.branch_status_embed.weight, std=0.02)

    def forward(
        self,
        edge_feat: torch.Tensor,
        branch_type_ids: torch.Tensor,
        branch_status_ids: torch.Tensor,
        endpoint_pos_encoding: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_x = self.value_proj(edge_feat)
        edge_x = edge_x + self.branch_type_embed(branch_type_ids).unsqueeze(0)
        edge_x = edge_x + self.branch_status_embed(branch_status_ids)
        edge_x = edge_x + endpoint_pos_encoding
        edge_x = self.out_norm(edge_x)
        edge_x = edge_x * candidate_mask.unsqueeze(-1).to(edge_x.dtype)
        edge_x = self.dropout(edge_x)
        attn_bias = self.attn_bias_proj(edge_x).permute(0, 3, 1, 2).contiguous()
        attn_bias = attn_bias * candidate_mask.unsqueeze(1).to(attn_bias.dtype)
        return edge_x, attn_bias


class DynamicLocalMessagePassing(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.node_proj = nn.Linear(dim, dim)
        self.edge_proj = nn.Linear(dim, dim)
        self.msg_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_x: torch.Tensor,
        closed_mask: torch.Tensor,
        node_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        node_j = self.node_proj(node_x).unsqueeze(1)
        edge_term = self.edge_proj(edge_x)
        raw_msg = torch.tanh(node_j + edge_term)
        gate = torch.sigmoid(self.gate_proj(edge_term))
        msg = self.msg_proj(raw_msg * gate) * closed_mask.unsqueeze(-1).to(raw_msg.dtype)
        agg = msg.sum(dim=2)
        deg = closed_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(agg.dtype)
        agg = agg / deg
        delta = self.out_proj(self.dropout(agg))
        delta = delta * node_valid_mask.unsqueeze(-1).to(delta.dtype)
        return delta


class GlobalGraphAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, node_valid_mask: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, num_nodes, dim = x.shape
        qkv = self.qkv(x).view(bsz, num_nodes, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            scores = scores + attn_bias
        pair_mask = node_valid_mask.unsqueeze(1) & node_valid_mask.unsqueeze(2)
        scores = scores.masked_fill(~pair_mask.unsqueeze(1), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(bsz, num_nodes, dim)
        out = self.out_proj(out)
        return self.proj_drop(out)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.local_norm = nn.LayerNorm(dim)
        self.local_mp = DynamicLocalMessagePassing(dim=dim, dropout=dropout)
        self.global_norm = nn.LayerNorm(dim)
        self.global_attn = GlobalGraphAttention(dim=dim, num_heads=num_heads, attn_drop=dropout, proj_drop=dropout)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_x: torch.Tensor,
        edge_x: torch.Tensor,
        closed_mask: torch.Tensor,
        node_valid_mask: torch.Tensor,
        attn_bias: torch.Tensor,
    ) -> torch.Tensor:
        node_x = node_x + self.dropout(self.local_mp(self.local_norm(node_x), edge_x=edge_x, closed_mask=closed_mask, node_valid_mask=node_valid_mask))
        node_x = node_x + self.dropout(self.global_attn(self.global_norm(node_x), node_valid_mask=node_valid_mask, attn_bias=attn_bias))
        node_x = node_x + self.dropout(self.ffn(self.ffn_norm(node_x)))
        node_x = node_x * node_valid_mask.unsqueeze(-1).to(node_x.dtype)
        return node_x


class HybridNodeEdgeGraphTransformer(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 6,
        edge_feat_dim: int = 4,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        edge_threshold: float = 1e-8,
        max_num_nodes: int = 2048,
        network_metadata: Optional[Dict] = None,
        dynamic_depth_sampling: bool = True,
    ):
        super().__init__()
        if network_metadata is None:
            raise ValueError("network_metadata is required")
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.edge_threshold = float(edge_threshold)
        self.max_num_nodes = int(max_num_nodes)
        self.dynamic_depth_sampling = bool(dynamic_depth_sampling)

        bus_ids = get_sorted_bus_ids(network_metadata)
        branch_catalog = build_branch_catalog(network_metadata)
        bus_type_ids = build_bus_type_vector(network_metadata, device=None)

        candidate_mask = torch.zeros((len(bus_ids), len(bus_ids)), dtype=torch.bool)
        branch_type_matrix = torch.zeros((len(bus_ids), len(bus_ids)), dtype=torch.long)
        for rec in branch_catalog:
            i = int(rec["u_pos"])
            j = int(rec["v_pos"])
            t = int(rec["type_id"])
            candidate_mask[i, j] = True
            candidate_mask[j, i] = True
            branch_type_matrix[i, j] = t
            branch_type_matrix[j, i] = t

        self.register_buffer("candidate_mask", candidate_mask, persistent=True)
        self.register_buffer("branch_type_matrix", branch_type_matrix, persistent=True)
        self.register_buffer("bus_type_ids", bus_type_ids, persistent=True)

        self.node_encoder = NodeInputEncoder(node_feat_dim=node_feat_dim, d_model=d_model, max_num_nodes=max_num_nodes, dropout=dropout)
        self.edge_encoder = EdgeInputEncoder(edge_feat_dim=edge_feat_dim, d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.blocks = nn.ModuleList([HybridBlock(dim=d_model, num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def _build_edge_inputs(self, Y_dense: torch.Tensor, node_valid_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        bsz, num_nodes, _ = Y_dense.shape
        if self.candidate_mask.shape[0] < num_nodes:
            raise ValueError(f"candidate_mask size={tuple(self.candidate_mask.shape)} smaller than num_nodes={num_nodes}")
        eye = torch.eye(num_nodes, dtype=torch.bool, device=Y_dense.device).unsqueeze(0)
        valid_pair_mask = node_valid_mask.unsqueeze(1) & node_valid_mask.unsqueeze(2)
        candidate_mask = self.candidate_mask[:num_nodes, :num_nodes].to(Y_dense.device).unsqueeze(0) & valid_pair_mask & (~eye)

        Y_off = Y_dense.masked_fill(eye, 0.0 + 0.0j)
        y_abs = torch.abs(Y_off)
        closed_mask = candidate_mask & (y_abs > self.edge_threshold)
        branch_status = closed_mask.long()

        real_part = Y_off.real.float()
        imag_part = Y_off.imag.float()
        abs_part = y_abs.float()
        status_part = branch_status.float()
        edge_feat = torch.stack([real_part, imag_part, abs_part, status_part], dim=-1)
        edge_feat = edge_feat * candidate_mask.unsqueeze(-1).to(edge_feat.dtype)

        branch_type_ids = self.branch_type_matrix[:num_nodes, :num_nodes].to(Y_dense.device)
        return {
            "edge_feat": edge_feat,
            "candidate_mask": candidate_mask,
            "closed_mask": closed_mask,
            "branch_type_ids": branch_type_ids,
            "branch_status_ids": branch_status,
        }

    def forward(
        self,
        H: torch.Tensor,
        Y: TensorLikeY,
        node_valid_mask: Optional[torch.Tensor] = None,
        feature_visible_mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if H.dim() == 2:
            H = H.unsqueeze(0)
        Y_dense = y_to_dense_complex(Y, device=H.device)
        if Y_dense.dim() == 2:
            Y_dense = Y_dense.unsqueeze(0)
        if H.dim() != 3 or Y_dense.dim() != 3:
            raise ValueError("H must be (B,N,F) and Y must be (B,N,N)")
        if H.shape[0] != Y_dense.shape[0] or H.shape[1] != Y_dense.shape[1] or Y_dense.shape[1] != Y_dense.shape[2]:
            raise ValueError(f"H shape {tuple(H.shape)} and Y shape {tuple(Y_dense.shape)} are inconsistent")

        bsz, num_nodes, _ = H.shape
        if node_valid_mask is None:
            node_valid_mask = torch.ones((bsz, num_nodes), dtype=torch.bool, device=H.device)
        else:
            node_valid_mask = node_valid_mask.to(H.device).bool()
            if node_valid_mask.dim() == 1:
                node_valid_mask = node_valid_mask.unsqueeze(0)

        if feature_visible_mask is not None:
            feature_visible_mask = feature_visible_mask.to(device=H.device, dtype=torch.bool)
            if feature_visible_mask.dim() == 2:
                feature_visible_mask = feature_visible_mask.unsqueeze(0)
            if feature_visible_mask.shape != H.shape:
                raise ValueError(f"feature_visible_mask shape {tuple(feature_visible_mask.shape)} != H shape {tuple(H.shape)}")
            feature_visible_mask = feature_visible_mask & node_valid_mask.unsqueeze(-1)

        node_x = self.node_encoder(H, self.bus_type_ids.to(H.device), feature_visible_mask=feature_visible_mask)
        node_x = node_x * node_valid_mask.unsqueeze(-1).to(node_x.dtype)

        edge_inputs = self._build_edge_inputs(Y_dense, node_valid_mask=node_valid_mask)
        pos_ids = torch.arange(num_nodes, dtype=torch.long, device=H.device)
        pos_encoding = self.node_encoder.node_pos_embed(pos_ids).unsqueeze(0).expand(bsz, -1, -1)
        endpoint_pos = pos_encoding.unsqueeze(2) + pos_encoding.unsqueeze(1)
        edge_x, attn_bias = self.edge_encoder(
            edge_feat=edge_inputs["edge_feat"],
            branch_type_ids=edge_inputs["branch_type_ids"],
            branch_status_ids=edge_inputs["branch_status_ids"],
            endpoint_pos_encoding=endpoint_pos,
            candidate_mask=edge_inputs["candidate_mask"],
        )

        if self.training and self.dynamic_depth_sampling and self.num_layers > 1:
            active_layers = int(torch.randint(1, self.num_layers + 1, (1,), device=H.device).item())
        else:
            active_layers = self.num_layers

        for layer_idx in range(active_layers):
            node_x = self.blocks[layer_idx](
                node_x=node_x,
                edge_x=edge_x,
                closed_mask=edge_inputs["closed_mask"],
                node_valid_mask=node_valid_mask,
                attn_bias=attn_bias,
            )

        node_x = self.final_norm(node_x)
        node_x = node_x * node_valid_mask.unsqueeze(-1).to(node_x.dtype)
        out = {
            "node_repr": node_x,
            "edge_repr": edge_x,
            "closed_mask": edge_inputs["closed_mask"],
            "candidate_mask": edge_inputs["candidate_mask"],
            "attn_bias": attn_bias,
            "active_layers": torch.tensor(active_layers, device=H.device),
        }
        if return_aux:
            out["branch_type_ids"] = edge_inputs["branch_type_ids"]
            out["branch_status_ids"] = edge_inputs["branch_status_ids"]
            out["feature_visible_mask"] = feature_visible_mask
        return out
