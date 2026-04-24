from __future__ import annotations

from typing import Dict, Optional, Union

import torch
import torch.nn as nn

from pf_topology_encoder import HybridNodeEdgeGraphTransformer, TensorLikeY


class PFPredictionHead(nn.Module):
    def __init__(self, dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 2 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridGTForPowerFlow(nn.Module):
    def __init__(
        self,
        node_feat_dim: int = 6,
        edge_feat_dim: int = 4,
        output_dim: int = 6,
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
        self.backbone = HybridNodeEdgeGraphTransformer(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            edge_threshold=edge_threshold,
            max_num_nodes=max_num_nodes,
            network_metadata=network_metadata,
            dynamic_depth_sampling=dynamic_depth_sampling,
        )
        self.head = PFPredictionHead(dim=d_model, output_dim=output_dim, dropout=dropout)

    def forward(
        self,
        H: torch.Tensor,
        Y: TensorLikeY,
        node_valid_mask: Optional[torch.Tensor] = None,
        feature_visible_mask: Optional[torch.Tensor] = None,
        return_backbone_outputs: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        squeeze_output = H.dim() == 2
        out = self.backbone(H, Y, node_valid_mask=node_valid_mask, feature_visible_mask=feature_visible_mask)
        pred = self.head(out["node_repr"])
        if squeeze_output:
            pred = pred.squeeze(0)
        if return_backbone_outputs:
            out["pred"] = pred
            return out
        return pred
