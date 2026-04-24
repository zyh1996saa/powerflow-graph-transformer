from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.sparse import load_npz
from torch.utils.data import DataLoader, Dataset, Subset, random_split


class PowerFlowDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        indices: Optional[Sequence[int]] = None,
        return_metadata: bool = True,
        y_as_dense: bool = True,
        y_dtype: torch.dtype = torch.complex64,
        h_dtype: torch.dtype = torch.float32,
        cache_metadata: bool = True,
        cache_arrays_in_memory: bool = False,
        h_mmap_mode: Optional[str] = "r",
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")

        self.return_metadata = return_metadata
        self.y_as_dense = y_as_dense
        self.y_dtype = y_dtype
        self.h_dtype = h_dtype
        self.cache_metadata = cache_metadata
        self.cache_arrays_in_memory = cache_arrays_in_memory
        self.h_mmap_mode = h_mmap_mode
        self._metadata_cache: Dict[int, Dict[str, Any]] = {}
        self._tensor_cache: Dict[int, Dict[str, Any]] = {}

        all_indices = self._discover_indices(self.data_dir)
        if not all_indices:
            raise RuntimeError(f"在目录 {self.data_dir} 中未发现任何样本文件")

        if indices is None:
            self.indices = all_indices
        else:
            allow = {int(i) for i in indices}
            self.indices = [i for i in all_indices if i in allow]
        if not self.indices:
            raise RuntimeError("筛选后的样本列表为空")

        if self.cache_metadata and self.return_metadata:
            for sample_idx in self.indices:
                _, _, m_path = self._sample_paths(sample_idx)
                with open(m_path, "r", encoding="utf-8") as f:
                    self._metadata_cache[sample_idx] = json.load(f)

    @staticmethod
    def _discover_indices(data_dir: Path) -> List[int]:
        indices: List[int] = []
        for fname in os.listdir(data_dir):
            if not (fname.startswith("metadata_") and fname.endswith(".json")):
                continue
            try:
                idx = int(fname.split("_")[1].split(".")[0])
            except Exception:
                continue
            if (data_dir / f"H_{idx}.npy").exists() and (data_dir / f"Y_{idx}.npz").exists():
                indices.append(idx)
        return sorted(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def _sample_paths(self, sample_idx: int) -> Tuple[Path, Path, Path]:
        return (
            self.data_dir / f"H_{sample_idx}.npy",
            self.data_dir / f"Y_{sample_idx}.npz",
            self.data_dir / f"metadata_{sample_idx}.json",
        )

    def _load_metadata(self, sample_idx: int, m_path: Path) -> Dict[str, Any]:
        if self.cache_metadata and sample_idx in self._metadata_cache:
            return self._metadata_cache[sample_idx]
        with open(m_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        if self.cache_metadata:
            self._metadata_cache[sample_idx] = meta
        return meta

    @staticmethod
    def _infer_num_valid_nodes(H_np: np.ndarray, metadata: Optional[Dict[str, Any]]) -> int:
        if isinstance(metadata, dict):
            h_shape = metadata.get("h_shape", None)
            if isinstance(h_shape, (list, tuple)) and len(h_shape) >= 1:
                try:
                    return max(0, min(int(h_shape[0]), int(H_np.shape[0])))
                except Exception:
                    pass
        return int(H_np.shape[0])

    @staticmethod
    def _infer_state_valid_mask(H_np: np.ndarray, metadata: Optional[Dict[str, Any]]) -> np.ndarray:
        num_nodes = int(H_np.shape[0])
        if isinstance(metadata, dict):
            for key in ["state_valid_mask", "pf_valid_mask", "connected_bus_mask", "solver_bus_mask"]:
                value = metadata.get(key, None)
                if isinstance(value, list) and len(value) == num_nodes:
                    try:
                        return np.asarray(value, dtype=bool)
                    except Exception:
                        pass
        vm = H_np[:, 4] if H_np.shape[1] > 4 else np.zeros((num_nodes,), dtype=np.float32)
        va = H_np[:, 5] if H_np.shape[1] > 5 else np.zeros((num_nodes,), dtype=np.float32)
        return np.isfinite(vm) & np.isfinite(va) & (np.abs(vm) > 0)

    def __getitem__(self, item: int) -> Dict[str, Any]:
        sample_idx = self.indices[item]
        if self.cache_arrays_in_memory and sample_idx in self._tensor_cache:
            cached = dict(self._tensor_cache[sample_idx])
            if self.return_metadata and "metadata" not in cached:
                _, _, m_path = self._sample_paths(sample_idx)
                cached["metadata"] = self._load_metadata(sample_idx, m_path)
            return cached

        h_path, y_path, m_path = self._sample_paths(sample_idx)
        H_np = np.load(h_path, mmap_mode=self.h_mmap_mode)
        H = torch.as_tensor(np.asarray(H_np).copy(), dtype=self.h_dtype)

        if self.y_as_dense:
            Y_np = load_npz(y_path).toarray()
            Y = torch.as_tensor(Y_np, dtype=self.y_dtype)
        else:
            Y_sp = load_npz(y_path).tocoo()
            ij = np.vstack([Y_sp.row, Y_sp.col])
            Y = torch.sparse_coo_tensor(
                indices=torch.as_tensor(ij, dtype=torch.long),
                values=torch.as_tensor(Y_sp.data.astype(np.complex64), dtype=self.y_dtype),
                size=Y_sp.shape,
                dtype=self.y_dtype,
            ).coalesce()

        sample: Dict[str, Any] = {
            "H": H,
            "Y": Y,
            "sample_idx": sample_idx,
            "num_nodes": int(H.shape[0]),
        }

        metadata = None
        if self.return_metadata:
            metadata = self._load_metadata(sample_idx, m_path)
            sample["metadata"] = metadata

        num_valid_nodes = self._infer_num_valid_nodes(np.asarray(H_np), metadata)
        node_valid_mask = torch.zeros(H.shape[0], dtype=torch.bool)
        node_valid_mask[:num_valid_nodes] = True
        state_valid_mask = torch.as_tensor(self._infer_state_valid_mask(np.asarray(H_np), metadata), dtype=torch.bool)
        state_valid_mask &= node_valid_mask

        sample["node_valid_mask"] = node_valid_mask
        sample["num_valid_nodes"] = int(node_valid_mask.sum().item())
        sample["state_valid_mask"] = state_valid_mask
        sample["num_state_valid_nodes"] = int(state_valid_mask.sum().item())

        if self.cache_arrays_in_memory:
            self._tensor_cache[sample_idx] = dict(sample)
        return sample


def _pad_and_stack_dense_tensors(tensors: List[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    if not tensors:
        raise ValueError("tensors 不能为空")
    shapes = [tuple(t.shape) for t in tensors]
    if all(s == shapes[0] for s in shapes):
        return torch.stack(tensors, dim=0)
    max_dims = [max(shape[d] for shape in shapes) for d in range(len(shapes[0]))]
    padded: List[torch.Tensor] = []
    for t in tensors:
        out = torch.full(max_dims, pad_value, dtype=t.dtype)
        slices = tuple(slice(0, s) for s in t.shape)
        out[slices] = t
        padded.append(out)
    return torch.stack(padded, dim=0)


class PowerFlowCollator:
    def __init__(self, pad_to_max: bool = True):
        self.pad_to_max = pad_to_max

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        H_list = [item["H"] for item in batch]
        Y_list = [item["Y"] for item in batch]
        node_valid_mask_list = [item["node_valid_mask"] for item in batch]
        state_valid_mask_list = [item["state_valid_mask"] for item in batch]
        if any(not isinstance(y, torch.Tensor) or y.is_sparse for y in Y_list):
            raise RuntimeError("当前训练流程要求 collate 后的 Y 为 dense tensor，请保持 y_as_dense=True")

        if self.pad_to_max:
            H = _pad_and_stack_dense_tensors(H_list, pad_value=0.0)
            Y = _pad_and_stack_dense_tensors(Y_list, pad_value=0.0)
            node_valid_mask = _pad_and_stack_dense_tensors(node_valid_mask_list, pad_value=0).bool()
            state_valid_mask = _pad_and_stack_dense_tensors(state_valid_mask_list, pad_value=0).bool()
        else:
            if not all(tuple(x.shape) == tuple(H_list[0].shape) for x in H_list):
                raise RuntimeError("pad_to_max=False 时，batch 内 H 形状必须一致")
            if not all(tuple(x.shape) == tuple(Y_list[0].shape) for x in Y_list):
                raise RuntimeError("pad_to_max=False 时，batch 内 Y 形状必须一致")
            H = torch.stack(H_list, dim=0)
            Y = torch.stack(Y_list, dim=0)
            node_valid_mask = torch.stack(node_valid_mask_list, dim=0)
            state_valid_mask = torch.stack(state_valid_mask_list, dim=0)

        out: Dict[str, Any] = {
            "H": H,
            "Y": Y,
            "node_valid_mask": node_valid_mask,
            "state_valid_mask": state_valid_mask,
            "sample_idx": [item["sample_idx"] for item in batch],
            "num_nodes": [item["num_nodes"] for item in batch],
            "num_valid_nodes": [item["num_valid_nodes"] for item in batch],
            "num_state_valid_nodes": [item["num_state_valid_nodes"] for item in batch],
        }
        if "metadata" in batch[0]:
            out["metadata"] = [item.get("metadata", None) for item in batch]
        return out


def _split_dataset(dataset: Dataset, train_split: float, val_split: float, seed: int) -> Tuple[Subset, Subset, Subset]:
    total = len(dataset)
    if total < 3:
        raise ValueError(f"样本数过少，无法划分 train/val/test: total={total}")
    train_len = int(round(total * train_split))
    val_len = int(round(total * val_split))
    test_len = total - train_len - val_len
    if train_len <= 0:
        train_len = 1
    if val_len <= 0:
        val_len = 1
    if test_len <= 0:
        test_len = 1
        if train_len > val_len and train_len > 1:
            train_len -= 1
        elif val_len > 1:
            val_len -= 1
    current_total = train_len + val_len + test_len
    if current_total != total:
        train_len += total - current_total
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_len, val_len, test_len], generator=generator)


def _count_discoverable_samples(data_dir: Path) -> int:
    count = 0
    for fname in os.listdir(data_dir):
        if not (fname.startswith("metadata_") and fname.endswith(".json")):
            continue
        try:
            idx = int(fname.split("_")[1].split(".")[0])
        except Exception:
            continue
        if (data_dir / f"H_{idx}.npy").exists() and (data_dir / f"Y_{idx}.npz").exists():
            count += 1
    return count


def _resolve_dataset_dirs(data_dir: str) -> Tuple[Path, Optional[Path]]:
    root = Path(data_dir)
    train_dir = root / "train"
    test_dir = root / "test"
    if train_dir.exists() and test_dir.exists():
        return train_dir, test_dir

    legacy_test_dir = root.parent / f"{root.name}_test"
    if root.exists() and legacy_test_dir.exists():
        train_count = _count_discoverable_samples(root)
        test_count = _count_discoverable_samples(legacy_test_dir)
        if train_count > 0 and test_count > 0:
            warnings.warn(
                (
                    f"检测到旧版数据布局：训练集={root}，测试集={legacy_test_dir}。"
                    f"建议迁移到规范目录 {root / 'train'} 与 {root / 'test'}。"
                ),
                stacklevel=2,
            )
            return root, legacy_test_dir

    return root, None


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    train_split: float = 0.8,
    val_split: float = 0.1,
    pad_to_max: bool = True,
    device: Optional[str] = None,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: Optional[bool] = None,
    y_as_dense: bool = True,
    shuffle_train: bool = True,
    cache_metadata: bool = True,
    cache_arrays_in_memory: bool = False,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
):
    if device is not None:
        del device

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = bool(num_workers > 0)
    if prefetch_factor is None and num_workers > 0:
        prefetch_factor = 2

    train_dir, test_dir = _resolve_dataset_dirs(data_dir)
    collate_fn = PowerFlowCollator(pad_to_max=pad_to_max)
    common_loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
        "collate_fn": collate_fn,
    }
    if num_workers > 0 and prefetch_factor is not None:
        common_loader_kwargs["prefetch_factor"] = prefetch_factor

    if test_dir is not None:
        trainval_dataset = PowerFlowDataset(
            data_dir=str(train_dir),
            return_metadata=True,
            y_as_dense=y_as_dense,
            cache_metadata=cache_metadata,
            cache_arrays_in_memory=cache_arrays_in_memory,
        )
        train_subset, val_subset, _ = _split_dataset(
            trainval_dataset,
            train_split=train_split,
            val_split=val_split,
            seed=seed,
        )
        test_dataset = PowerFlowDataset(
            data_dir=str(test_dir),
            return_metadata=True,
            y_as_dense=y_as_dense,
            cache_metadata=cache_metadata,
            cache_arrays_in_memory=cache_arrays_in_memory,
        )
        train_loader = DataLoader(train_subset, shuffle=shuffle_train, **common_loader_kwargs)
        val_loader = DataLoader(val_subset, shuffle=False, **common_loader_kwargs)
        test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_kwargs)
        return train_loader, val_loader, test_loader

    full_dataset = PowerFlowDataset(
        data_dir=str(train_dir),
        return_metadata=True,
        y_as_dense=y_as_dense,
        cache_metadata=cache_metadata,
        cache_arrays_in_memory=cache_arrays_in_memory,
    )
    train_subset, val_subset, test_subset = _split_dataset(
        full_dataset,
        train_split=train_split,
        val_split=val_split,
        seed=seed,
    )
    train_loader = DataLoader(train_subset, shuffle=shuffle_train, **common_loader_kwargs)
    val_loader = DataLoader(val_subset, shuffle=False, **common_loader_kwargs)
    test_loader = DataLoader(test_subset, shuffle=False, **common_loader_kwargs)
    return train_loader, val_loader, test_loader
