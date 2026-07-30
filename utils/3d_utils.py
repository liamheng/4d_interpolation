from __future__ import annotations
from typing import Optional, Union, Tuple
import numbers

try:
    import torch
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

import numpy as np
import random


AxisLike = Union[str, int]


class SliceProcessor:
    """
    A processor that extracts a single 2D slice from 3D medical volumes (batched),
    with the axis and slice position determined (optionally at random) and *frozen*
    at instantiation time for consistent slicing across many volumes.

    Input shape:  (B, C, D, H, W)
    Output shape:
        axis='z' (slice along D) -> (B, C, H, W)
        axis='y' (slice along H) -> (B, C, D, W)
        axis='x' (slice along W) -> (B, C, D, H)
    """

    _AXIS_NAME_TO_DIM = {
        # Logical axes -> spatial dims of (D,H,W)
        "z": 2, "d": 2,  # depth
        "y": 3, "h": 3,  # height
        "x": 4, "w": 4,  # width
    }

    _INT_TO_AXIS = {
        0: "z",  # allow 0/1/2 as aliases for (z,y,x)
        1: "y",
        2: "x",
        -3: "z",  # torch/numpy negative indexing analogs for (D,H,W)
        -2: "y",
        -1: "x",
    }

    def __init__(
        self,
        axis: Optional[AxisLike] = None,
        pos: Optional[Union[float, int]] = None,
        rand_range: Tuple[float, float] = (0.25, 0.75),
        seed: Optional[int] = None,
    ):
        """
        Args:
            axis: 'x'/'y'/'z' 或 'd'/'h'/'w' 或 0/1/2（分别表示 z,y,x）。
                  若为 None 则随机从 {'x','y','z'} 选取并固定。
            pos:  切片位置。
                  - float: 相对比例 [0,1]（推荐；用于不同尺寸也保持相对一致）
                  - int:   绝对索引（针对给定长度）
                  若为 None 则在 rand_range 内随机均匀采样（float）并固定。
            rand_range: 随机相对比例的范围（闭区间），默认 [0.25, 0.75]。
            seed: 随机种子（仅用于 __init__ 的一次性随机选择）。
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # 1) axis 解析与冻结
        if axis is None:
            self.axis: str = random.choice(["x", "y", "z"])
        else:
            self.axis = self._normalize_axis(axis)

        # 2) 位置解析与冻结
        if pos is None:
            lo, hi = rand_range
            assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0 and lo <= hi, \
                "rand_range must be within [0,1] and lo <= hi"
            self._pos_is_fraction = True
            self._pos_value = float(np.random.uniform(lo, hi))
        else:
            if isinstance(pos, numbers.Real) and not isinstance(pos, bool):
                if float(pos) == int(pos):  # looks like integer
                    self._pos_is_fraction = False
                    self._pos_value = int(pos)      # absolute index
                else:
                    self._pos_is_fraction = True
                    self._pos_value = float(pos)    # fraction
            else:
                raise TypeError("pos must be int (index) or float (fraction in [0,1]).")

        # 冻结的元信息快照
        self._frozen = True

    # -------- public API --------

    def __call__(self, x: Union[np.ndarray, "torch.Tensor"]) -> Union[np.ndarray, "torch.Tensor"]:
        """
        Slice a batched 5D volume: x with shape (B, C, D, H, W).
        Returns a 4D batch of 2D slices with shape depending on self.axis.
        """
        return self.slice_batch(x)

    def slice_batch(self, x: Union[np.ndarray, "torch.Tensor"]) -> Union[np.ndarray, "torch.Tensor"]:
        self._check_ndim5(x)
        axis_dim = self._AXIS_NAME_TO_DIM[self.axis]
        L = self._size_of_dim(x, axis_dim)
        idx = self._resolve_index(L)  # integer index

        if self._is_torch(x):
            slicer = [slice(None)] * x.ndim
            slicer[axis_dim] = idx
            out = x[tuple(slicer)]
            return out  # torch: (B,C,H,W) / (B,C,D,W) / (B,C,D,H)
        else:
            slicer = [slice(None)] * x.ndim
            slicer[axis_dim] = idx
            out = x[tuple(slicer)]
            return out  # numpy

    def describe(self) -> str:
        """Return a human-friendly summary of the frozen selection."""
        mode = "fraction" if self._pos_is_fraction else "index"
        return f"SliceProcessor(axis='{self.axis}', pos_{mode}={self._pos_value})"

    def resolved_index_for_length(self, length: int) -> int:
        """Given a spatial length along the chosen axis, tell the concrete integer index."""
        return self._resolve_index(length)

    # -------- helpers --------

    @staticmethod
    def _is_torch(x) -> bool:
        return _HAS_TORCH and isinstance(x, torch.Tensor)

    @staticmethod
    def _size_of_dim(x, dim: int) -> int:
        return int(x.shape[dim])

    @staticmethod
    def _check_ndim5(x) -> None:
        if not hasattr(x, "ndim") or x.ndim != 5:
            raise ValueError(f"input must be 5D (B,C,D,H,W), but got ndim={getattr(x, 'ndim', None)} and shape={getattr(x, 'shape', None)}")

    @classmethod
    def _normalize_axis(cls, axis: AxisLike) -> str:
        if isinstance(axis, str):
            a = axis.lower()
            if a in cls._AXIS_NAME_TO_DIM:
                # map 'd','h','w' to 'z','y','x'
                if a in ("d", "h", "w"):
                    return {"d": "z", "h": "y", "w": "x"}[a]
                return a  # 'x','y','z'
            raise ValueError("axis must be one of {'x','y','z','d','h','w'} (case-insensitive)")
        elif isinstance(axis, numbers.Integral):
            if axis in cls._INT_TO_AXIS:
                return cls._INT_TO_AXIS[int(axis)]
            raise ValueError("int axis must be in {0,1,2,-3,-2,-1} mapping to z,y,x respectively")
        else:
            raise TypeError("axis must be str or int")

    def _resolve_index(self, length: int) -> int:
        if length <= 0:
            raise ValueError("length must be positive")
        if self._pos_is_fraction:
            # round to nearest valid index in [0, length-1]
            idx = int(round(self._pos_value * (length - 1)))
            idx = max(0, min(length - 1, idx))
            return idx
        else:
            # absolute index -> clamp into valid range
            idx = int(self._pos_value)
            if not (0 <= idx < length):
                idx = max(0, min(length - 1, idx))
            return idx


# -----------------------
# Demo / Quick test
# -----------------------
if __name__ == "__main__":
    B, C, D, H, W = 2, 1, 16, 20, 24

    # Numpy demo
    vol_np = np.arange(B * C * D * H * W, dtype=np.float32).reshape(B, C, D, H, W)
    proc1 = SliceProcessor(seed=42)  # random axis & fraction in [0.25,0.75], frozen
    out_np = proc1(vol_np)
    print(proc1.describe())
    print("numpy out shape:", out_np.shape)  # depends on chosen axis

    # Torch demo (if available)
    if _HAS_TORCH:
        vol_t = torch.randn(B, C, D, H, W)
        # explicit axis and fraction (e.g., axis='z', pos=0.5 -> middle slice along D)
        proc2 = SliceProcessor(axis='z', pos=0.5)
        out_t = proc2(vol_t)
        print(proc2.describe())
        print("torch out shape:", tuple(out_t.shape))
