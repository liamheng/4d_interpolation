from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union, List
import numbers
import torch

AxisLike = Union[str, int]
PosLike = Optional[Union[int, float]]


def _clamp_int(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _resolve_index(length: int, mode: str, val: Union[int, float]) -> int:
    if length <= 0:
        raise ValueError("Axis length must be positive.")
    if mode == "int":
        idx = int(val)
    elif mode == "frac":
        frac = float(val)
        idx = int(round(frac * (length - 1)))
    else:
        raise RuntimeError(f"Unknown pos mode '{mode}'.")
    return _clamp_int(idx, 0, length - 1)


def _normalize_axis_xyz(axis: AxisLike) -> str:
    """Normalize to one of {'z','y','x'}."""
    if isinstance(axis, numbers.Integral):
        idx = int(axis)
        # Relative to xyz: 0/1/2 == z/y/x ; allow -3..-1
        if idx < 0:
            idx = 3 + idx
        if idx not in (0, 1, 2):
            raise ValueError("axis int for xyz must be in {0,1,2} or {-3,-2,-1}.")
        return ["z", "y", "x"][idx]

    if not isinstance(axis, str):
        raise TypeError("axis must be str or int")

    a = axis.strip().lower()
    if a in ("z", "d", "depth"):
        return "z"
    if a in ("y", "h", "height"):
        return "y"
    if a in ("x", "w", "width"):
        return "x"
    raise ValueError(f"Unknown xyz axis '{axis}'. Use one of z/y/x (or aliases).")


def _freeze_pos(
    pos: PosLike,
    rand_range: Tuple[float, float],
    gen: Optional[torch.Generator],
) -> Tuple[str, Union[int, float]]:
    """Return (mode, value) where mode is 'frac' or 'int'. Random only happens here (init-time)."""
    if pos is None:
        lo, hi = float(rand_range[0]), float(rand_range[1])
        if gen is None:
            frac = float((lo + (hi - lo) * torch.rand((),).item()))
        else:
            frac = float((lo + (hi - lo) * torch.rand((), generator=gen).item()))
        return "frac", frac

    if isinstance(pos, numbers.Integral):
        return "int", int(pos)

    if isinstance(pos, numbers.Real):
        frac = float(pos)
        if not (0.0 <= frac <= 1.0):
            raise ValueError("pos as float must be in [0,1].")
        return "frac", frac

    raise TypeError("pos must be int, float, or None")


@dataclass(frozen=True)
class _Plan:
    batch: bool
    channel: bool
    time: bool
    # slicing steps: list of (axis_name, mode, val)
    # axis_name: 't' or 'z'/'y'/'x'
    steps: List[Tuple[str, str, Union[int, float]]]


class SliceProcessor:
    """
    Behavior per your updated requirements:

    1) Always slice ONE of the last-three axes (D/H/W i.e. z/y/x). This is the default.
    2) If a time axis exists (i.e., spatial dims are T,D,H,W), it MUST also be sliced.
       - time position is controlled by `t_pos` (None => random frozen at init).
    3) Channel axis is optional: inputs may omit C entirely.

    Supported layouts (time, if present, is immediately before D/H/W):
      - With channel:
          * no batch: (C, D, H, W) or (C, T, D, H, W)
          * batch   : (B, C, D, H, W) or (B, C, T, D, H, W)
      - No channel:
          * no batch: (D, H, W) or (T, D, H, W)
          * batch   : (B, D, H, W) or (B, T, D, H, W)

    Inference defaults (legacy-friendly):
      - expect_batch=None -> ndim>=5 => batch=True else False
      - has_channel=None  -> prefer channel when possible (legacy).
        If your tensor is (B,D,H,W) or (B,T,D,H,W), pass has_channel=False.

    Random choices are frozen at __init__ time.
    """

    def __init__(
        self,
        expect_batch: Optional[bool] = None,
        has_channel: Optional[bool] = None,
        has_time: Optional[bool] = None,
        # xyz slicing (always performed)
        axis: Optional[AxisLike] = None,
        pos: PosLike = None,
        # time slicing (performed iff time exists)
        t_pos: PosLike = None,
        # randomness controls
        rand_range: Tuple[float, float] = (0.25, 0.75),
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        # restrict random xyz axis choices
        random_xyz_pool: Sequence[AxisLike] = ("z", "y", "x"),
    ) -> None:
        self.expect_batch = expect_batch
        self.has_channel = has_channel
        self.has_time = has_time

        if not (isinstance(rand_range, (tuple, list)) and len(rand_range) == 2):
            raise ValueError("rand_range must be a (low, high) tuple.")
        lo, hi = float(rand_range[0]), float(rand_range[1])
        if not (0.0 <= lo <= hi <= 1.0):
            raise ValueError("rand_range must satisfy 0 <= low <= high <= 1.")
        self.rand_range = (lo, hi)

        if generator is not None:
            self._gen = generator
        elif seed is not None:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(seed))
            self._gen = g
        else:
            self._gen = None

        # Freeze xyz axis: default random from pool if axis is None
        pool_xyz = [_normalize_axis_xyz(a) for a in list(random_xyz_pool)]
        seen = set()
        pool_xyz = [a for a in pool_xyz if (a not in seen and not seen.add(a))]
        if not pool_xyz:
            raise ValueError("random_xyz_pool must contain at least one valid xyz axis.")

        if axis is None:
            pick = self._randint(0, len(pool_xyz) - 1)
            self._xyz_axis = pool_xyz[pick]
        else:
            self._xyz_axis = _normalize_axis_xyz(axis)

        self._xyz_mode, self._xyz_val = _freeze_pos(pos, self.rand_range, self._gen)

        # Freeze time pos value as well (used only if time axis exists after shape inference)
        self._t_mode, self._t_val = _freeze_pos(t_pos, self.rand_range, self._gen)

        self._planned: Optional[_Plan] = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.slice_to_2d(x)

    def slice_to_2d(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        plan = self._get_or_make_plan(x.ndim)

        out = x

        # Slice time first (so last 3 dims remain D/H/W)
        if plan.time:
            t_dim = self._time_dim_index(out.ndim, plan.batch, plan.channel)
            L = out.shape[t_dim]
            idx_t = _resolve_index(L, self._t_mode, self._t_val)
            out = out.select(t_dim, idx_t)

        # Always slice one of z/y/x on the *current* tensor (still last 3 dims)
        axis_dim = self._xyz_dim_index(out.ndim, self._xyz_axis)
        L = out.shape[axis_dim]
        idx = _resolve_index(L, self._xyz_mode, self._xyz_val)
        out = out.select(axis_dim, idx)

        return out

    def describe(self) -> str:
        if self._planned is None:
            return (
                "SliceProcessor(planned=False; call once with a tensor to infer batch/channel/time; "
                f"xyz_axis={self._xyz_axis}, xyz_pos={self._xyz_mode}={self._xyz_val}, "
                f"t_pos={self._t_mode}={self._t_val}, expect_batch={self.expect_batch!r}, "
                f"has_channel={self.has_channel!r}, has_time={self.has_time!r})"
            )
        p = self._planned
        return (
            f"SliceProcessor(batch={p.batch}, channel={p.channel}, time={p.time}, "
            f"xyz_axis={self._xyz_axis}, xyz_pos={self._xyz_mode}={self._xyz_val}, "
            f"t_pos={self._t_mode}={self._t_val})"
        )

    # ----------------------------
    # Planning / inference
    # ----------------------------
    def _get_or_make_plan(self, ndim: int) -> _Plan:
        if self._planned is not None:
            return self._planned

        batch = self._infer_batch(ndim)
        channel = self._infer_channel(ndim, batch)

        remaining = ndim - (1 if batch else 0) - (1 if channel else 0)
        if remaining not in (3, 4):
            raise ValueError(
                "Unsupported tensor shape. Expected remaining spatial dims to be 3 (D,H,W) or 4 (T,D,H,W). "
                f"Got ndim={ndim}, inferred batch={batch}, channel={channel}, remaining={remaining}."
            )

        time = (remaining == 4)
        if self.has_time is not None:
            time = bool(self.has_time)
            if time and remaining != 4:
                raise ValueError(
                    f"has_time=True but remaining spatial dims is {remaining}. Expected 4 for (T,D,H,W)."
                )
            if (not time) and remaining != 3:
                raise ValueError(
                    f"has_time=False but remaining spatial dims is {remaining}. Expected 3 for (D,H,W)."
                )

        steps: List[Tuple[str, str, Union[int, float]]] = []
        if time:
            steps.append(("t", self._t_mode, self._t_val))
        steps.append((self._xyz_axis, self._xyz_mode, self._xyz_val))

        self._planned = _Plan(batch=batch, channel=channel, time=time, steps=steps)
        return self._planned

    def _infer_batch(self, ndim: int) -> bool:
        if self.expect_batch is True:
            return True
        if self.expect_batch is False:
            return False
        # legacy-friendly default:
        return ndim >= 5

    def _infer_channel(self, ndim: int, batch: bool) -> bool:
        if self.has_channel is True:
            return True
        if self.has_channel is False:
            return False

        # Default: prefer channel when possible (legacy behavior).
        # If user has tensors like (B,D,H,W) or (B,T,D,H,W), they should pass has_channel=False.
        if batch:
            return ndim >= 5
        else:
            return ndim >= 4

    # ----------------------------
    # Dimension index helpers
    # ----------------------------
    @staticmethod
    def _time_dim_index(ndim: int, batch: bool, channel: bool) -> int:
        base = 0
        if batch:
            base += 1
        if channel:
            base += 1
        return base  # (T, D, H, W) starts here

    @staticmethod
    def _xyz_dim_index(ndim: int, axis: str) -> int:
        # D/H/W are always the last three dims at the time of xyz slicing
        if axis == "z":
            return ndim - 3
        if axis == "y":
            return ndim - 2
        if axis == "x":
            return ndim - 1
        raise RuntimeError(f"Unknown xyz axis '{axis}'.")

    # ----------------------------
    # RNG helpers
    # ----------------------------
    def _randint(self, low: int, high: int) -> int:
        if low > high:
            raise ValueError("low must be <= high")
        if self._gen is None:
            return int(torch.randint(low, high + 1, (1,)).item())
        return int(torch.randint(low, high + 1, (1,), generator=self._gen).item())
