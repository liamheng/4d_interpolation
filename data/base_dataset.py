# -*- coding: utf-8 -*-
"""
BaseDataset with per-sample *paired* transforms.

Key ideas:
- For every sample, call BaseDataset.new_transform_pair(opt) to get a *fresh* pair of
  transform instances. Random parameters are sampled once inside the instance and
  reused for image & label → strictly parallel geometry on the same sample.
- 2D external tensor: C×W×H
- 3D external tensor: C×D×H×W
- 2D tokens DO NOT carry a "_2d" suffix; 3D tokens keep "*3d".
- Intensity/noise ops apply to image only; labels only receive geometry ops
  (image interpolates bilinear/trilinear; label uses nearest).
- Spacing is original metadata from the file, not augmentation params.
  - load_nifti(path) → (tensor C×D×H×W, spacing=(sz,sy,sx), affine)
  - spacing_to_str((sz,sy,sx)) → "sz,sy,sx" for output dict convenience.
- 3D transforms that need spacing (e.g., resample3d) accept & return spacing.

Dependencies: numpy, torch, (optional) Pillow for 2D, nibabel for NIfTI I/O.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Union

import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import nibabel as nib

    _NIB_OK = True
except Exception:
    _NIB_OK = False


# =========================================================
# Abstract BaseDataset
# =========================================================
class BaseDataset(Dataset, ABC):
    def __init__(self, opt):
        self.opt = opt
        self.root = getattr(opt, "data_dirname", None)

    @staticmethod
    def modify_commandline_options(parser, is_train: bool):
        # keep CLI minimal on purpose; all transform params live in `opt.preprocess`
        return parser

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def __getitem__(self, index: int):
        ...

    # -----------------------------------------------------
    # Public: per-sample paired transform factory
    # -----------------------------------------------------
    @staticmethod
    def new_transform_pair(opt):
        """
        Return a new pair-transform instance per sample.
        - For 2D: returns PairTransform2D(preprocess_str)
        - For 3D: returns PairTransform3D(preprocess_str)
        """
        preprocess = getattr(opt, "preprocess", "")
        if getattr(opt, "is_3d", False):
            return PairTransform3D(preprocess)
        else:
            return PairTransform2D(preprocess)

    # =====================================================
    # I/O helpers
    # =====================================================
    @staticmethod
    def load_nifti(path: str) -> Tuple[torch.Tensor, Tuple[float, float, float], np.ndarray]:
        """
        Read .nii/.nii.gz → unify to RAS.
        Returns:
          tensor:  C×D×H×W  (float32)
          spacing: (sz, sy, sx) in mm (Z,Y,X mapped to D,H,W)
          affine:  4×4 numpy array
        """
        if not _NIB_OK:
            raise ImportError("nibabel is required for load_nifti but not installed")
        nii = nib.load(path)
        nii = nib.as_closest_canonical(nii)  # standardize to RAS (X=LR, Y=PA, Z=IS)
        arr = np.asarray(nii.get_fdata(dtype=np.float32))  # X×Y×Z[×C?]
        sx, sy, sz = nii.header.get_zooms()[:3]

        # to C×D×H×W
        if arr.ndim == 3:
            vol = np.transpose(arr, (2, 1, 0))[None, ...]  # 1×D×H×W
        elif arr.ndim == 4:
            # heuristic: if last dim small, assume channels at last → X×Y×Z×C
            vol = np.transpose(arr, (3, 2, 1, 0))  # C×D×H×W
            # if arr.shape[-1] <= 8:
            #     vol = np.transpose(arr, (3, 2, 1, 0))  # C×D×H×W
            # else:
            #     assume channels-first C×X×Y×Z
                # vol = np.transpose(arr, (0, 3, 2, 1))  # C×D×H×W
        else:
            raise ValueError(f"Unsupported NIfTI ndim={arr.ndim} for {path}")

        t = torch.from_numpy(vol.astype(np.float32))
        spacing = (float(sz), float(sy), float(sx))  # map Z,Y,X → D,H,W
        return t, spacing, nii.affine

    @staticmethod
    def spacing_to_str(spacing: Tuple[float, float, float]) -> str:
        sz, sy, sx = spacing
        return f"{sz},{sy},{sx}"


# =========================================================
# Per-sample 2D paired transform
# =========================================================
class PairTransform2D:
    """
    2D paired transform for *one* sample.
    - Build from a preprocess string once
    - Sample random params lazily at first apply, then cache (fixed for this instance)
    - Apply geometry to both image & label (modes differ); apply intensity to image only
    External tensor layout: C×W×H
    """

    def __init__(self, preprocess: str):
        self.geom_ops: List[_Geom2D] = []
        self.img_ops: List[_ImgOnly2D] = []
        self._parse(preprocess)

    # ---------- public API ----------
    def apply_image(self, x: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
        t = _to_tensor_wh(x)
        for op in self.geom_ops:
            op.sample_if_needed(t.shape)
            t = op.apply(t, mode="bilinear")
        for op in self.img_ops:
            t = op.apply(t)
        return t

    def apply_label(self, y: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
        t = _to_tensor_wh(y)
        for op in self.geom_ops:
            op.sample_if_needed(t.shape)
            t = op.apply(t, mode="nearest")
        return t

    # ---------- parser with mini docs ----------
    def _parse(self, preprocess: str):
        tokens = _as_tokens(preprocess)
        for tk in tokens:
            key, *args = tk.split("_")
            key = key.lower()

            # resize_W_H : resize to (W,H). ex: resize_256_256
            if key == "resize":
                W, H = _parse_ints(args, expect=(2,))
                self.geom_ops.append(_Resize2D((W, H)))

            # center_crop_W_H : center crop; zero-pad if smaller. ex: center_crop_320_320
            elif key == "center" and len(args) >= 1 and args[0] == "crop":
                W, H = _parse_ints(args[1:], expect=(2,))
                self.geom_ops.append(_CenterCrop2D((W, H)))

            # random_crop_W_H : ONE random crop; zero-pad if needed. ex: random_crop_256_256
            elif key == "random" and len(args) >= 1 and args[0] == "crop":
                W, H = _parse_ints(args[1:], expect=(2,))
                self.geom_ops.append(_RandomCrop2D((W, H)))

            # flip_xy : random flip on x/y (width/height), p=0.5 each. ex: flip_xy / flip_x / flip_y
            elif key == "flip":
                axes = _parse_axes_2d(args)
                self.geom_ops.append(_RandomFlip2D(axes))

            # rot_any_deg : rotate by angle in [-deg,+deg]. ex: rot_any_10 (default 10)
            elif key == "rot" and len(args) >= 1 and args[0] == "any":
                deg = float(args[1]) if len(args) >= 2 else 10.0
                self.geom_ops.append(_RotateAny2D(deg))

            # rot_rightangle : rotate by k*90°. ex: rot_rightangle
            elif key == "rot" and len(args) >= 1 and args[0] == "rightangle":
                self.geom_ops.append(_RotateRightAngle2D())

            # minmax : per-channel min-max to [0,1]. ex: minmax
            elif key == "minmax":
                self.img_ops.append(_MinMax2D())

            # linear_a_b : x' = a*x + b. ex: linear_1.2_-0.1
            elif key == "linear":
                a, b = _parse_floats(args, expect=(2,))
                self.img_ops.append(_Linear2D(a, b))

            # poisson[_I0] : Poisson noise in [0,1]. ex: poisson_8000  (default 8000)
            elif key == "poisson":
                I0 = _parse_floats(args, default=[8000.0], expect=(0, 1))[0]
                self.img_ops.append(_Poisson2D(I0))

            # awgn[_sigma] : additive Gaussian. ex: awgn_0.006 (default 0.006)
            elif key == "awgn":
                sigma = _parse_floats(args, default=[0.006], expect=(0, 1))[0]
                self.img_ops.append(_AWGN2D(sigma))

            # speckle[_k] : multiplicative Gamma noise (mean 1). ex: speckle_9 (default 9)
            elif key == "speckle":
                k = _parse_floats(args, default=[9.0], expect=(0, 1))[0]
                self.img_ops.append(_Speckle2D(k))

            else:
                # unknown token → ignore
                pass


# =========================================================
# Per-sample 3D paired transform
# =========================================================
class PairTransform3D:
    """
    3D paired transform for *one* sample.
    - Build from preprocess string once
    - Random params sampled lazily, cached per instance
    - apply_image / apply_label accept and return spacing when necessary
    External tensor layout: C×D×H×W
    """

    def __init__(self, preprocess: str):
        self.geom_ops: List[_Geom3D] = []
        self.img_ops: List[_ImgOnly3D] = []
        self._parse(preprocess)

    # ---------- public API ----------
    def apply_image(self, x: Union[torch.Tensor, np.ndarray], spacing: Optional[Tuple[float, float, float]] = None
                    ) -> Tuple[torch.Tensor, Optional[Tuple[float, float, float]]]:
        no_spacing = spacing is None
        t = _to_tensor4(x)
        sp = spacing
        for op in self.geom_ops:
            op.sample_if_needed(t.shape)
            t, sp = op.apply(t, mode="trilinear", spacing=sp)
        for op in self.img_ops:
            t = op.apply(t)
        if no_spacing: return t
        return t, sp

    def apply_label(self, y: Union[torch.Tensor, np.ndarray], spacing: Optional[Tuple[float, float, float]] = None
                    ) -> Tuple[torch.Tensor, Optional[Tuple[float, float, float]]]:
        t = _to_tensor4(y)
        sp = spacing
        for op in self.geom_ops:
            op.sample_if_needed(t.shape)
            t, sp = op.apply(t, mode="nearest", spacing=sp)
        return t, sp

    # 一次性对 image 与 label 同步应用，避免你在子类里手动两次传spacing
    def apply_pair(self,
                   image: Union[torch.Tensor, np.ndarray],
                   label: Optional[Union[torch.Tensor, np.ndarray]] = None,
                   spacing: Optional[Tuple[float, float, float]] = None):
        img_t, sp = self.apply_image(image, spacing)
        lab_t = None
        if label is not None:
            lab_t, sp = self.apply_label(label, sp)
        return img_t, lab_t, sp

    # ---------- parser with mini docs ----------
    def _parse(self, preprocess: str):
        # FIX: accept list/tuple/str via _as_tokens (was preprocess.split(","))
        tokens = _as_tokens(preprocess)
        for tk in tokens:
            key, *args = tk.split("_")
            key = key.lower()

            # resample3d_tz_ty_tx : resample to target spacing (mm) if spacing is provided; else identity.
            #                        ex: resample3d_1.0_1.0_1.0
            if key == "resample3d":
                tz, ty, tx = _parse_floats(args, expect=(3,))
                self.geom_ops.append(_ResampleToSpacing3D((tz, ty, tx)))

            # setspacing3d_sz_sy_sx : write spacing=(sz,sy,sx) without resampling (for non-NIfTI sources).
            #                          ex: setspacing3d_2.5_1.2_1.2
            elif key == "setspacing3d":
                sz, sy, sx = _parse_floats(args, expect=(3,))
                self.geom_ops.append(_SetSpacing3D((sz, sy, sx)))

            # resize3d_D_H_W : resize to exact voxel grid (D,H,W). ex: resize3d_64_128_128
            elif key == "resize3d":
                D, H, W = _parse_ints(args, expect=(3,))
                self.geom_ops.append(_Resize3D((D, H, W)))

            # center_crop3d_D_H_W : center crop; zero-pad if smaller. ex: center_crop3d_96_160_160
            elif key == "center" and len(args) >= 1 and args[0] == "crop3d":
                D, H, W = _parse_ints(args[1:], expect=(3,))
                self.geom_ops.append(_CenterCrop3D((D, H, W)))

            # randompatch3d_D_H_W[_N] : ONE random patch; zero-pad if needed. ex: randompatch3d_64_128_128
            elif key == "randompatch3d":
                if len(args) == 3:
                    D, H, W = _parse_ints(args, expect=(3,))
                else:
                    D, H, W, _ = _parse_ints(args, expect=(4,))
                self.geom_ops.append(_RandomPatch3D((D, H, W)))

            # flip3d_axes : random flip subset of {x,y,z}, p=0.5 each. ex: flip3d_xyz / flip3d_xz
            elif key == "flip3d":
                axes = _parse_axes_3d(args)
                self.geom_ops.append(_RandomFlip3D(axes))

            # rot3d_any_dx_dy_dz : per-axis angle in [-d,+d]°, default 10,10,10. ex: rot3d_any_10_10_10
            elif key == "rot3d" and len(args) >= 1 and args[0] == "any":
                if len(args) >= 4:
                    dx, dy, dz = _parse_floats(args[1:], expect=(3,))
                else:
                    dx, dy, dz = 10.0, 10.0, 10.0
                self.geom_ops.append(_RotateAny3D((dx, dy, dz)))

            # rot3d_rightangle_axes : pick one axis and rotate by k*90°. ex: rot3d_rightangle_z
            elif key == "rot3d" and len(args) >= 1 and args[0] == "rightangle":
                axes = _parse_axes_3d(args[1:])
                self.geom_ops.append(_RotateRightAngle3D(axes))

            # minmax3d : per-channel min-max to [0,1]. ex: minmax3d
            elif key == "minmax3d":
                self.img_ops.append(_MinMax3D())

            # linear3d_a_b : x' = a*x + b. ex: linear3d_1.2_-0.1
            elif key == "linear3d":
                a, b = _parse_floats(args, expect=(2,))
                self.img_ops.append(_Linear3D(a, b))

            # poisson3d[_I0] : Poisson noise in [0,1]. ex: poisson3d_8000
            elif key == "poisson3d":
                I0 = _parse_floats(args, default=[8000.0], expect=(0, 1))[0]
                self.img_ops.append(_Poisson3D(I0))

            # awgn3d[_sigma[_sigcorr]] : additive Gaussian + optional spatial corr (voxel sigma). ex: awgn3d_0.006_1.0
            elif key == "awgn3d":
                if len(args) == 0:
                    sigma, sigcorr = 0.006, 1.0
                elif len(args) == 1:
                    sigma = float(args[0])
                    sigcorr = 1.0
                else:
                    sigma, sigcorr = _parse_floats(args, expect=(2,))
                self.img_ops.append(_AWGN3D(sigma=sigma, sigma_corr=sigcorr))

            # speckle3d[_k[_sigcorr]] : multiplicative Gamma; corr in log-domain; mean≈1. ex: speckle3d_9_1.0
            elif key == "speckle3d":
                if len(args) == 0:
                    k, sigcorr = 9.0, 1.0
                elif len(args) == 1:
                    k = float(args[0])
                    sigcorr = 1.0
                else:
                    k, sigcorr = _parse_floats(args, expect=(2,))
                self.img_ops.append(_Speckle3D(k=k, sigma_corr=sigcorr))

            else:
                # unknown token → ignore
                pass


# =========================================================
# 2D op implementations (stateful geom + image-only intensity)
# External layout: C×W×H  (internally convert to C×H×W for interpolate)
# =========================================================
def _to_tensor_wh(x: Union[torch.Tensor, np.ndarray, Image.Image]) -> torch.Tensor:
    if isinstance(x, Image.Image):
        arr = np.asarray(x)
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = arr.astype(np.float32)
        if arr.dtype == np.uint8:
            arr = arr / 255.0
        chw = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(np.transpose(chw, (0, 2, 1)))
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[None, ...]
        return torch.from_numpy(arr)
    if isinstance(x, torch.Tensor):
        t = x
        if t.ndim == 2:
            t = t.unsqueeze(0)
        return t.to(dtype=torch.float32)
    raise TypeError(f"Unsupported input type: {type(x)}")


def _wh_to_hw(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1).contiguous()


def _hw_to_wh(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1).contiguous()


class _Geom2D:
    def __init__(self):
        self._sampled = False

    def sample_if_needed(self, shape_cwh: Tuple[int, int, int]):
        if not self._sampled:
            self._sample(shape_cwh)
            self._sampled = True

    def _sample(self, shape_cwh: Tuple[int, int, int]):
        pass

    def apply(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        raise NotImplementedError


class _ImgOnly2D:
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _Resize2D(_Geom2D):
    # resize_W_H : resize to (W,H). ex: resize_256_256
    def __init__(self, size_wh: Tuple[int, int]):
        super().__init__()
        self.W, self.H = int(size_wh[0]), int(size_wh[1])

    def apply(self, x, mode="bilinear"):
        chw = _wh_to_hw(x)
        chw = F.interpolate(chw.unsqueeze(0), size=(self.H, self.W), mode=mode, align_corners=False).squeeze(0)
        return _hw_to_wh(chw)


class _CenterCrop2D(_Geom2D):
    # center_crop_W_H : center crop; zero-pad if smaller. ex: center_crop_320_320
    def __init__(self, size_wh: Tuple[int, int]):
        super().__init__()
        self.W, self.H = int(size_wh[0]), int(size_wh[1])
        self.x0 = self.y0 = None

    def _sample(self, shape_cwh):
        C, W, H = shape_cwh
        pw = max(0, self.W - W)
        ph = max(0, self.H - H)
        if pw or ph:
            # pad when apply
            self.x0 = None
            self.y0 = None
        else:
            self.x0 = (W - self.W) // 2
            self.y0 = (H - self.H) // 2

    def apply(self, x, mode="bilinear"):
        C, W, H = x.shape
        tw, th = self.W, self.H
        pw = max(0, tw - W)
        ph = max(0, th - H)
        if pw or ph:
            chw = _wh_to_hw(x).unsqueeze(0)
            chw = F.pad(chw, (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2), "constant", 0.0)
            x = _hw_to_wh(chw.squeeze(0))
            C, W, H = x.shape
            self.x0 = (W - tw) // 2
            self.y0 = (H - th) // 2
        return x[:, self.x0:self.x0 + tw, self.y0:self.y0 + th]


class _RandomCrop2D(_Geom2D):
    # random_crop_W_H : ONE random crop; zero-pad if needed. ex: random_crop_256_256
    def __init__(self, size_wh: Tuple[int, int]):
        super().__init__()
        self.W, self.H = int(size_wh[0]), int(size_wh[1])
        self.x0 = self.y0 = None

    def _sample(self, shape_cwh):
        C, W, H = shape_cwh
        tw, th = self.W, self.H
        pw = max(0, tw - W)
        ph = max(0, th - H)
        if pw or ph:
            # do padding at apply(), then recompute a centered start
            self.x0 = None
            self.y0 = None
        else:
            self.x0 = 0 if W == tw else random.randint(0, W - tw)
            self.y0 = 0 if H == th else random.randint(0, H - th)

    def apply(self, x, mode="bilinear"):
        C, W, H = x.shape
        tw, th = self.W, self.H
        pw = max(0, tw - W)
        ph = max(0, th - H)
        if pw or ph:
            chw = _wh_to_hw(x).unsqueeze(0)
            chw = F.pad(chw, (0, pw, 0, ph), "constant", 0.0)
            x = _hw_to_wh(chw.squeeze(0))
            C, W, H = x.shape
            self.x0 = (W - tw) // 2
            self.y0 = (H - th) // 2
        return x[:, self.x0:self.x0 + tw, self.y0:self.y0 + th]


class _RandomFlip2D(_Geom2D):
    # flip_xy : random flip on x/y (width/height), p=0.5 each. ex: flip_xy / flip_x / flip_y
    def __init__(self, axes: Sequence[str]):
        super().__init__()
        a = set(axes) if axes else {"x", "y"}
        self.use_x = "x" in a
        self.use_y = "y" in a
        self.fx = self.fy = False

    def _sample(self, shape_cwh):
        if self.use_x:
            self.fx = random.random() < 0.5
        if self.use_y:
            self.fy = random.random() < 0.5

    def apply(self, x, mode="bilinear"):
        if self.fx:
            x = torch.flip(x, dims=[1])
        if self.fy:
            x = torch.flip(x, dims=[2])
        return x


class _RotateAny2D(_Geom2D):
    # rot_any_deg : rotate random angle in [-deg,+deg]. ex: rot_any_10 (default 10)
    def __init__(self, deg: float):
        super().__init__()
        self.deg = float(deg)
        self.angle = 0.0

    def _sample(self, shape_cwh):
        if abs(self.deg) > 1e-6:
            self.angle = random.uniform(-self.deg, self.deg)

    def apply(self, x, mode="bilinear"):
        if abs(self.angle) < 1e-6:
            return x
        chw = _wh_to_hw(x)
        C, H, W = chw.shape
        rad = math.radians(self.angle)
        c, s = math.cos(rad), math.sin(rad)
        A = chw.new_tensor([[c, -s, 0.0],
                            [s, c, 0.0]])
        grid = F.affine_grid(A.unsqueeze(0), size=(1, C, H, W), align_corners=False)
        y = F.grid_sample(chw.unsqueeze(0), grid, mode=mode, padding_mode='zeros', align_corners=False).squeeze(0)
        return _hw_to_wh(y)


class _RotateRightAngle2D(_Geom2D):
    # rot_rightangle : rotate by k*90°. ex: rot_rightangle
    def __init__(self):
        super().__init__()
        self.k = 0

    def _sample(self, shape_cwh):
        self.k = random.randint(0, 3)

    def apply(self, x, mode="bilinear"):
        if self.k == 0:
            return x
        chw = _wh_to_hw(x)
        chw = torch.rot90(chw, k=self.k, dims=(1, 2))
        return _hw_to_wh(chw)


class _MinMax2D(_ImgOnly2D):
    # minmax : per-channel min-max to [0,1]. ex: minmax
    def apply(self, x):
        for c in range(x.shape[0]):
            v = x[c]
            vmin = float(v.min())
            vmax = float(v.max())
            x[c] = (v - vmin) / (vmax - vmin) if vmax > vmin else torch.zeros_like(v)
        return x.clamp_(0.0, 1.0)


class _Linear2D(_ImgOnly2D):
    # linear_a_b : x' = a*x + b. ex: linear_1.2_-0.1
    def __init__(self, a: float, b: float):
        self.a, self.b = float(a), float(b)

    def apply(self, x):
        return x * self.a + self.b


class _Poisson2D(_ImgOnly2D):
    # poisson[_I0] : Poisson noise in [0,1]. ex: poisson_8000 (default 8000)
    def __init__(self, I0: float = 8000.0):
        self.I0 = float(I0)

    def apply(self, x):
        x = x.clamp_(0.0, 1.0)
        noisy = torch.poisson(x * self.I0) / self.I0
        return noisy.clamp_(0.0, 1.0)


class _AWGN2D(_ImgOnly2D):
    # awgn[_sigma] : additive Gaussian. ex: awgn_0.006 (default 0.006)
    def __init__(self, sigma: float = 0.006):
        self.sigma = float(sigma)

    def apply(self, x):
        return (x + torch.randn_like(x) * self.sigma).clamp_(0.0, 1.0)


class _Speckle2D(_ImgOnly2D):
    # speckle[_k] : multiplicative Gamma (mean≈1). ex: speckle_9 (default 9)
    def __init__(self, k: float = 9.0):
        self.k = float(k)

    def apply(self, x):
        gamma = torch.distributions.Gamma(concentration=self.k, rate=self.k)
        s = gamma.sample(x.shape).to(x.device).to(x.dtype)
        y = x * s
        return y.clamp_(0.0, 1.0)


# =========================================================
# 3D op implementations (stateful geom + image-only intensity)
# External layout: C×D×H×W
# =========================================================
def _to_tensor4(x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        arr = x
        if arr.ndim == 3:
            arr = arr[None, ...]
        return torch.from_numpy(arr.astype(np.float32))
    if isinstance(x, torch.Tensor):
        t = x
        if t.ndim == 3:
            t = t.unsqueeze(0)
        return t.to(dtype=torch.float32)
    raise TypeError(f"Unsupported input type: {type(x)}")


class _Geom3D:
    def __init__(self):
        self._sampled = False

    def sample_if_needed(self, shape_cd_hw: Tuple[int, int, int, int]):
        if not self._sampled:
            self._sample(shape_cd_hw)
            self._sampled = True

    def _sample(self, shape_cd_hw: Tuple[int, int, int, int]):
        pass

    def apply(self, x: torch.Tensor, mode: str, spacing: Optional[Tuple[float, float, float]]):
        """
        Return (tensor, spacing_out). spacing may pass-through or be updated (e.g., resample).
        """
        raise NotImplementedError


class _ImgOnly3D:
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _SetSpacing3D(_Geom3D):
    # setspacing3d_sz_sy_sx : write spacing without resampling. ex: setspacing3d_2.0_2.0_2.0
    def __init__(self, spacing_zyx: Tuple[float, float, float]):
        super().__init__()
        self.spacing = tuple(float(v) for v in spacing_zyx)

    def apply(self, x, mode="trilinear", spacing=None):
        return x, self.spacing


class _ResampleToSpacing3D(_Geom3D):
    # resample3d_tz_ty_tx : resample to target spacing (mm) if spacing provided; else identity.
    # ex: resample3d_1.0_1.0_1.0
    def __init__(self, target_spacing_zyx: Tuple[float, float, float]):
        super().__init__()
        self.tz, self.ty, self.tx = [float(v) for v in target_spacing_zyx]

    def apply(self, x, mode="trilinear", spacing=None):
        if spacing is None or min(self.tz, self.ty, self.tx) <= 0:
            return x, spacing
        sz, sy, sx = spacing
        C, D, H, W = x.shape
        tD = max(1, int(round(D * sz / self.tz)))
        tH = max(1, int(round(H * sy / self.ty)))
        tW = max(1, int(round(W * sx / self.tx)))
        y = F.interpolate(x.unsqueeze(0), size=(tD, tH, tW),
                          mode=mode, align_corners=False).squeeze(0)
        return y, (self.tz, self.ty, self.tx)


class _Resize3D(_Geom3D):
    # resize3d_D_H_W : resize to exact voxel size. ex: resize3d_64_128_128
    def __init__(self, size_dhw: Tuple[int, int, int]):
        super().__init__()
        self.size = (int(size_dhw[0]), int(size_dhw[1]), int(size_dhw[2]))

    def apply(self, x, mode="trilinear", spacing=None):
        y = F.interpolate(x.unsqueeze(0), size=self.size, mode=mode, align_corners=False).squeeze(0)
        return y, spacing


class _CenterCrop3D(_Geom3D):
    # center_crop3d_D_H_W : center crop; zero-pad if smaller. ex: center_crop3d_96_160_160
    def __init__(self, size_dhw: Tuple[int, int, int]):
        super().__init__()
        self.d, self.h, self.w = map(int, size_dhw)
        self.z0 = self.y0 = self.x0 = None

    def _sample(self, shape_cdhw):
        C, D, H, W = shape_cdhw
        pz = max(0, self.d - D)
        py = max(0, self.h - H)
        px = max(0, self.w - W)
        if pz or py or px:
            self.z0 = self.y0 = self.x0 = None
        else:
            self.z0 = (D - self.d) // 2
            self.y0 = (H - self.h) // 2
            self.x0 = (W - self.w) // 2

    def apply(self, x, mode="trilinear", spacing=None):
        C, D, H, W = x.shape
        pz = max(0, self.d - D)
        py = max(0, self.h - H)
        px = max(0, self.w - W)
        if pz or py or px:
            x = F.pad(x, (px // 2, px - px // 2, py // 2, py - py // 2, pz // 2, pz - pz // 2), "constant", 0.0)
            C, D, H, W = x.shape
            self.z0 = (D - self.d) // 2
            self.y0 = (H - self.h) // 2
            self.x0 = (W - self.w) // 2
        x = x[:, self.z0:self.z0 + self.d, self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        return x, spacing


class _RandomPatch3D(_Geom3D):
    # randompatch3d_D_H_W[_N] : ONE random patch; zero-pad if needed. ex: randompatch3d_64_128_128
    def __init__(self, size_dhw: Tuple[int, int, int]):
        super().__init__()
        self.d, self.h, self.w = map(int, size_dhw)
        self.z0 = self.y0 = self.x0 = None

    def _sample(self, shape_cdhw):
        C, D, H, W = shape_cdhw
        pz = max(0, self.d - D)
        py = max(0, self.h - H)
        px = max(0, self.w - W)
        if pz or py or px:
            self.z0 = self.y0 = self.x0 = None
        else:
            maxz, maxy, maxx = D - self.d, H - self.h, W - self.w
            self.z0 = 0 if maxz == 0 else random.randint(0, maxz)
            self.y0 = 0 if maxy == 0 else random.randint(0, maxy)
            self.x0 = 0 if maxx == 0 else random.randint(0, maxx)

    def apply(self, x, mode="trilinear", spacing=None):
        C, D, H, W = x.shape
        pz = max(0, self.d - D)
        py = max(0, self.h - H)
        px = max(0, self.w - W)
        if pz or py or px:
            x = F.pad(x, (0, px, 0, py, 0, pz), "constant", 0.0)
            C, D, H, W = x.shape
            self.z0 = (D - self.d) // 2
            self.y0 = (H - self.h) // 2
            self.x0 = (W - self.w) // 2
        x = x[:, self.z0:self.z0 + self.d, self.y0:self.y0 + self.h, self.x0:self.x0 + self.w]
        return x, spacing


class _RandomFlip3D(_Geom3D):
    # flip3d_axes : random flip subset of {x,y,z}, p=0.5 each. ex: flip3d_xyz / flip3d_xz
    def __init__(self, axes: Sequence[str]):
        super().__init__()
        a = set(axes) if axes else {"z"}
        self.fx = "x" in a and False
        self.fy = "y" in a and False
        self.fz = "z" in a and False
        self.use_x = "x" in a
        self.use_y = "y" in a
        self.use_z = "z" in a

    def _sample(self, shape_cdhw):
        if self.use_x:
            self.fx = random.random() < 0.5
        if self.use_y:
            self.fy = random.random() < 0.5
        if self.use_z:
            self.fz = random.random() < 0.5

    def apply(self, x, mode="trilinear", spacing=None):
        if self.fz:
            x = torch.flip(x, dims=[1])
        if self.fy:
            x = torch.flip(x, dims=[2])
        if self.fx:
            x = torch.flip(x, dims=[3])
        return x, spacing


class _RotateAny3D(_Geom3D):
    # rot3d_any_dx_dy_dz : per-axis angle in [-d,+d]°, default 10,10,10. ex: rot3d_any_10_10_10
    def __init__(self, deg_xyz: Tuple[float, float, float]):
        super().__init__()
        self.dx, self.dy, self.dz = [float(v) for v in deg_xyz]
        self.ax = self.ay = self.az = 0.0

    def _sample(self, shape_cdhw):
        if max(abs(self.dx), abs(self.dy), abs(self.dz)) < 1e-6:
            self.ax = self.ay = self.az = 0.0
        else:
            self.ax = math.radians(random.uniform(-self.dx, self.dx))
            self.ay = math.radians(random.uniform(-self.dy, self.dy))
            self.az = math.radians(random.uniform(-self.dz, self.dz))

    def apply(self, x, mode="trilinear", spacing=None):
        if max(abs(self.ax), abs(self.ay), abs(self.az)) < 1e-6:
            return x, spacing

        # grid_sample 接口：3D 体数据的“三线性”也要用 'bilinear' 这个名字
        if mode == "trilinear":
            gs_mode = "bilinear"
        elif mode == "linear":
            gs_mode = "bilinear"
        else:
            gs_mode = mode  # e.g. 'nearest'

        def Rx3(a, device, dtype):
            ca, sa = math.cos(a), math.sin(a)
            return torch.tensor([[1, 0, 0],
                                 [0, ca, -sa],
                                 [0, sa, ca]], dtype=dtype, device=device)

        def Ry3(a, device, dtype):
            ca, sa = math.cos(a), math.sin(a)
            return torch.tensor([[ca, 0, sa],
                                 [0, 1, 0],
                                 [-sa, 0, ca]], dtype=dtype, device=device)

        def Rz3(a, device, dtype):
            ca, sa = math.cos(a), math.sin(a)
            return torch.tensor([[ca, -sa, 0],
                                 [sa, ca, 0],
                                 [0, 0, 1]], dtype=dtype, device=device)

        C, D, H, W = x.shape

        # 先合成 3x3 旋转矩阵
        R = (Rz3(self.az, x.device, x.dtype) @
             Ry3(self.ay, x.device, x.dtype) @
             Rx3(self.ax, x.device, x.dtype))

        # 再拼成 affine_grid 所需的 3x4： [R | 0]
        M = torch.cat([R, torch.zeros((3, 1), dtype=x.dtype, device=x.device)], dim=1)

        grid = F.affine_grid(M.unsqueeze(0), size=(1, C, D, H, W), align_corners=False)
        y = F.grid_sample(x.unsqueeze(0), grid, mode=gs_mode, padding_mode='zeros', align_corners=False).squeeze(0)
        return y, spacing


class _RotateRightAngle3D(_Geom3D):
    # rot3d_rightangle_axes : pick one axis and rotate by k*90°. ex: rot3d_rightangle_z
    def __init__(self, axes: Sequence[str]):
        super().__init__()
        self.axes = list(axes) if axes else ['z']
        self.axis = 'z'
        self.k = 0

    def _sample(self, shape_cdhw):
        self.axis = random.choice(self.axes)
        self.k = random.randint(0, 3)

    def apply(self, x, mode="trilinear", spacing=None):
        if self.k == 0:
            return x, spacing
        if self.axis == 'z':
            x = torch.rot90(x, k=self.k, dims=(2, 3))
        elif self.axis == 'y':
            x = torch.rot90(x, k=self.k, dims=(1, 3))
        else:
            x = torch.rot90(x, k=self.k, dims=(1, 2))
        return x, spacing


class _MinMax3D(_ImgOnly3D):
    # minmax3d : per-channel min-max to [0,1]. ex: minmax3d
    def apply(self, x):
        for c in range(x.shape[0]):
            v = x[c]
            vmin = float(v.min())
            vmax = float(v.max())
            x[c] = (v - vmin) / (vmax - vmin) if vmax > vmin else torch.zeros_like(v)
        return x.clamp_(0.0, 1.0)


class _Linear3D(_ImgOnly3D):
    # linear3d_a_b : x' = a*x + b. ex: linear3d_1.2_-0.1
    def __init__(self, a: float, b: float):
        self.a, self.b = float(a), float(b)

    def apply(self, x):
        return x * self.a + self.b


class _Poisson3D(_ImgOnly3D):
    # poisson3d[_I0] : Poisson noise in [0,1]. ex: poisson3d_8000
    def __init__(self, I0: float = 8000.0):
        self.I0 = float(I0)

    def apply(self, x):
        x = x.clamp_(0.0, 1.0)
        noisy = torch.poisson(x * self.I0) / self.I0
        return noisy.clamp_(0.0, 1.0)


def _gaussian_kernel1d(sigma: float, radius: int) -> torch.Tensor:
    if sigma <= 1e-6 or radius <= 0:
        return torch.tensor([1.0], dtype=torch.float32)
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32)
    w = torch.exp(-0.5 * (xs / sigma) ** 2)
    w = w / w.sum()
    return w


def _gaussian_blur3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 1e-6:
        return x
    radius = max(1, int(math.ceil(3 * sigma)))
    k1 = _gaussian_kernel1d(sigma, radius).to(x.device).to(x.dtype)
    kx = k1.view(1, 1, 1, 1, -1)
    ky = k1.view(1, 1, 1, -1, 1)
    kz = k1.view(1, 1, -1, 1, 1)
    C = x.shape[0]
    weight_x = kx.repeat(C, 1, 1, 1, 1)
    weight_y = ky.repeat(C, 1, 1, 1, 1)
    weight_z = kz.repeat(C, 1, 1, 1, 1)
    y = F.conv3d(x.unsqueeze(0), weight_z, padding=(radius, 0, 0), groups=C)
    y = F.conv3d(y, weight_y, padding=(0, radius, 0), groups=C)
    y = F.conv3d(y, weight_x, padding=(0, 0, radius), groups=C)
    return y.squeeze(0)


class _AWGN3D(_ImgOnly3D):
    # awgn3d[_sigma[_sigcorr]] : additive Gaussian + optional spatial corr. ex: awgn3d_0.006_1.0
    def __init__(self, sigma: float = 0.006, sigma_corr: float = 1.0):
        self.sigma = float(sigma)
        self.sigma_corr = float(sigma_corr)

    def apply(self, x):
        noise = torch.randn_like(x) * self.sigma
        if self.sigma_corr > 1e-6:
            noise = _gaussian_blur3d(noise, self.sigma_corr)
        return (x + noise).clamp_(0.0, 1.0)


class _Speckle3D(_ImgOnly3D):
    # speckle3d[_k[_sigcorr]] : multiplicative Gamma; log-domain corr; mean≈1. ex: speckle3d_9_1.0
    def __init__(self, k: float = 9.0, sigma_corr: float = 1.0):
        self.k = float(k)
        self.sigma_corr = float(sigma_corr)

    def apply(self, x):
        gamma = torch.distributions.Gamma(concentration=self.k, rate=self.k)
        s = gamma.sample(x.shape).to(x.device).to(x.dtype)
        if self.sigma_corr > 1e-6:
            ls = torch.log(torch.clamp(s, 1e-8))
            ls = _gaussian_blur3d(ls, self.sigma_corr)
            ls = ls - ls.mean()
            s = torch.exp(ls)
        y = x * s
        return y.clamp_(0.0, 1.0)


# =========================================================
# Parsing helpers & axis parsers
# =========================================================
def _parse_ints(args: List[str], expect: Tuple[int, ...]) -> Tuple[int, ...]:
    if len(args) not in expect:
        raise ValueError(f"Expect {expect} ints, got {len(args)} in {args}")
    return tuple(int(round(float(a))) for a in args)


def _parse_floats(args: List[str], expect: Tuple[int, ...] = (1,), default: Optional[List[float]] = None) -> Tuple[
    float, ...]:
    if len(args) == 0 and default is not None:
        return tuple(float(v) for v in default)
    if len(args) not in expect:
        raise ValueError(f"Expect {expect} floats, got {len(args)} in {args}")
    return tuple(float(a) for a in args)


def _parse_axes_2d(args: List[str]) -> List[str]:
    if not args:
        return ['x', 'y']
    s = "".join(args).lower()
    axes = []
    for a in "xy":
        if a in s:
            axes.append(a)
    return axes or ['x', 'y']


def _parse_axes_3d(args: List[str]) -> List[str]:
    if not args:
        return ['z']
    s = "".join(args).lower()
    axes = []
    for a in "xyz":
        if a in s:
            axes.append(a)
    return axes or ['z']


def _as_tokens(preprocess):
    # 支持三种输入：list/tuple、逗号分隔字符串、空字符串/None
    if preprocess is None:
        return []
    if isinstance(preprocess, (list, tuple)):
        return [t.strip() for t in preprocess if isinstance(preprocess, (list, tuple)) for t in preprocess if
                isinstance(t, str) and t.strip()]  # will be overridden below to keep semantics
    if isinstance(preprocess, str):
        return [t.strip() for t in preprocess.split(",") if t.strip()]
    raise TypeError(f"Unsupported preprocess type: {type(preprocess)}")
