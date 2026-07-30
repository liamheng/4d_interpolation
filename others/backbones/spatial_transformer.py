# -*- coding: utf-8 -*-
"""
SpatialTransformer (2D/3D) — FIXED
Key fixes:
- Always convert vector field (B,C,...) -> (B,...,C) before adding to xgrid.
- New helpers: _vecC_to_last, _coords_from_u.
- Applied consistently in displacement, compose, TVF/SVF integration (Euler/RK/RK45), feedback paths.

Other features kept:
- Modes: 'displacement' | 'svf' | 'tvf' | 'tvf_feedback'
- 2D/3D via is_3d
- padding_mode, align_corners
- image_interp (and per-call override), field_interp, state_interp
- t supports float or (B,)
- return_disp to return (warped, disp)
"""

from typing import Callable, Optional, Literal, Union
import math
import torch
import torch.nn.functional as F

InterpMode = Literal['bilinear', 'nearest']
PaddingMode = Literal['zeros', 'border', 'reflection']
ModeKind = Literal['displacement', 'svf', 'tvf', 'tvf_feedback']
Integrator = Literal['ss', 'euler', 'rk2', 'rk4', 'rk45']
StateSrc = Literal['image', 'feature']


class SpatialTransformer(torch.nn.Module):
    def __init__(
            self,
            mode: ModeKind = 'displacement',
            is_3d: bool = True,
            padding_mode: PaddingMode = 'border',
            align_corners: bool = False,
            image_interp: InterpMode = 'bilinear',
            field_interp: Optional[InterpMode] = None,  # None -> same as image_interp
            # Feedback-only knobs
            state_source: StateSrc = 'image',
            state_encoder: Optional[torch.nn.Module] = None,
            state_interp: Optional[InterpMode] = None,  # None -> image_interp
            feedback_detach: bool = False,
            cache_base_grid: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.is_3d = is_3d
        self.padding_mode = padding_mode
        self.align_corners = align_corners
        self.image_interp = image_interp
        self.field_interp = field_interp or image_interp

        # Performance
        self.cache_base_grid = bool(cache_base_grid)
        self._base_grid_cache = {}

        self.state_source = state_source
        self.state_encoder = state_encoder
        self.state_interp = state_interp or image_interp
        self.feedback_detach = bool(feedback_detach)

        if mode not in ('displacement', 'svf', 'tvf', 'tvf_feedback'):
            raise ValueError("mode must be 'displacement'|'svf'|'tvf'|'tvf_feedback'")
        if padding_mode not in ('zeros', 'border', 'reflection'):
            raise ValueError("padding_mode must be 'zeros'|'border'|'reflection'")
        if self.image_interp not in ('bilinear', 'nearest'):
            raise ValueError("image_interp must be 'bilinear'|'nearest'")
        if self.field_interp not in ('bilinear', 'nearest'):
            raise ValueError("field_interp must be 'bilinear'|'nearest'")
        if self.state_interp not in ('bilinear', 'nearest'):
            raise ValueError("state_interp must be 'bilinear'|'nearest'")
        if self.state_source not in ('image', 'feature'):
            raise ValueError("state_source must be 'image' or 'feature'")

    # ------------------------------- Public API -------------------------------

    def forward(
            self,
            img: torch.Tensor,
            *,
            return_disp: bool = False,
            image_interp: Optional[InterpMode] = None,  # per-call override
            mode: ModeKind = None,
            **kwargs
    ):
        if not self.is_3d and img.dim() != 4:
            raise ValueError("Expected img of shape (B,C,H,W) for 2D.")
        if self.is_3d and img.dim() != 5:
            raise ValueError("Expected img of shape (B,C,D,H,W) for 3D.")

        mode = self.mode if mode is None else mode

        img_interp = image_interp if image_interp is not None else self.image_interp
        if img_interp not in ('bilinear', 'nearest'):
            raise ValueError("image_interp must be 'bilinear' or 'nearest'")

        B = img.shape[0]
        xgrid = self._make_base_grid(img)  # (B,*,*, vec-last)

        # --- displacement ---
        if mode == 'displacement':
            disp = kwargs.get('disp', None)
            if disp is None:
                raise ValueError("mode='displacement' requires kwarg: disp")
            self._check_field_shape(disp, B, img.shape)
            y = self._coords_from_u(xgrid, disp)  # FIX: convert to coords-last before adding
            warped = self._sample_tensor(img, y, interp=img_interp)
            return (warped, disp) if return_disp else warped

        # parse t (float or (B,))
        t_end_in = kwargs.get('t', 1.0)
        t_vec = self._parse_t_end(t_end_in, B, img.device)

        # --- svf ---
        if mode == 'svf':
            vel = kwargs.get('vel', None)
            if vel is None:
                raise ValueError("mode='svf' requires kwarg: vel")
            self._check_field_shape(vel, B, img.shape)

            integrator: Integrator = kwargs.get('integrator', kwargs.get('method', 'ss'))
            num_steps: Optional[int] = kwargs.get('num_steps', None)
            step_size: Optional[float] = kwargs.get('step_size', None)
            rtol: float = float(kwargs.get('rtol', 1e-3))
            atol: float = float(kwargs.get('atol', 1e-6))
            max_steps: int = int(kwargs.get('max_steps', 2048))

            if integrator == 'ss':
                ss_steps = kwargs.get('ss_steps', None)
                ss_init_max_disp = float(kwargs.get('ss_init_max_disp', 0.5))
                u = self._integrate_svf_ss_batched(vel, t_vec, xgrid, ss_steps=ss_steps, init_thresh=ss_init_max_disp)
            elif integrator in ('euler', 'rk2', 'rk4', 'rk45'):
                def v_const(_t: float) -> torch.Tensor:
                    return vel

                u = self._integrate_tvf_batched(
                    v_of_t=v_const, xgrid=xgrid, t_vec=t_vec,
                    method=integrator, num_steps=num_steps,
                    step_size=step_size, rtol=rtol, atol=atol, max_steps=max_steps
                )
            else:
                raise ValueError(f"Unknown integrator for 'svf': {integrator}")

            warped = self._sample_tensor(img, self._coords_from_u(xgrid, u), interp=img_interp)
            return (warped, u) if return_disp else warped

        # --- tvf ---
        if mode == 'tvf':
            v_func = kwargs.get('v_of_t', None)
            if v_func is None or not callable(v_func):
                raise ValueError("mode='tvf' requires kwarg: v_of_t (callable: t -> velocity field)")
            method = kwargs.get('method', kwargs.get('integrator', 'rk45'))
            num_steps: Optional[int] = kwargs.get('num_steps', None)
            step_size: Optional[float] = kwargs.get('step_size', None)
            rtol: float = float(kwargs.get('rtol', 1e-3))
            atol: float = float(kwargs.get('atol', 1e-6))
            max_steps: int = int(kwargs.get('max_steps', 4096))

            u = self._integrate_tvf_batched(
                v_of_t=v_func, xgrid=xgrid, t_vec=t_vec,
                method=method, num_steps=num_steps,
                step_size=step_size, rtol=rtol, atol=atol, max_steps=max_steps
            )
            warped = self._sample_tensor(img, self._coords_from_u(xgrid, u), interp=img_interp)
            return (warped, u) if return_disp else warped

        # --- tvf_feedback ---
        if mode == 'tvf_feedback':
            v_of_state = kwargs.get('v_of_state', None)
            if v_of_state is None or not callable(v_of_state):
                raise ValueError("mode='tvf_feedback' requires kwarg: v_of_state(t, state) -> velocity field")
            method = kwargs.get('method', kwargs.get('integrator', 'rk45'))
            if method == 'ss':
                raise ValueError("'tvf_feedback' does not support 'ss'.")
            num_steps: Optional[int] = kwargs.get('num_steps', None)
            step_size: Optional[float] = kwargs.get('step_size', None)
            rtol: float = float(kwargs.get('rtol', 1e-3))
            atol: float = float(kwargs.get('atol', 1e-6))
            max_steps: int = int(kwargs.get('max_steps', 4096))
            pre_state = kwargs.get('precomputed_state', None)

            u = self._integrate_tvf_feedback_batched(
                img=img, xgrid=xgrid,
                v_of_state=v_of_state,
                t_vec=t_vec, method=method, num_steps=num_steps,
                step_size=step_size, rtol=rtol, atol=atol, max_steps=max_steps,
                precomputed_state=pre_state
            )
            warped = self._sample_tensor(img, self._coords_from_u(xgrid, u), interp=img_interp)
            return (warped, u) if return_disp else warped

        raise RuntimeError("Invalid mode dispatch.")

    __call__ = forward

    # --------------------------- Grid & Sampling utils ---------------------------

    def _make_base_grid(self, img: torch.Tensor) -> torch.Tensor:
        B = img.shape[0]
        device = img.device
        dtype = img.dtype

        if not self.is_3d:
            _, _, H, W = img.shape
            key = ('2d', device.type, getattr(device, 'index', None), str(dtype), int(H), int(W))
            if self.cache_base_grid and key in self._base_grid_cache:
                base = self._base_grid_cache[key]
            else:
                ys = torch.arange(H, device=device, dtype=dtype)
                xs = torch.arange(W, device=device, dtype=dtype)
                yy, xx = torch.meshgrid(ys, xs, indexing='ij')
                base = torch.stack([xx, yy], dim=-1).unsqueeze(0)  # (1,H,W,2)
                if self.cache_base_grid:
                    self._base_grid_cache[key] = base
            grid = base.expand(B, H, W, 2)
        else:
            _, _, D, H, W = img.shape
            key = ('3d', device.type, getattr(device, 'index', None), str(dtype), int(D), int(H), int(W))
            if self.cache_base_grid and key in self._base_grid_cache:
                base = self._base_grid_cache[key]
            else:
                zs = torch.arange(D, device=device, dtype=dtype)
                ys = torch.arange(H, device=device, dtype=dtype)
                xs = torch.arange(W, device=device, dtype=dtype)
                zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing='ij')
                base = torch.stack([xx, yy, zz], dim=-1).unsqueeze(0)  # (1,D,H,W,3)
                if self.cache_base_grid:
                    self._base_grid_cache[key] = base
            grid = base.expand(B, D, H, W, 3)
        return grid

    def _tensor_spatial(self, ten: torch.Tensor) -> Union[tuple, tuple]:
        return ten.shape[-2:] if not self.is_3d else ten.shape[-3:]

    def _pixel_to_normalized(self, coords: torch.Tensor, spatial: tuple) -> torch.Tensor:
        if not self.is_3d:
            H, W = spatial
            x = coords[..., 0]
            y = coords[..., 1]
            if self.align_corners:
                gx = 2.0 * x / max(W - 1, 1) - 1.0
                gy = 2.0 * y / max(H - 1, 1) - 1.0
            else:
                gx = 2.0 * (x + 0.5) / max(W, 1) - 1.0
                gy = 2.0 * (y + 0.5) / max(H, 1) - 1.0
            g = torch.stack([gx, gy], dim=-1)
        else:
            D, H, W = spatial
            x = coords[..., 0]
            y = coords[..., 1]
            z = coords[..., 2]
            if self.align_corners:
                gx = 2.0 * x / max(W - 1, 1) - 1.0
                gy = 2.0 * y / max(H - 1, 1) - 1.0
                gz = 2.0 * z / max(D - 1, 1) - 1.0
            else:
                gx = 2.0 * (x + 0.5) / max(W, 1) - 1.0
                gy = 2.0 * (y + 0.5) / max(H, 1) - 1.0
                gz = 2.0 * (z + 0.5) / max(D, 1) - 1.0
            g = torch.stack([gx, gy, gz], dim=-1)
        return g

    def _sample_tensor(self, tensor: torch.Tensor, coords_pixel: torch.Tensor, interp: InterpMode) -> torch.Tensor:
        spatial = self._tensor_spatial(tensor)
        # NOTE: keep geometric sampling in FP32 even under outer autocast (AMP)
        with torch.cuda.amp.autocast(enabled=False):
            tensor_f = tensor.float()
            coords_pixel_f = coords_pixel.float()
            g = self._pixel_to_normalized(coords_pixel_f, spatial)
            out = F.grid_sample(
                tensor_f, g, mode=interp,
                padding_mode=self.padding_mode, align_corners=self.align_corners
            )
        return out.to(tensor.dtype)

    def _sample_field(self, field: torch.Tensor, coords_pixel: torch.Tensor) -> torch.Tensor:
        return self._sample_tensor(field, coords_pixel, interp=self.field_interp)

    def _check_field_shape(self, field: torch.Tensor, B: int, img_shape: torch.Size):
        if not self.is_3d:
            if field.dim() != 4 or field.shape[0] != B or field.shape[1] != 2 or field.shape[2:] != img_shape[-2:]:
                raise ValueError(f"Field shape must be (B,2,H,W) with H,W={img_shape[-2:]}, got {tuple(field.shape)}")
        else:
            if field.dim() != 5 or field.shape[0] != B or field.shape[1] != 3 or field.shape[2:] != img_shape[-3:]:
                raise ValueError(
                    f"Field shape must be (B,3,D,H,W) with D,H,W={img_shape[-3:]}, got {tuple(field.shape)}")

    # ------------------------ New helpers for coords addition ------------------------

    def _vecC_to_last(self, u: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W)->(B,H,W,2) or (B,C,D,H,W)->(B,D,H,W,3)"""
        return u.permute(0, 2, 3, 1) if not self.is_3d else u.permute(0, 2, 3, 4, 1)

    def _coords_from_u(self, xgrid: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Return coords = xgrid + u (with u converted to coords-last)."""
        return xgrid + self._vecC_to_last(u)

    # -------------------------- Composition & SVF (S&S) --------------------------

    def _compose_displacements(self, u_a: torch.Tensor, u_b: torch.Tensor, xgrid: torch.Tensor) -> torch.Tensor:
        """
        Compose φ_a ∘ φ_b where φ(x)=x+u(x). Returns displacement u_ab:
            u_ab(x) = u_b(x) + u_a( x + u_b(x) )
        """
        coords = self._coords_from_u(xgrid, u_b)  # FIX
        ua_warp = self._sample_field(u_a, coords)
        return u_b + ua_warp

    def _compute_ss_steps_batched(self, vel: torch.Tensor, t_vec: torch.Tensor,
                                  init_thresh: float = 0.5) -> torch.Tensor:
        if self.is_3d:
            vmax = torch.amax(torch.abs(vel), dim=(1, 2, 3, 4))
        else:
            vmax = torch.amax(torch.abs(vel), dim=(1, 2, 3))
        scaled = torch.maximum(t_vec * vmax, torch.full_like(t_vec, 1e-12))
        m = torch.clamp(torch.ceil(torch.log2(scaled / max(init_thresh, 1e-12))), min=0).to(torch.int64)
        return m

    def _integrate_svf_ss_batched(
            self,
            vel: torch.Tensor,
            t_vec: torch.Tensor,
            xgrid: torch.Tensor,
            ss_steps: Optional[int] = None,
            init_thresh: float = 0.5,
    ) -> torch.Tensor:
        B = vel.shape[0]
        device = vel.device
        if ss_steps is not None:
            m = torch.full((B,), int(ss_steps), device=device, dtype=torch.int64)
        else:
            m = self._compute_ss_steps_batched(vel, t_vec, init_thresh)

        scale = t_vec / torch.pow(2.0, m.to(torch.float32))
        scale_bc = self._dt_broadcast(scale, vel)
        u = vel * scale_bc

        max_m = int(m.max().item())
        for s in range(max_m):
            do_mask = (m > s).view([B] + [1] * (u.dim() - 1))
            u_composed = self._compose_displacements(u, u, xgrid)  # uses coords_from_u internally
            u = torch.where(do_mask, u_composed, u)
        return u

    # --------------------------- TVF integration (no feedback) ---------------------------

    def _integrate_tvf_batched(
            self,
            v_of_t: Callable[[float], torch.Tensor],
            xgrid: torch.Tensor,
            t_vec: torch.Tensor,  # (B,)
            method: Integrator = 'rk4',
            num_steps: Optional[int] = None,
            step_size: Optional[float] = None,
            rtol: float = 1e-3,
            atol: float = 1e-6,
            max_steps: int = 4096,
    ) -> torch.Tensor:
        B = xgrid.shape[0]
        u = torch.zeros_like(self._zeros_like_field_from_grid(xgrid))

        def eval_v_batch(times: torch.Tensor) -> torch.Tensor:
            v_list = []
            for i in range(B):
                ti = float(times[i].item())
                v_full = v_of_t(ti).to(device=u.device, dtype=u.dtype)
                if v_full.shape[0] == B:
                    v_i = v_full[i:i + 1]
                elif v_full.shape[0] == 1:
                    v_i = v_full
                else:
                    v_i = v_full.unsqueeze(0)
                self._check_field_shape(v_i, 1, self._img_shape_from_grid(xgrid[i:i + 1]))
                v_list.append(v_i)
            return torch.cat(v_list, dim=0)

        if method in ('euler', 'rk2', 'rk4'):
            S_default = {'euler': 8, 'rk2': 6, 'rk4': 4}[method]
            S = int(num_steps) if (num_steps is not None) else S_default
            S = max(1, S)
            dt_vec = t_vec / float(S)
            t_curr = torch.zeros_like(t_vec)

            for _ in range(S):
                dt_bc = self._dt_broadcast(dt_vec, u)

                if method == 'euler':
                    v = eval_v_batch(t_curr)
                    k1 = self._sample_field(v, self._coords_from_u(xgrid, u))  # FIX
                    u = u + dt_bc * k1
                    t_curr = t_curr + dt_vec

                elif method == 'rk2':
                    v1 = eval_v_batch(t_curr)
                    k1 = self._sample_field(v1, self._coords_from_u(xgrid, u))
                    u_half = u + 0.5 * dt_bc * k1
                    v2 = eval_v_batch(t_curr + 0.5 * dt_vec)
                    k2 = self._sample_field(v2, self._coords_from_u(xgrid, u_half))
                    u = u + dt_bc * k2
                    t_curr = t_curr + dt_vec

                else:  # 'rk4'
                    v1 = eval_v_batch(t_curr)
                    k1 = self._sample_field(v1, self._coords_from_u(xgrid, u))

                    v2 = eval_v_batch(t_curr + 0.5 * dt_vec)
                    k2 = self._sample_field(v2, self._coords_from_u(xgrid, u + 0.5 * dt_bc * k1))

                    v3 = eval_v_batch(t_curr + 0.5 * dt_vec)
                    k3 = self._sample_field(v3, self._coords_from_u(xgrid, u + 0.5 * dt_bc * k2))

                    v4 = eval_v_batch(t_curr + dt_vec)
                    k4 = self._sample_field(v4, self._coords_from_u(xgrid, u + dt_bc * k3))

                    u = u + (dt_bc / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                    t_curr = t_curr + dt_vec

            return u

        elif method == 'rk45':
            outs = []
            for i in range(B):
                xi = xgrid[i:i + 1]
                ui0 = torch.zeros_like(self._zeros_like_field_from_grid(xi))
                t_end_i = float(t_vec[i].item())

                def f_i(t_scalar: float, u_state: torch.Tensor) -> torch.Tensor:
                    v_i_full = v_of_t(float(t_scalar)).to(device=u_state.device, dtype=u_state.dtype)
                    if v_i_full.shape[0] == B:
                        v_i = v_i_full[i:i + 1]
                    elif v_i_full.shape[0] == 1:
                        v_i = v_i_full
                    else:
                        v_i = v_i_full.unsqueeze(0)
                    self._check_field_shape(v_i, 1, self._img_shape_from_grid(xi))
                    coords_i = self._coords_from_u(xi, u_state)  # FIX
                    return self._sample_field(v_i, coords_i)

                ui = self._integrate_rk45(f_i, u0=ui0, t_end=t_end_i,
                                          dt_init=step_size, rtol=rtol, atol=atol, max_steps=max_steps)
                outs.append(ui)
            return torch.cat(outs, dim=0)

        else:
            raise ValueError(f"Unknown method: {method}")

    # --------------------------- TVF integration (feedback) ---------------------------

    def _integrate_tvf_feedback_batched(
            self,
            img: torch.Tensor,
            xgrid: torch.Tensor,
            v_of_state: Callable[[float, torch.Tensor], torch.Tensor],
            t_vec: torch.Tensor,
            method: Integrator = 'rk45',
            num_steps: Optional[int] = None,
            step_size: Optional[float] = None,
            rtol: float = 1e-3,
            atol: float = 1e-6,
            max_steps: int = 4096,
            precomputed_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = xgrid.shape[0]

        if self.state_source == 'feature':
            if precomputed_state is not None:
                base_state = precomputed_state
            else:
                if self.state_encoder is None:
                    raise ValueError("state_source='feature' requires a state_encoder or precomputed_state.")
                base_state = self.state_encoder(img)
            if self._tensor_spatial(base_state) != self._tensor_spatial(img):
                raise ValueError("Feature's spatial size must match image spatial size.")
        else:
            base_state = img

        u = torch.zeros_like(self._zeros_like_field_from_grid(xgrid))

        def build_vel_from_state(times: torch.Tensor, coords: torch.Tensor,
                                 base_state_tensor: torch.Tensor) -> torch.Tensor:
            v_list = []
            for i in range(B):
                st_i = self._sample_tensor(base_state_tensor[i:i + 1], coords[i:i + 1], interp=self.state_interp)
                if self.feedback_detach:
                    st_i = st_i.detach()
                vi = v_of_state(float(times[i].item()), st_i)
                if vi.dim() == st_i.dim() - 1:
                    vi = vi.unsqueeze(0)
                elif vi.shape[0] != 1:
                    vi = vi[i:i + 1]
                vi = vi.to(device=u.device, dtype=u.dtype)
                self._check_field_shape(vi, 1, self._img_shape_from_grid(xgrid[i:i + 1]))
                v_list.append(vi)
            return torch.cat(v_list, dim=0)

        if method in ('euler', 'rk2', 'rk4'):
            S_default = {'euler': 8, 'rk2': 6, 'rk4': 4}[method]
            S = int(num_steps) if (num_steps is not None) else S_default
            S = max(1, S)
            dt_vec = t_vec / float(S)
            t_curr = torch.zeros_like(t_vec)

            for _ in range(S):
                dt_bc = self._dt_broadcast(dt_vec, u)
                coords = self._coords_from_u(xgrid, u)  # FIX

                if method == 'euler':
                    v = build_vel_from_state(t_curr, coords, base_state)
                    k1 = self._sample_field(v, coords)
                    u = u + dt_bc * k1
                    t_curr = t_curr + dt_vec

                elif method == 'rk2':
                    v1 = build_vel_from_state(t_curr, coords, base_state)
                    k1 = self._sample_field(v1, coords)
                    u_half = u + 0.5 * dt_bc * k1
                    coords_half = self._coords_from_u(xgrid, u_half)  # FIX
                    v2 = build_vel_from_state(t_curr + 0.5 * dt_vec, coords_half, base_state)
                    k2 = self._sample_field(v2, coords_half)
                    u = u + dt_bc * k2
                    t_curr = t_curr + dt_vec

                else:  # 'rk4'
                    v1 = build_vel_from_state(t_curr, coords, base_state)
                    k1 = self._sample_field(v1, coords)

                    u2 = u + 0.5 * dt_bc * k1
                    coords2 = self._coords_from_u(xgrid, u2)  # FIX
                    v2 = build_vel_from_state(t_curr + 0.5 * dt_vec, coords2, base_state)
                    k2 = self._sample_field(v2, coords2)

                    u3 = u + 0.5 * dt_bc * k2
                    coords3 = self._coords_from_u(xgrid, u3)  # FIX
                    v3 = build_vel_from_state(t_curr + 0.5 * dt_vec, coords3, base_state)
                    k3 = self._sample_field(v3, coords3)

                    u4 = u + dt_bc * k3
                    coords4 = self._coords_from_u(xgrid, u4)  # FIX
                    v4 = build_vel_from_state(t_curr + dt_vec, coords4, base_state)
                    k4 = self._sample_field(v4, coords4)

                    u = u + (dt_bc / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                    t_curr = t_curr + dt_vec

            return u

        elif method == 'rk45':
            outs = []
            for i in range(B):
                xi = xgrid[i:i + 1]
                ui0 = torch.zeros_like(self._zeros_like_field_from_grid(xi))
                t_end_i = float(t_vec[i].item())
                base_i = base_state[i:i + 1]

                def f_i(t_scalar: float, u_state: torch.Tensor) -> torch.Tensor:
                    coords_i = self._coords_from_u(xi, u_state)  # FIX
                    st_i = self._sample_tensor(base_i, coords_i, interp=self.state_interp)
                    if self.feedback_detach:
                        st_i = st_i.detach()
                    vi = v_of_state(float(t_scalar), st_i)
                    if vi.dim() == st_i.dim() - 1:
                        vi = vi.unsqueeze(0)
                    vi = vi.to(device=u_state.device, dtype=u_state.dtype)
                    self._check_field_shape(vi, 1, self._img_shape_from_grid(xi))
                    return self._sample_field(vi, coords_i)

                ui = self._integrate_rk45(f_i, u0=ui0, t_end=t_end_i,
                                          dt_init=step_size, rtol=rtol, atol=atol, max_steps=max_steps)
                outs.append(ui)
            return torch.cat(outs, dim=0)

        else:
            raise ValueError(f"Unknown method for 'tvf_feedback': {method}")

    # ------------------------------- RK45 core -------------------------------

    def _integrate_rk45(
            self,
            f: Callable[[float, torch.Tensor], torch.Tensor],  # 函数：t -> 速度场
            u0: torch.Tensor,
            t_end: float,
            dt_init: Optional[float] = None,
            rtol: float = 1e-3,
            atol: float = 1e-6,
            max_steps: int = 4096
    ) -> torch.Tensor:
        """
        Runge-Kutta 4(5) integrator with adaptive step size for ODEs
        """
        # RK45 coefficients
        c = [0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0]  # Time coefficients
        a = [
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [44 / 45, -56 / 15, 32 / 9],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
        ]  # Stage coefficients
        b = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0]  # Weights for solution
        b_hat = [5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100,
                 1 / 40]  # Weights for error estimate

        t = 0.0
        u = u0
        dt = dt_init if (dt_init is not None and dt_init > 0) else min(0.25, max(0.05, t_end / 4.0))

        while t < t_end - 1e-12:
            k = [None] * 6  # Only 6 stages for RK45

            # Stage 1: Compute k1
            k[0] = f(t + c[0] * dt, u)
            if k[0] is None:
                print(f"Warning: k[0] is None at t={t}")

            # Stage 2: Compute k2
            u2 = u + dt * (a[1][0] * k[0]) if k[0] is not None else u
            k[1] = f(t + c[1] * dt, u2)
            if k[1] is None:
                print(f"Warning: k[1] is None at t={t}")

            # Stage 3: Compute k3
            u3 = u + dt * (a[2][0] * k[0] + a[2][1] * k[1]) if k[1] is not None else u
            k[2] = f(t + c[2] * dt, u3)
            if k[2] is None:
                print(f"Warning: k[2] is None at t={t}")

            # Stage 4: Compute k4
            u4 = u + dt * (a[3][0] * k[0] + a[3][1] * k[1] + a[3][2] * k[2]) if k[2] is not None else u
            k[3] = f(t + c[3] * dt, u4)
            if k[3] is None:
                print(f"Warning: k[3] is None at t={t}")

            # Stage 5: Compute k5
            u5 = u + dt * (a[4][0] * k[0] + a[4][1] * k[1] + a[4][2] * k[2] + a[4][3] * k[3]) if k[3] is not None else u
            k[4] = f(t + c[4] * dt, u5)
            if k[4] is None:
                print(f"Warning: k[4] is None at t={t}")

            # Stage 6: Compute k6
            u6 = u + dt * (a[5][0] * k[0] + a[5][1] * k[1] + a[5][2] * k[2] + a[5][3] * k[3] + a[5][4] * k[4]) if k[
                                                                                                                      4] is not None else u
            k[5] = f(t + c[5] * dt, u6)
            if k[5] is None:
                print(f"Warning: k[5] is None at t={t}")

            # Now compute the final weighted sum of the k's
            u_5 = u
            u_4 = u
            for j in range(6):  # Only 6 stages, not 7
                if k[j] is None:
                    print(f"Error: k[{j}] is None, skipping.")
                    continue  # Skip None values

                bj = b[j]
                bhj = b_hat[j]
                if bj != 0.0:
                    u_5 = u_5 + dt * (bj * k[j])
                if bhj != 0.0:
                    u_4 = u_4 + dt * (bhj * k[j])

            # Ensure u_5 and u_4 are valid tensors
            if u_5 is None or u_4 is None:
                raise ValueError(f"u_5 or u_4 became None at t={t}")

            # Compute the error
            err_tensor = u_5 - u_4
            scale_tensor = atol + rtol * torch.maximum(u.abs(), u_5.abs())
            err = (err_tensor.abs() / (scale_tensor + 1e-12)).amax().detach().item()

            # Step size adjustment based on the error
            if err <= 1.0 or dt <= 1e-6:
                u = u_5
                t += dt
                factor = 5.0 if err == 0.0 else float(0.9 * (err ** (-0.2)))
                factor = min(5.0, max(0.2, factor))
                dt = max(1e-6, min(dt * factor, t_end - t))
            else:
                factor = float(0.9 * (err ** (-0.2)))
                dt = max(1e-6, dt * max(0.2, min(5.0, factor)))

        return u

    # ------------------------------- Misc helpers -------------------------------

    def _img_shape_from_grid(self, xgrid: torch.Tensor) -> torch.Size:
        if not self.is_3d:
            B, H, W, _ = xgrid.shape
            return torch.Size([B, 1, H, W])
        else:
            B, D, H, W, _ = xgrid.shape
            return torch.Size([B, 1, D, H, W])

    def _zeros_like_field_from_grid(self, xgrid: torch.Tensor) -> torch.Tensor:
        B = xgrid.shape[0]
        device = xgrid.device
        dtype = xgrid.dtype
        if not self.is_3d:
            H, W = xgrid.shape[1], xgrid.shape[2]
            return torch.zeros((B, 2, H, W), device=device, dtype=dtype)
        else:
            D, H, W = xgrid.shape[1], xgrid.shape[2], xgrid.shape[3]
            return torch.zeros((B, 3, D, H, W), device=device, dtype=dtype)

    def _parse_t_end(self, t_end, B: int, device) -> torch.Tensor:
        if isinstance(t_end, torch.Tensor):
            if t_end.dim() == 0:
                t_vec = t_end.expand(B)
            elif t_end.dim() == 1:
                if t_end.numel() != B:
                    raise ValueError(f"t tensor must have shape (B,), got {tuple(t_end.shape)} vs B={B}")
                t_vec = t_end
            else:
                raise ValueError("t tensor must be scalar or 1D of shape (B,).")
            t_vec = t_vec.to(device=device, dtype=torch.float32)
        else:
            t_val = float(t_end)
            t_vec = torch.full((B,), t_val, device=device, dtype=torch.float32)
        t_vec = torch.clamp(t_vec, 0.0, 1.0)
        return t_vec

    def _dt_broadcast(self, dt_vec: torch.Tensor, u_like: torch.Tensor) -> torch.Tensor:
        view_shape = [u_like.shape[0]] + [1] * (u_like.dim() - 1)
        return dt_vec.view(view_shape).type_as(u_like)
