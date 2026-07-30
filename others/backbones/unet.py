from typing import Optional, Sequence, Set, List, Union, Dict

import torch
from torch import nn
import torch.nn.functional as F

import torch.utils.checkpoint as checkpoint

from others.backbones.modules import dropout_modified
from others.backbones.modules.common_blocks import MultiConv


# ==============================================================================
# Legacy 2D UNet (kept for backward compatibility)
# ==============================================================================
class Unet(nn.Module):
    """
    Legacy 2D U-Net implementation (MultiConv + MaxPool + ConvTranspose).

    NOTE:
      - This class is kept for backward compatibility.
      - For a more configurable modern UNet (ResBlocks/Attention/Configurable resampling),
        use `Unet2d` (see below), which can also be reached via backbone="modern" here.
    """
    ALL_OUTPUT = 0
    PART_OUTPUT = 1
    ONE_OUTPUT = 2

    def __init__(
        self,
        input_nc,
        output_nc,
        depth=5,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=dropout_modified.DROPOUT_NONE,
        last_layer='Sigmoid',
        activation_func=nn.ReLU,
        kernel_size=3,
        conv_num=2,
        use_bias=False,
        output_mode=ONE_OUTPUT,
        # new (optional): allow switching to modern backbone inside the same class
        backbone: str = "legacy",  # "legacy" | "modern"
        modern_cfg: Optional[dict] = None,
    ):
        super(Unet, self).__init__()

        if isinstance(output_mode, str):
            output_mode = getattr(Unet, output_mode.upper())
        self.output_mode = output_mode

        self.backbone = (backbone or "legacy").lower()
        if self.backbone not in ("legacy", "modern"):
            raise ValueError(f"Unknown backbone={backbone}. Use 'legacy' or 'modern'.")

        if self.backbone == "modern":
            cfg = dict(
                input_nc=input_nc,
                output_nc=output_nc,
                base_nc=ngf,
                nc_multiplier=(1, 2, 4, 8)[: max(2, depth)],  # default
                num_blocks=conv_num,
                norm_type="batch",  # closest to legacy default
                activation="relu",
                dropout=0.0,
                attn_levels=(),
                num_heads=4,
                residual=True,
                downsample_mode="maxpool",
                upsample_mode="transpose",
                skip_mode="concat",
                kernel_size=kernel_size,
                use_bias=use_bias,
                out_activation=last_layer,
                output_mode=output_mode,
                deep_supervision=False,
                ds_levels=(),
            )
            if modern_cfg:
                cfg.update(modern_cfg)
            self._modern = Unet2d(**cfg)
            return

        # ----- legacy path -----
        self.down_down = nn.MaxPool2d(2, 2)
        self.down_conv_list = nn.ModuleList(
            [
                MultiConv(
                    input_nc if i == 0 else ngf * 2 ** (i - 1),
                    ngf * 2 ** i,
                    conv_num,
                    kernel_size,
                    use_bias,
                    norm_layer,
                    activation_func,
                    use_dropout,
                )
                for i in range(depth)
            ]
        )

        self.up_conv_list = nn.ModuleList(
            [
                MultiConv(
                    ngf * 2 ** (i - 1),
                    ngf * 2 ** (i - 2),
                    conv_num,
                    kernel_size,
                    use_bias,
                    norm_layer,
                    activation_func,
                    use_dropout,
                )
                for i in range(depth, 1, -1)
            ]
        )

        self.up_up_list = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    ngf * 2 ** (i - 1),
                    ngf * 2 ** (i - 2),
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=use_bias,
                )
                for i in range(depth, 1, -1)
            ]
        )

        self.out = nn.Sequential(
            nn.Conv2d(ngf, output_nc, kernel_size=1, padding=0, bias=use_bias),
            getattr(nn, last_layer)(),
        )

    def forward(self, x):
        if getattr(self, "backbone", "legacy") == "modern":
            return self._modern(x)

        x_list = [self.down_conv_list[0](x)]
        for i in range(1, len(self.down_conv_list)):
            x_list.append(self.down_conv_list[i](self.down_down(x_list[-1])))

        y_list = [x_list[-1]]
        for i in range(len(self.up_conv_list)):
            y = torch.cat([x_list[-i - 2], self.up_up_list[i](y_list[-1])], dim=1)
            y_list.append(self.up_conv_list[i](y))

        o = self.out(y_list[-1])

        if self.output_mode == Unet.ALL_OUTPUT:
            return o, *y_list[::-1]  # from shallow to deep
        elif self.output_mode == Unet.PART_OUTPUT:
            return o, y_list[-1], x_list[0]
        else:
            return o


# ==============================================================================
# Legacy 3D building blocks (kept)
# ==============================================================================
class MultiConv3d(nn.Module):
    """
    A stack of (Conv3d -> Norm -> Act -> Dropout).
    - norm_layer: e.g. nn.BatchNorm3d / nn.InstanceNorm3d or None
    - activation_func: e.g. nn.ReLU / nn.LeakyReLU (class, not instance)
    - use_dropout: False/0/None (no), float p (Dropout3d), or nn.Module
    - residual: optional skip connection
    """

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        conv_num: int = 2,
        kernel_size: int = 3,
        use_bias: bool = False,
        norm_layer=nn.BatchNorm3d,
        activation_func=nn.ReLU,
        use_dropout=None,
        residual: bool = False,
    ):
        super().__init__()
        self.residual = residual
        padding = kernel_size // 2

        layers = []
        c_in = input_nc
        for _ in range(conv_num):
            c_out = output_nc
            layers.append(nn.Conv3d(c_in, c_out, kernel_size=kernel_size, padding=padding, bias=use_bias))
            if norm_layer is not None:
                layers.append(norm_layer(c_out))

            layers.append(
                activation_func(inplace=True)
                if 'inplace' in activation_func.__init__.__code__.co_varnames
                else activation_func()
            )

            if use_dropout is not None and use_dropout is not False and use_dropout != 0:
                if isinstance(use_dropout, nn.Module):
                    layers.append(use_dropout)
                elif isinstance(use_dropout, (int, float)):
                    layers.append(nn.Dropout3d(float(use_dropout)))
                else:
                    raise TypeError(f"Unsupported use_dropout type: {type(use_dropout)}")

            c_in = c_out

        self.net = nn.Sequential(*layers)
        self.skip = None
        if residual:
            if input_nc == output_nc:
                self.skip = nn.Identity()
            else:
                self.skip = nn.Conv3d(input_nc, output_nc, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        if self.residual and self.skip is not None:
            y = y + self.skip(x)
        return y


class Unet3d(nn.Module):
    """
    Legacy 3D U-Net (MultiConv3d + MaxPool3d + ConvTranspose3d).

    NOTE:
      - kept for backward compatibility.
      - For a more configurable modern UNet, set backbone="modern" and pass modern_cfg,
        or use `Unet3D` directly.
    """
    ALL_OUTPUT = 0
    PART_OUTPUT = 1
    ONE_OUTPUT = 2

    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        depth: int = 5,
        ngf: int = 64,
        norm_layer=nn.BatchNorm3d,
        use_dropout=None,
        last_layer: str = 'Sigmoid',
        activation_func=nn.ReLU,
        kernel_size: int = 3,
        conv_num: int = 2,
        use_bias: bool = False,
        residual: bool = False,
        output_mode=ONE_OUTPUT,
        # new (optional): allow switching to modern backbone inside the same class
        backbone: str = "legacy",  # "legacy" | "modern"
        modern_cfg: Optional[dict] = None,
    ):
        super().__init__()

        if isinstance(output_mode, str):
            output_mode = getattr(Unet3d, output_mode.upper())
        self.output_mode = output_mode
        self.residual = residual

        self.backbone = (backbone or "legacy").lower()
        if self.backbone not in ("legacy", "modern"):
            raise ValueError(f"Unknown backbone={backbone}. Use 'legacy' or 'modern'.")

        if self.backbone == "modern":
            cfg = dict(
                input_nc=input_nc,
                output_nc=output_nc,
                base_nc=ngf,
                nc_multiplier=(1, 2, 4, 8)[: max(2, depth)],
                num_blocks=conv_num,
                norm_type="batch",
                activation="relu",
                dropout=0.0,
                attn_levels=(len((1, 2, 4, 8)[: max(2, depth)]) - 1,),  # safest default for 3D
                num_heads=4,
                residual=True,
                downsample_mode="maxpool",
                upsample_mode="transpose",
                skip_mode="concat",
                kernel_size=kernel_size,
                use_bias=use_bias,
                out_activation=last_layer,
                output_mode=output_mode,
                deep_supervision=False,
                ds_levels=(),
            )
            if modern_cfg:
                cfg.update(modern_cfg)
            self._modern = Unet3D(**cfg)
            return

        # ----- legacy path -----
        self.down_down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.down_conv_list = nn.ModuleList([
            MultiConv3d(
                input_nc=input_nc if i == 0 else ngf * 2 ** (i - 1),
                output_nc=ngf * 2 ** i,
                conv_num=conv_num,
                kernel_size=kernel_size,
                use_bias=use_bias,
                norm_layer=norm_layer,
                activation_func=activation_func,
                use_dropout=use_dropout,
                residual=residual
            )
            for i in range(depth)
        ])

        self.up_up_list = nn.ModuleList([
            nn.ConvTranspose3d(
                in_channels=ngf * 2 ** (i - 1),
                out_channels=ngf * 2 ** (i - 2),
                kernel_size=2, stride=2, padding=0, bias=use_bias
            )
            for i in range(depth, 1, -1)
        ])

        self.up_conv_list = nn.ModuleList([
            MultiConv3d(
                input_nc=ngf * 2 ** (i - 1),
                output_nc=ngf * 2 ** (i - 2),
                conv_num=conv_num,
                kernel_size=kernel_size,
                use_bias=use_bias,
                norm_layer=norm_layer,
                activation_func=activation_func,
                use_dropout=use_dropout
            )
            for i in range(depth, 1, -1)
        ])

        self.out = nn.Sequential(
            nn.Conv3d(ngf, output_nc, kernel_size=1, padding=0, bias=use_bias),
            getattr(nn, last_layer)() if hasattr(nn, last_layer) else nn.Identity()
        )

    def forward(self, x: torch.Tensor):
        if getattr(self, "backbone", "legacy") == "modern":
            return self._modern(x)

        x_list = [self.down_conv_list[0](x)]
        for i in range(1, len(self.down_conv_list)):
            x_list.append(self.down_conv_list[i](self.down_down(x_list[-1])))

        y_list = [x_list[-1]]
        for i in range(len(self.up_conv_list)):
            up = self.up_up_list[i](y_list[-1])
            y = torch.cat([x_list[-i - 2], up], dim=1)
            y_list.append(self.up_conv_list[i](y))

        o = self.out(y_list[-1])

        if self.output_mode == Unet3d.ALL_OUTPUT:
            return o, *y_list[::-1]
        elif self.output_mode == Unet3d.PART_OUTPUT:
            return o, y_list[-1], x_list[0]
        else:
            return o


# ==============================================================================
# Modern, configurable UNet blocks (ND via is_3d)
# ==============================================================================
def get_activation(name: str) -> nn.Module:
    name = (name or "silu").lower()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("lrelu", "leaky_relu"):
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    if name == "elu":
        return nn.ELU(inplace=True)
    raise ValueError(f"Unknown activation: {name}")


def _pick(is_3d: bool, two, three):
    return three if is_3d else two


def make_conv(
    is_3d: bool,
    in_c: int,
    out_c: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: Optional[int] = None,
    bias: bool = True,
) -> nn.Module:
    Conv = _pick(is_3d, nn.Conv2d, nn.Conv3d)
    if padding is None:
        padding = kernel_size // 2
    return Conv(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)


def make_norm(
    is_3d: bool,
    norm_type: str,
    num_channels: int,
    num_groups: int = 32,
    eps: float = 1e-5
) -> nn.Module:
    typ = (norm_type or "group").lower()
    if typ in ("group", "gn"):
        g = min(num_groups, num_channels)
        while g > 1 and (num_channels % g != 0):
            g -= 1
        return nn.GroupNorm(g, num_channels, eps=eps)
    if typ in ("batch", "bn"):
        BN = _pick(is_3d, nn.BatchNorm2d, nn.BatchNorm3d)
        return BN(num_channels, eps=eps)
    if typ in ("instance", "in"):
        IN = _pick(is_3d, nn.InstanceNorm2d, nn.InstanceNorm3d)
        return IN(num_channels, eps=eps, affine=True)
    if typ in ("layer", "ln"):
        return nn.GroupNorm(1, num_channels, eps=eps)
    if typ in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"Unknown norm type: {norm_type}")


def make_avgpool(is_3d: bool, k: int = 2, s: int = 2) -> nn.Module:
    Pool = _pick(is_3d, nn.AvgPool2d, nn.AvgPool3d)
    return Pool(kernel_size=k, stride=s)


def make_maxpool(is_3d: bool, k: int = 2, s: int = 2) -> nn.Module:
    Pool = _pick(is_3d, nn.MaxPool2d, nn.MaxPool3d)
    return Pool(kernel_size=k, stride=s)


def _safe_cat_or_add(skip_mode: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mode = (skip_mode or "concat").lower()
    if mode in ("concat", "cat"):
        return torch.cat([a, b], dim=1)
    if mode in ("add", "sum", "+"):
        if a.shape[1] != b.shape[1]:
            raise ValueError(f"skip_mode='add' requires same channels, got {a.shape[1]} vs {b.shape[1]}")
        return a + b
    raise ValueError(f"Unknown skip_mode: {skip_mode}")


def _align_like(x: torch.Tensor, ref: torch.Tensor, mode: str = "pad") -> torch.Tensor:
    xs = list(x.shape[2:])
    rs = list(ref.shape[2:])
    if xs == rs:
        return x

    if len(xs) != len(rs):
        raise ValueError("Tensor rank mismatch.")

    if mode == "pad":
        pads = []
        for xd, rd in zip(reversed(xs), reversed(rs)):
            diff = rd - xd
            pads.extend([0, max(diff, 0)])
        y = F.pad(x, pads)
        return _align_like(y, ref, mode="crop")

    if mode == "crop":
        slices = [slice(None), slice(None)]
        for xd, rd in zip(xs, rs):
            if xd == rd:
                slices.append(slice(None))
            elif xd > rd:
                start = (xd - rd) // 2
                slices.append(slice(start, start + rd))
            else:
                return _align_like(x, ref, mode="pad")
        return x[tuple(slices)]

    raise ValueError(f"Unknown align mode: {mode}")


class ResBlockPlain(nn.Module):
    def __init__(
        self,
        is_3d: bool,
        in_ch: int,
        out_ch: int,
        norm_type: str = "group",
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.0,
        kernel_size: int = 3,
        use_bias: bool = True,
    ):
        super().__init__()
        self.norm1 = make_norm(is_3d, norm_type, in_ch, num_groups)
        self.act1 = get_activation(activation)
        self.conv1 = make_conv(is_3d, in_ch, out_ch, kernel_size=kernel_size, bias=use_bias)

        self.norm2 = make_norm(is_3d, norm_type, out_ch, num_groups)
        self.act2 = get_activation(activation)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.conv2 = make_conv(is_3d, out_ch, out_ch, kernel_size=kernel_size, bias=use_bias)

        self.skip = nn.Identity() if in_ch == out_ch else make_conv(is_3d, in_ch, out_ch, kernel_size=1, padding=0, bias=use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.skip(x)


class SpatialSelfAttention(nn.Module):
    def __init__(
        self,
        is_3d: bool,
        channels: int,
        num_heads: int = 4,
        norm_type: str = "group",
        num_groups: int = 32,
        dropout: float = 0.0,
        use_bias: bool = True,
    ):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = make_norm(is_3d, norm_type, channels, num_groups)
        self.qkv = make_conv(is_3d, channels, channels * 3, kernel_size=1, padding=0, bias=use_bias)
        self.proj = make_conv(is_3d, channels, channels, kernel_size=1, padding=0, bias=use_bias)
        self.attn_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.proj_drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        spatial = x.shape[2:]
        n = 1
        for s in spatial:
            n *= s

        x_norm = self.norm(x)
        qkv = self.qkv(x_norm)
        q, k, v = qkv.chunk(3, dim=1)

        q = q.view(b, self.num_heads, self.head_dim, n).transpose(2, 3)
        k = k.view(b, self.num_heads, self.head_dim, n)
        v = v.view(b, self.num_heads, self.head_dim, n).transpose(2, 3)

        attn = (q @ k) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v
        out = out.transpose(2, 3).contiguous().view(b, c, *spatial)

        out = self.proj(out)
        out = self.proj_drop(out)
        return x + out


class Downsample(nn.Module):
    def __init__(self, is_3d: bool, channels: int, mode: str = "conv", use_bias: bool = True):
        super().__init__()
        m = (mode or "conv").lower()
        if m in ("conv", "stride", "strided"):
            self.op = make_conv(is_3d, channels, channels, kernel_size=3, stride=2, padding=1, bias=use_bias)
        elif m in ("avg", "avgpool"):
            self.op = make_avgpool(is_3d, k=2, s=2)
        elif m in ("max", "maxpool"):
            self.op = make_maxpool(is_3d, k=2, s=2)
        else:
            raise ValueError(f"Unknown downsample_mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(
        self,
        is_3d: bool,
        in_channels: int,
        out_channels: int,
        mode: str = "nearest",
        use_conv: bool = True,
        kernel_size: int = 3,
        use_bias: bool = True,
    ):
        super().__init__()
        m = (mode or "nearest").lower()
        self.is_3d = is_3d

        if m in ("transpose", "deconv", "convtranspose"):
            ConvT = _pick(is_3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
            self.op = ConvT(in_channels, out_channels, kernel_size=2, stride=2, padding=0, bias=use_bias)
            self.post = nn.Identity()
            self.interp_mode = None
        elif m in ("nearest", "bilinear", "trilinear"):
            self.op = None
            self.interp_mode = "nearest" if m == "nearest" else ("trilinear" if is_3d else "bilinear")
            if use_conv:
                self.post = make_conv(is_3d, in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2, bias=use_bias)
            else:
                self.post = make_conv(is_3d, in_channels, out_channels, kernel_size=1, padding=0, bias=use_bias) if in_channels != out_channels else nn.Identity()
        else:
            raise ValueError(f"Unknown upsample_mode: {mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.op is not None:
            return self.op(x)
        x = F.interpolate(
            x,
            scale_factor=2.0,
            mode=self.interp_mode,
            align_corners=False if self.interp_mode != "nearest" else None
        )
        return self.post(x)


def _make_out_activation(name: str) -> nn.Module:
    n = (name or "identity").lower()
    if n in ("identity", "none", ""):
        return nn.Identity()
    if hasattr(nn, name):
        return getattr(nn, name)()
    if n == "sigmoid":
        return nn.Sigmoid()
    if n == "tanh":
        return nn.Tanh()
    if n == "softmax":
        return nn.Softmax(dim=1)
    raise ValueError(f"Unknown out_activation: {name}")


class UNetPlain(nn.Module):
    ALL_OUTPUT = 0
    PART_OUTPUT = 1
    ONE_OUTPUT = 2

    def __init__(
        self,
        is_3d: bool,
        input_nc: int,
        output_nc: int,
        base_nc: int = 64,
        nc_multiplier: Sequence[int] = (1, 2, 4, 8),
        num_blocks: Union[int, Sequence[int]] = 2,
        norm_type: str = "group",
        num_groups: int = 32,
        activation: str = "silu",
        dropout: float = 0.0,
        residual: bool = True,
        attn_levels: Optional[Sequence[int]] = None,
        num_heads: int = 4,
        kernel_size: int = 3,
        use_bias: bool = True,
        downsample_mode: str = "conv",
        upsample_mode: str = "nearest",
        skip_mode: str = "concat",
        align_mode: str = "pad",
        # memory-saving options (do not change math, only storage/recompute)
        use_checkpoint: bool = False,
        # how to STORE skips during encoder: 'dense' (per-block, default) or 'per_level'
        skip_store_mode: str = "dense",
        out_activation: str = "Identity",
        output_mode: Union[int, str] = ONE_OUTPUT,
        deep_supervision: bool = False,
        ds_levels: Sequence[int] = (),
    ):
        super().__init__()
        if isinstance(output_mode, str):
            output_mode = getattr(UNetPlain, output_mode.upper())
        self.output_mode = output_mode

        self.is_3d = bool(is_3d)
        self.base_nc = int(base_nc)
        self.nc_multiplier = list(nc_multiplier)
        self.depth = len(self.nc_multiplier)

        # ----------------------
        # Skip configuration
        #   - skip_mode: how to MERGE skip and current feature ('concat'|'add')
        #   - skip_store_mode: how to STORE skips during encoder ('dense'|'per_level')
        # Backward-compatible convenience:
        #   If user passes skip_mode in ('dense','per_level',...), interpret it as skip_store_mode
        #   and default merge to 'concat'.
        merge_mode = (skip_mode or "concat").lower()
        store_mode = (skip_store_mode or "dense").lower()
        if merge_mode in ("dense", "per_level", "perlevel", "level", "single", "per-level"):
            store_mode = "dense" if merge_mode == "dense" else "per_level"
            merge_mode = "concat"

        if store_mode not in ("dense", "per_level"):
            raise ValueError(f"Unknown skip_store_mode: {skip_store_mode}. Use 'dense' or 'per_level'.")
        if merge_mode not in ("concat", "cat", "add", "sum", "+"):
            raise ValueError(f"Unknown skip_mode: {skip_mode}. Use 'concat' or 'add' (or pass store mode).")

        self.skip_mode = merge_mode  # merge
        self.skip_store_mode = store_mode  # store strategy
        self.align_mode = align_mode
        self.use_checkpoint = bool(use_checkpoint)
        self.deep_supervision = bool(deep_supervision)
        self.ds_levels: Set[int] = set(ds_levels)

        if attn_levels is None:
            attn_levels = ()
        self.attn_levels: Set[int] = set(attn_levels)

        if isinstance(num_blocks, int):
            nb = [int(num_blocks)] * self.depth
        else:
            nb = list(num_blocks)
            if len(nb) != self.depth:
                raise ValueError(f"num_blocks length must match depth={self.depth}")
        self.num_blocks_per_level = nb

        self.in_conv = make_conv(self.is_3d, input_nc, self.base_nc, kernel_size=kernel_size, bias=use_bias)

        self.down = nn.ModuleList()
        ch = self.base_nc
        self._skip_channels: List[int] = []

        for level, mult in enumerate(self.nc_multiplier):
            out_ch = self.base_nc * int(mult)
            for _ in range(self.num_blocks_per_level[level]):
                blk = ResBlockPlain(
                    self.is_3d, ch, out_ch,
                    norm_type=norm_type, num_groups=num_groups, activation=activation,
                    dropout=dropout, kernel_size=kernel_size, use_bias=use_bias
                ) if residual else nn.Sequential(
                    make_norm(self.is_3d, norm_type, ch, num_groups),
                    get_activation(activation),
                    make_conv(self.is_3d, ch, out_ch, kernel_size=kernel_size, bias=use_bias),
                    make_norm(self.is_3d, norm_type, out_ch, num_groups),
                    get_activation(activation),
                    nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
                    make_conv(self.is_3d, out_ch, out_ch, kernel_size=kernel_size, bias=use_bias),
                )
                self.down.append(blk)
                ch = out_ch
                if level in self.attn_levels:
                    self.down.append(
                        SpatialSelfAttention(
                            self.is_3d, ch, num_heads=num_heads,
                            norm_type=norm_type, num_groups=num_groups, dropout=dropout, use_bias=use_bias
                        )
                    )
                if self.skip_store_mode == "dense":
                    self._skip_channels.append(ch)

            if self.skip_store_mode == "per_level":
                self._skip_channels.append(ch)

            if level != self.depth - 1:
                self.down.append(Downsample(self.is_3d, ch, mode=downsample_mode, use_bias=use_bias))

        self.mid1 = ResBlockPlain(
            self.is_3d, ch, ch,
            norm_type=norm_type, num_groups=num_groups, activation=activation,
            dropout=dropout, kernel_size=kernel_size, use_bias=use_bias
        ) if residual else nn.Identity()
        self.mid_attn = SpatialSelfAttention(
            self.is_3d, ch, num_heads=num_heads,
            norm_type=norm_type, num_groups=num_groups, dropout=dropout, use_bias=use_bias
        ) if (self.depth - 1) in self.attn_levels else nn.Identity()
        self.mid2 = ResBlockPlain(
            self.is_3d, ch, ch,
            norm_type=norm_type, num_groups=num_groups, activation=activation,
            dropout=dropout, kernel_size=kernel_size, use_bias=use_bias
        ) if residual else nn.Identity()

        self.up = nn.ModuleList()
        self.ups = nn.ModuleList()

        for level, mult in reversed(list(enumerate(self.nc_multiplier))):
            out_ch = self.base_nc * int(mult)
            level_skip_ch = None
            if self.skip_store_mode == "per_level":
                level_skip_ch = self._skip_channels.pop()
            for block_i in range(self.num_blocks_per_level[level]):
                skip_ch = self._skip_channels.pop() if self.skip_store_mode == "dense" else (level_skip_ch if block_i == 0 else 0)
                in_ch = (ch + skip_ch) if ((self.skip_mode or "concat").lower() in ("concat", "cat") and skip_ch) else ch
                blk = ResBlockPlain(
                    self.is_3d, in_ch, out_ch,
                    norm_type=norm_type, num_groups=num_groups, activation=activation,
                    dropout=dropout, kernel_size=kernel_size, use_bias=use_bias
                ) if residual else nn.Sequential(
                    make_norm(self.is_3d, norm_type, in_ch, num_groups),
                    get_activation(activation),
                    make_conv(self.is_3d, in_ch, out_ch, kernel_size=kernel_size, bias=use_bias),
                    make_norm(self.is_3d, norm_type, out_ch, num_groups),
                    get_activation(activation),
                    nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity(),
                    make_conv(self.is_3d, out_ch, out_ch, kernel_size=kernel_size, bias=use_bias),
                )
                self.up.append(blk)
                ch = out_ch

                if level in self.attn_levels:
                    self.up.append(
                        SpatialSelfAttention(
                            self.is_3d, ch, num_heads=num_heads,
                            norm_type=norm_type, num_groups=num_groups, dropout=dropout, use_bias=use_bias
                        )
                    )

            if level != 0:
                self.ups.append(
                    Upsample(
                        self.is_3d,
                        in_channels=ch,
                        out_channels=ch,
                        mode=upsample_mode,
                        use_conv=True,
                        kernel_size=kernel_size,
                        use_bias=use_bias,
                    )
                )

        self.out_norm = make_norm(self.is_3d, norm_type, ch, num_groups)
        self.out_act = get_activation(activation)
        self.out_conv = make_conv(self.is_3d, ch, output_nc, kernel_size=1, padding=0, bias=use_bias)
        self.out_act2 = _make_out_activation(out_activation)

        self.ds_heads = nn.ModuleDict()
        if self.deep_supervision:
            for lv in self.ds_levels:
                if lv < 0 or lv >= self.depth:
                    raise ValueError(f"ds_levels contains invalid level {lv} for depth={self.depth}")
                ch_lv = self.base_nc * int(self.nc_multiplier[lv])
                self.ds_heads[str(lv)] = nn.Sequential(
                    make_conv(self.is_3d, ch_lv, output_nc, kernel_size=1, padding=0, bias=use_bias),
                    _make_out_activation(out_activation),
                )

    def _run_maybe_ckpt(self, module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run a module, optionally under gradient checkpointing to save activation memory.

        - Only applies during training when `use_checkpoint=True` and `x` requires grad.
        - This does NOT change the math of forward; it trades extra compute in backward for memory.
        """
        if self.use_checkpoint and self.training and x.requires_grad:
            # use_reentrant=False is recommended in newer PyTorch; fallback if not available.
            try:
                return checkpoint.checkpoint(module, x, use_reentrant=False)
            except TypeError:
                return checkpoint.checkpoint(module, x)
        return module(x)

    def forward(self, x: torch.Tensor):
        h = self.in_conv(x)
        skips: List[torch.Tensor] = []
        idx = 0

        dense_skips = (self.skip_store_mode == "dense")

        # -------- encoder --------
        for level in range(self.depth):
            for _ in range(self.num_blocks_per_level[level]):
                m = self.down[idx]
                idx += 1
                h = self._run_maybe_ckpt(m, h)
                if idx < len(self.down) and isinstance(self.down[idx], SpatialSelfAttention):
                    h = self._run_maybe_ckpt(self.down[idx], h)
                    idx += 1
                if dense_skips:
                    skips.append(h)

            if not dense_skips:
                # store one skip per level (after the last block/attn at this resolution)
                skips.append(h)

            if level != self.depth - 1:
                h = self._run_maybe_ckpt(self.down[idx], h)  # downsample
                idx += 1

        # -------- middle --------
        h = self._run_maybe_ckpt(self.mid1, h)
        h = self._run_maybe_ckpt(self.mid_attn, h)
        h = self._run_maybe_ckpt(self.mid2, h)

        # -------- decoder --------
        up_idx = 0
        ups_idx = 0
        ds_outputs: Dict[int, torch.Tensor] = {}

        for level in reversed(range(self.depth)):
            level_skip = None
            if not dense_skips:
                level_skip = skips.pop()

            for block_i in range(self.num_blocks_per_level[level]):
                skip = skips.pop() if dense_skips else (level_skip if block_i == 0 else None)
                if skip is not None:
                    if h.shape[2:] != skip.shape[2:]:
                        h = _align_like(h, skip, mode=self.align_mode)
                    h = _safe_cat_or_add(self.skip_mode, h, skip)

                m = self.up[up_idx]
                up_idx += 1
                h = self._run_maybe_ckpt(m, h)
                if up_idx < len(self.up) and isinstance(self.up[up_idx], SpatialSelfAttention):
                    h = self._run_maybe_ckpt(self.up[up_idx], h)
                    up_idx += 1

            if self.deep_supervision and (level in self.ds_levels):
                ds_outputs[level] = self.ds_heads[str(level)](h)

            if level != 0:
                h = self._run_maybe_ckpt(self.ups[ups_idx], h)
                ups_idx += 1

        out = self.out_act2(self.out_conv(self.out_act(self.out_norm(h))))

        if self.output_mode == UNetPlain.ALL_OUTPUT:
            if self.deep_supervision and ds_outputs:
                ordered = [ds_outputs[k] for k in sorted(ds_outputs.keys())]
                return out, *ordered
            return out
        elif self.output_mode == UNetPlain.PART_OUTPUT:
            return out, h, None
        else:
            return out


class Unet2d(UNetPlain):
    """Modern configurable 2D UNet."""
    def __init__(self, **kwargs):
        super().__init__(is_3d=False, **kwargs)


class Unet3D(UNetPlain):
    """Modern configurable 3D UNet."""
    def __init__(self, **kwargs):
        super().__init__(is_3d=True, **kwargs)


# ==============================================================================
# Diffusion-style UNet (time-conditioned) - restored from original implementation
# ==============================================================================
# (以下内容与本次改动无关，保持原样)
# ... 省略：后续文件内容保持不变（你下载的 unet_modified.py 里是完整的）
