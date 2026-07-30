import torch
import torch.nn.functional as F


# ---------------- 基础网格生成 ----------------
def _base_grid_2d(N, H, W, device, dtype):
    ys = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(N, 1, 1, 1)
    return grid


def _base_grid_3d(N, D, H, W, device, dtype):
    zs = torch.linspace(-1, 1, steps=D, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, steps=H, device=device, dtype=dtype)
    xs = torch.linspace(-1, 1, steps=W, device=device, dtype=dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing='ij')
    grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0).repeat(N, 1, 1, 1, 1)
    return grid


# ---------------- 位移场 → 归一化 grid ----------------
def _disp_to_normgrid_2d(disp: torch.Tensor):
    """
    disp: (N, 2, H, W), 像素单位
    return: (N, H, W, 2) in [-1,1], align_corners=True
    """
    assert disp.dim() == 4 and disp.size(1) == 2, \
        f"2D displacement must be (N,2,H,W), got {tuple(disp.shape)}"
    N, _, H, W = disp.shape
    # 像素 -> 归一化偏移
    gx = 2.0 * disp[:, 0:1] / max(W - 1, 1)  # (N,1,H,W)
    gy = 2.0 * disp[:, 1:2] / max(H - 1, 1)  # (N,1,H,W)
    # base grid
    ys = torch.linspace(-1, 1, steps=H, device=disp.device, dtype=disp.dtype)
    xs = torch.linspace(-1, 1, steps=W, device=disp.device, dtype=disp.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).repeat(N, 1, 1, 1)  # (N,H,W,2)
    # 加偏移（用切片，避免 permute）
    grid[..., 0] = grid[..., 0] + gx[:, 0]  # (N,H,W)
    grid[..., 1] = grid[..., 1] + gy[:, 0]
    return grid


def _disp_to_normgrid_3d(disp: torch.Tensor):
    """
    disp: (N, 3, D, H, W), 像素单位
    return: (N, D, H, W, 3) in [-1,1], align_corners=True
    """
    assert disp.dim() == 5 and disp.size(1) == 3, \
        f"3D displacement must be (N,3,D,H,W), got {tuple(disp.shape)}"
    N, _, D, H, W = disp.shape
    gx = 2.0 * disp[:, 0:1] / max(W - 1, 1)  # (N,1,D,H,W)
    gy = 2.0 * disp[:, 1:2] / max(H - 1, 1)
    gz = 2.0 * disp[:, 2:3] / max(D - 1, 1)
    # base grid
    zs = torch.linspace(-1, 1, steps=D, device=disp.device, dtype=disp.dtype)
    ys = torch.linspace(-1, 1, steps=H, device=disp.device, dtype=disp.dtype)
    xs = torch.linspace(-1, 1, steps=W, device=disp.device, dtype=disp.dtype)
    zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing='ij')
    grid = torch.stack((xx, yy, zz), dim=-1).unsqueeze(0).repeat(N, 1, 1, 1, 1)  # (N,D,H,W,3)
    # 加偏移（切片方式）
    grid[..., 0] = grid[..., 0] + gx[:, 0]  # (N,D,H,W)
    grid[..., 1] = grid[..., 1] + gy[:, 0]
    grid[..., 2] = grid[..., 2] + gz[:, 0]
    return grid


# ---------------- 位移场 warp ----------------
def _warp_displacement_2d(u, v):
    grid = _disp_to_normgrid_2d(v)
    return F.grid_sample(u, grid, mode='bilinear', padding_mode='border', align_corners=True)


def _warp_displacement_3d(u, v):
    grid = _disp_to_normgrid_3d(v)
    return F.grid_sample(u, grid, mode='bilinear', padding_mode='border', align_corners=True)


# ---------------- 位移场合成 ----------------
def _compose_displacements_2d(u, v):
    u_warped = _warp_displacement_2d(u, v)
    return v + u_warped


def _compose_displacements_3d(u, v):
    u_warped = _warp_displacement_3d(u, v)
    return v + u_warped


# ---------------- Scaling & Squaring 主函数 ----------------
def scaling_and_squaring(velocity: torch.Tensor, steps: int, is_3d: bool = None) -> torch.Tensor:
    """
    velocity -> displacement via Scaling & Squaring.
    仅考虑像素/voxel单位，不考虑 spacing。
    """
    assert steps >= 0
    # 自动/显式判定维度
    if is_3d is None:
        if velocity.dim() == 4:
            is_3d = False
        elif velocity.dim() == 5:
            is_3d = True
        else:
            raise ValueError(f"velocity must be 4D or 5D, got {velocity.dim()}D")

    # 维度一致性检查
    if is_3d:
        assert velocity.dim() == 5 and velocity.size(1) == 3, \
            f"is_3d=True but velocity shape is {tuple(velocity.shape)}; expected (N,3,D,H,W)"
    else:
        assert velocity.dim() == 4 and velocity.size(1) == 2, \
            f"is_3d=False but velocity shape is {tuple(velocity.shape)}; expected (N,2,H,W)"

    # 初始缩放
    scale = 1.0 / (2 ** steps)
    u = velocity * scale

    # 反复自合成
    if is_3d:
        for _ in range(steps):
            u = _compose_displacements_3d(u, u)
    else:
        for _ in range(steps):
            u = _compose_displacements_2d(u, u)
    return u


# ---------------- 对外 API：像素位移 → grid ----------------
def displacement_to_sampling_grid(disp: torch.Tensor, is_3d: bool = None):
    """
    像素位移 -> grid_sample 需要的标准化网格
    自动/显式 2D/3D
    """
    if is_3d is None:
        # 自动从维度推断
        if disp.dim() == 4:
            is_3d = False
        elif disp.dim() == 5:
            is_3d = True
        else:
            raise ValueError(f"displacement tensor must be 4D or 5D, got {disp.dim()}D")
    if is_3d:
        return _disp_to_normgrid_3d(disp)
    else:
        return _disp_to_normgrid_2d(disp)


# ---------------- 对外 API：位移场合并 ----------------
def compose_displacements(u: torch.Tensor, v: torch.Tensor, is_3d: bool) -> torch.Tensor:
    """
    合并两个位移场（像素单位）：
    等价于返回 w = u ∘ v + v。
    即：先应用 v，再应用 u。
    """
    if is_3d:
        return _compose_displacements_3d(u, v)
    else:
        return _compose_displacements_2d(u, v)


def warp_image_by_displacement(image: torch.Tensor,
                               displacement: torch.Tensor,
                               is_3d: bool,
                               mode: str = 'bilinear',
                               padding_mode: str = 'border',
                               align_corners: bool = True) -> torch.Tensor:
    """
    用给定位移场对图像进行形变采样（warp）。

    Args:
        image: torch.Tensor
            形状：
                - 2D: (N, C, H, W)
                - 3D: (N, C, D, H, W)
            可以是 moving 图像或特征图，值域任意。
        displacement: torch.Tensor
            位移场（像素单位）：
                - 2D: (N, 2, H, W)
                - 3D: (N, 3, D, H, W)
        is_3d: bool
            True 表示 3D 数据，False 表示 2D。
        mode: str
            grid_sample 的插值方式 ('bilinear', 'nearest', 'bicubic')。
        padding_mode: str
            采样越界的填充值方式 ('zeros', 'border', 'reflection')。
        align_corners: bool
            是否使用 align_corners=True 的网格定义。

    Returns:
        warped_image: torch.Tensor，与输入 image 形状相同
    """
    if is_3d:
        grid = _disp_to_normgrid_3d(displacement)
    else:
        grid = _disp_to_normgrid_2d(displacement)

    warped = F.grid_sample(image, grid, mode=mode,
                           padding_mode=padding_mode,
                           align_corners=align_corners)
    return warped
