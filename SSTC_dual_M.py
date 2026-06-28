"""
用 PyTorch “通过 T^-1(T(x)+t) 拟合中毒生成”的样本特定性度量实验。

功能：
1) 使用 CIFAR-10，像素取值保持 [0,1]（只做 ToTensor，不做标准化）。
2) 定义 P、Q 为由若干个残差 Block 串联（ReLU -> Conv + 旁路）。每个 Conv 使用谱范数约束。
3) 自定义固定的中毒样本生成函数：。
4) 训练目标：L = (1-lamda) * MSE(x, Q(P(x))) + lamda * MSE(xp, Q(P(x)+t)) ]，t 每次更新后投影到 L2=1。
5) 轻量正则：Hutchinson 估计的雅可比范数正则（P、Q 各一次），系数很小。
6) 动态加深：用“最近 10 个 epoch 的验证损失，若倒数第10个是窗口最小”判定为无进步 -> P/Q 末端各加 1 个 Block。
7) 停止阈值使用方案 B：扩深后 epsilon = max(epsilon, (1+alpha) * best_val)。
8) 每轮训练结束：打印累计用时（时:分:秒），并在测试集上打印 clean/poison 的 MSE、PSNR、SSIM。
9) 触发加深时打印：当前验证 loss、加深后的层数、新阈值；训练结束打印最终 loss 和层数。

说明：此脚本默认只用测试集做“验证”。若需要更严格划分，可自行拆分 train/val/test。
"""
import os
import math
import time
import random
import numpy as np
from collections import deque
from typing import Tuple
import logging
import matplotlib.pyplot as plt
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.interpolate import RectBivariateSpline

# =========================
# 工具：时间格式化 & 设备
# =========================
def format_hms(seconds: float) -> str:
    secs = int(seconds)
    h = secs // 3600
    m = (secs % 3600) // 60
    s = secs % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    logger.handlers = []  # 防止重复打印
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

logger = setup_logger("logs/training_log.log")
logger.info("Experiment started")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#logger.info("Device:", device)

# 设定随机种子以便复现
seed = random.randint(1000, 9999)
#seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# =========================
# 数据：CIFAR-10（仅 ToTensor）
# =========================
def load_data():
    train_path = '../data/CIFAR10_train.npz'
    test_path = '../data/CIFAR10_test.npz'
    train_data = np.load(train_path, mmap_mode='r')
    test_data = np.load(test_path, mmap_mode='r')
    X_train = train_data['X'] / 255
    Y_train = train_data['Y']
    X_test = test_data['X'] / 255
    Y_test = test_data['Y']
    return X_train, Y_train, X_test, Y_test  # x in range of [0, 1]

# =========================
# 中毒样本生成函数
# =========================
def generate_poi_sample(x):  # [b,c,m,n]
    #xp = x.clone()
    xp = Badnets(x)
    #xp = Patch(x, 6, 0.4)
    #xp = Patch(x, 16, 0.11)
    #xp = Patch(x, 32, 0.05)
    #xp = DCT_replace(x)
    #xp = DCT_patch(x)
    #xp = DCT_3s(x)
    #xp = FFT_replace(x)
    #xp = FFT_patch(x)
    #xp = FFT_FIBA(x)
    #xp = Bppattack(x, intensity=0.05)
    #xp = Bppattack_v2(x, intensity=1)
    #xp = Wanet(x)
    #xp = ISSBA(x)
    #xp = pure_noise(x)
    xp.clamp_(0.0, 1.0)
    return xp


def Badnets(x: torch.Tensor, tri_size=4, intensity=1) -> torch.Tensor:
    mask = torch.zeros_like(x)
    mask[:, :, 2:2+tri_size, 2:2+tri_size] = 1
    xp = x * (1 - mask) + mask * intensity
    xp = xp.clamp(0.0, 1.0)
    return xp

def Patch(x: torch.Tensor, size = 5, intensity: float = 0.5) -> torch.Tensor:
    mask = torch.zeros_like(x)
    mask[:, :, 0:0+size, 0:0+size] = 1
    xp = x + mask * intensity
    xp = xp.clamp(0.0, 1.0)
    return xp

def DCT_replace(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 4 and x.shape[1] == 3 and x.shape[2] == x.shape[3], "输入需为 [B,3,N,N] 且 N×N"
    B, Cc, N, _ = x.shape
    device = x.device
    dtype_in = x.dtype

    # 计算用 float32 保持稳定；最后再转回原 dtype
    x32 = x.to(torch.float32)

    # ---- 构造 N×N 正交 DCT-II 矩阵（torch 内完成）----
    n = torch.arange(N, device=device, dtype=torch.float32)
    u = n.view(-1, 1)  # [N,1]
    alpha0 = torch.sqrt(torch.tensor(1.0 / N, device=device))
    alpha  = torch.sqrt(torch.tensor(2.0 / N, device=device))
    C = torch.cos((torch.pi * (2 * n + 1) * u) / (2 * N))  # [N,N]
    C = C * alpha
    C[0, :] = alpha0
    Ct = C.t()

    # ---- 2D DCT ----
    X = (C @ x32) @ Ct   # [B,3,N,N]

    # ---- 读取 trigger 并立刻转 torch（仅这一步可能用到 numpy）----
    trig_np = np.load('DCT_trigger.npy') / 255                 # e.g. [1,3,N,N] 或 [3,N,N]
    trig = torch.from_numpy(trig_np)
    trig = trig.to(device=device, dtype=torch.float32)

    # ---- 频域替换：trigger 非零处用 trigger 值替换 ----
    mask = (trig != 0)
    X = torch.where(mask, trig, X)

    # ---- 2D IDCT ----
    x_hat = (Ct @ X) @ C

    # 裁剪并转回原 dtype
    x_hat = x_hat.clamp(0.0, 1.0).to(dtype_in)
    return x_hat

def DCT_patch(x: torch.Tensor, p_d_ratio: float = 1.6) -> torch.Tensor:
    assert x.ndim == 4 and x.shape[1] == 3 and x.shape[2] == x.shape[3], "输入需为 [B,3,N,N] 且为方阵"
    B, _, N, _ = x.shape
    device = x.device
    in_dtype = x.dtype

    # 计算用 float32 以保证数值稳定，最后再转回原 dtype
    x32 = x.to(torch.float32)

    # 构造 N×N 正交 DCT-II 矩阵 C（torch 内完成）
    n = torch.arange(N, device=device, dtype=torch.float32)
    u = n.view(-1, 1)                            # [N,1]
    alpha0 = torch.sqrt(torch.tensor(1.0/N, device=device))
    alpha  = torch.sqrt(torch.tensor(2.0/N, device=device))
    C = torch.cos((torch.pi * (2*n + 1) * u) / (2 * N))  # [N,N]
    C = C * alpha
    C[0, :] = alpha0
    Ct = C.t()

    # 2D DCT：先沿行(高)乘，再沿列(宽)乘；torch.matmul 会在 batch 维广播
    tmp   = C @ x32                  # [N,N] @ [B,3,N,N] -> [B,3,N,N]
    x_dct = tmp @ Ct                 # [B,3,N,N] @ [N,N] -> [B,3,N,N]

    # 读取触发器并立刻转 torch（仅这一步用到 numpy）
    trig_np = np.load('DCT_trigger.npy') / 255          # 允许形状 [3,N,N] 或 [1,3,N,N]
    trigger = torch.from_numpy(trig_np).to(device=device, dtype=torch.float32)
    if trigger.ndim == 3:
        trigger = trigger.unsqueeze(0)           # -> [1,3,N,N]
    assert trigger.shape == (1, 3, N, N), f"trigger 形状应为 [1,3,{N},{N}] 或 [3,{N},{N}]"

    # 频域加触发（全在 torch 中进行；广播到 batch 维）
    x_dct = x_dct + p_d_ratio * trigger

    # 2D IDCT：x_hat = C^T @ X @ C
    tmp2 = Ct @ x_dct
    xp32 = tmp2 @ C

    # 裁剪并转回原 dtype
    xp32 = xp32.clamp(0.0, 1.0)
    xp = xp32.to(in_dtype)
    return xp

def DCT_3s(x: torch.Tensor, p_d_ratio: float = 0.5):
    device = x.device

    # 构造 32×32 的正交 DCT-II 变换矩阵 C
    B, _, _, N = x.shape
    n = torch.arange(N, device=device, dtype=torch.float32)
    u = n.view(-1, 1)  # 纵向索引
    # 标准正交化系数
    alpha0 = torch.sqrt(torch.tensor(1.0/N, device=device))
    alpha  = torch.sqrt(torch.tensor(2.0/N, device=device))
    C = torch.cos((torch.pi * (2*n + 1) * u) / (2 * N))
    C = C * alpha
    C[0, :] = alpha0  # 第一行系数不同
    Ct = C.t()
    # DCT process
    tmp = C @ x
    x_dct = tmp @ Ct

    trig_np = np.load('DCT_trigger.npy') / 255          # 允许形状 [3,N,N] 或 [1,3,N,N]
    trigger = torch.from_numpy(trig_np).to(device=device, dtype=torch.float32)
    mask = (trigger != 0)
    x_dct = x_dct + p_d_ratio * (trigger - x_dct) * mask.to(x_dct.dtype)
    # IDCT process
    tmp2 = Ct @ x_dct
    xp = tmp2 @ C
    xp = xp.clamp(0.0, 1.0)
    return xp

def FFT_replace(x: torch.Tensor, ratio: float = 0.1, patch_half: int = 4) -> torch.Tensor:
    B, Cc, N, _ = x.shape
    device = x.device
    dtype_in = x.dtype
    # ---- 读取触发图像，并变为 [3,32,32]（容错不同布局） ----
    tri_npz = np.load("sel_img.npz", mmap_mode="r")
    tri_np = tri_npz["X"]  # 可能是 [3,32,32] 或 [32,32,3] 或带 batch 的
    tri = torch.from_numpy(tri_np)
    tri = tri.permute(2, 0, 1)  # HWC -> CHW
    tri = tri.to(device=device, dtype=torch.float32)

    # ---- 计算触发图频谱实部 ----
    tri_fft = torch.fft.fft2(tri, dim=(-2, -1), norm="backward")  # [3,N,N] complex
    trigger_real = tri_fft.real  # [3,N,N]
    trigger_real = trigger_real.unsqueeze(0)  # [1,3,N,N] 用于对 B 广播

    # ---- 构造掩膜：以 (N//2-1, N//2-1) 为中心的方块 ----
    center = N // 2 - 1  # 对 32 来说是 15，与原代码一致
    y0, y1 = center - patch_half, center + patch_half
    x0, x1 = center - patch_half, center + patch_half
    mask = torch.zeros((1, Cc, N, N), device=device, dtype=torch.float32)
    mask[:, :, y0:y1, x0:x1] = 1.0  # [1,3,N,N]

    # ---- 对输入整体做 FFT ----
    x32 = x.to(torch.float32)
    X = torch.fft.fft2(x32, dim=(-2, -1), norm="backward")        # [B,3,N,N] complex
    Xr, Xi = X.real, X.imag                                       # [B,3,N,N]

    # ---- 仅在掩膜区域对实部做线性插值 ---\
    new_real = Xr + mask * ratio * (trigger_real - Xr)

    # ---- 复合频谱并反变换 ----
    Xp = torch.complex(new_real, Xi)
    x_rec_c = torch.fft.ifft2(Xp, dim=(-2, -1), norm="backward")  # complex
    x_rec = x_rec_c.real

    # ---- 裁剪并还原 dtype ----
    x_rec = x_rec.clamp(0, 1)

    return x_rec.to(dtype_in)

def FFT_patch(x: torch.Tensor, ratio: float = 0.19, patch_half: int = 3) -> torch.Tensor:
    B, Cc, N, _ = x.shape
    device = x.device
    dtype_in = x.dtype
    # ---- 读取触发图像，并变为 [3,32,32]（容错不同布局） ----
    tri_npz = np.load("sel_img.npz", mmap_mode="r")
    tri_np = tri_npz["X"]  # 可能是 [3,32,32] 或 [32,32,3] 或带 batch 的
    tri = torch.from_numpy(tri_np)
    tri = tri.permute(2, 0, 1)  # HWC -> CHW
    tri = tri.to(device=device, dtype=torch.float32)

    # ---- 计算触发图频谱实部 ----
    tri_fft = torch.fft.fft2(tri, dim=(-2, -1), norm="backward")  # [3,N,N] complex
    trigger_real = tri_fft.real  # [3,N,N]
    trigger_real = trigger_real.unsqueeze(0)  # [1,3,N,N] 用于对 B 广播

    # ---- 构造掩膜：以 (N//2-1, N//2-1) 为中心的方块 ----
    center = N // 2 - 1  # 对 32 来说是 15，与原代码一致
    y0, y1 = center - patch_half, center + patch_half
    x0, x1 = center - patch_half, center + patch_half
    mask = torch.zeros((1, Cc, N, N), device=device, dtype=torch.float32)
    mask[:, :, y0:y1, x0:x1] = 1.0  # [1,3,N,N]

    # ---- 对输入整体做 FFT ----
    x32 = x.to(torch.float32)
    X = torch.fft.fft2(x32, dim=(-2, -1), norm="backward")        # [B,3,N,N] complex
    Xr, Xi = X.real, X.imag                                       # [B,3,N,N]

    # ---- 仅在掩膜区域对实部做线性插值 ---\
    new_real = Xr + mask * ratio * trigger_real

    # ---- 复合频谱并反变换 ----
    Xp = torch.complex(new_real, Xi)
    x_rec_c = torch.fft.ifft2(Xp, dim=(-2, -1), norm="backward")  # complex
    x_rec = x_rec_c.real

    # ---- 裁剪并还原 dtype ----
    x_rec = x_rec.clamp(0, 1)

    return x_rec.to(dtype_in)

def FFT_FIBA(x: torch.Tensor, ratio: float = 0.249, patch_half: int = 4) -> torch.Tensor:
    device = x.device
    dtype_in = x.dtype
    # --- load trigger image (np) then move to torch (minimal change) ---
    tri_npz = np.load("sel_img.npz", mmap_mode="r")
    tri_np = tri_npz["X"] / 255.0                      # [0,255]->[0,1], shape [32,32,3]
    tri_np = tri_np.transpose((2, 0, 1))               # -> [3,32,32] (same as your code)
    tri = torch.from_numpy(tri_np)
    # --- trigger amplitude in Fourier domain (torch) ---
    tri_fft = torch.fft.fft2(tri, dim=(-2, -1))        # complex
    tri_fshift = torch.fft.fftshift(tri_fft, dim=(-2, -1))
    trigger_amplitude = torch.abs(tri_fshift)
    trigger_amplitude = trigger_amplitude.to(device=device, dtype=torch.float32)

    # --- benign FFT (torch) ---
    fft_benign = torch.fft.fft2(x, dim=(-2, -1))
    fshift_benign = torch.fft.fftshift(fft_benign, dim=(-2, -1))
    amplitude_benign = torch.abs(fshift_benign)
    phase_benign = torch.angle(fshift_benign)

    # --- low-frequency mask (torch), keep your original masking logic ---
    B, Cc, N, _ = x.shape
    center = N // 2 - 1
    y0, y1 = center - patch_half, center + patch_half
    x0, x1 = center - patch_half, center + patch_half
    mask = torch.zeros((1, Cc, N, N), device=device, dtype=torch.float32)
    mask[:, :, y0:y1, x0:x1] = 1.0  # [1,3,N,N]

    # --- mix amplitude in masked region (torch) ---
    new_amplitude = (1 - mask) * amplitude_benign + mask * ((1 - ratio) * amplitude_benign + ratio * trigger_amplitude)
    # --- reconstruct complex spectrum using original phase (torch) ---
    fft_poisoned = torch.polar(new_amplitude, phase_benign)  # = new_amp * exp(1j*phase)

    ifftshift_poisoned = torch.fft.ifftshift(fft_poisoned, dim=(-2, -1))
    poisoned_image = torch.fft.ifft2(ifftshift_poisoned, dim=(-2, -1)).real

    poisoned_image = torch.clamp(poisoned_image, 0, 255)
    return poisoned_image.to(dtype_in)

def Bppattack(x: torch.Tensor, intensity: float, dist: int = 4) -> torch.Tensor:
    assert x.ndim == 4 and x.shape[1:] == (3, 32, 32), "x 必须是 [B, 3, 32, 32]"
    B, C, H, W = x.shape
    assert 1 <= dist < H and 1 <= dist < W, "dist 必须在 [1, 31]"

    out = x.to(torch.float32).clone()               # 等价于 zeros_like + x
    intensity = torch.tensor(float(intensity), device=x.device, dtype=out.dtype)

    # 对应原始 numpy 版本的四次偏移叠加
    out[:, :, :,      dist: ] += x[:, :, :,      :-dist].to(torch.float32) * intensity
    out[:, :, :-dist, dist: ] += x[:, :,  dist:, :-dist].to(torch.float32) * intensity
    out[:, :, :-dist, :     ] += x[:, :,  dist:, :     ].to(torch.float32) * intensity
    out[:, :, :-dist, :-dist] += x[:, :,  dist:,  dist:].to(torch.float32) * intensity

    out = out.clamp(0.0, 1.0)
    return out

def Bppattack_v2(x: torch.Tensor, intensity: float, grid: float = 0.1, dist: int = 4) -> torch.Tensor:
    assert x.ndim == 4 and x.shape[1:] == (3, 32, 32), "x 必须是 [B, 3, 32, 32]"
    B, C, H, W = x.shape
    assert 1 <= dist < H and 1 <= dist < W, "dist 必须在 [1, 31]"

    #out = x.to(torch.float32).clone()               # 等价于 zeros_like + x
    intensity = torch.tensor(float(intensity), device=x.device, dtype=x.dtype)
    error = x % grid - grid/2
    out = x - error
    # 对应原始 numpy 版本的四次偏移叠加
    out[:, :, :,      dist: ] += error[:, :, :,      :-dist].to(torch.float32) * intensity
    out[:, :, :-dist, dist: ] += error[:, :,  dist:, :-dist].to(torch.float32) * intensity
    out[:, :, :-dist, :     ] += error[:, :,  dist:, :     ].to(torch.float32) * intensity
    out[:, :, :-dist, :-dist] += error[:, :,  dist:,  dist:].to(torch.float32) * intensity

    out = out.clamp(0.0, 1.0)
    return out

def generate_warp(img_size=[32, 32], grid_size=4, warping_strength=2.5):
    h, w = img_size
    grid_x, grid_y = np.meshgrid(
        np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size)
    )
    random_offsets = np.random.uniform(-1, 1, (grid_size, grid_size, 2))
    random_offsets *= warping_strength

    field_x = gaussian_filter(random_offsets[:, :, 0], sigma=0.5, mode='reflect')
    field_y = gaussian_filter(random_offsets[:, :, 1], sigma=0.5, mode='reflect')
    #logger.info(field_x.shape, field_y.shape)
    x = np.linspace(0, grid_size - 1, grid_size)
    y = np.linspace(0, grid_size - 1, grid_size)
    interp_x = RectBivariateSpline(x, y, field_x, kx=2, ky=2)
    interp_y = RectBivariateSpline(x, y, field_y, kx=2, ky=2)
    x_new = np.linspace(0, grid_size - 1, w)
    y_new = np.linspace(0, grid_size - 1, h)
    field_x = interp_x(x_new, y_new)
    field_y = interp_y(x_new, y_new)
    warp_field = np.stack([field_x, field_y], axis=2)  # [32, 32, 2]
    #logger.info(warp_field.shape)
    np.save('warp_field.npy', warp_field)
    return warp_field
#generate_warp()
def Wanet(x: torch.Tensor, warp_field_path: str = 'warp_field.npy', align_corners: bool = True) -> torch.Tensor:
    B, C, H, W = x.shape
    device = x.device
    dtype = x.dtype
    x = x.to(dtype)

    # 读取并转换 warp_field（像素位移，单位≈像素）
    wf_np = np.load(warp_field_path)                # 期望形状 (W, H, 2) 或 (H, W, 2)
    wf = torch.from_numpy(wf_np).to(device=device, dtype=dtype)

    if wf.ndim != 3 or wf.shape[-1] != 2:
        raise ValueError(f"warp_field 形状应为 (H, W, 2) 或 (W, H, 2)，实际 {tuple(wf.shape)}")

    # 构建基础采样网格（归一化到[-1, 1]）
    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                            torch.arange(W, device=device, dtype=dtype), indexing='ij')
    base_x = 2.0 * xx / (W - 1) - 1.0   # [-1,1]
    base_y = 2.0 * yy / (H - 1) - 1.0   # [-1,1]

    # 将像素位移归一化到[-1,1]的坐标系
    dx = 2.0 * wf[..., 0] / (W - 1)
    dy = 2.0 * wf[..., 1] / (H - 1)

    grid = torch.stack([base_x + dx, base_y + dy], dim=-1)  # [H, W, 2]，顺序为(x, y)
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1)          # [B, H, W, 2]

    # 以反射填充、双线性插值进行采样
    y = F.grid_sample(x, grid, mode='bilinear', padding_mode='reflection', align_corners=align_corners)
    return y

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        # 图像特征提取
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # one-hot信息映射
        self.fc_embed = nn.Linear(100, 32 * 32)  # 将100维信息映射到32x32的特征
        self.conv_embed = nn.Conv2d(1, 128, kernel_size=3, padding=1)  # 用1x32x32形式融合

        # 合并信息
        self.conv_fusion = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        # 还原图像
        self.conv_out1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_out2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv_out3 = nn.Conv2d(32, 3, kernel_size=3, padding=1)

    def forward(self, img, message):
        # 处理图像
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # 处理信息
        message_embed = self.fc_embed(message)  # (batch, 1024)
        message_embed = message_embed.view(-1, 1, 32, 32)  # reshape为图像形状
        message_embed = F.relu(self.conv_embed(message_embed))  # (batch, 128, 32, 32)

        # 融合信息
        x = torch.cat([x, message_embed], dim=1)  # (batch, 256, 32, 32)
        x = F.relu(self.conv_fusion(x))  # (batch, 128, 32, 32)

        # 生成隐写图像
        x = F.relu(self.conv_out1(x))
        x = F.relu(self.conv_out2(x))
        stego_img = torch.sigmoid(self.conv_out3(x))  # 输出范围 0-255

        return stego_img
@lru_cache(maxsize=1)
def get_issba_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "../models"
    encoder = Encoder().to(device)
    state_dict_path = os.path.join(model_path, "ISSBA_encoder_full.pt")
    encoder.load_state_dict(torch.load(state_dict_path, weights_only=True))
    encoder.eval()
    return encoder

def ISSBA(x):  # x: [-1, 3, 32, 32], [0, 1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = get_issba_encoder()
    x = x.to(torch.float32).to(device)
    message = np.zeros((10, 10), dtype=np.float32)
    for i in range(10):
        message[i, i] = 1  # 生成 one-hot 标签
    message = message.flatten()
    message = torch.tensor(message).unsqueeze(0).to(device)
    B = x.size(0)
    message = message.expand(B, -1)

    x = encoder(x, message)
    x = x.clamp(0, 1).to(torch.float32)  # 反归一化 & 转为 uint8
    return x

def pure_noise(x: torch.Tensor) -> torch.Tensor:
    sigma_noise = 0.104
    xp = (x + sigma_noise * torch.randn_like(x)).clamp(0.0, 1.0)
    return xp
# =========================
# 模型定义：Block、网络（支持动态加深）、谱范数
# =========================
def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class ResBlock(nn.Module):
    """基本单元：ReLU -> Conv(3x3, same padding) + 残差旁路"""
    def __init__(self, channels: int = 3, kernel_size: int = 3, spectral_norm: bool = True):
        super().__init__()
        conv = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.conv = nn.utils.spectral_norm(conv) if spectral_norm else conv
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(x)
        out = self.conv(out)
        out = out + x
        return out

class ResBlockNet(nn.Module):
    """由若干 ResBlock 串联构成的网络，支持 add_block 动态加深"""
    def __init__(self, num_blocks: int = 1, channels: int = 3, spectral_norm: bool = True):
        super().__init__()
        self.channels = channels
        self.spectral_norm = spectral_norm
        self.blocks = nn.ModuleList([ResBlock(channels, spectral_norm=spectral_norm) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for b in self.blocks:
            x = b(x)
        return x

    def add_block(self):
        self.blocks.append(ResBlock(self.channels, spectral_norm=self.spectral_norm))

    @property
    def depth(self) -> int:
        return len(self.blocks)

# =========================
# 轻量 Jacobian 正则（Hutchinson 估计）
# =========================
def jacobian_frobenius_estimate(f, x: torch.Tensor) -> torch.Tensor:
    """估计 ||J_f(x)||_F^2，使用一次 Hutchinson（随机向量 v ~ N(0, I)）。
    注意：x 必须 requires_grad=True；该函数会创建二阶梯度图以便回传到参数。
    返回：标量张量（batch 内取平均）。
    """
    # y = f(x) 形状与 x 相同
    y = f(x)
    v = torch.randn_like(y)
    # 计算 v^T * J_f(x) 的向量-雅可比乘积（等价于 J_f(x)^T v 的梯度）
    (jtv,) = torch.autograd.grad(y, x, grad_outputs=v, retain_graph=True, create_graph=True)
    # 对每个样本求和，再取 batch 均值
    return (jtv.pow(2).sum(dim=(1, 2, 3))).mean()

# =========================
# 图像质量指标：MSE、PSNR、SSIM（简单实现）
# =========================
def mse_metric(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.mse_loss(a, b, reduction='mean').item()

def psnr_metric(a: torch.Tensor, b: torch.Tensor, max_val: float = 1.0) -> float:
    mse = F.mse_loss(a, b, reduction='mean').item()
    if mse == 0:
        return 99.0
    return 10.0 * math.log10((max_val ** 2) / mse)

def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    gauss = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1d = gauss.unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.unsqueeze(0).unsqueeze(0)  # [1,1,K,K]
    window = window.repeat(channels, 1, 1, 1)  # [C,1,K,K]
    return window

def ssim_metric(a: torch.Tensor, b: torch.Tensor, window_size: int = 8, sigma: float = 1.5) -> float:
    """简单版 SSIM（与 skimage/torchmetrics 可能略有数值差异）。输入范围假定在 [0,1]。
    返回整个批次的平均 SSIM。
    """
    C = a.shape[1]
    window = _gaussian_window(window_size, sigma, C).to(a.device)
    K1, K2 = 0.01, 0.03
    L = 1.0
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    mu1 = F.conv2d(a, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(b, window, padding=window_size // 2, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(a * a, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(b * b, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(a * b, window, padding=window_size // 2, groups=C) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / (den + 1e-12)
    # 对通道与空间求平均
    return ssim_map.mean().item()

# =========================
# 评估：在测试集上计算 clean/poison 的 MSE/PSNR/SSIM
# =========================
def evaluate(P: nn.Module, Q: nn.Module, t: torch.Tensor, test_loader: DataLoader, sigma: float) -> Tuple[float, float]:
    P.eval(); Q.eval()
    mse_clean_sum, mse_poison_sum = 0.0, 0.0
    n_pix = 0

    psnr_clean_sum, psnr_poison_sum = 0.0, 0.0
    ssim_clean_sum, ssim_poison_sum = 0.0, 0.0
    n_batches = 0

    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            # 生成中毒样本（对测试同样加噪后叠加补丁）
            xb = (xb + sigma * torch.randn_like(xb)).clamp(0.0, 1.0)
            xp = generate_poi_sample(xb)

            # 前向
            y_clean = Q(P(xb))
            y_poison = Q(P(xb) + t)
            # y_clean.clamp_(0.0, 1.0)
            # y_poison.clamp_(0.0, 1.0)

            # 计算指标（截断到 [0,1] 再评估 PSNR/SSIM 更稳）
            y_clean_clamped = y_clean.clamp(0.0, 1.0)
            y_poison_clamped = y_poison.clamp(0.0, 1.0)

            mse_clean_sum += F.mse_loss(y_clean, xb, reduction='sum').item()
            mse_poison_sum += F.mse_loss(y_poison, xp, reduction='sum').item()
            n_pix += xb.numel()

            psnr_clean_sum += psnr_metric(y_clean_clamped, xb)
            psnr_poison_sum += psnr_metric(y_poison_clamped, xp)
            ssim_clean_sum += ssim_metric(y_clean_clamped, xb)
            ssim_poison_sum += ssim_metric(y_poison_clamped, xp)
            n_batches += 1

    mse_clean = mse_clean_sum / n_pix
    mse_poison = mse_poison_sum / n_pix
    psnr_clean = psnr_clean_sum / n_batches
    psnr_poison = psnr_poison_sum / n_batches
    ssim_clean = ssim_clean_sum / n_batches
    ssim_poison = ssim_poison_sum / n_batches

    return mse_clean, psnr_clean, ssim_clean, mse_poison, psnr_poison, ssim_poison

# =========================
# 训练主循环
# =========================
def train(
    epochs_per_depth: int = 1000,  # 每个深度最多训练 n 轮
    patience: int = 20,  # 判定“无进步”的窗口
    alpha: float = 0.05,  # 方案 B 的 (1+alpha)
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    sigma_noise: float = 0.1,  # 训练/评估时对干净样本加的高斯噪声标准差
    jacobian_lambda: float = 1e-6,  # Jacobian 正则系数（很小）
    max_depth: int = 100,  # 防止无限加深
    lamda: float = 0.6,  # weight for poisoned loss
):
    # 数据
    X_train, Y_train, X_test, Y_test = load_data()
    X_train = X_train.transpose(0, 3, 1, 2)  # X: [-1, 3, 128, 128]
    X_test = X_test.transpose(0, 3, 1, 2)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)  # 输入数据
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)  # 标签 (确保标签是long类型)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    xb = X_train_tensor[1:1000]  # [b,c,m,n]
    xb = xb.to(device)
    xp = generate_poi_sample(xb)
    mse = mse_metric(xb, xp)
    psnr = psnr_metric(xb, xp)
    ssim = ssim_metric(xb, xp)
    logger.info(f"Poison Disturbance: MSE {mse:.6f}  PSNR {psnr:.2f}  SSIM {ssim:.4f}")

    # 模型与触发器 t
    P = ResBlockNet(num_blocks=1, spectral_norm=True).to(device)
    Q = ResBlockNet(num_blocks=1, spectral_norm=True).to(device)
    P.apply(init_weights)
    Q.apply(init_weights)
    t = nn.Parameter(torch.randn(1, 3, 32, 32, device=device))
    with torch.no_grad():
        #t.copy_(t / (t.norm(p=2) + 1e-12))  # L2 归一化
        #t = torch.clamp(t, 0.0, 1.0)
        t = t

    # 优化器（AdamW）；将 t 单独一个参数组（可选择不同 wd）
    params = list(P.parameters()) + list(Q.parameters())
    opt = torch.optim.AdamW([
        {"params": params, "weight_decay": weight_decay, "lr": lr},
        {"params": [t], "weight_decay": 0.0, "lr": lr},
    ])
    def rebuild_optimizer():
        return torch.optim.AdamW([
            {"params": list(P.parameters()) + list(Q.parameters()), "weight_decay": weight_decay, "lr": lr},
            {"params": [t], "weight_decay": 0.0, "lr": lr},
        ])

    # 训练控制
    start_time = time.time()
    epsilon_init = mse / 100
    epsilon = epsilon_init * (1.1) ** (P.depth - 1)
    logger.info("initial threshold: {:.6f} (perturbation MSE / 100)".format(epsilon))
    best_val = float('inf')
    loss_window = deque([float('inf')]*patience, maxlen=patience)  # 最近10轮验证loss的FIFO窗口

    deepen_once = False

    while True:
        logger.info(f"Start training: P:{P.depth}, Q:{Q.depth}")
        # 每个深度最多训练 epochs_per_depth 轮
        for ep in range(epochs_per_depth):
            P.train(); Q.train()
            for xb, _ in train_loader:  # [b,c,m,n]
                xb = xb.to(device)
                # 生成中毒样本：加高斯噪声 -> 生成中毒样本
                xb = (xb + sigma_noise * torch.randn_like(xb)).clamp(0.0, 1.0)
                xp = generate_poi_sample(xb)

                # 主损失
                y_clean = Q(P(xb))
                y_poison = Q(P(xb) + t)
                # y_clean.clamp_(0.0, 1.0)
                # y_poison.clamp_(0.0, 1.0)
                loss_main =(1 - lamda) * F.mse_loss(y_clean, xb) + lamda * F.mse_loss(y_poison, xp)

                # 轻量 Jacobian 正则（需要对输入开 requires_grad）
                # 为了效率，只对一个小批开启二阶梯度，可按需改进采样频率。
                xb.requires_grad_(True)
                def fP(inp):
                    return P(inp)
                def fQ(inp):
                    return Q(inp)
                jacP = jacobian_frobenius_estimate(fP, xb)
                jacQ = jacobian_frobenius_estimate(fQ, P(xb).detach().requires_grad_(True))
                xb.requires_grad_(False)

                loss = loss_main #+ jacobian_lambda * (jacP + jacQ)

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                # t 做 L2 投影（保持范数=1）
                # with torch.no_grad():
                #     t.copy_(t / (t.norm(p=2) + 1e-12))

            # ---- epoch 结束：验证与日志 ----
            # 当前验证 loss：用测试集上的同一目标
            P.eval(); Q.eval()
            val_loss_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for xb, _ in test_loader:
                    xb = xb.to(device)
                    xb = (xb + sigma_noise * torch.randn_like(xb)).clamp(0.0, 1.0)
                    xp = generate_poi_sample(xb)

                    y_clean = Q(P(xb))
                    y_poison = Q(P(xb) + t)
                    # y_clean.clamp_(0.0, 1.0)
                    y_poison.clamp_(0.0, 1.0)
                    loss_batch = (1 - lamda) * F.mse_loss(y_clean, xb, reduction='sum') + lamda * F.mse_loss(y_poison, xp, reduction='sum')
                    val_loss_sum += loss_batch.item()
                    n_val += xb.numel()
            cur_val = val_loss_sum / n_val
            loss_window.append(cur_val)
            best_val = min(best_val, cur_val)

            # 计算并打印测试集指标（MSE / PSNR / SSIM）
            mse_c, psnr_c, ssim_c, mse_p, psnr_p, ssim_p = evaluate(P, Q, t, test_loader, sigma_noise)
            elapsed = format_hms(time.time() - start_time)
            logger.info(f"[Epoch {ep+1}] time: {elapsed} | Test Loss={cur_val:.6f}\n"
                  f"Clean: MSE {mse_c:.6f} PSNR {psnr_c:.2f} SSIM {ssim_c:.4f}  Poison: MSE {mse_p:.6f} PSNR {psnr_p:.2f}  SSIM {ssim_p:.4f}")

            # ---- 停止判据（方案：用“当前”验证损失与 epsilon 比较）----
            if cur_val <= epsilon:
                logger.info(t.min().item(), t.max().item())
                save_image(1 - t, "./tri_img/Layer_{}.png".format(P.depth))
                logger.info(f"train complete: loss {cur_val:.6f} < threshold {epsilon:.6f} .layers: P={P.depth}, Q={Q.depth}")
                return P, Q, t, epsilon, cur_val

            # ---- 无进步 -> 尝试加深（窗口长度 = patience）----
            # 判断连续10轮无进步：检查FIFO窗口内最小值是否在索引0
            if loss_window.index(min(loss_window)) == 0:
                logger.info(t.min().item(), t.max().item())
                save_image(1-t, "./tri_img/Layer_{}.png".format(P.depth))
                # 触发加深
                logger.info(f"Engage next layer: current loss {cur_val:.6f}")
                # 调整模型前：重置FIFO窗口为inf
                loss_window.clear(); loss_window.extend([float('inf')]*patience)
                P.add_block(); Q.add_block()
                P = P.to(device); Q = Q.to(device)
                #P.apply(init_weights); Q.apply(init_weights)
                opt = rebuild_optimizer()  # 关键：重建优化器，清空动量/二阶矩
                # 更新阈值（方案 B）
                #epsilon = max(epsilon, (1.0 + alpha) * best_val)
                epsilon = epsilon_init * (1.1) ** (P.depth - 1)
                best_val = float('inf')

                logger.info(f"Next layer: P:{P.depth}, Q:{Q.depth}: New threshold epsilon {epsilon:.6f}")

                # 若达到最大深度限制，直接返回
                if P.depth >= max_depth or Q.depth >= max_depth:
                    logger.info(f"Maximum depth reached({max_depth}). Forced stop. Current loss:{cur_val:.6f}")
                    return P, Q, t, epsilon, cur_val

                # 加深后，跳出本深度的剩余 epoch，进入新深度训练
                deepen_once = True
                break

        if not deepen_once:
            logger.info(t.min().item(), t.max().item())
            save_image(1-t, "./tri_img/Layer_{}.png".format(P.depth))
            # 当前深度训练完 epochs_per_depth 仍未达标，也未触发加深(可能阈值太严或学习受限）
            # 这里也选择加深（工程止损），以免停滞。
            logger.info(f"Maximum epoch reached, engage next layer. current loss {cur_val:.6f}")
            # 调整模型前：重置FIFO窗口为inf
            loss_window.clear(); loss_window.extend([float('inf')]*patience)
            P.add_block(); Q.add_block()
            P = P.to(device); Q = Q.to(device)
            #P.apply(init_weights); Q.apply(init_weights)
            opt = rebuild_optimizer()  # 关键：重建优化器，清空动量/二阶矩
            #epsilon = max(epsilon, (1.0 + alpha) * best_val)
            epsilon = epsilon_init * (1.1) ** (P.depth - 1)
            best_val = float('inf')
            logger.info(f"Next layer: P:{P.depth}, Q:{Q.depth}: New threshold epsilon:{epsilon:.6f}")
            if P.depth >= max_depth or Q.depth >= max_depth:
                logger.info(f"Maximum depth reached({max_depth}). Forced stop. Current loss:{cur_val:.6f}")
                return P, Q, t, epsilon, cur_val
        deepen_once = False  # 重置，进入下一轮深度训练


for i in range(10):
    P, Q, t, epsilon, final_loss = train()
    #logger.info("Train complete。final loss=%.6f, threshold=%.6f, layers：P=%d, Q=%d" % (final_loss, epsilon, P.depth, Q.depth))
