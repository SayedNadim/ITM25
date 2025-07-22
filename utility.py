# -*- coding: utf-8 -*-

import math
import os
import random
import shutil
import time
from collections.abc import Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
from math import exp, pi

import kornia
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
import yaml
from easydict import EasyDict
from matplotlib.colors import hsv_to_rgb
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.transform import resize


from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

MAX_HDR: float = 1.0  # cd/m^2
MU: float = 5000.0  # mu‑law parameter


def read_yaml(file_path):
    with open(file_path, "r") as file:
        data = yaml.safe_load(file)
    return EasyDict(data)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════


class LDRHDRAug:
    def __init__(self, crop_size=224):
        # Accept int or (h, w) tuple
        if isinstance(crop_size, int):
            h, w = crop_size, crop_size
        elif isinstance(crop_size, (tuple, list)) and len(crop_size) == 2:
            h, w = crop_size
        else:
            raise ValueError("crop_size must be int or (h, w) tuple")

        # ── geometry applied to LDR **and** HDR ────────────────────────────────────────
        self.shared = A.Compose(
            [
                A.HorizontalFlip(p=0.5),  # keep as-is
                A.VerticalFlip(p=0.5),  # keep as-is
                A.Rotate(
                    limit=(-15, 15),
                    border_mode=cv2.BORDER_REFLECT_101,
                    interpolation=cv2.INTER_LINEAR,
                    p=0.25,
                ),
                A.Perspective(
                    scale=(0.01, 0.03),
                    keep_size=True,
                    pad_mode=cv2.BORDER_REFLECT_101,
                    p=0.15,
                ),
            ],
            additional_targets={"hdr": "image"},
        )

        # ── photometric noise **LDR-only** ────────────────────────────────────────────
        self.ldr_only = A.Compose(
            [
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.30
                ),
                A.ImageCompression(quality_lower=40, quality_upper=80, p=0.20),
            ]
        )

    @staticmethod
    def _to_same_type(arr, proto):
        return Image.fromarray(arr) if isinstance(proto, Image.Image) else arr

    def __call__(self, hdr, ldr):
        sample = self.shared(image=np.asarray(ldr), hdr=np.asarray(hdr))
        ldr_aug = self.ldr_only(image=sample["image"])["image"]

        return (
            self._to_same_type(sample["hdr"], hdr),
            self._to_same_type(ldr_aug, ldr),
        )


def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = random.random() < 0.5

    def _augment(img):
        if hflip:
            img = img[:, :, ::-1]
        if vflip:
            img = img[:, ::-1, :]
        if rot90:
            img = np.rot90(img, axes=(0, 1))
        return np.ascontiguousarray(img)

    return [_augment(a) for a in args]


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING & CROPPING
# ═══════════════════════════════════════════════════════════════════════════════


def mod_crop(img, modulo):
    if len(img.shape) == 2:
        h, w = img.shape
        return img[: h - (h % modulo), : w - (w % modulo)]
    else:
        h, w, _ = img.shape
        return img[: h - (h % modulo), : w - (w % modulo), :]


def get_patch(*args, patch_size=64, scale=4):
    ih, iw = args[0].shape[:2]
    tp = scale * patch_size
    ip = tp // scale

    iy = random.randrange(0, ih - ip + 1)
    ix = random.randrange(0, iw - ip + 1)
    tx, ty = scale * ix, scale * iy
    ret = [
        args[0][iy : iy + ip, ix : ix + ip, :],
        *[a[ty : ty + tp, tx : tx + tp, :] for a in args[1:]],
    ]

    return ret


def np_to_tensor(*args, input_data_range=1.0, process_data_range=1.0):
    def _np_to_tensor(img):
        np_transpose = img.transpose(2, 0, 1).astype(np.float32)
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(process_data_range / input_data_range)
        return tensor.float()

    return [_np_to_tensor(a) for a in args]


# ═══════════════════════════════════════════════════════════════════════════════
# COLOR SPACE CONVERSIONS
# ═══════════════════════════════════════════════════════════════════════════════


def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: torch.Tensor) -> torch.Tensor:
    """Inverse of :pyfunc:`srgb_to_linear`. Input expected in [0, 1]."""
    a = 0.0031308
    return torch.where(lin <= a, lin * 12.92, 1.055 * lin.pow(1 / 2.4) - 0.055)


def srgb_to_linear_np(srgb: np.ndarray) -> np.ndarray:
    srgb = np.asarray(srgb, dtype=np.float32)  # ensure float32
    mask = srgb <= 0.04045
    out = np.empty_like(srgb)
    out[mask] = srgb[mask] / 12.92
    out[~mask] = np.power((srgb[~mask] + 0.055) / 1.055, 2.4)
    return out


def linear_to_srgb_np(lin: np.ndarray) -> np.ndarray:
    lin = np.asarray(lin, dtype=np.float32)
    mask = lin <= 0.0031308
    out = np.empty_like(lin)
    out[mask] = lin[mask] * 12.92
    out[~mask] = 1.055 * np.power(lin[~mask], 1 / 2.4) - 0.055
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# HDR/LDR PROCESSING & TONE MAPPING
# ═══════════════════════════════════════════════════════════════════════════════


def range_compressor_cuda(
    hdr: torch.Tensor, mu: float = MU, max_hdr: float = MAX_HDR
) -> torch.Tensor:
    x = torch.clamp(hdr, 0.0, max_hdr) / max_hdr  # normalise first
    return torch.log1p(mu * x) / math.log1p(mu)


def hdr_to_pu(hdr, c: float = 10_000.0, max_hdr: float = MAX_HDR) -> torch.Tensor:
    """Perceptual-Uniform (PU) encoding approximated by Mantiuk 2015 formula.

    Y_pu = log10(1 + c · Y) / log10(1 + c)
    Works channel-wise; input must be normalised to [0, max_hdr].
    """
    x = torch.clamp(hdr, 0.0, max_hdr) / max_hdr
    return torch.log10(1.0 + c * x) / math.log10(1.0 + c)


def tonemap(hdr: torch.Tensor) -> torch.Tensor:
    """Wrapper used throughout the file."""
    return range_compressor_cuda(hdr, MU, MAX_HDR)


def ldr_to_hdr(imgs, expo, gamma):
    return (imgs**gamma) / (expo + 1e-8)


def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def range_compressor_tensor(x, device):
    a = torch.tensor(1.0, device=device, requires_grad=False)
    mu = torch.tensor(5000.0, device=device, requires_grad=False)
    return (torch.log(a + mu * x)) / torch.log(a + mu)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE QUALITY METRICS
# ═══════════════════════════════════════════════════════════════════════════════


def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1 / sqrdErr)


def batch_psnr(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(
            Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range
        )
    return psnr / Img.shape[0]


def batch_psnr_mu(img, imclean, data_range):
    img = range_compressor_cuda(img)
    imclean = range_compressor_cuda(imclean)
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    psnr = 0
    for i in range(Img.shape[0]):
        psnr += peak_signal_noise_ratio(
            Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range
        )
    return psnr / Img.shape[0]


def root_mean_sqrt_error(im_pred, im_true):
    """Compute RMSE between predicted and true images."""
    # Calculate RMSE
    mse = torch.mean((im_true - im_pred) ** 2)
    rmse = torch.sqrt(mse)

    return round(rmse.item(), 5)


def _gaussian_kernel(kernel_size, sigma, dtype, device):
    half = kernel_size // 2
    arange = torch.arange(-half, half + 1, dtype=dtype, device=device)
    kern1d = torch.exp(-(arange**2) / (2 * sigma**2))
    kern1d /= kern1d.sum()
    return kern1d[:, None] * kern1d[None, :]  # 2-D


def calculate_ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    border: int = 0,
    reduce: bool = True,
):
    if x.shape != y.shape:
        raise ValueError("Input tensors must share shape")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    B, C, H, W = x.shape

    # gaussian window replicated per channel
    window = (
        _gaussian_kernel(kernel_size, sigma, dtype=x.dtype, device=x.device)
        .unsqueeze(0)
        .unsqueeze(0)
    )  # 1×1×k×k
    window = window.expand(C, 1, -1, -1).contiguous()

    mu_x = F.conv2d(x, window, groups=C)
    mu_y = F.conv2d(y, window, groups=C)

    mu_x2, mu_y2, mu_xy = mu_x.pow(2), mu_y.pow(2), mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, groups=C) - mu_y2
    sigma_xy = F.conv2d(x * y, window, groups=C) - mu_xy

    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )

    if border:
        ssim_map = ssim_map[..., border:-border, border:-border]

    per_img = ssim_map.mean(dim=(1, 2, 3))  # N-vector
    return per_img.mean().item() if reduce else per_img


# ═══════════════════════════════════════════════════════════════════════════════
# FILE I/O & SAVING
# ═══════════════════════════════════════════════════════════════════════════════


def radiance_writer(out_path, image):

    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


def save_hdr(path, image):
    return radiance_writer(path, image)


def save_visualization(
    ldr_img, out_img, gt_img, min_value, max_value, index, prefix, save_dir
):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{prefix}_{index}.jpg")

    combined = torch.cat([ldr_img, out_img, gt_img], dim=3)
    vutils.save_image(combined, save_path, nrow=1)
    return save_path


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════


def make_optimizer(args, targets):
    if hasattr(targets, "pca_basis") and args.optimizer in ("Adam", "AdamW"):
        base_params = []
        crf_params = []
        for name, p in targets.named_parameters():
            if not p.requires_grad:
                continue
            if "pca_basis" in name:
                crf_params.append(p)
            else:
                base_params.append(p)
        OptimCls = optim.Adam if args.optimizer == "Adam" else optim.AdamW
        return OptimCls(
            [
                {"params": base_params, "lr": args.lr},
                {"params": crf_params, "lr": args.lr * 0.01},
            ],
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

    if args.optimizer == "Adam":
        return optim.Adam(
            targets.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif args.optimizer == "AdamW":
        return optim.AdamW(
            targets.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif args.optimizer == "SGD":
        return optim.SGD(
            targets.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "RMSprop":
        return optim.RMSprop(
            targets.parameters(),
            lr=args.lr,
            alpha=0.9,
            eps=1e-8,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {args.optimizer}")


def make_checkpoint_dir(file_name):
    path = "./checkpoints/{}".format(file_name)
    os.makedirs(path, exist_ok=True)
    return path


def init_state(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)
