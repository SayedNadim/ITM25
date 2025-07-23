import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback

try:
    from .model_layers import (
        BasicBlock,
        UpBlock,
        FusionModule,
        ProjectionLayer,
        LearnableExposureMask,
        ResidualDenoiseModule,
        NoiseEstimationModule,
        BitRecoverNet,
        BitExpansionNet,
        DetailEnhancementNet,
        _kaiming_init,
    )
except ImportError:
    from model_layers import (
        BasicBlock,
        UpBlock,
        FusionModule,
        ProjectionLayer,
        LearnableExposureMask,
        ResidualDenoiseModule,
        NoiseEstimationModule,
        BitRecoverNet,
        BitExpansionNet,
        DetailEnhancementNet,
        _kaiming_init,
    )

__all__ = [
    "LearnableExposureMask",
    "BitRecoverNet",
    "BitExpansionNet",
    "DetailEnhancementNet",
    "ResidualDenoiseModule",
    "NoiseEstimationModule",
    "SoftClip",
    "DITM",
]


def srgb_to_linear(srgb) -> torch.Tensor:
    return torch.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


class DITM(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base: int = 32,
        levels: int = 3,
    ):
        super().__init__()
        self.deq = BitRecoverNet(in_channels, base=base, levels=levels)
        self.exp = BitExpansionNet(in_channels, base=base, levels=levels)
        self.detail = DetailEnhancementNet(in_channels, base, out_channels)
        self.apply(_kaiming_init)

    def forward(self, x):
        # Pad input to multiple of 32 for compatibility with the model
        # This is necessary for the model to work correctly with various input sizes
        x_in, pad_info = self._pad_to_multiple(x, 32)
        # Convert sRGB to linear RGB
        x_in = srgb_to_linear(x_in)
        # Denoise and dequantize the input
        x_denoised, x_deq = self.deq(x_in)
        # Progressive hdr expansion
        o1, o2, o3 = self.exp(x_deq)
        # Detail enhancement
        hdr = self.detail(o3)
        outs = [x_denoised, x_deq, o1, o2, o3, hdr]
        # Unpad the outputs to match the original input size
        outs = [self._unpad(o, pad_info) for o in outs]
        return outs[-1] if not self.training else outs

    # --------------------------- padding -----------------------------
    @staticmethod
    def _pad_to_multiple(x, m: int):
        h_pad = (m - x.shape[2] % m) % m
        w_pad = (m - x.shape[3] % m) % m
        lh, lw = h_pad // 2, w_pad // 2
        rh, rw = h_pad - lh, w_pad - lw
        if h_pad or w_pad:
            x = F.pad(x, (lw, rw, lh, rh), mode="reflect")
        return x, (lh, rh, lw, rw)

    @staticmethod
    def _unpad(x, pad_info):
        lh, rh, lw, rw = pad_info
        h_end = -rh if rh else None
        w_end = -rw if rw else None
        return x[:, :, lh:h_end, lw:w_end]
