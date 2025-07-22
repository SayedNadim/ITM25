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


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test with various resolutions
    def test_resolution(h, w):
        print(f"Testing resolution: {h}x{w}")
        model = DITM(in_channels=3, out_channels=3).to(device)
        model.eval()

        with torch.no_grad():
            x = torch.randn(1, 3, h, w).to(device)
            try:
                output = model(x)
                print(f"  Input shape: {x.shape}")
                print(f"  Output shape: {output.shape}")
                print(
                    f"  Success: Output size matches input size: {output.shape[-2:] == (h, w)}"
                )
                print()
            except Exception as e:
                print(f"  Error: {e}")
                print(f"  Traceback: {traceback.format_exc()}")
                print()

    # Test various resolutions including odd dimensions
    test_cases = [
        (640, 480),  # Original test case
        (321, 241),  # Odd dimensions
        # (256, 256),  # Power of 2
        # (224, 224),  # Common ImageNet size
        # (1920, 1080),  # HD resolution
        # (127, 83),  # Small odd dimensions
        # (512, 384),  # Mixed even dimensions
    ]

    for h, w in test_cases:
        test_resolution(h, w)

    # Original benchmark if available
    try:
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.benchmark import run_benchmark

        # Hyperparameters
        input_config = {
            "batch_size": 1,
            "out_channels": 3,
            "in_channels": 3,
            "H": 256,
            "W": 256,
            "num_pyramid": 2,
        }

        upscale_factor_list = [1]
        model = DITM
        run_benchmark(model, input_config, upscale_factor_list)
    except ImportError:
        print("Benchmark script not available, skipping benchmark test")
