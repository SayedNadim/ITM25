import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------
# Basic building blocks for DITM
# ---------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=True,
        use_norm=False,
        use_act=True,
        norm_type="bn",
    ):
        super(BasicBlock, self).__init__()
        bias = bias if use_norm is False else False
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode="reflect",
        )
        self.use_norm = use_norm

        if use_norm:
            if norm_type == "bn":
                self.norm = nn.BatchNorm2d(out_channels)
            elif norm_type == "in":
                self.norm = nn.InstanceNorm2d(out_channels)
            else:
                raise ValueError("Unsupported normalization type")
        self.use_act = use_act
        if use_act:
            self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_norm:
            x = self.norm(x)
        if self.use_act:
            x = self.act(x)
        return x


# ---------------------------------------------------
# Residual block with two convolutional layers
# ---------------------------------------------------


# class ResBlock(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size=3,
#         stride=1,
#         padding=1,
#         dilation=1,
#         groups=1,
#     ):
#         super(ResBlock, self).__init__()
#         self.conv1 = BasicBlock(
#             in_channels, out_channels, kernel_size, stride, padding, dilation, groups
#         )
#         self.conv2 = BasicBlock(
#             out_channels,
#             out_channels,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             dilation=1,
#             groups=1,
#             use_act=False,  # No activation before residual add
#         )
#         self.shortcut = (
#             BasicBlock(
#                 in_channels,
#                 out_channels,
#                 kernel_size=1,  # Use 1x1 for channel matching
#                 stride=stride,  # Match stride
#                 padding=0,
#                 dilation=1,
#                 groups=1,
#                 use_act=False,
#             )
#             if in_channels != out_channels or stride != 1
#             else nn.Identity()
#         )
#         self.act = nn.LeakyReLU(0.1, inplace=True)

#     def forward(self, x):
#         residual = self.shortcut(x)
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = x + residual
#         x = self.act(x)  # Activation after residual add
#         return x


# ---------------------------------------------------
# Upsampling block with convolution
# ---------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.upsample = nn.Upsample(
            scale_factor=scale, mode="bilinear", align_corners=False
        )
        self.conv = BasicBlock(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


# ---------------------------------------------------
# Dense block with multiple dense units
# ---------------------------------------------------
class DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_layers: int = 2,
        growth_rate: int = 12,
        *,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_layers = n_layers
        self.growth_rate = growth_rate

        pad = kernel_size // 2
        dense_units = []
        num_channels = in_channels
        for _ in range(n_layers):
            # one dense unit ≡ two convs + concat
            unit = nn.Sequential(
                BasicBlock(
                    num_channels,
                    growth_rate,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=pad,
                ),
                BasicBlock(
                    growth_rate,
                    growth_rate,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=pad,
                ),
            )
            dense_units.append(unit)
            num_channels += growth_rate  # after concat
        self.dense_units = nn.ModuleList(dense_units)

        # 1×1 projection back to `in_channels` before residual add
        self.conv1x1 = nn.Conv2d(num_channels, in_channels, kernel_size=1, bias=False)

    # ---------------------------------------------------------------------
    def forward(self, x):  # noqa: N802
        out = x
        for unit in self.dense_units:
            new_feat = unit(out)
            out = torch.cat((out, new_feat), dim=1)
        out = self.conv1x1(out)
        return out + x


# ---------------------------------------------------
# Feature extractor with residual connections
# ---------------------------------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = BasicBlock(in_ch, out_ch)
        body = [
            BasicBlock(out_ch, out_ch),
            BasicBlock(out_ch, out_ch),
            BasicBlock(out_ch, out_ch),
        ]
        self.body = nn.Sequential(*body)

    def forward(self, x):
        skip = self.conv1(x)
        x = self.body(skip)
        return skip + x  # residual connection


# ---------------------------------------------------
# Fusion module for multi-scale feature fusion
# ---------------------------------------------------
class FusionModule(nn.Module):
    def __init__(
        self,
        channels: int,
        prev_channels: int,
        *,
        layers: int = 3,
        eca_k: int = 3,
        spa_k: int = 7,
    ):
        super().__init__()

        blocks = [BasicBlock(channels, channels)]
        for _ in range(layers - 1):
            blocks.append(BasicBlock(channels, channels))
        self.body = nn.Sequential(*blocks)

        self.eca_conv = nn.Conv1d(
            1, 1, kernel_size=eca_k, padding=eca_k // 2, bias=False
        )

        pad = spa_k // 2
        self.spa_att = nn.Sequential(
            nn.Conv2d(
                1, 1, (spa_k, 1), padding=(pad, 0), bias=False, padding_mode="reflect"
            ),
            nn.Conv2d(
                1, 1, (1, spa_k), padding=(0, pad), bias=False, padding_mode="reflect"
            ),
        )

        self.x_conv = FeatureExtractor(channels + prev_channels, channels)
        self.y_conv = BasicBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.r_conv = BasicBlock(
            prev_channels, channels, kernel_size=3, stride=1, padding=1
        )
        self.ridge_refine = BasicBlock(channels, channels)
        self.proj = nn.Conv2d(channels, channels, 1)

    # -----------------------------------------------------------------------
    def _attention_mask(self, feat):
        B, C, H, W = feat.shape
        # Efficient Channel Attention (ECA)
        chn = feat.mean((-1, -2), keepdim=False).unsqueeze(1)  # (B,1,C)
        chn_weight = torch.sigmoid(self.eca_conv(chn)).view(B, C, 1, 1)

        # Separable spatial attention
        spa = feat.mean(1, keepdim=True)  # (B,1,H,W)
        spa_weight = torch.sigmoid(self.spa_att(spa))  # (B,1,H,W)

        return chn_weight * spa_weight  # (B,C,H,W) - Remove redundant sigmoid

    # -----------------------------------------------------------------------
    def forward(self, src_feat, prev_feat):
        # Concatenate and compress
        feats = self.x_conv(torch.cat([src_feat, prev_feat], dim=1))

        # Inlined IndexNet body + residual
        body_out = self.body(feats)
        feat = body_out + feats

        # Attention mask
        w = self._attention_mask(feat)

        # Fuse
        y = self.y_conv(feat)
        fused = y * w
        ridge = self.ridge_refine(fused)
        return self.proj(ridge)


# ---------------------------------------------------
# Learnable exposure mask for HDR reconstruction
# ---------------------------------------------------
class LearnableExposureMask(nn.Module):
    def __init__(
        self,
        n: int = 3,
        slope: float = 8.0,
        k: int = 5,
    ) -> None:
        super().__init__()
        if n < 2:
            raise ValueError("n must be ≥ 2")
        self.logits_tau = nn.Parameter(torch.zeros(n - 1))
        self.slope = nn.Parameter(torch.tensor(slope).log())  # log-space
        blur = torch.ones(1, 1, k, k) / (k * k)
        self.register_buffer("blur", blur, False)

    def forward(self, ldr):
        # Linear RGB to luminance (ITU-R BT.709 linear coefficients)
        luma = 0.2126 * ldr[:, 0:1] + 0.7152 * ldr[:, 1:2] + 0.0722 * ldr[:, 2:3]
        avg = F.conv2d(luma, self.blur, padding=2)

        taus = torch.cumsum(torch.softmax(self.logits_tau, 0), 0)  # (n-1,)
        alpha = (
            torch.exp(self.slope) * 10.0 + 3.0
        )  # Use exp instead of sigmoid for better range

        cdfs = [torch.sigmoid(alpha * (avg - t)) for t in taus]  # n-1 sigm.
        masks = [1.0 - cdfs[0]]  # dark
        masks += [cdfs[i - 1] - cdfs[i] for i in range(1, len(cdfs))]  # mids
        masks += [cdfs[-1]]  # bright

        stack = torch.stack(masks, 1)
        stack = stack / stack.sum(1, keepdim=True).clamp_min(1e-6)
        return tuple(stack.unbind(1))


# ---------------------------------------------------
# Noise estimation and denoising modules
# ---------------------------------------------------
class NoiseEstimationModule(nn.Module):
    def __init__(self, in_channels, feat=64, levels=3):
        super().__init__()
        self.feature = BasicBlock(in_channels, feat, kernel_size=3, stride=1, padding=1)
        self.blocks = nn.Sequential(
            BasicBlock(in_channels, feat, kernel_size=3, stride=1, padding=1),
            BasicBlock(feat, feat, kernel_size=3, stride=1, padding=2, dilation=2),
            BasicBlock(feat, feat, kernel_size=3, stride=1, padding=4, dilation=4),
        )
        self.noise_map = nn.Sequential(
            BasicBlock(feat, feat, kernel_size=3, stride=1, padding=1),
            BasicBlock(feat, feat, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(feat, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),  # Bound noise to [-1, 1]
        )

    def forward(self, x):
        # Extract features
        res = self.feature(x)
        # Extract features using dense blocks
        features = self.blocks(x)
        # Estimate noise map
        noise_map = self.noise_map(features)
        return noise_map


class ResidualDenoiseModule(nn.Module):
    def __init__(self, in_channels, feat=64):
        super().__init__()
        self.fuse = BasicBlock(
            in_channels * 2, feat, kernel_size=3, stride=1, padding=1  # 2 instead of 3
        )

        self.refine = nn.Sequential(
            BasicBlock(feat, feat, kernel_size=3, stride=1, padding=1),
            BasicBlock(feat, feat, kernel_size=3, stride=1, padding=1),
        )

        self.out = nn.Conv2d(
            feat,
            in_channels,
            3,
            1,
            1,
        )

        # Learnable weight for residual connection
        self.beta = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, s, n):
        # Concatenate input and noise
        signal = torch.cat([s, n], 1)

        fuse = self.fuse(signal)
        refine = self.refine(fuse)
        out = self.out(refine)

        # Residual connection with learnable weight
        beta = torch.tanh(self.beta)  # Ensure beta is in [-1, 1]
        return s * (1 + beta * out)  # Multiplicative enhancement


# ---------------------------------------------------
# BitRecoverNet for HDR reconstruction
# ---------------------------------------------------
class BitRecoverNet(nn.Module):
    def __init__(self, in_channels: int = 3, *, base: int = 32, levels: int = 3):
        super().__init__()

        self.exposure_mask = LearnableExposureMask()
        self.noise_map = NoiseEstimationModule(in_channels=in_channels, feat=base)
        self.denoiser = ResidualDenoiseModule(in_channels=in_channels, feat=base)

        self.weight_net = nn.Sequential(
            BasicBlock(in_channels * 3, base),  # concat under/over/mid
            BasicBlock(base, base, kernel_size=3, stride=1, padding=1),
            BasicBlock(base, base, kernel_size=3, stride=1, padding=1),
            BasicBlock(base, 3, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=1),
        )

        self.fuse = nn.Sequential(
            BasicBlock(in_channels, base),
            BasicBlock(base, base, kernel_size=3, stride=1, padding=1),
            BasicBlock(base, base, kernel_size=3, stride=1, padding=1),
            BasicBlock(base, in_channels, kernel_size=3, stride=1, padding=1),
        )

        self.alpha = nn.Parameter(torch.tensor(0.1), requires_grad=True)

    def forward(self, x: torch.Tensor):
        noise = self.noise_map(x)
        x_clean = self.denoiser(x, noise)  # cleaned input

        under_m, mid_m, over_m = self.exposure_mask(x_clean)  # each [B,1,H,W]
        x_under = x_clean * under_m
        x_over = x_clean * over_m
        x_mid = x_clean * mid_m

        exposure_concat = torch.cat([x_under, x_over, x_mid], dim=1)  # [B,3C,H,W]
        weights = self.weight_net(exposure_concat)  # [B,3,H,W]
        weights = weights.unsqueeze(2)  # [B,3,1,H,W]

        stacked = torch.stack([x_under, x_over, x_mid], dim=1)  # [B,3,C,H,W]
        # Weighted sum without division - weights already sum to 1
        fused = (stacked * weights).sum(dim=1)  # [B,C,H,W]

        residual = self.fuse(fused)
        alpha = torch.sigmoid(self.alpha)  # Ensure alpha is in [0, 1]
        x_deq = fused * (1 + alpha * residual)  # Multiplicative enhancement

        return x_clean, x_deq


# ---------------------------------------------------
# BitExpansionNet for multi-scale HDR prediction
# ---------------------------------------------------
class BitExpansionNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        *,
        base: int = 32,
        out_channels: int = 3,
        levels: int = 3,
    ):
        super().__init__()
        in_c = in_channels
        c1, c2, c3 = base, int(base * 1.5), int(base * 2)

        # Encoder
        self.conv1 = BasicBlock(in_c, c1, stride=1)
        self.conv2 = BasicBlock(c1, c2, stride=2)
        self.conv3 = BasicBlock(c2, c3, stride=2)

        # Bottleneck – at least 2 blocks
        b_blocks = max(2, levels - 1)
        self.bottleneck = nn.Sequential(*[BasicBlock(c3, c3) for _ in range(b_blocks)])

        # Decoder / fusion
        self.fm1 = FusionModule(c3, c3)
        self.fm1_up = UpBlock(c3, c2)
        self.fm1_res = BasicBlock(c2, c2)

        self.fm2 = FusionModule(c2, c2)
        self.fm2_up = UpBlock(c2, c1)
        self.fm2_res = BasicBlock(c1, c1)

        self.fm3 = FusionModule(c1, c1)
        self.fm3_res = BasicBlock(c1, c1)

        # Projections at multiple scales
        self.proj1 = ProjectionLayer(c2, c2, out_channels, scale=1)
        self.proj2 = ProjectionLayer(c1, c2, out_channels, scale=2)
        self.proj3 = ProjectionLayer(c1, c1, out_channels, scale=1)

    def forward(self, x_lin):
        # Encoder
        x1 = self.conv1(x_lin)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        # Bottleneck
        b3 = self.bottleneck(x3)

        # Decoder w/ fusion
        f1 = self.fm1(b3, x3)
        f1 = self.fm1_up(f1)
        f1 = self.fm1_res(f1)

        f2 = self.fm2(f1, x2)
        f2 = self.fm2_up(f2)
        f2 = self.fm2_res(f2)

        f3 = self.fm3(f2, x1)
        f3 = self.fm3_res(f3)

        # Multi-scale HDR predictions
        f1, o1 = self.proj1(f1, add_feat=None, img_skip=x_lin)
        f2, o2 = self.proj2(f2, add_feat=f1, img_skip=o1)
        f3, o3 = self.proj3(f3, add_feat=f2, img_skip=o2)

        return o1, o2, o3


# ---------------------------------------------------
# Detail enhancement network for HDR refinement
# ---------------------------------------------------
class DetailEnhancementNet(nn.Module):
    def __init__(self, in_channels, base=64, out_channels=3):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(in_channels, 16)
        self.enc2 = self._conv_block(16, 32, stride=2)
        self.enc3 = self._conv_block(32, 64, stride=2)
        self.enc4 = self._conv_block(64, 128, stride=2)
        self.enc5 = self._conv_block(128, 256, stride=2)

        # Decoder with resize-convolution to avoid checkerboard artifacts
        self.dec5 = self._resize_conv_block(256, 128)
        self.dec4 = self._resize_conv_block(256, 64)  # 128 + 128 from skip
        self.dec3 = self._resize_conv_block(128, 32)  # 64 + 64 from skip
        self.dec2 = self._resize_conv_block(64, 16)  # 32 + 32 from skip
        self.dec1 = self._conv_block(32, 16)  # 16 + 16 from skip

        self.final_conv = nn.Conv2d(16, out_channels, 1)
        self.alpha = nn.Parameter(
            torch.tensor(0.1), requires_grad=True  # Better initialization
        )

    def _conv_block(self, in_ch, out_ch, stride=1):
        return BasicBlock(
            in_ch,
            out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

    def _resize_conv_block(self, in_ch, out_ch):
        return UpBlock(
            in_ch,
            out_ch,
            scale=2,
        )

    def forward(self, s, gamma=0.95):
        e1 = self.enc1(s)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        d5 = self.dec5(e5)
        d5 = F.interpolate(d5, size=e4.shape[2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d5, e4], 1))
        d4 = F.interpolate(d4, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d3 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d2 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d2, e1], 1))

        residuals = self.final_conv(d1)
        alpha = torch.sigmoid(self.alpha)  # Bound to [0, 1]
        hdr_image = s * (1 + alpha * residuals)  # Multiplicative enhancement

        return hdr_image


# ---------------------------------------------------
# Projection layer for final HDR prediction
# ---------------------------------------------------
class ProjectionLayer(nn.Module):
    def __init__(self, num_features, interm_features, out_channels=3, scale=2):
        super().__init__()
        self.align = BasicBlock(interm_features, num_features, kernel_size=3, padding=1)
        self.head = nn.Conv2d(
            num_features, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x, add_feat=None, img_skip=None):
        if add_feat is not None:
            y = F.interpolate(
                add_feat, size=x.shape[2:], mode="bilinear", align_corners=False
            )
            y = self.align(y)
            x = x + y

        pred = self.head(x)

        if img_skip is not None:
            pred = F.interpolate(
                pred, size=img_skip.shape[2:], mode="bilinear", align_corners=False
            )
            pred = img_skip + pred

        return x, pred


# ---------------------------------------------------
# UNet architecture for HDR reconstruction
# ---------------------------------------------------
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        *,
        base: int = 16,
        levels: int = 4,
    ):
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be ≥ 1")

        # -------- channel plan ------------------------------------------------
        widths = [base * 2**i for i in range(levels)]

        # -------- encoders ----------------------------------------------------
        encs = []
        for i, w in enumerate(widths):
            if i == 0:
                encs.append(BasicBlock(in_channels, w, stride=1))
            else:
                encs.append(BasicBlock(widths[i - 1], w, stride=2))
        self.encs = nn.ModuleList(encs)

        # -------- bottleneck (optional extra conv without down-sample) --------
        self.bottleneck = BasicBlock(widths[-1], widths[-1])

        # -------- up-projections + decoders -----------------------------------
        ups, decs = [], []
        for i in reversed(range(1, levels)):
            ups.append(UpBlock(widths[i], widths[i - 1]))
            decs.append(BasicBlock(widths[i], widths[i - 1]))
        self.ups = nn.ModuleList(ups)
        self.decs = nn.ModuleList(decs)

        # -------- output head -------------------------------------------------
        self.out_conv = nn.Conv2d(widths[0], out_channels, 1)

    # -------------------------------------------------------------------------
    # forward
    # -------------------------------------------------------------------------
    def forward(self, x):
        skips = []
        for enc in self.encs:
            x = enc(x)
            skips.append(x)

        x = self.bottleneck(x)  # deepest representation

        for up, dec, skip in zip(self.ups, self.decs, reversed(skips[:-1])):
            x = up(x)
            x = dec(torch.cat([x, skip], dim=1))

        return self.out_conv(x)


# -----------------------------------------------------------------------------
# Weight initialisation helper
# -----------------------------------------------------------------------------
def _kaiming_init(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(
            m.weight, a=0.0, mode="fan_in", nonlinearity="leaky_relu"
        )
        if m.bias is not None:
            nn.init.zeros_(m.bias)
