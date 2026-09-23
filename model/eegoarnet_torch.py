"""
EEGOAR-Net -- native PyTorch implementation.

This is a direct, layer-by-layer port of the original TensorFlow/Keras
architecture in ``EEGOARNET_architecture.py``. Every layer uses the same
kernel sizes, channel counts, padding behaviour, activation, and connectivity
as the original, so that a model loaded with the converted weights produces
numerically identical output to the original Keras model (see
``test_parity.py``).

Data layout
-----------
Internally this module uses PyTorch's native NCHW layout:

    x      : (N, 1, T, C)   T = time samples (128 by default), C = channels (64)
    mask   : (N, C) or (N, 1, 1, C)  -- boolean/float electrode-presence mask

The original Keras model used NHWC (``Input((T, C, 1))``), i.e. time as the
"height" axis and electrodes as the "width" axis, with the singleton channel
axis last. ``EEGOARNet.forward`` accepts input in that same convention
(``(N, T, C, 1)``) as well, auto-detected, so it's a drop-in replacement for
the Keras call signature -- see ``forward()`` below.

Key implementation notes (see README for the full derivation):

* All stride-1 convolutions use TF's "SAME" padding, which for an even
  kernel size pads asymmetrically (``floor`` extra pixels before,
  ``ceil`` after). PyTorch's ``padding="same"`` follows the exact same
  convention for stride-1 convolutions, so it is used directly.
* ``DepthwiseConv2D`` (TF) maps to a grouped ``nn.Conv2d`` with
  ``groups=in_channels`` and ``out_channels=in_channels*depth_multiplier``.
  TF orders the depth-multiplied output channels as
  ``k*multiplier + q`` for input channel ``k`` and multiplier index ``q``,
  which is exactly PyTorch's default grouped-conv output-channel ordering,
  so no channel re-ordering is required -- only a weight reshape.
* ``BatchNormalization`` in Keras defaults to ``epsilon=1e-3``; PyTorch's
  ``BatchNorm2d`` defaults to ``eps=1e-5``, so ``eps=1e-3`` is set explicitly
  everywhere to match.
* ``UpSampling2D`` defaults to nearest-neighbour interpolation, matching
  ``nn.Upsample(mode="nearest")``.
* ``SpatialDropout2D`` is only active during training and is a no-op at
  inference; it is included here (as ``nn.Dropout2d``) for training-time
  parity but has zero effect when the model is in ``eval()`` mode.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionUnit(nn.Module):
    """One branch of a Block-1/Block-2 "inception" unit:
    Conv2D(k,1) -> BN -> ELU -> DepthwiseConv2D(1, 2F) -> BN -> ELU.
    """

    def __init__(self, in_ch: int, filters_per_branch: int, kernel_h: int,
                 use_bias: bool, activation: str = "elu"):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, filters_per_branch,
                               kernel_size=(kernel_h, 1),
                               padding="same", bias=use_bias)
        self.bn1 = nn.BatchNorm2d(filters_per_branch, eps=1e-3)
        self.dwconv = nn.Conv2d(filters_per_branch, filters_per_branch * 2,
                                 kernel_size=(1, 2 * filters_per_branch),
                                 groups=filters_per_branch,
                                 padding="same", bias=False)
        self.bn2 = nn.BatchNorm2d(filters_per_branch * 2, eps=1e-3)
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv(x)))
        x = self.act(self.bn2(self.dwconv(x)))
        return x


class ConvBnAct(nn.Module):
    """Conv2D -> BN -> activation (+ optional SpatialDropout2D, inference no-op)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size, dropout_rate: float = 0.0,
                 activation: str = "elu", bias: bool = False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size,
                               padding="same", bias=bias)
        self.bn = nn.BatchNorm2d(out_ch, eps=1e-3)
        self.act = _activation(activation)
        self.drop = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        return self.drop(x)


def _activation(name: str) -> nn.Module:
    if name == "elu":
        return nn.ELU(alpha=1.0)
    if name == "linear":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class EEGOARNET(nn.Module):
    """Native PyTorch re-implementation of EEGOAR-Net.

    Parameters mirror ``EEGOARNET_architecture.EEGOARNET`` exactly. Only
    ``fs=128`` / ``input_time=1000`` / ``scales_time=(500,250,125)`` /
    ``filters_per_branch=8`` (i.e. the defaults used to train the shipped
    weights) have been validated bit-for-bit against TensorFlow -- other
    configurations follow the same rules but haven't been numerically
    re-verified.
    """

    def __init__(self, input_time: int = 1000, fs: int = 128, ncha: int = 64,
                 filters_per_branch: int = 8,
                 scales_time: Sequence[int] = (500, 250, 125),
                 dropout_rate: float = 0.25, activation: str = "elu"):
        super().__init__()
        self.ncha = ncha
        self.input_samples = int(input_time * fs / 1000)
        scales_samples = [int(s * fs / 1000) for s in scales_time]
        n_scales = len(scales_samples)
        F_ = filters_per_branch

        # ---- Block 1: inception (3 branches over the raw signal) ----
        self.b1_units = nn.ModuleList([
            InceptionUnit(1, F_, k, use_bias=True, activation=activation)
            for k in scales_samples
        ])
        self.b1_pool = nn.MaxPool2d((4, 2))

        # ---- Block 2: inception (3 branches over pooled Block-1 output) ----
        b1_out_ch = F_ * 2 * n_scales
        self.b2_units = nn.ModuleList([
            InceptionUnit(b1_out_ch, F_, max(1, k // 4), use_bias=False, activation=activation)
            for k in scales_samples
        ])
        self.b2_pool = nn.MaxPool2d((2, 2))

        # ---- Block 3: encoder ----
        b2_out_ch = F_ * 2 * n_scales
        c3 = F_ * n_scales * 4
        self.b3_u1 = ConvBnAct(b2_out_ch, c3, (3, 1), dropout_rate, activation)
        self.b3_u2 = ConvBnAct(c3, c3, (3, 1), dropout_rate, activation)
        self.b3_pool = nn.MaxPool2d((2, 2))

        # ---- Block 4: encoder (bottleneck) ----
        c4 = F_ * n_scales * 6
        self.b4_u1 = ConvBnAct(c3, c4, (3, 1), 0.0, activation)          # no dropout (matches original: commented out)
        self.b4_u2 = ConvBnAct(c4, c4, (3, 1), dropout_rate, activation)

        # ---- Block 5: decoder ----
        c5a = F_ * n_scales * 4
        self.up1 = nn.Upsample(scale_factor=(2, 2), mode="nearest")
        self.b5_u1a = ConvBnAct(c4, c5a, (3, 3), 0.0, activation)
        self.b5_u1b = ConvBnAct(c3 + c5a, c5a, (3, 3), 0.0, activation)  # concat with b3_u2

        c5b = F_ * n_scales * 2
        self.up2 = nn.Upsample(scale_factor=(2, 2), mode="nearest")
        self.b5_u2a = ConvBnAct(c5a, c5b, (3, 3), 0.0, activation)
        self.b5_u2b = ConvBnAct(b2_out_ch + c5b, c5b, (3, 3), 0.0, activation)  # concat with b2_units

        c5c = F_ * n_scales * 2
        self.up3 = nn.Upsample(scale_factor=(4, 2), mode="nearest")
        self.b5_u3a = ConvBnAct(c5b, c5c, (3, 3), 0.0, activation)
        self.b5_u3b = ConvBnAct(b1_out_ch + c5c, c5c, (3, 3), 0.0, activation)  # concat with b1_units

        # ---- Output head ----
        self.out_conv = nn.Conv2d(c5c, 1, kernel_size=(1, 1), padding="same", bias=False)
        self.out_bn = nn.BatchNorm2d(1, eps=1e-3)

    # ------------------------------------------------------------------ #
    def forward(self, signal: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        signal : (N, 1, T, C) NCHW float tensor, OR (N, T, C, 1) NHWC float
                 tensor (Keras convention) -- auto-detected from shape.
        mask   : (N, C) or (N, C, 1) or (N, 1, 1, C) -- electrode-presence
                 mask (1 = electrode present, 0 = absent/padded).

        Returns
        -------
        (N, 1, T, C) NCHW tensor (mirrors the input's convention: if the
        input was given NHWC, the output is returned NHWC as (N, T, C, 1)).
        """
        nhwc_input = signal.shape[-1] == 1 and signal.shape[1] != 1
        if nhwc_input:
            x = signal.permute(0, 3, 1, 2).contiguous()  # NHWC -> NCHW
        else:
            x = signal

        mask = mask.reshape(mask.shape[0], -1).to(x.dtype)  # (N, C)
        mask4d = mask.view(mask.shape[0], 1, 1, self.ncha)  # broadcast over (1, T, C)

        b1_outs = [unit(x) for unit in self.b1_units]
        b1_out = torch.cat(b1_outs, dim=1)
        b1_out = self.b1_pool(b1_out)

        b2_outs = [unit(b1_out) for unit in self.b2_units]
        b2_out = torch.cat(b2_outs, dim=1)
        b2_out = self.b2_pool(b2_out)

        b3_u1 = self.b3_u1(b2_out)
        b3_u2 = self.b3_u2(b3_u1)
        b3_out = self.b3_pool(b3_u2)

        b4_u1 = self.b4_u1(b3_out)
        b4_u2 = self.b4_u2(b4_u1)

        b5_u1 = self.b5_u1a(self.up1(b4_u2))
        b5_u1 = torch.cat([b3_u2, b5_u1], dim=1)
        b5_u1 = self.b5_u1b(b5_u1)

        b5_u2 = self.b5_u2a(self.up2(b5_u1))
        b5_u2 = torch.cat([torch.cat(b2_outs, dim=1), b5_u2], dim=1)
        b5_u2 = self.b5_u2b(b5_u2)

        b5_u3 = self.b5_u3a(self.up3(b5_u2))
        b5_u3 = torch.cat([torch.cat(b1_outs, dim=1), b5_u3], dim=1)
        b5_u3 = self.b5_u3b(b5_u3)

        out = self.out_bn(self.out_conv(b5_u3))  # linear activation = identity
        out = out * mask4d

        if nhwc_input:
            out = out.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
        return out


def custom_mse_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """PyTorch equivalent of the original ``custom_mse`` Keras loss:
    time-domain MSE + MSE of the (complex) FFT along the time axis.

    Expects NCHW tensors (N, 1, T, C); the FFT is taken along the time
    axis (dim=2), matching the original's FFT over the "height" axis.
    """
    mse_time = F.mse_loss(y_pred, y_true)
    fft_true = torch.fft.fft(y_true.to(torch.complex64), dim=2)
    fft_pred = torch.fft.fft(y_pred.to(torch.complex64), dim=2)
    # complex MSE == mean squared magnitude of the difference
    mse_fft = torch.mean(torch.abs(fft_true - fft_pred) ** 2)
    return mse_time + mse_fft
