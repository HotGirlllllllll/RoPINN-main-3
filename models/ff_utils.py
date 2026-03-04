import math
import torch
import torch.nn as nn


class FourierInputAdapter(nn.Module):
    """Map [x, t] to Fourier features and keep an even final width."""

    def __init__(
        self,
        in_dim=2,
        ff_dim=64,
        ff_scale=1.0,
        ff_seed=None,
        include_raw_input=False,
        raw_input_scale=1.0,
    ):
        super().__init__()
        in_dim = int(in_dim)
        ff_dim = max(8, int(ff_dim))
        ff_scale = float(ff_scale)
        self.include_raw_input = bool(include_raw_input)
        self.raw_input_scale = float(raw_input_scale)

        if ff_seed is None:
            b = torch.randn(in_dim, ff_dim)
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(ff_seed))
            b = torch.randn(in_dim, ff_dim, generator=gen)
        b = b * ff_scale
        self.register_buffer("fourier_b", b)

        self.out_dim = 2 * ff_dim + (in_dim if self.include_raw_input else 0)
        if self.out_dim % 2 != 0:
            raise ValueError(f"Fourier adapter output dimension must be even, got {self.out_dim}")

    def forward(self, x, t):
        src = torch.cat((x, t), dim=-1)
        proj = 2.0 * math.pi * src @ self.fourier_b
        feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
        if self.include_raw_input:
            feat = torch.cat([feat, self.raw_input_scale * src], dim=-1)
        return feat


def split_feature_as_xt(feat):
    last = int(feat.shape[-1])
    if last % 2 != 0:
        raise ValueError(f"Feature width must be even to split as x/t, got {last}")
    half = last // 2
    return feat[..., :half], feat[..., half:]
