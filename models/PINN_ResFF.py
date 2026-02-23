# Residual PINN with Fourier feature embedding.
# This keeps the same call signature as PINN for drop-in use.

import math
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, dim),
        )
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(x + self.net(x))


class Model(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        num_layer,
        ff_dim=64,
        ff_scale=1.0,
        ff_scale_x=None,
        ff_scale_t=None,
        use_characteristic=False,
        adv_speed=50.0,
        ff_scale_char=None,
        ff_basis='gaussian',
        include_raw_input=False,
        raw_input_scale=1.0,
    ):
        super(Model, self).__init__()
        ff_dim = max(8, int(ff_dim))
        ff_scale = float(ff_scale)
        self.use_characteristic = bool(use_characteristic)
        self.adv_speed = float(adv_speed)
        self.include_raw_input = bool(include_raw_input)
        self.raw_input_scale = float(raw_input_scale)

        # Fixed random Fourier matrix.
        # Default input is [x, t]. If enabled, characteristic feature [x - c t] is appended.
        eff_in_dim = int(in_dim) + (1 if self.use_characteristic else 0)

        # Allow anisotropic scaling by coordinates to test directional inductive bias.
        sx = float(ff_scale if ff_scale_x is None else ff_scale_x)
        st = float(ff_scale if ff_scale_t is None else ff_scale_t)
        sc = float(ff_scale if ff_scale_char is None else ff_scale_char)
        scale_vec = [sx, st] + ([sc] if self.use_characteristic else [])
        scale = torch.tensor(scale_vec, dtype=torch.float32).view(eff_in_dim, 1)
        basis = str(ff_basis).strip().lower()
        if basis == "gaussian":
            b = torch.randn(eff_in_dim, ff_dim)
        elif basis == "axis":
            # Deterministic axis-aligned frequency basis to reduce seed sensitivity.
            b = torch.zeros(eff_in_dim, ff_dim, dtype=torch.float32)
            for j in range(ff_dim):
                axis = j % eff_in_dim
                level = 1.0 + float(j // eff_in_dim)
                b[axis, j] = level
        else:
            raise ValueError(f"Unknown ff_basis: {ff_basis}")
        b = b * scale
        self.register_buffer("fourier_b", b)

        input_width = 2 * ff_dim + (eff_in_dim if self.include_raw_input else 0)
        self.input_proj = nn.Linear(input_width, hidden_dim)
        self.input_act = nn.Tanh()

        depth = max(1, num_layer - 1)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.out = nn.Linear(hidden_dim, out_dim)

    def fourier_features(self, src):
        proj = 2.0 * math.pi * src @ self.fourier_b
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x, t):
        if self.use_characteristic:
            char = x - self.adv_speed * t
            src = torch.cat((x, t, char), dim=-1)
        else:
            src = torch.cat((x, t), dim=-1)
        feat = self.fourier_features(src)
        if self.include_raw_input:
            feat = torch.cat((feat, self.raw_input_scale * src), dim=-1)
        h = self.input_act(self.input_proj(feat))
        for block in self.blocks:
            h = block(h)
        return self.out(h)
