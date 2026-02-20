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
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, ff_dim=64, ff_scale=1.0):
        super(Model, self).__init__()
        ff_dim = max(8, int(ff_dim))
        ff_scale = float(ff_scale)

        # Fixed random Fourier matrix. Input: [x, t] -> features of size 2*ff_dim.
        b = torch.randn(in_dim, ff_dim) * ff_scale
        self.register_buffer("fourier_b", b)

        self.input_proj = nn.Linear(2 * ff_dim, hidden_dim)
        self.input_act = nn.Tanh()

        depth = max(1, num_layer - 1)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.out = nn.Linear(hidden_dim, out_dim)

    def fourier_features(self, src):
        proj = 2.0 * math.pi * src @ self.fourier_b
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

    def forward(self, x, t):
        src = torch.cat((x, t), dim=-1)
        feat = self.fourier_features(src)
        h = self.input_act(self.input_proj(feat))
        for block in self.blocks:
            h = block(h)
        return self.out(h)
