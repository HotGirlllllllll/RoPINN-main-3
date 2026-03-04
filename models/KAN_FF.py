import torch
import torch.nn as nn

from .KAN import Model as KANModel
from .ff_utils import FourierInputAdapter, split_feature_as_xt


class Model(nn.Module):
    def __init__(
        self,
        width=None,
        grid=3,
        k=3,
        noise_scale=0.1,
        noise_scale_base=0.1,
        base_fun=None,
        symbolic_enabled=True,
        bias_trainable=True,
        grid_eps=1.0,
        grid_range=None,
        sp_trainable=True,
        sb_trainable=True,
        device='cpu',
        seed=0,
        ff_dim=64,
        ff_scale=1.0,
        ff_seed=None,
        include_raw_input=False,
        raw_input_scale=1.0,
    ):
        super(Model, self).__init__()
        if base_fun is None:
            base_fun = torch.nn.SiLU()
        if grid_range is None:
            grid_range = [-1, 1]

        in_dim = 2
        self.ff = FourierInputAdapter(
            in_dim=in_dim,
            ff_dim=ff_dim,
            ff_scale=ff_scale,
            ff_seed=ff_seed,
            include_raw_input=include_raw_input,
            raw_input_scale=raw_input_scale,
        )

        if width is None:
            width = [self.ff.out_dim, 5, 1]
        else:
            width = list(width)
            width[0] = self.ff.out_dim

        self.base = KANModel(
            width=width,
            grid=grid,
            k=k,
            noise_scale=noise_scale,
            noise_scale_base=noise_scale_base,
            base_fun=base_fun,
            symbolic_enabled=symbolic_enabled,
            bias_trainable=bias_trainable,
            grid_eps=grid_eps,
            grid_range=grid_range,
            sp_trainable=sp_trainable,
            sb_trainable=sb_trainable,
            device=device,
            seed=seed,
        )

    def forward(self, x, t):
        feat = self.ff(x, t)
        x_ff, t_ff = split_feature_as_xt(feat)
        return self.base(x_ff, t_ff)
