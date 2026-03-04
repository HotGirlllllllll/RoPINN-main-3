import torch.nn as nn

from .PINNsFormer_Enc_Only import Model as PINNsFormerEncOnlyModel
from .ff_utils import FourierInputAdapter, split_feature_as_xt


class Model(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        hidden_dim,
        num_layer,
        hidden_d_ff=512,
        heads=2,
        ff_dim=64,
        ff_scale=1.0,
        ff_seed=None,
        include_raw_input=False,
        raw_input_scale=1.0,
    ):
        super(Model, self).__init__()
        self.ff = FourierInputAdapter(
            in_dim=in_dim,
            ff_dim=ff_dim,
            ff_scale=ff_scale,
            ff_seed=ff_seed,
            include_raw_input=include_raw_input,
            raw_input_scale=raw_input_scale,
        )
        self.base = PINNsFormerEncOnlyModel(
            in_dim=self.ff.out_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            hidden_d_ff=hidden_d_ff,
            heads=heads,
        )

    def forward(self, x, t):
        feat = self.ff(x, t)
        x_ff, t_ff = split_feature_as_xt(feat)
        return self.base(x_ff, t_ff)
