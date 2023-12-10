
import torch
import torch.nn as nn

from ddpm_disc.diffusion.helpers import SinusoidalPosEmb


class MLP(nn.Module):
    """
    MLP Model
    """

    def __init__(self,
                 x_dim,
                 hid_dim,
                 device,
                 t_dim=16):
        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = x_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, hid_dim),
                                       nn.Mish(),
                                       nn.Linear(hid_dim, hid_dim),
                                       nn.Mish(),
                                       nn.Linear(hid_dim, hid_dim),
                                       nn.Mish())

        self.final_layer = nn.Linear(hid_dim, x_dim)

    def forward(self, x, time):
        t = self.time_mlp(time)
        x = torch.cat([x, t], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)
