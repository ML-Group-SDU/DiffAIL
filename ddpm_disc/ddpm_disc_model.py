import torch
from ddpm_disc.diffusion.diffusion import Diffusion
from ddpm_disc.diffusion.model import MLP
import rlkit.torch.utils.pytorch_util as ptu


class DDPM_Disc(torch.nn.Module):

    def __init__(self, x_dim, action_dim, max_action, beta_schedule, n_timesteps, device, disc_hid_dim, disc_momentum,
                 lr=0.0003, clamp_magnitude=10.0, ):
        super().__init__()
        self.model = MLP(x_dim=x_dim, hid_dim=disc_hid_dim, device=device)
        self.diffusion = Diffusion(x_dim, action_dim, self.model, max_action, beta_schedule=beta_schedule,
                                   n_timesteps=n_timesteps, clamp_magnitude=clamp_magnitude).to(device)

        self.diff_opti = torch.optim.Adam(self.diffusion.parameters(), lr=lr, betas=(disc_momentum, 0.999))


    def disc_reward(self, x):
        batch = x.to(ptu.device)
        disc_cost = self.diffusion.calc_reward(batch)
        return disc_cost
