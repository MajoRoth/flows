import numpy as np
import torch
from torch import nn
from torch.nn import init
from models.affine_coupling import AffineCoupling
from models.permutation import Permutation


class FlowMatching(nn.Module):
    def __init__(self, layer_dim, dt):
        super(FlowMatching, self).__init__()
        self.dt = dt
        self.layers = nn.Sequential(
            nn.Linear(layer_dim + 1, 64).double(),
            nn.LeakyReLU(),
            nn.Linear(64, 64).double(),
            nn.LeakyReLU(),
            nn.Linear(64, 64).double(),
            nn.LeakyReLU(),
            nn.Linear(64, 64).double(),
            nn.LeakyReLU(),
            nn.Linear(64, layer_dim).double(),
        )


    def forward(self, x, t):
        x = torch.concat([x, t], dim=1)
        x = self.layers(x)
        return x

    def integrate(self, y, t_max=1, dt=None):
        if dt is None:
            dt = self.dt

        with torch.no_grad():
            for time in np.arange(0, t_max, dt):
                y += self.forward(y, torch.full((y.shape[0],1), time)) * dt
            return y

    def inverse_integrate(self, y, t_max=1, dt=None):
        if dt is None:
            dt = self.dt

        with torch.no_grad():
            for time in np.arange(t_max, -dt, -dt):
                y -= self.forward(y, torch.full((y.shape[0],1), time)) * dt
            return y


