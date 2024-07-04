import numpy as np
import torch
from torch import nn
from torch.nn import init
from models.affine_coupling import AffineCoupling
from models.permutation import Permutation


class FlowMatchingCond(nn.Module):
    COND_EMB_SIZE = 3
    def __init__(self, layer_dim, dt):
        super(FlowMatchingCond, self).__init__()
        self.dt = dt
        self.layers = nn.Sequential(
            nn.Linear(layer_dim + 1 + FlowMatchingCond.COND_EMB_SIZE, 64).double(),
            nn.LeakyReLU(),
            nn.Linear(64, 64).double(),
            nn.LeakyReLU(),
            nn.Linear(64, 64).double(),
            nn.LeakyReLU(),
            nn.Linear(64, 64).double(),
            nn.LeakyReLU(),
            nn.Linear(64, layer_dim).double(),
        )

        self.cond_layer = torch.nn.Embedding(5, FlowMatchingCond.COND_EMB_SIZE)


    def forward(self, x, t, cond):
        cond_emb = self.cond_layer(cond)
        x = torch.concat([x, t, cond_emb], dim=1)
        x = self.layers(x)
        return x

    def integrate(self, y, cond, t_max=1, dt=None):
        if dt is None:
            dt = self.dt

        with torch.no_grad():
            for time in np.arange(0, t_max, dt):
                y += self.forward(y, torch.full((y.shape[0], 1), time), cond) * dt
            return y


