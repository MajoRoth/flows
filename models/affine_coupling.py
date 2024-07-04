import torch
from torch import nn
from torch.nn import init


class AffineCoupling(nn.Module):
    def __init__(self, layer_dim):
        """"
            maps z_l to (log(s), b)
        """
        super(AffineCoupling, self).__init__()

        assert layer_dim % 2 == 0
        self.layer_dim = layer_dim

        self.f_logs = nn.Sequential(
            nn.Linear(layer_dim // 2, 8).double(),
            nn.LeakyReLU(),
            nn.Linear(8, 8).double(),
            nn.LeakyReLU(),
            nn.Linear(8, 8).double(),
            nn.LeakyReLU(),
            nn.Linear(8, 8).double(),
            nn.LeakyReLU(),
            nn.Linear(8, layer_dim // 2).double(),
        )

        self.f_b = nn.Sequential(
            nn.Linear(layer_dim // 2, 8).double(),
            nn.LeakyReLU(),
            nn.Linear(8, 8).double(),
            nn.LeakyReLU(),
            nn.Linear(8, 8).double(),
            nn.LeakyReLU(),
            nn.Linear(8, 8).double(),
            nn.LeakyReLU(),
            nn.Linear(8, layer_dim // 2).double(),
        )

    def forward(self, z):
        """
        computes h_t(z) given z
        :param x:
        :return:
        """
        zl, zr = self.split(z)

        log_s = self.f_logs(zl)
        b = self.f_b(zl)

        yr = log_s.exp() * zr + b

        return torch.cat([zl, yr], dim=1)

    def inverse(self, y):
        """
        computes h_t^-1(y) given y
        :param y:
        :return:
        """
        yl, yr = self.split(y)

        logs = self.f_logs(yl)
        b = self.f_b(yl)

        zr = (yr - b) / logs.exp()
        return torch.cat([yl, zr], dim=1)

    def log_det_jacobian(self, y):
        """
        computes log det jacobian of h_t
        :param z:
        :return:
        """
        yl, yr = self.split(y)

        logs = self.f_logs(yl)
        return -torch.sum(logs, dim=1)

    def split(self, x):
        """
        split x into xl and xr
        :param x:
        :return:
        """
        assert x.shape[1] == self.layer_dim
        xl = x[:, :self.layer_dim // 2]
        xr = x[:, self.layer_dim // 2:]

        return xl, xr

