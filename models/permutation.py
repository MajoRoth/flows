import torch
from torch import nn
from torch.nn import init


class Permutation(nn.Module):
    def __init__(self, layer_dim):
        """"
            maps z_l to (log(s), b)
        """
        super(Permutation, self).__init__()
        assert layer_dim % 2 == 0

        self.layer_dim = layer_dim

        # by chatGPT
        self.register_buffer('permutation', torch.randperm(self.layer_dim))
        self.register_buffer('inverse_permutation', torch.argsort(self.permutation))

    def forward(self, x):
        return x[:, self.permutation]

    def inverse(self, x):
        return x[:, self.inverse_permutation]

    def log_det_jacobian(self, x):
        """
        returns a 0 like tensor
        """
        return torch.zeros(x.shape[0])



