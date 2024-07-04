import torch
from torch import nn
from torch.nn import init
from models.affine_coupling import AffineCoupling
from models.permutation import Permutation


class NormalizingFlow(nn.Module):
    def __init__(self, layer_dim, num_layers):
        super(NormalizingFlow, self).__init__()

        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(Permutation(layer_dim))
            self.layers.append(AffineCoupling(layer_dim))


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def inverse(self, y):
        for layer in reversed(self.layers):
            y = layer.inverse(y)
        return y

    def inverse_log_det_jacobian(self, y):
        """
        computes log det jacobian
        :param y:
        :return:
        """
        sum_log_det = torch.zeros(y.shape[0])
        for layer in reversed(self.layers):
            y = layer.inverse(y)
            sum_log_det += layer.log_det_jacobian(y)

        return sum_log_det

    def loss(self, y):
        """
                computes log det jacobian
                :param y:
                :return:
                """
        sum_log_det = torch.zeros(y.shape[0])
        for layer in reversed(self.layers):
            y = layer.inverse(y)
            sum_log_det += layer.log_det_jacobian(y)

        x = y
        return x, sum_log_det



