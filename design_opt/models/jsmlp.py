import math
import torch.nn as nn
import torch
from torch.nn import init


class IndexLinear(nn.Module):
    def __init__(self, input_dim, out_dim, max_index=256, zero_init=False):
        super().__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.zeros(max_index, out_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(max_index, out_dim))
        if not zero_init:
            self.reset_parameters()

    def forward(self, x, ind):
        uni_ind = ind.unique()
        out = torch.zeros((x.shape[0], self.out_dim), device=x.device)
        for ind_i in uni_ind:
            W = self.W[ind_i]
            b = self.b[ind_i]
            x_ind = ind == ind_i
            out[x_ind] = torch.addmm(b, x[x_ind], W.t())
        return out

    def reset_parameters(self):
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)


class JSMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), linear_dim=None, max_index=256, activation='tanh', rescale_linear=False, zero_init=False):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid

        self.affine_layers = nn.ModuleList()
        cur_dim = input_dim
        for nh in hidden_dims:
            self.affine_layers.append(IndexLinear(cur_dim, nh, max_index, zero_init))
            cur_dim = nh

        if linear_dim is not None:
            self.linear = IndexLinear(cur_dim, linear_dim, max_index, zero_init)
            cur_dim = linear_dim
            if rescale_linear:
                self.linear.W.data.mul_(0.1)
                self.linear.b.data.mul_(0.0)

        self.out_dim = cur_dim

    def forward(self, x, ind):
        for affine in self.affine_layers:
            x = self.activation(affine(x, ind))
        if self.linear is not None:
            x = self.linear(x, ind)
        return x
