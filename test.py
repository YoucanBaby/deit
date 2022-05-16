import os
from math import pi

import torch
from einops import rearrange, repeat
import torchvision.models as models
from timm.models.layers import trunc_normal_
from torch import nn, Tensor

from models.convnet.resnet import *

from fvcore.nn import flop_count, FlopCountAnalysis, flop_count_str


def matrix(X: Tensor, Y: Tensor) -> Tensor:
    o_h, o_w = X.shape[0], Y.shape[1]
    inner_dim = X.shape[1]
    out = torch.zeros((o_h, o_w))

    for i in range(o_h):
        for j in range(o_w):
            for k in range(inner_dim):
                out[i, j] += X[i, k] * Y[k, j]

    return out


def main():
    X = torch.ones(2, 3)
    Y = torch.ones(3, 5)
    Z = matrix(X, Y)
    print(Z)


if __name__ == '__main__':
    main()
