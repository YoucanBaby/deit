import os
from math import pi
import random

import torch
from einops import rearrange, repeat
import torchvision.models as models
from timm.models.layers import trunc_normal_
from torch import nn, Tensor

from models.convnet.resnet import *

from fvcore.nn import flop_count, FlopCountAnalysis, flop_count_str


def main():
    random_list = random.sample(range(50), 10)
    random_list.sort()
    print(random_list)


if __name__ == '__main__':
    main()
