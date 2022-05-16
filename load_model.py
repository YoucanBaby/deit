import torch
import timm
assert timm.__version__ == "0.3.2"

model = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)