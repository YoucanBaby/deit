import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from models.transformer.vit import ViT
from models.model import Baseline, ResNetBaseline


__all__ = [

]


@register_model
def vit_small(pretrained=False, **kwargs):
    model = ViT(
        patch_size=16,
        num_patches=196,
        dim=384,

        depth=12,

        num_self_attention_heads=6,
        self_attention_head_dim=64,

        num_classes=1000,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def cnn_baseline_small(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[0],
        ca_out=[i for i in range(12)],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def resnet_baseline_small(pretrained=False, **kwargs):
    model = Baseline(
        embedding='ResNet',         # 'CNN', 'ResNet'
        ca_in=[0],
        ca_out=[i for i in range(12)],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def cnn_multi_small_v1(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[],
        ca_out=[11],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def cnn_multi_small_v2(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[],
        ca_out=[10, 11],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def cnn_multi_small_v3(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[0],
        ca_out=[0, 11],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def cnn_multi_small_v4(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[],
        ca_out=[0, 4, 7, 11],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def cnn_multi_small_v5(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[],
        ca_out=[i for i in range(12)],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def cnn_multi_small_v6(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[],
        ca_out=[6, 11],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def cnn_multi_small_v7(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[],
        ca_out=[10, 11],
        skip_connection_out=True,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def cnn_multi_small_v8(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[],
        ca_out=[9, 10, 11],
        skip_connection_out=False,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_small_v1(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[0],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_small_v2(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[0, 1],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_small_v3(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[0, 6],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_small_v4(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[0, 3, 6, 9],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_small_v5(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        skip_connection_in=True,
        ca_in=[0],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_small_v6(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        skip_connection_in=True,
        ca_in=[0, 6],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_small_v7(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        skip_connection_in=True,
        ca_in=[0, 3, 6, 9],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_small_v8(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        skip_connection_in=True,
        ca_in=[0, 5, 9],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_small_v9(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        skip_connection_in=True,
        ca_in=[0, 1, 2],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_small_v10(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        skip_connection_in=True,
        ca_in=[0, 3, 6],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_multi_small_v1(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[0, 6],
        ca_out=[10, 11],
        skip_connection_in=True,
        skip_connection_out=False,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_cnn_multi_small_v2(pretrained=False, **kwargs):
    model = Baseline(
        embedding='CNN',  # 'CNN', 'ResNet'
        ca_in=[0, 6],
        ca_out=[10, 11],
        skip_connection_in=True,
        skip_connection_out=True,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_resnet_multi_small_v1(pretrained=False, **kwargs):
    model = ResNetBaseline(
        resnet_blocks=[2, 2, 2],
        input_layer=[3, 3],
        ca_in=[0, 6],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_resnet_multi_small_v2(pretrained=False, **kwargs):
    model = ResNetBaseline(
        resnet_blocks=[2, 2, 2, 2],
        input_layer=[3, 4],
        ca_in=[0, 6],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_resnet_multi_small_v3(pretrained=False, **kwargs):
    model = ResNetBaseline(
        resnet_blocks=[2, 2, 6],
        input_layer=[3, 3],
        ca_in=[0, 6],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_resnet_multi_small_v4(pretrained=False, **kwargs):
    model = ResNetBaseline(
        resnet_blocks=[2, 2, 6, 2],
        input_layer=[3, 4],
        ca_in=[0, 6],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_resnet_multi_small_v5(pretrained=False, **kwargs):
    model = ResNetBaseline(
        resnet_blocks=[1, 1, 1],
        input_layer=[3, 3],
        ca_in=[0, 6],
        ca_out=[],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_resnet_multi_small_v6(pretrained=False, **kwargs):
    model = ResNetBaseline(
        resnet_blocks=[2, 2, 2],
        input_layer=[3, 3],
        ca_in=[0, 6],
        ca_out=[10, 11],
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_resnet_multi_small_v7(pretrained=False, **kwargs):
    model = ResNetBaseline(
        resnet_blocks=[2, 2, 2],
        input_layer=[3, 3],
        ca_in=[0, 6],
        ca_out=[],
        skip_connection_in=True,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def multi_resnet_multi_small_v8(pretrained=False, **kwargs):
    model = ResNetBaseline(
        resnet_blocks=[2, 2, 2],
        input_layer=[3, 3],
        ca_in=[0, 6],
        ca_out=[10, 11],
        skip_connection_in=True,
    )
    model.default_cfg = _cfg()
    return model
