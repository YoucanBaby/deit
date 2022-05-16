from math import pi, log

import torch
from timm.models.layers import trunc_normal_
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops.layers.torch import Reduce, Rearrange
from typing import Type, Any, Callable, Union, List, Optional

from models.convnet.resnet import resnet


def mlp(in_dim: int, mlp_ratio: int = 4):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, in_dim * mlp_ratio),
        nn.GELU(),
        nn.Linear(in_dim * mlp_ratio, in_dim),
    )


def cross_attention_layer(
    q_dim: int, kv_dim: int, num_heads: int, head_dim: int, dropout: float = 0.
):
    layer = Sequential(
        Residual(CrossAttention(q_dim, kv_dim, num_heads, head_dim, dropout), dropout),
        Residual(mlp(q_dim), dropout),
    )
    return layer


def self_attention_layer(
    x_dim: int, num_heads: int, head_dim: int, dropout: float = 0.
):
    layer = Sequential(
        Residual(SelfAttention(x_dim, num_heads, head_dim, dropout), dropout),
        Residual(mlp(x_dim), dropout)
    )
    return layer


def self_attention_block(
    num_blocks: int, x_dim: int, num_heads: int, head_dim: int, dropout: float = 0.
):
    layers = [self_attention_layer(x_dim, num_heads, head_dim, dropout) for _ in range(num_blocks)]
    return nn.Sequential(*layers)


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float, dropout_p=0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, kv_dim=None, heads=8, head_dim=64, dropout=0.):
        super().__init__()
        inner_dim = head_dim * heads

        if kv_dim is None:
            kv_dim = q_dim

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(q_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(kv_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, q_dim)

    def forward(self, x, kv=None):
        h = self.heads

        q = self.to_q(x)
        if kv is None:
            kv = q

        k, v = self.to_kv(kv).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, num_heads: int, head_dim: int, dropout: float):
        super().__init__()
        self.q_norm = nn.LayerNorm(q_dim)
        self.kv_norm = nn.LayerNorm(kv_dim)
        self.attention = MultiHeadAttention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(self, x_q, x_kv):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv)


class SelfAttention(nn.Module):
    def __init__(self, x_dim: int, num_heads: int, head_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(x_dim)
        self.attention = MultiHeadAttention(
            q_dim=x_dim,
            kv_dim=x_dim,
            heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
        )

    def forward(self, x):
        x = self.norm(x)
        return self.attention(x, x)


class Baseline(nn.Module):
    def __init__(
        self,
        embedding: str = 'CNN',             # 'CNN', 'ResNet'
        skip_connection_in: bool = True,
        skip_connection_out: bool = False,

        num_latents: int = 196,
        latent_dim: int = 384,

        num_self_attention: int = 12,

        num_heads: int = 6,
        head_dim: int = 64,

        ca_in: List[int] = [0],
        ca_out: List[int] = [i for i in range(12)],

        num_classes: int = 1000,
    ):
        super().__init__()

        self.embedding = embedding
        self.skip_connection_in = skip_connection_in
        self.skip_connection_out = skip_connection_out

        self.num_self_attention = num_self_attention
        self.ca_in = set(ca_in)
        self.ca_out = set(ca_out)

        input_dim = 384

        if embedding == 'ResNet':
            self.proj = Sequential(
                resnet(layers=[1, 1, 1, 1], size='small', width_per_group=48, return_features=False),
                Rearrange('b c h w -> b (h w) c'),
            )
        else:
            patch_size = 16
            self.proj = Sequential(
                nn.Conv2d(3, input_dim, patch_size, stride=patch_size),
                Rearrange('b c h w -> b (h w) c'),
                nn.LayerNorm(latent_dim)
            )

        self.ca_input = nn.ModuleList([
            cross_attention_layer(latent_dim, input_dim, num_heads, head_dim) for _ in range(len(ca_in))
        ])

        self.sa = nn.ModuleList([
            self_attention_layer(latent_dim, num_heads, head_dim) for _ in range(num_self_attention)
        ])

        self.ca_output = nn.ModuleList([
            cross_attention_layer(latent_dim, latent_dim, num_heads, head_dim) for _ in range(len(ca_out))
        ])

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b 1 d', 'mean') if len(ca_out) == 0 else nn.Identity(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

        self.pos_embed = nn.Parameter(torch.zeros(num_latents, latent_dim))
        self.latent = nn.Parameter(torch.zeros(num_latents, latent_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, latent_dim))

        self._init_parameters()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_parameters(self):
        with torch.no_grad():
            trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.latent, std=.02)
            trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        x = self.proj(x)        # (b, hw, c)

        b, *_ = x.shape
        pos_embed = repeat(self.pos_embed, "... -> b ...", b=b)
        x_latent = repeat(self.latent, "... -> b ...", b=b)
        cls_token = repeat(self.cls_token, "... -> b ...", b=b)

        if self.embedding != 'ResNet':
            x = x + pos_embed
            x_latent = x_latent + pos_embed

        ca_in_i = 0
        ca_out_i = 0

        if len(self.ca_in) == 0:
            x_latent = x

        for sa_i in range(self.num_self_attention):
            if sa_i in self.ca_in:
                x_latent = self.ca_input[ca_in_i](x_latent, x)

                if self.skip_connection_in:
                    x_latent = x_latent + x

                ca_in_i += 1

            x_latent = self.sa[sa_i](x_latent)

            if sa_i in self.ca_out:
                cls_token = self.ca_output[ca_out_i](cls_token, x_latent)

                if self.skip_connection_out:
                    cls_token = cls_token + reduce(x_latent, "b n d -> b 1 d", "mean")

                ca_out_i += 1

        if len(self.ca_out) == 0:
            return self.to_logits(x_latent).squeeze(dim=1)

        return self.to_logits(cls_token).squeeze(dim=1)


class ResNetBaseline(nn.Module):
    def __init__(
        self,
        skip_connection_in: bool = False,

        resnet_blocks: List[int] = [2, 2, 2],
        input_layer: List[int] = [3, 3],

        num_latents: int = 196,
        latent_dim: int = 384,

        num_self_attention: int = 12,

        num_heads: int = 6,
        head_dim: int = 64,

        ca_in: List[int] = [0, 6],
        ca_out: List[int] = [i for i in range(12)],

        num_classes: int = 1000,
    ):
        super().__init__()

        self.skip_connection_in = skip_connection_in
        self.input_layer = input_layer
        self.num_self_attention = num_self_attention
        self.ca_in = set(ca_in)
        self.ca_out = set(ca_out)

        self.proj = resnet(layers=resnet_blocks, width_per_group=64, return_features=True)

        # input_dim = {1: 96, 2: 192, 3: 384, 4: 768}
        input_dim = {1: 64, 2: 128, 3: 256, 4: 512}
        self.ca_input = nn.ModuleList([
            cross_attention_layer(latent_dim, input_dim[input_layer[i]], num_heads, head_dim) for i in range(len(ca_in))
        ])

        self.sa = nn.ModuleList([
            self_attention_layer(latent_dim, num_heads, head_dim) for _ in range(num_self_attention)
        ])

        self.ca_output = nn.ModuleList([
            cross_attention_layer(latent_dim, latent_dim, num_heads, head_dim) for _ in range(len(ca_out))
        ])

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b 1 d', 'mean') if len(ca_out) == 0 else nn.Identity(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

        self.latent = nn.Parameter(torch.zeros(num_latents, latent_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, latent_dim))

        self._init_parameters()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)
            self.cls_token.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def pre(self, x) -> List:
        features = self.proj(x)
        res = []

        for layer in self.input_layer:
            feature = rearrange(features[layer - 1], 'b c h w -> b (h w) c')
            res.append(feature)

        return res

    def forward(self, x):
        features = self.pre(x)

        b, *_ = features[0].shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)
        cls_token = repeat(self.cls_token, "... -> b ...", b=b)

        ca_in_i = 0
        ca_out_i = 0

        for sa_i in range(self.num_self_attention):
            if sa_i in self.ca_in:
                x_latent = self.ca_input[ca_in_i](x_latent, features[ca_in_i])

                if self.skip_connection_in:
                    x_latent = x_latent + x

                ca_in_i += 1

            x_latent = self.sa[sa_i](x_latent)

            if sa_i in self.ca_out:
                cls_token = self.ca_output[ca_out_i](cls_token, x_latent)
                ca_out_i += 1

        if len(self.ca_out) == 0:
            return self.to_logits(x_latent).squeeze(dim=1)

        return self.to_logits(cls_token).squeeze(dim=1)
