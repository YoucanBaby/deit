from math import pi, log

import torch
from timm.models.layers import trunc_normal_
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce
from typing import Tuple

from models.resnet import resnet18, resnet48


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def fourier_encode(x, max_freq=224, num_bands=64):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device=device, dtype=dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim=-1)
    x = torch.cat((x, orig_x), dim=-1)
    return x


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


class Perceiver(nn.Module):
    def __init__(
        self,
        token_location: str = 'None',       # 'None', 'Encoder-CA', 'Encoder-SA', 'Decoder'

        max_freq: int = 224,
        num_bands: int = 10,

        num_latents: int = 196,
        latent_dim: int = 384,

        num_blocks: int = 3,
        self_attention_per_block: int = 4,

        num_cross_attention_heads: int = 6,
        num_self_attention_heads: int = 6,

        cross_attention_head_dim: int = 64,
        self_attention_head_dim: int = 64,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.token_location = token_location
        self.max_freq = max_freq
        self.num_freq_bands = num_bands

        # 加入傅里叶变换
        input_dim = 3 + 2 * (num_bands * 2 + 1)

        if token_location == 'None':
            self.to_logits = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif token_location == 'Encoder_CA':
            self.cls_token = nn.Parameter(torch.zeros(1, latent_dim))
            self.to_logits = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif token_location == 'Encoder_SA':
            self.cls_token = nn.Parameter(torch.zeros(1, latent_dim))
            self.to_logits = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif token_location == 'Decoder':
            self.cls_token = nn.Parameter(torch.zeros(1, num_classes))
            self.decoder = cross_attention_layer(
                q_dim=num_classes,
                kv_dim=latent_dim,
                num_heads=num_cross_attention_heads,
                head_dim=cross_attention_head_dim,
            )
            self.to_logits = nn.Sequential(
                nn.LayerNorm(num_classes),
                nn.Linear(num_classes, num_classes)
            )

        self.ca = cross_attention_layer(
            q_dim=latent_dim,
            kv_dim=input_dim,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
        )
        self.sa = self_attention_block(
            num_blocks * self_attention_per_block,
            x_dim=latent_dim,
            num_heads=num_self_attention_heads,
            head_dim=self_attention_head_dim,
        )

        # 初始化latent
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        b, *axis, _, device, dtype = *x.shape, x.device, x.dtype

        # 加入傅里叶变换
        axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
        pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
        enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
        enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
        enc_pos = repeat(enc_pos, '... -> b ...', b=b)
        x = torch.cat((x, enc_pos), dim=-1)

        x = rearrange(x, "b ... c -> b (...) c")

        x_latent = repeat(self.latent, "... -> b ...", b=b)

        if self.token_location == 'Encoder_CA':
            cls_token = repeat(self.cls_token, "... -> b ...", b=b)
            x_latent = torch.cat([cls_token, x_latent], 1)

        x_latent = self.ca(x_latent, x)

        if self.token_location == 'Encoder_SA':
            cls_token = repeat(self.cls_token, "... -> b ...", b=b)
            x_latent = torch.cat([cls_token, x_latent], 1)

        x_latent = self.sa(x_latent)

        if self.token_location == 'None':
            y = self.to_logits(x_latent)
        elif self.token_location == 'Encoder_CA':
            x_latent = x_latent[:, 0, :]
            y = self.to_logits(x_latent)
        elif self.token_location == 'Encoder_SA':
            x_latent = x_latent[:, 0, :]
            y = self.to_logits(x_latent)
        elif self.token_location == 'Decoder':
            cls_token = repeat(self.cls_token, "... -> b ...", b=b)
            y = self.decoder(cls_token, x_latent)
            y = self.to_logits(y).squeeze(dim=1)

        return y


class CNNPerceiver(nn.Module):
    def __init__(
        self,
        embedding: str = 'CNN',             # 'CNN', 'CNN-Position', 'ResNet', 'MobileNet'

        token_location: str = 'None',       # 'None', 'Encoder-CA', 'Encoder-SA', 'Decoder'

        num_latents: int = 196,
        latent_dim: int = 384,

        num_blocks: int = 4,
        self_attention_per_block: int = 3,

        num_cross_attention_heads: int = 6,
        num_self_attention_heads: int = 6,

        cross_attention_head_dim: int = 64,
        self_attention_head_dim: int = 64,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.embedding = embedding
        self.token_location = token_location

        if embedding == 'CNN':
            patch_size = 16
            input_dim = 384
            self.proj = nn.Conv2d(3, input_dim, kernel_size=patch_size, stride=patch_size)
            self.ln = nn.LayerNorm(latent_dim)
        elif embedding == 'CNN-Position':
            pass
        elif embedding == 'ResNet':
            input_dim = 256
            self.proj = resnet18(layers=[1, 1, 1, 1], size='small')
        elif embedding == 'MobileNet':
            pass

        self.ca = cross_attention_layer(
            q_dim=latent_dim,
            kv_dim=input_dim,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
        )
        self.sa = self_attention_block(
            num_blocks * self_attention_per_block,
            x_dim=latent_dim,
            num_heads=num_self_attention_heads,
            head_dim=self_attention_head_dim,
        )

        if token_location == 'None':
            self.to_logits = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif token_location == 'Encoder_CA':
            self.cls_token = nn.Parameter(torch.zeros(1, latent_dim))
            self.to_logits = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif token_location == 'Encoder_SA':
            self.cls_token = nn.Parameter(torch.zeros(1, latent_dim))
            self.to_logits = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif token_location == 'Decoder':
            self.cls_token = nn.Parameter(torch.zeros(1, num_classes))
            self.decoder = cross_attention_layer(
                q_dim=num_classes,
                kv_dim=latent_dim,
                num_heads=num_cross_attention_heads,
                head_dim=cross_attention_head_dim,
            )
            self.to_logits = nn.Sequential(
                nn.LayerNorm(num_classes),
                nn.Linear(num_classes, num_classes)
            )

        # 初始化latent
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        if self.embedding == 'CNN':
            x = self.proj(x)
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.ln(x)
        elif self.embedding == 'ResNet':
            x = self.proj(x)
            x = rearrange(x, 'b c h w -> b (h w) c')

        b, *_ = x.shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        if self.token_location == 'Encoder_CA':
            cls_token = repeat(self.cls_token, "... -> b ...", b=b)
            x_latent = torch.cat([cls_token, x_latent], 1)

        x_latent = self.ca(x_latent, x)

        if self.token_location == 'Encoder_SA':
            cls_token = repeat(self.cls_token, "... -> b ...", b=b)
            x_latent = torch.cat([cls_token, x_latent], 1)

        x_latent = self.sa(x_latent)

        if self.token_location == 'None':
            y = self.to_logits(x_latent)
        elif self.token_location == 'Encoder_CA':
            x_latent = x_latent[:, 0, :]
            y = self.to_logits(x_latent)
        elif self.token_location == 'Encoder_SA':
            x_latent = x_latent[:, 0, :]
            y = self.to_logits(x_latent)
        elif self.token_location == 'Decoder':
            cls_token = repeat(self.cls_token, "... -> b ...", b=b)
            y = self.decoder(cls_token, x_latent)
            y = self.to_logits(y).squeeze(dim=1)

        return y


class MultiPerceiver(nn.Module):
    def __init__(
        self,
        embedding: str = 'ResNet',             # 'ResNet'

        token_location: str = 'None',       # 'None', 'Encoder-CA', 'Decoder'

        num_latents: int = 196,
        latent_dim: int = 384,

        num_blocks: int = 4,
        self_attention_per_block: int = 2,

        num_cross_attention_heads: int = 6,
        num_self_attention_heads: int = 6,

        cross_attention_head_dim: int = 64,
        self_attention_head_dim: int = 64,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.embedding = embedding
        self.token_location = token_location
        self.num_blocks = num_blocks

        input_dim = [48, 96, 192, 384]
        # input_dim = [64, 128, 256, 512]
        # input_dim = [192, 384, 768, 1536]
        self.proj = resnet48(layers=[1, 1, 3, 1], return_features=True)

        self.ca_block = nn.ModuleList([
            cross_attention_layer(latent_dim, input_dim[i], num_cross_attention_heads, cross_attention_head_dim) for i in range(num_blocks)
        ])

        self.sa_block = nn.ModuleList([
            self_attention_block(self_attention_per_block, latent_dim, num_self_attention_heads, self_attention_head_dim) for _ in range(num_blocks)
        ])

        if token_location == 'None':
            self.to_logits = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif token_location == 'Encoder_CA':
            self.cls_token = nn.Parameter(torch.zeros(1, latent_dim))
            self.to_logits = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif token_location == 'Decoder':
            self.cls_token = nn.Parameter(torch.zeros(1, num_classes))
            self.decoder = cross_attention_layer(
                q_dim=num_classes,
                kv_dim=latent_dim,
                num_heads=num_cross_attention_heads,
                head_dim=cross_attention_head_dim,
            )
            self.to_logits = nn.Sequential(
                nn.LayerNorm(num_classes),
                nn.Linear(num_classes, num_classes)
            )

        # 初始化latent
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        b, *_ = x.shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        if self.token_location == 'Encoder_CA':
            cls_token = repeat(self.cls_token, "... -> b ...", b=b)
            x_latent = torch.cat([cls_token, x_latent], 1)

        features = self.proj(x)

        for i in range(self.num_blocks):
            features[i] = rearrange(features[i], 'b c h w -> b (h w) c')

        for i in range(self.num_blocks):
            x_latent = self.ca_block[i](x_latent, features[i])
            x_latent = self.sa_block[i](x_latent)

        if self.token_location == 'None':
            y = self.to_logits(x_latent)
        elif self.token_location == 'Encoder_CA':
            x_latent = x_latent[:, 0, :]
            y = self.to_logits(x_latent)
        elif self.token_location == 'Decoder':
            cls_token = repeat(self.cls_token, "... -> b ...", b=b)
            y = self.decoder(cls_token, x_latent)
            y = self.to_logits(y).squeeze(dim=1)

        return y


class MultiPerceiver2(nn.Module):
    def __init__(
        self,
        resnet_blocks: list = [2, 2, 2, 2],
        token_location: str = 'None',       # 'None', 'Encoder-CA', 'Decoder'

        num_latents: int = 98,
        latent_dim: int = 192,

        num_blocks: int = 4,
        self_attention_per_block: tuple = (3, 3, 9, 3),

        num_cross_attention_heads: tuple = (3, 3, 12, 3),
        num_self_attention_heads: tuple = (3, 3, 12, 3),

        cross_attention_head_dim: int = 64,
        self_attention_head_dim: int = 64,
        num_classes: int = 1000,
    ):
        super().__init__()

        self.token_location = token_location
        self.num_blocks = num_blocks

        input_dim = [48, 96, 192, 384]
        # input_dim = [64, 128, 256, 512]
        # input_dim = [192, 384, 768, 1536]
        self.proj = resnet48(layers=resnet_blocks, return_features=True)

        self.ca_block = nn.ModuleList([
            cross_attention_layer(latent_dim, input_dim[i], num_cross_attention_heads[i], cross_attention_head_dim) for i in range(num_blocks)
        ])

        self.sa_block = nn.ModuleList([
            self_attention_block(self_attention_per_block[i], latent_dim, num_self_attention_heads[i], self_attention_head_dim) for i in range(num_blocks)
        ])

        if token_location == 'None':
            self.to_logits = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif token_location == 'Encoder_CA':
            self.cls_token = nn.Parameter(torch.zeros(1, latent_dim))
            self.to_logits = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, num_classes)
            )
        elif token_location == 'Decoder':
            self.cls_token = nn.Parameter(torch.zeros(1, num_classes))
            self.decoder = cross_attention_layer(
                q_dim=num_classes,
                kv_dim=latent_dim,
                num_heads=6,
                head_dim=cross_attention_head_dim,
            )
            self.to_logits = nn.Sequential(
                nn.LayerNorm(num_classes),
                nn.Linear(num_classes, num_classes)
            )

        # 初始化latent
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            # nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        b, *_ = x.shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        if self.token_location == 'Encoder_CA':
            cls_token = repeat(self.cls_token, "... -> b ...", b=b)
            x_latent = torch.cat([cls_token, x_latent], 1)

        features = self.proj(x)

        for i in range(self.num_blocks):
            features[i] = rearrange(features[i], 'b c h w -> b (h w) c')

        for i in range(self.num_blocks):
            x_latent = self.ca_block[i](x_latent, features[i])
            x_latent = self.sa_block[i](x_latent)

        if self.token_location == 'None':
            y = self.to_logits(x_latent)
        elif self.token_location == 'Encoder_CA':
            x_latent = x_latent[:, 0, :]
            y = self.to_logits(x_latent)
        elif self.token_location == 'Decoder':
            cls_token = repeat(self.cls_token, "... -> b ...", b=b)
            y = self.decoder(cls_token, x_latent)
            y = self.to_logits(y).squeeze(dim=1)

        return y


class MultiMulti(nn.Module):
    def __init__(
        self,
        embedding: str = 'ResNet',             # 'ResNet'

        num_latents: int = 196,
        latent_dim: int = 384,

        num_blocks: int = 4,
        self_attention_per_block: int = 2,

        num_cross_attention_heads: int = 3,
        num_self_attention_heads: int = 6,

        cross_attention_head_dim: int = 64,
        self_attention_head_dim: int = 64,
        num_classes: int = 1000,

        num_task_tokens: int = 1,
        task_token_dim: int = 1000,

        decoder_location: tuple = (0, 7),
        last_embedding: bool = True,
    ):
        super().__init__()

        self.embedding = embedding

        self.decoder_location = decoder_location
        self.last_embedding = last_embedding
        self.task_token = nn.Parameter(torch.zeros(num_task_tokens, task_token_dim))

        self.num_blocks = num_blocks
        self.self_attention_per_block = self_attention_per_block

        input_dim = [64, 128, 256, 512]
        self.proj = resnet18(layers=[1, 1, 1, 1], return_features=True)

        self.ca_block = nn.ModuleList([
            cross_attention_layer(latent_dim, input_dim[i], num_cross_attention_heads, cross_attention_head_dim) for i in range(num_blocks)
        ])

        sa_block = []
        for i in range(num_blocks):
            for j in range(self_attention_per_block):
                sa_block.append(self_attention_layer(latent_dim, num_self_attention_heads, self_attention_head_dim))
        self.sa_block = nn.ModuleList(sa_block)

        self.decoder_block = nn.ModuleList([
            cross_attention_layer(task_token_dim, latent_dim, num_cross_attention_heads, cross_attention_head_dim) for _ in range(len(decoder_location))
        ])

        if num_task_tokens > 1:
            self.to_logits = nn.Sequential(
                Reduce('b n d -> b d', 'mean'),
                nn.LayerNorm(task_token_dim),
                nn.Linear(task_token_dim, num_classes)
            )
        else:
            self.to_logits = nn.Sequential(
                nn.LayerNorm(task_token_dim),
                nn.Linear(task_token_dim, num_classes)
            )

        # 初始化latent
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        b, *_ = x.shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)
        task_token = repeat(self.task_token, "... -> b ...", b=b)

        features = self.proj(x)

        for i in range(self.num_blocks):
            features[i] = rearrange(features[i], 'b c h w -> b (h w) c')

        decoder_count = 0

        for i in range(self.num_blocks):
            x_latent = self.ca_block[i](x_latent, features[i])

            for j in range(self.self_attention_per_block):
                sa_count = i * self.self_attention_per_block + j
                x_latent = self.sa_block[sa_count](x_latent)

                if sa_count in self.decoder_location:
                    task_token = self.decoder_block[decoder_count](task_token, x_latent)
                    decoder_count += 1

        if self.last_embedding:
            task_token = self.to_logits(task_token).squeeze(dim=1)

        return task_token
