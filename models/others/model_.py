from math import pi, log

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Reduce
from typing import Tuple


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


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


def mlp(x_dim: int, mult: int = 4):
    return Sequential(
        nn.LayerNorm(x_dim),
        nn.Linear(x_dim, x_dim * mult * 2),
        GEGLU(),
        nn.Linear(x_dim * mult, x_dim),
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
    return Sequential(*layers)


def get_conv(
    in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False
):
    conv = nn.Sequential(
        nn.Conv2d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            bias = bias),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )
    return conv


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
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


class Encoder(nn.Module):
    def __init__(
        self,
        encoding: str = '',
        max_freq: int = 224,
        num_bands: int = 64,

        num_latents: int = 128,
        latent_dim: int = 128,

        num_blocks: int = 3,
        num_cross_attention_heads: int = 4,
        cross_attention_head_dim: int = 16,

        self_attention_per_block: int = 0,
        num_self_attention_heads: int = 4,
        self_attention_head_dim: int = 16,

        dropout: float = 0.0,
    ):
        super().__init__()

        self.encoding = encoding
        self.max_freq = max_freq
        self.num_freq_bands = num_bands

        num_kv_channels = 3
        if encoding == 'Fourier':
            num_kv_channels += 2 * (num_bands * 2 + 1)

        def create_perceiver_block():
            if self_attention_per_block == 0:
                return Sequential(
                    cross_attention_layer(
                        q_dim=latent_dim,
                        kv_dim=num_kv_channels,
                        num_heads=num_cross_attention_heads,
                        head_dim=cross_attention_head_dim,
                        dropout=dropout,
                    )
                )
            else:
                return Sequential(
                    cross_attention_layer(
                        q_dim=latent_dim,
                        kv_dim=num_kv_channels,
                        num_heads=num_cross_attention_heads,
                        head_dim=cross_attention_head_dim,
                        dropout=dropout,
                    ),
                    self_attention_block(
                        num_blocks=self_attention_per_block,
                        x_dim=latent_dim,
                        num_heads=num_self_attention_heads,
                        head_dim=self_attention_head_dim,
                        dropout=dropout,
                    ),
                )

        self.perceiver_blocks = nn.ModuleList([create_perceiver_block() for _ in range(num_blocks)])

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        b, *axis, _, device, dtype = *x.shape, x.device, x.dtype

        if self.encoding == 'Fourier':
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)
            x = torch.cat((x, enc_pos), dim=-1)

        x = rearrange(x, "b ... c -> b (...) c")

        x_latent = repeat(self.latent, "... -> b ...", b=b)

        for block in self.perceiver_blocks:
            x_latent = block(x_latent, x)

        return x_latent


class Decoder(nn.Module):
    def __init__(
        self,
        output_array_shape: Tuple = (1, 1000),
        num_classes: int = 1000,

        latent_dim: int = 256,
        num_cross_attention_heads: int = 4,
        cross_attention_head_dim: int = 64,

        dropout: float = 0.0,
    ):
        super().__init__()

        self.output = nn.Parameter(torch.empty(output_array_shape))
        self._init_parameters()

        output_dim = output_array_shape[-1]
        self.cross_attention = cross_attention_layer(
            q_dim=output_dim,
            kv_dim=latent_dim,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
            dropout=dropout,
        )

        self.linear = nn.Linear(output_dim, num_classes)

    def _init_parameters(self):
        with torch.no_grad():
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x_latent):
        b, *_ = x_latent.shape

        output = repeat(self.output, "... -> b ...", b=b)
        output = self.cross_attention(output, x_latent)

        return self.linear(output).squeeze(dim=1)


class PerceiverIO(nn.Module):
    def __init__(
        self,
        encoding: str = 'Fourier',
        max_freq: int = 224,
        num_bands: int = 64,

        num_latents: int = 128,
        latent_dim: int = 128,

        num_blocks: int = 3,
        num_cross_attention_heads: int = 4,
        cross_attention_head_dim: int = 16,

        self_attention_per_block: int = 0,
        num_self_attention_heads: int = 4,
        self_attention_head_dim: int = 16,

        num_classes: int = 1000,
    ):
        super().__init__()
        self.encoder = Encoder(
            encoding=encoding,
            max_freq=max_freq,
            num_bands=num_bands,

            num_latents=num_latents,
            latent_dim=latent_dim,

            num_blocks=num_blocks,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,

            self_attention_per_block=self_attention_per_block,
            num_self_attention_heads=num_self_attention_heads,
            self_attention_head_dim=self_attention_head_dim,
        )
        self.decoder = Decoder(
            output_array_shape=(1, num_classes),
            latent_dim=latent_dim,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
        )

    def forward(self, x):
        x_latent = self.encoder(x)
        y = self.decoder(x_latent)
        return y


class PerceiverIOWithSA(nn.Module):
    def __init__(
        self,
        encoding: str = 'Fourier',
        max_freq: int = 224,
        num_bands: int = 10,

        num_latents: int = 128,
        latent_dim: int = 256,

        num_cross_attention_heads: int = 4,
        cross_attention_head_dim: int = 64,

        num_blocks: int = 5,
        self_attention_per_block: int = 4,
        num_self_attention_heads: int = 4,
        self_attention_head_dim: int = 64,

        num_classes: int = 1000,
    ):
        super().__init__()
        self.encoding = encoding
        self.max_freq = max_freq
        self.num_freq_bands = num_bands
        self.num_blocks = num_blocks

        num_kv_channels = 3
        if encoding == 'Fourier':
            num_kv_channels += 2 * (num_bands * 2 + 1)

        self.ca = cross_attention_layer(
            q_dim=latent_dim,
            kv_dim=num_kv_channels,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
        )

        self.sa_block = self_attention_block(
            num_blocks=self_attention_per_block,
            x_dim=latent_dim,
            num_heads=num_self_attention_heads,
            head_dim=self_attention_head_dim,
        )

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

        self.decoder = Decoder(
            output_array_shape=(1, num_classes),
            latent_dim=latent_dim,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
        )

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        b, *axis, _, device, dtype = *x.shape, x.device, x.dtype

        if self.encoding == 'Fourier':
            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing='ij'), dim=-1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)
            x = torch.cat((x, enc_pos), dim=-1)

        x = rearrange(x, "b ... c -> b (...) c")

        x_latent = repeat(self.latent, "... -> b ...", b=b)

        x_latent = self.ca(x_latent, x)

        for _ in range(self.num_blocks):
            x_latent = self.sa_block(x_latent)

        y = self.decoder(x_latent)
        return y


# 没有加入SA
class EncoderCNN(nn.Module):
    def __init__(
        self,
        num_latents: int = 128,
        latent_dim: int = 256,

        num_blocks: int = 5,
        num_cross_attention_heads: int = 4,
        cross_attention_head_dim: int = 64,

        dropout: float = 0.0,
    ):
        super().__init__()

        channels = [96, 192, 384, 768, 1536]
        # channels = [96, 192, 192, 384, 384]
        conv1 = get_conv(3,           channels[0], kernel_size=7, stride=2, padding=3)
        conv2 = get_conv(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        conv3 = get_conv(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
        conv4 = get_conv(channels[2], channels[3], kernel_size=3, stride=2, padding=1)
        conv5 = get_conv(channels[3], channels[4], kernel_size=3, stride=2, padding=1)

        self.conv_blocks = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])

        def create_perceiver_block(kv_dim):
            return Sequential(
                cross_attention_layer(
                    q_dim=latent_dim,
                    kv_dim=kv_dim,
                    num_heads=num_cross_attention_heads,
                    head_dim=cross_attention_head_dim,
                    dropout=dropout,
                )
            )

        self.perceiver_blocks = nn.ModuleList([create_perceiver_block(channels[i]) for i in range(num_blocks)])

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        b, *_ = x.shape

        x_latent = repeat(self.latent, "... -> b ...", b=b)

        for conv, pcr in zip(self.conv_blocks, self.perceiver_blocks):
            x = conv(x)
            x_input = rearrange(x, "b c h w -> b (h w) c")
            x_latent = pcr(x_latent, x_input)

        return x_latent, x


class EncoderCNNDecoder(nn.Module):
    def __init__(
        self,
        num_latents: int = 128,
        latent_dim: int = 256,

        num_blocks: int = 5,
        num_cross_attention_heads: int = 4,
        cross_attention_head_dim: int = 16,

        num_classes: int = 1000,
    ):
        super().__init__()
        self.encoder = EncoderCNN(
            num_latents=num_latents,
            latent_dim=latent_dim,

            num_blocks=num_blocks,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
        )
        self.decoder = Decoder(
            output_array_shape=(1, num_classes),
            latent_dim=latent_dim,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
        )

    def forward(self, x):
        x_latent, _ = self.encoder(x)
        y = self.decoder(x_latent)
        return y


class EncoderCNNDecoderCNN(nn.Module):
    def __init__(
        self,
        num_latents: int = 256,
        latent_dim: int = 512,

        num_blocks: int = 5,
        num_cross_attention_heads: int = 12,
        cross_attention_head_dim: int = 64,

        num_classes: int = 1000,
    ):
        super().__init__()
        self.encoder = EncoderCNN(
            num_latents=num_latents,
            latent_dim=latent_dim,

            num_blocks=num_blocks,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1536, num_classes)

        self.cross_attention = cross_attention_layer(
            q_dim=num_classes,
            kv_dim=latent_dim,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
        )

    def forward(self, input):
        x_latent, x = self.encoder(input)
        b, *_ = x_latent.shape

        # (b, 1536, 7, 7)
        x = self.avgpool(x)
        # (b, 1536, 1, 1)

        x = torch.flatten(x, 1)
        # (b, 1536)

        x = self.fc(x)
        # (b, num_classes)

        x = x.unsqueeze(1)
        # (b, 1, num_classes)

        y = self.cross_attention(x, x_latent).squeeze(1)
        # (b, num_classes)

        return y


class EncoderDecoderCNN(nn.Module):
    def __init__(
        self,
        encoding: str = 'Fourier',
        max_freq: int = 224,
        num_bands: int = 10,

        num_latents: int = 256,
        latent_dim: int = 512,

        num_blocks: int = 3,
        num_cross_attention_heads: int = 12,
        cross_attention_head_dim: int = 64,

        num_classes: int = 1000,
    ):
        super().__init__()
        self.encoder = Encoder(
            encoding=encoding,
            max_freq=max_freq,
            num_bands=num_bands,

            num_latents=num_latents,
            latent_dim=latent_dim,

            num_blocks=num_blocks,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
        )

        channels = [96, 192, 384, 768, 1536]
        conv1 = get_conv(3, channels[0], kernel_size=7, stride=2, padding=3)
        conv2 = get_conv(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        conv3 = get_conv(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
        conv4 = get_conv(channels[2], channels[3], kernel_size=3, stride=2, padding=1)
        conv5 = get_conv(channels[3], channels[4], kernel_size=3, stride=2, padding=1)

        self.conv_blocks = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1536, num_classes)

        self.cross_attention = cross_attention_layer(
            q_dim=num_classes,
            kv_dim=latent_dim,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
        )

    def forward(self, img):
        x_latent = self.encoder(img)
        b, *_ = x_latent.shape

        for conv in self.conv_blocks:
            img = conv(img)

        # (b, 1536, 7, 7)
        x = self.avgpool(img)
        # (b, 1536, 1, 1)

        x = torch.flatten(x, 1)
        # (b, 1536)
        x = self.fc(x)
        # (b, num_classes)

        x = x.unsqueeze(1)
        # (b, 1, num_classes)

        y = self.cross_attention(x, x_latent).squeeze(1)
        # (b, num_classes)

        return y


class EncoderCNNDecoderWithSA(nn.Module):
    def __init__(
        self,
        num_latents: int = 128,
        latent_dim: int = 256,

        num_blocks: int = 5,
        num_cross_attention_heads: int = 4,
        cross_attention_head_dim: int = 64,

        self_attention_per_block: int = 4,
        num_self_attention_heads: int = 4,
        self_attention_head_dim: int = 64,

        num_classes: int = 1000,
    ):
        super().__init__()

        # channels = [96, 192, 384, 768, 1536]
        channels = [48, 96, 192, 384, 768]
        conv1 = get_conv(3, channels[0], kernel_size=7, stride=2, padding=3)
        conv2 = get_conv(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        conv3 = get_conv(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
        conv4 = get_conv(channels[2], channels[3], kernel_size=3, stride=2, padding=1)
        conv5 = get_conv(channels[3], channels[4], kernel_size=3, stride=2, padding=1)

        self.conv_blocks = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])

        def create_perceiver_block(kv_dim):
            if self_attention_per_block == 0:
                return Sequential(
                    cross_attention_layer(
                        q_dim=latent_dim,
                        kv_dim=kv_dim,
                        num_heads=num_cross_attention_heads,
                        head_dim=cross_attention_head_dim,
                    )
                )
            else:
                return Sequential(
                    cross_attention_layer(
                        q_dim=latent_dim,
                        kv_dim=kv_dim,
                        num_heads=num_cross_attention_heads,
                        head_dim=cross_attention_head_dim,
                    ),
                    self_attention_block(
                        num_blocks=self_attention_per_block,
                        x_dim=latent_dim,
                        num_heads=num_self_attention_heads,
                        head_dim=self_attention_head_dim,
                    ),
                )

        self.perceiver_blocks = nn.ModuleList([create_perceiver_block(channels[i]) for i in range(num_blocks)])

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

        self.decoder = Decoder(
            output_array_shape=(1, num_classes),
            latent_dim=latent_dim,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
        )

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        b, *_ = x.shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        for conv, pcr in zip(self.conv_blocks, self.perceiver_blocks):
            x = conv(x)
            x_input = rearrange(x, "b c h w -> b (h w) c")
            x_latent = pcr(x_latent, x_input)

        y = self.decoder(x_latent)
        return y


class EncoderCNNDecoderCNNWithSA(nn.Module):
    def __init__(
        self,
        num_latents: int = 128,
        latent_dim: int = 256,

        num_blocks: int = 5,
        num_cross_attention_heads: int = 4,
        cross_attention_head_dim: int = 64,

        self_attention_per_block: int = 4,
        num_self_attention_heads: int = 4,
        self_attention_head_dim: int = 64,

        num_classes: int = 1000,
    ):
        super().__init__()

        channels = [96, 192, 384, 768, 1536]
        conv1 = get_conv(3, channels[0], kernel_size=7, stride=2, padding=3)
        conv2 = get_conv(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        conv3 = get_conv(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
        conv4 = get_conv(channels[2], channels[3], kernel_size=3, stride=2, padding=1)
        conv5 = get_conv(channels[3], channels[4], kernel_size=3, stride=2, padding=1)

        self.conv_blocks = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])

        def create_perceiver_block(kv_dim):
            return Sequential(
                cross_attention_layer(
                    q_dim=latent_dim,
                    kv_dim=kv_dim,
                    num_heads=num_cross_attention_heads,
                    head_dim=cross_attention_head_dim,
                ),
                self_attention_block(
                    num_blocks=self_attention_per_block,
                    x_dim=latent_dim,
                    num_heads=num_self_attention_heads,
                    head_dim=self_attention_head_dim,
                ),
            )

        self.perceiver_blocks = nn.ModuleList([create_perceiver_block(channels[i]) for i in range(num_blocks)])

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1536, num_classes)

        self.cross_attention = cross_attention_layer(
            q_dim=num_classes,
            kv_dim=latent_dim,
            num_heads=num_cross_attention_heads,
            head_dim=cross_attention_head_dim,
        )

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        b, *_ = x.shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        for conv, pcr in zip(self.conv_blocks, self.perceiver_blocks):
            x = conv(x)
            x_input = rearrange(x, "b c h w -> b (h w) c")
            x_latent = pcr(x_latent, x_input)

        # (b, 1536, 7, 7)
        x = self.avgpool(x)
        # (b, 1536, 1, 1)

        x = torch.flatten(x, 1)
        # (b, 1536)

        x = self.fc(x)
        # (b, num_classes)

        x = x.unsqueeze(1)
        # (b, 1, num_classes)

        y = self.cross_attention(x, x_latent).squeeze(1)
        # (b, num_classes)
        return y


class MultiScaleDecoder(nn.Module):
    def __init__(
        self,
        num_latents: int = 128,
        latent_dim: int = 256,

        num_blocks: int = 5,
        num_cross_attention_heads: int = 4,
        cross_attention_head_dim: int = 64,

        self_attention_per_block: int = 4,
        num_self_attention_heads: int = 4,
        self_attention_head_dim: int = 64,

        num_classes: int = 1000,
    ):
        super().__init__()

        # channels = [96, 192, 384, 768, 1536]
        channels = [48, 96, 192, 384, 768]
        conv1 = get_conv(3, channels[0], kernel_size=7, stride=2, padding=3)
        conv2 = get_conv(channels[0], channels[1], kernel_size=3, stride=2, padding=1)
        conv3 = get_conv(channels[1], channels[2], kernel_size=3, stride=2, padding=1)
        conv4 = get_conv(channels[2], channels[3], kernel_size=3, stride=2, padding=1)
        conv5 = get_conv(channels[3], channels[4], kernel_size=3, stride=2, padding=1)

        self.conv_blocks = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])

        def create_perceiver_block(kv_dim):
            return Sequential(
                cross_attention_layer(
                    q_dim=latent_dim,
                    kv_dim=kv_dim,
                    num_heads=num_cross_attention_heads,
                    head_dim=cross_attention_head_dim,
                ),
                self_attention_block(
                    num_blocks=self_attention_per_block,
                    x_dim=latent_dim,
                    num_heads=num_self_attention_heads,
                    head_dim=self_attention_head_dim,
                ),
            )

        self.perceiver_blocks = nn.ModuleList([create_perceiver_block(channels[i]) for i in range(num_blocks)])

        # learnable initial latent vectors
        self.latent = nn.Parameter(torch.empty(num_latents, latent_dim))
        self._init_parameters()

        self.ca_blocks = nn.ModuleList([])
        self.avgpool_blocks = nn.ModuleList([])
        for i in range(num_blocks):
            self.ca_blocks.append(
                cross_attention_layer(
                    q_dim=channels[i],
                    kv_dim=latent_dim,
                    num_heads=num_cross_attention_heads,
                    head_dim=cross_attention_head_dim,
                )
            )
            self.avgpool_blocks.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.fc = nn.Linear(sum(channels), num_classes)

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x):
        b, *_ = x.shape
        x_latent = repeat(self.latent, "... -> b ...", b=b)

        outputs = []

        for conv, pcr, pool, ca in zip(self.conv_blocks, self.perceiver_blocks, self.avgpool_blocks, self.ca_blocks):
            x = conv(x)

            q = pool(x)
            # (b, c, 1, 1)

            x_input = rearrange(x, "b c h w -> b (h w) c")
            x_latent = pcr(x_latent, x_input)

            q = torch.flatten(q, 1)
            # (b, c)

            q = torch.unsqueeze(q, 1)
            # (b, 1, c)

            output = ca(q, x_latent)
            # (b, 1, c)

            output = torch.squeeze(output)
            # (b, c)

            outputs.append(output)

        y = torch.cat(outputs, dim=-1)
        # (b, c1+c2+c3+c4+c5)

        y = self.fc(y)
        # (b, num_classes)
        return y
