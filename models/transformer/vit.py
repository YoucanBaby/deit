import torch
from timm.models.layers import trunc_normal_
from torch import nn, einsum

from einops import rearrange, repeat


def mlp(in_dim: int, mlp_ratio: int = 4):
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, in_dim * mlp_ratio),
        nn.GELU(),
        nn.Linear(in_dim * mlp_ratio, in_dim),
    )


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


class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float, drop_path=0.0):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.drop_path = drop_path

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention(nn.Module):
    def __init__(self, q_dim, kv_dim=None, heads=12, head_dim=64, dropout=0.):
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

        # 矩阵乘法
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


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


class ViT(nn.Module):
    def __init__(
        self,

        patch_size: int = 16,
        num_patches: int = 196,
        dim: int = 384,

        depth: int = 12,

        num_self_attention_heads: int = 6,
        self_attention_head_dim: int = 64,

        num_classes: int = 1000,
    ):
        super().__init__()

        self.proj = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.ln = nn.LayerNorm(dim),

        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        trunc_normal_(self.pos_embedding, std=.02)
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        self.sa = self_attention_block(
            num_blocks=depth,
            x_dim=dim,
            num_heads=num_self_attention_heads,
            head_dim=self_attention_head_dim,
        )

        self.to_latent = nn.Identity()
        self.to_logit = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.proj(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.ln(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "... -> b ...", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        x = self.sa(x)
        x = x[:, 0, :]

        return self.to_latent(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.to_logit(x)
        return x
