###############################################################################
# Code adapted from
# https://github.com/rwightman/pytorch-image-models
# https://github.com/facebookresearch/deit
# Modified the original code to explore different attention mechanisms
###############################################################################

import torch
import torch.nn as nn

from .spectral import get_spectral_norm
from tools.utils import from_ctx, to_ctx


class MultiBlocks(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.multi_blocks = nn.ModuleList([Block(**kwargs) for _ in range(kwargs["depth"])])

    def forward(self, z, **kwargs):
        for block in self.multi_blocks:
            z = block(z, **kwargs)
        return z


class Block(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        dim = kwargs["dim"]
        norm_layer = kwargs["norm_layer"]
        spectral_norm_layer = kwargs["spectral_norm_layer"]
        block_type = kwargs["block_type"]
        cond_norm = block_type in ["full_with_cond_norm"]
        if cond_norm:
            assert norm_layer != "ln"
            self.ab = Mlp(dim, spectral_norm_layer, out_dim=4 * dim)
        self.norm1 = CustomNorm(norm_layer, dim)
        self.attn = CustomAttention(**kwargs)
        self.norm2 = CustomNorm(norm_layer, dim)
        self.mlp = Mlp(dim, spectral_norm_layer)
        self.block_type = block_type
        self.cond_norm = cond_norm
        if "dropout" in kwargs and kwargs["dropout"] > 0:
            self.dp1 = nn.Dropout(kwargs["dropout"])
            self.dp2 = nn.Dropout(kwargs["dropout"])
        else:
            self.dp1 = lambda x: x
            self.dp2 = lambda x: x

    def forward(self, x, **kwargs):
       if self.cond_norm:
           ab = self.ab(kwargs["z_cond"]).view(x.size(0), 1, 4, -1)
           a1, b1, a2, b2 = ab[:, :, 0], ab[:, :, 1], ab[:, :, 2], ab[:, :, 3]
       else:
           a1, b1, a2, b2 = 1., 0., 1., 0.
       x = x + self.dp1(self.attn(a1 * self.norm1(x) + b1, **kwargs))
       x = x + self.dp2(self.mlp(a2 * self.norm2(x) + b2))
       return x


class CustomAttention(nn.Module):
    def __init__(self, block_type, **kwargs):
        super().__init__()
        if block_type in ["full", "full_with_cond_norm"]:
            self.attn = FullAttention(**kwargs)
        elif block_type == "cross":
            self.attn = CrossAttention(**kwargs)
        elif block_type == "obj":
            self.attn = ObjAttention(**kwargs)
        elif block_type == "cls":
            self.attn = ClsAttention(**kwargs)
        elif block_type == "ctx":
            self.attn = CtxAttention(**kwargs)
        elif block_type == "block_causal":
            self.attn = BlockCausalAttention(**kwargs)
        elif block_type == "skip":
            self.attn = SkipAttention(**kwargs)
        elif block_type == "skip2":
            self.attn = Skip2Attention(**kwargs)
        else:
            raise ValueError

    def forward(self, x, **kwargs):
        return self.attn(x, **kwargs)


class FullAttention(nn.Module):
    def __init__(self, dim, num_heads, spectral_norm_layer, noise, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        spec = get_spectral_norm(spectral_norm_layer)
        self.qkv = spec(nn.Linear(dim, dim * 3, bias=False))
        self.proj = spec(nn.Linear(dim, dim))
        self.noise = noise
        if noise:
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x, ctx_mask=None, **kwargs):
        B, N, C = x.shape
        if ctx_mask is not None:
            B, T = ctx_mask.shape # overwrite B with correct value when reshaping to full view from ctx
        if self.noise:
            x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength
        qkv = self.qkv(x)
        if ctx_mask is not None:
            qkv = from_ctx(qkv, ctx_mask)
        qkv = qkv.reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if ctx_mask is not None:
            mask = (~ctx_mask).view(B, T, 1).expand(-1, -1, N).contiguous()
            mask = (mask.view(B, 1, 1, T * N) * (~mask).view(B, 1, T * N, 1)) # TODO /!\ double check that the mask is correct
            attn = attn.masked_fill(mask.expand(-1, self.num_heads, -1, -1), float('-inf'))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        if ctx_mask is not None:
            x = x.view(B, T, N, C)
            x = to_ctx(x, ctx_mask)
        x = self.proj(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, spectral_norm_layer, noise, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        spec = get_spectral_norm(spectral_norm_layer)
        self.q = spec(nn.Linear(dim, dim, bias=False))
        self.kv = spec(nn.Linear(dim, dim * 2, bias=False))
        self.proj = spec(nn.Linear(dim, dim))
        self.noise = noise
        if noise:
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

    def forward(self, x_pred, x_ctx, ctx_mask=None, **kwargs):
        assert ctx_mask is not None
        pred_mask = ~ctx_mask
        _, N, C = x_pred.shape
        B, T = ctx_mask.shape
        if self.noise:
            x_pred = x_pred + torch.randn([x_pred.size(0), x_pred.size(1), 1], device=x_pred.device) * self.noise_strength
        q = from_ctx(self.q(x_pred), pred_mask).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = from_ctx(self.kv(x_ctx), ctx_mask).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        mask = (~ctx_mask).view(B, T, 1).expand(-1, -1, N).contiguous()
        mask = (mask.view(B, 1, 1, T * N) * mask.view(B, 1, T * N, 1)) # TODO /!\ double check that the mask is correct
        attn = attn.masked_fill(mask.expand(-1, self.num_heads, -1, -1), float('-inf'))
        attn = attn.softmax(dim=-1)
        x_pred = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x_pred = x_pred.view(B, T, N, C)
        x_pred = to_ctx(x_pred, pred_mask)
        x_pred = self.proj(x_pred)
        return x_pred


class ObjAttention(nn.Module):
    def __init__(self, dim, num_heads, spectral_norm_layer, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        spec = get_spectral_norm(spectral_norm_layer)
        self.q = spec(nn.Linear(dim, dim, bias=False))
        self.kv = spec(nn.Linear(dim, dim * 2, bias=False))
        self.proj = spec(nn.Linear(dim, dim))

    def forward(self, x_obj, x_ctx, **kwargs):
        B, NoLo, L, C = x_obj.size(0), x_obj.size(1), x_ctx.size(1), self.dim
        q_obj = self.q(x_obj).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv_obj = self.kv(x_obj).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(x_ctx).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_obj, v_obj = kv_obj[0], kv_obj[1]
        k, v = kv[0], kv[1]
        attn_obj = (q_obj @ k_obj.transpose(-2, -1)) * self.scale
        attn = (q_obj @ k.transpose(-2, -1)) * self.scale
        attn = torch.cat([attn_obj, attn], dim=-1)
        attn = attn.softmax(dim=-1)
        v = torch.cat([v_obj, v], dim=2)
        x_obj = (attn @ v).transpose(1, 2).reshape(B, NoLo, C)
        x_obj = self.proj(x_obj)
        return x_obj


class ClsAttention(nn.Module):
    def __init__(self, dim, num_heads, spectral_norm_layer, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        spec = get_spectral_norm(spectral_norm_layer)
        self.q = spec(nn.Linear(dim, dim, bias=False))
        self.kv = spec(nn.Linear(dim, dim * 2, bias=False))
        self.proj = spec(nn.Linear(dim, dim))

    def forward(self, z_cls, x_ctx, **kwargs):
        x = torch.cat([z_cls, x_ctx], dim=1)
        B, N, C = x.shape
        q_cls = self.q(z_cls).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q_cls @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        z_cls = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        z_cls = self.proj(z_cls)
        return z_cls


class CtxAttention(nn.Module):
    def __init__(self, dim, num_heads, spectral_norm_layer, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        spec = get_spectral_norm(spectral_norm_layer)
        self.q = spec(nn.Linear(dim, dim, bias=False))
        self.kv = spec(nn.Linear(dim, dim * 2, bias=False))
        self.proj = spec(nn.Linear(dim, dim))

    def forward(self, x_ctx, z_cls, **kwargs):
        B, N, C = x_ctx.shape
        x = torch.cat([z_cls, x_ctx], dim=1)
        q_ctx = self.q(x_ctx).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, N + 1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q_ctx @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x_ctx = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x_ctx = self.proj(x_ctx)
        return x_ctx


class SeedAttention(nn.Module):
    def __init__(self, dim, num_heads, spectral_norm_layer, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        spec = get_spectral_norm(spectral_norm_layer)
        self.qkv = spec(nn.Linear(dim, dim * 3, bias=False))
        self.kv_cls = spec(nn.Linear(2 * dim, dim * 2, bias=False))
        self.proj = spec(nn.Linear(dim, dim))

    def forward(self, x, z_cls, **kwargs):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        kv_cls = self.kv_cls(z_cls).reshape(B, 1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_cls, v_cls = kv_cls[0], kv_cls[1]
        k = torch.cat([k_cls, k], dim=2)
        v = torch.cat([v_cls, v], dim=2)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class BlockCausalAttention(nn.Module):
    def __init__(self, dim, num_heads, noise, spectral_norm_layer, causal_mask_sizes, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        spec = get_spectral_norm(spectral_norm_layer)
        self.qkv = spec(nn.Linear(dim, dim * 3, bias=False))
        self.proj = spec(nn.Linear(dim, dim))
        self.noise = noise
        if noise:
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))
        self.register_buffer("causal_mask", get_causal_mask(causal_mask_sizes))

    def forward(self, x, **kwargs):
        B, N, C = x.shape
        if self.noise:
            x = x + torch.randn([x.size(0), x.size(1), 1], device=x.device) * self.noise_strength
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.masked_fill(self.causal_mask[:, :, :N, :N], float('-inf'))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class SkipAttention(nn.Module):
    def __init__(self, dim, num_heads, spectral_norm_layer, latent_size, num_seeds, temporal_dropout, non_trivial, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        spec = get_spectral_norm(spectral_norm_layer)
        self.qkv = spec(nn.Linear(dim, dim * 3, bias=False))
        self.k = spec(nn.Linear(dim, dim, bias=False))
        self.v = spec(nn.Linear(dim, dim, bias=False))
        self.proj = spec(nn.Linear(dim, dim))
        self.latent_size = latent_size
        self.num_seeds = num_seeds
        self.temporal_dropout = temporal_dropout
        self.non_trivial = non_trivial

    def forward(self, x, x_ctx, dx_ctx, mode, ctx_mask, **kwargs):
        B, T, L, C = x_ctx.shape
        T0 = x.size(1) // L
        qkv = self.qkv(x).reshape(B, T0 * L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # B Nh T0*L Ch
        k_ctx = self.k(dx_ctx).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B Nh T*L Ch
        v_ctx = self.v(x_ctx).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B Nh T*L Ch
        attn = (q @ k_ctx.transpose(-2, -1)) * self.scale
        attn = attn.view(B, self.num_heads, T0 * L, T, L)
        if ctx_mask is not None:
            # mask unused ctx
            mask = (~ctx_mask).view(B, 1, 1, T, 1)
            attn = attn.masked_fill(mask.expand(-1, self.num_heads, T0 * L, -1, L), float('-inf'))
        if mode == "training" and self.non_trivial:
            # mask self reconstruction ctx
            assert T0 + self.num_seeds == T
            mask = torch.arange(T).cuda()
            mask = mask.view(-1, 1) + self.num_seeds == mask.view(1, -1)
            mask = mask[:T0, :T].view(1, 1, T0, 1, T, 1)
            attn = attn.masked_fill(mask.expand(B, self.num_heads, -1, L, -1, L).reshape(attn.shape), float('-inf'))
        if mode == "training" and self.temporal_dropout > 0:
            # randomly mask some timesteps for reconstruction / prediction
            mask = (torch.rand_like(attn[:, :, :, :, :1]) < self.temporal_dropout)
            attn = attn.masked_fill(mask.expand(-1, -1, -1, -1, L), float('-inf'))
        self_attn = (q.view(B, self.num_heads, T0, L, -1) @ k.view(B, self.num_heads, T0, L, -1).transpose(-2, -1)) * self.scale
        self_attn = self_attn.view(B, self.num_heads, T0 * L, 1, L)
        attn = torch.cat([attn, self_attn], dim=-2).view(B, self.num_heads, T0 * L, (T + 1) * L)
        attn = attn.softmax(dim=-1)
        x = attn[:, :, :, :-L] @ v_ctx
        x = x + (attn[:, :, :, -L:].view(B, self.num_heads, T0, L, L) @ v.view(B, self.num_heads, T0, L, -1)).view(B, self.num_heads, T0 * L, -1)
        x = x.transpose(1, 2).reshape(B, T0 * L, C)
        x = self.proj(x)
        return x


class Skip2Attention(nn.Module):
    def __init__(self, dim, num_heads, spectral_norm_layer, latent_size, num_seeds, temporal_dropout, non_trivial, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        spec = get_spectral_norm(spectral_norm_layer)
        self.qkv = spec(nn.Linear(dim, dim * 3, bias=False))
        self.k = spec(nn.Linear(dim, dim, bias=False))
        self.v = spec(nn.Linear(dim, dim, bias=False))
        self.proj = spec(nn.Linear(dim, dim))
        self.latent_size = latent_size
        self.num_seeds = num_seeds
        self.temporal_dropout = temporal_dropout
        self.non_trivial = non_trivial

    def forward(self, x, x_ctx, dx_ctx, mode, ctx_mask, **kwargs):
        L = self.latent_size
        B, T, N, C = dx_ctx.shape
        T0 = N // L
        qkv = self.qkv(x).reshape(B, T0 * L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # B Nh T0*L Ch
        k_ctx = self.k(dx_ctx).reshape(B, T, T0, L, self.num_heads, C // self.num_heads).permute(0, 4, 2, 1, 3, 5) # B Nh T0 T L Ch
        v_ctx = self.v(x_ctx).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B Nh T*L Ch
        attn = (q.view(B, self.num_heads, T0, 1, L, C // self.num_heads).expand(-1, -1, -1, T, -1, -1) @ k_ctx.transpose(-2, -1)) * self.scale
        attn = attn.permute(0, 1, 2, 4, 3, 5).reshape(B, self.num_heads, T0 * L, T, L)
        if ctx_mask is not None:
            # mask unused ctx
            mask = (~ctx_mask).view(B, 1, 1, T, 1)
            attn = attn.masked_fill(mask.expand(-1, self.num_heads, T0 * L, -1, L), float('-inf'))
        if mode == "training" and self.non_trivial:
            # mask self reconstruction ctx
            assert T0 + self.num_seeds == T
            mask = torch.arange(T).cuda()
            mask = mask.view(-1, 1) + self.num_seeds == mask.view(1, -1)
            mask = mask[:T0, :T].view(1, 1, T0, 1, T, 1)
            attn = attn.masked_fill(mask.expand(B, self.num_heads, -1, L, -1, L).reshape(attn.shape), float('-inf'))
        if mode == "training" and self.temporal_dropout > 0:
            # randomly mask some timesteps for reconstruction / prediction
            mask = (torch.rand_like(attn[:, :, :, :, :1]) < self.temporal_dropout)
            attn = attn.masked_fill(mask.expand(-1, -1, -1, -1, L), float('-inf'))
        self_attn = (q.view(B, self.num_heads, T0, L, -1) @ k.view(B, self.num_heads, T0, L, -1).transpose(-2, -1)) * self.scale
        self_attn = self_attn.view(B, self.num_heads, T0 * L, 1, L)
        attn = torch.cat([attn, self_attn], dim=-2).view(B, self.num_heads, T0 * L, (T + 1) * L)
        attn = attn.softmax(dim=-1)
        x = attn[:, :, :, :-L] @ v_ctx
        x = x + (attn[:, :, :, -L:].view(B, self.num_heads, T0, L, L) @ v.view(B, self.num_heads, T0, L, -1)).view(B, self.num_heads, T0 * L, -1)
        x = x.transpose(1, 2).reshape(B, T0 * L, C)
        x = self.proj(x)
        return x


class CustomNorm(nn.Module):
    def __init__(self, norm_layer, dim):
        super().__init__()
        self.norm_type = norm_layer
        if norm_layer == "ln":
            self.norm = nn.LayerNorm(dim)
        elif norm_layer == "ln_not_affine":
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        elif norm_layer == "pn":
            self.norm = PixelNorm()
        elif norm_layer == "bn2d":
            self.norm = nn.SyncBatchNorm(dim)
        elif norm_layer == "ln2d":
            self.norm = nn.GroupNorm(dim, dim)
        else:
            raise ValueError

    def forward(self, x):
        return self.norm(x)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=2, keepdim=True) + 1e-8)


class Mlp(nn.Module):
    def __init__(self, dim, spectral_norm_layer, mul=4, out_dim=None):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        spec = get_spectral_norm(spectral_norm_layer)
        self.fc1 = spec(nn.Linear(dim, mul * dim))
        self.act = nn.GELU()
        self.fc2 = spec(nn.Linear(mul * dim, out_dim))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


def get_causal_mask(causal_mask_sizes, mask_diag=False):
    cumrange = torch.tensor([i for i in range(len(causal_mask_sizes)) for _ in range(causal_mask_sizes[i])])
    row, col = torch.meshgrid(cumrange, cumrange)  # indexing="ij"
    if mask_diag:
        causal_mask = row <= col
    else:
        causal_mask = row < col
    return causal_mask.view(1, 1, len(cumrange), len(cumrange))