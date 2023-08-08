import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .transform import CustomNorm
from .weight_init import init_weights

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def deconv3x3(in_planes, out_planes, stride=1):
    """3x3 transposed convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=1, bias=False)


def build_layer(mode, in_planes, out_planes, norm_layer):
    conv = conv3x3 if mode == "conv" else deconv3x3
    return nn.Sequential(
        conv(in_planes, out_planes, stride=2),
        CustomNorm(norm_layer, out_planes),
        nn.GELU()
    )


class UNet(nn.Module):
    def __init__(self, num_channels_in, num_channels_out, embed_dim, norm_layer, depth, scale_hd, zero_init, upmode):
        super().__init__()
        # params
        self.depth = depth

        # layers
        self.to_emb = conv3x3(num_channels_in, embed_dim // (2 ** (depth - 1)))
        self.from_emb = conv3x3(2 * embed_dim // (2 ** (depth - 1)), num_channels_out)
        conv_layers = []
        deconv_layers = []
        for i in range(depth):
            planes = embed_dim // (2 ** (depth - 1 - i))
            conv_layers.append(build_layer("conv", planes, planes * 2, norm_layer))
            mul = 2 if i == depth - 1 else 4
            deconv_layers.append(build_layer("deconv", planes * mul, planes, norm_layer))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.deconv_layers = nn.ModuleList(deconv_layers)

        self.apply(init_weights)
        if zero_init:
            self.from_emb.weight.data.zero_()

    def forward(self, x):
        y_list = [self.to_emb(x)]
        for i in range(self.depth):
            y_list.append(self.conv_layers[i](y_list[-1]))
        y = y_list.pop()
        y = self.deconv_layers[-1](y)
        for i in range(1, self.depth):
            y_skip = y_list.pop()
            y = torch.cat([y, y_skip], dim=1)
            y = self.deconv_layers[-1 - i](y)
        y_skip = y_list.pop()
        y = torch.cat([y, y_skip], dim=1)
        x = self.from_emb(y)
        return x


class ConvPatchProj(nn.Module):
    def __init__(self, patch_size, embed_dim, norm_layer, num_channels, skip_channels=0, from_patch=True, hr_ratio=None, use_hr=False):
        super().__init__()
        # params
        self.from_patch = from_patch
        self.num_channels = num_channels
        self.use_hr = use_hr
        self.conv = conv3x3 if from_patch else deconv3x3
        self.skip_conv = conv3x3
        self.norm_layer = norm_layer
        self.patch_size = patch_size
        self.use_skip = skip_channels > 0

        # compute layer dims
        num_dims = int(math.log2(patch_size))
        dims = [embed_dim // (2 ** k) for k in range(num_dims)] + [num_channels]
        hr_dims = []
        if use_hr:
            assert hr_ratio is not None
            num_hr_dims = int(math.log2(hr_ratio))
            hr_dims = [embed_dim // (2 ** k) for k in range(num_dims - 1, num_dims + num_hr_dims)] + [num_channels]
        if from_patch:
            dims.reverse()
            hr_dims.reverse()

        # define layers
        layer_dims, proj_dims = (dims[1:], dims[:2]) if from_patch else (dims[:-1], dims[-2:])
        self.layers = self.build_layers(layer_dims, activate_last=not from_patch)
        if self.use_skip:
            if from_patch or use_hr:
                raise NotImplementedError
            self.skip_proj = self.skip_conv(skip_channels, proj_dims[0], stride=2)
            proj_dims = (2 * proj_dims[0], proj_dims[1])
        self.proj = self.conv(*proj_dims, stride=2)
        if use_hr:
            hr_layer_dims, hr_proj_dims = (hr_dims[1:], hr_dims[:2]) if from_patch else (hr_dims[:-1], hr_dims[-2:])
            self.hr_layers = self.build_layers(hr_layer_dims, activate_last=not from_patch)
            self.hr_proj = self.conv(*hr_proj_dims, stride=2)

    def build_layers(self, dims, activate_last=False):
        layers = []
        num_dims = len(dims) - 1 if activate_last else len(dims) - 2
        for i in range(num_dims):
            layers.append(nn.Sequential(
                self.conv(dims[i], dims[i + 1], stride=2),
                CustomNorm(self.norm_layer, dims[i + 1]),
                nn.GELU()
            ))
        if not activate_last:
            layers.append(self.conv(dims[-2], dims[-1], stride=2))
        return nn.ModuleList(layers)

    def append(self, x, x_list, return_list):
        if return_list:
            x_list.append(x)

    def fuse(self, x, x_list, idx, fuse_m):
        if x_list is not None:
            xi = x_list[idx]
            fuse_m = F.interpolate(fuse_m, size=x.shape[-2:], mode='bilinear', align_corners=False)
            x = fuse_m * xi + (1 - fuse_m) * x
        idx += 1
        return x, idx

    def forward(self, x, latent_shape=None, return_list=False, x_list=None, fuse_m=None, skip=None):
        if self.from_patch:
            x_list = []
            B, C, H, W = x.shape
            if C == self.num_channels - 1:
                # if no alpha in input and expect alpha channel, set alpha channel to one
                x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
            if C == self.num_channels + 1:
                # if alpha in input and do not expect alpha, remove alpha channel
                x = x[:, :self.num_channels]
            if self.use_hr:
                x = self.hr_proj(x)
                self.append(x, x_list, return_list)
                for l in self.hr_layers:
                    x = l(x)
                    self.append(x, x_list, return_list)
            else:
                x = self.proj(x)
                self.append(x, x_list, return_list)
            for l in self.layers:
                x = l(x)
                self.append(x, x_list, return_list)
            x = x.flatten(2).transpose(1, 2).contiguous()

        else:
            B, L, C = x.shape
            assert latent_shape is not None
            H, W = latent_shape
            x = x.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
            idx = 0
            for l in self.layers:
                x, idx = self.fuse(x, x_list, idx, fuse_m)
                x = l(x)
            if self.use_hr:
                for l in self.hr_layers:
                    x, idx = self.fuse(x, x_list, idx, fuse_m)
                    x = l(x)
                x, idx = self.fuse(x, x_list, idx, fuse_m)
                x = self.hr_proj(x)
            else:
                x, idx = self.fuse(x, x_list, idx, fuse_m)
                if self.use_skip:
                    x_skip = self.skip_proj(skip)
                    x = torch.cat([x, x_skip], dim=1)
                x = self.proj(x)

        if return_list:
            x_list.reverse()
            return x_list
        return x