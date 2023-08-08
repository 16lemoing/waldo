import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(code_dir, "models/modules/mat_utils"))

import torch
import torch.nn as nn
from torch.nn import functional as F

import dnnlib
from networks.mat import Generator
import legacy

from tools.utils import expand

def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = {name: tensor for name, tensor in named_params_and_buffers(src_module)}
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)


def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())




class MatInpainter(nn.Module):
    def __init__(self, opt):
        super().__init__()
        config = {'z_dim': 512,
                  'c_dim': 0,
                  'w_dim': 512,
                  'img_resolution': 512,
                  'img_channels': 3}
        self.net = Generator(**config).cuda().eval().requires_grad_(False)
        with dnnlib.util.open_url(opt.inpainter_path) as f:
            net_saved = legacy.load_network_pkl(f)['G_ema'].cuda().eval().requires_grad_(False)  # type: ignore
        copy_params_and_buffers(net_saved, self.net, require_all=True)

    def forward(self, x, mask, exp=True, is_masked=True):
        h, w = x.shape[2:]
        if h == w:
            h0, w0 = 512, 512
            if (h, w) != (h0, w0):
                x = F.interpolate(x, (h0, w0), mode="bilinear")
                # mask = F.interpolate(mask, (h0, w0), mode="bilinear")
                mask = F.interpolate(mask, (h0, w0), mode="nearest")
            if not is_masked:
                x = (1 - mask) * x
            # assert h == 512 and w == 1024
            z = torch.randn(x.size(0), 512).to(x.device)
            m = expand(mask, 3) if exp else mask
            x2 = self.net(x, 1 - m, z, None, noise_mode="const", truncation_psi=0.5)
            x = x2 * mask + x * (1. - mask)
            if (h, w) != (h0, w0):
                x = F.interpolate(x, (h, w), mode="bilinear")
            return x
        else:
            h, w = x.shape[2:]
            h0, w0 = 512, 1024
            if (h, w) != (h0, w0):
                x = F.interpolate(x, (h0, w0), mode="bilinear")
                # mask = F.interpolate(mask, (h0, w0), mode="bilinear")
                mask = F.interpolate(mask, (h0, w0), mode="nearest")
            if not is_masked:
                x = (1 - mask) * x
            # assert h == 512 and w == 1024
            x2 = torch.zeros_like(x)
            c = torch.zeros_like(mask)
            for i in range(3):
                s = int(256 * i)
                c_square = torch.cat([torch.linspace(1, 100, 256), torch.linspace(100, 1, 256)], dim=0).cuda()
                c_square = c_square.view(1, 1, 1, -1)
                x_square = x[:, :, :, s:s + 512]
                mask_square = mask[:, :, :, s:s + 512]
                z = torch.randn(x.size(0), 512).to(x.device)
                m = expand(mask_square, 3) if exp else mask_square
                x2_square = self.net(x_square, 1 - m, z, None, noise_mode="const", truncation_psi=0.5)
                x2[:, :, :, s:s + 512] += x2_square * c_square
                c[:, :, :, s:s + 512] += c_square
            x2 = x2 / c
            x = x2 * mask + x * (1. - mask)
            if (h, w) != (h0, w0):
                x = F.interpolate(x, (h, w), mode="bilinear")
            return x