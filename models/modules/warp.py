###############################################################################
# Code adapted from
# https://github.com/WarBean/tps_stn_pytorch/blob/master/tps_grid_gen.py
# Modified the original code to add warp inversion
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.utils import get_gaussian_kernel, get_grid


# phi(x1, x2) = r^2 * log(r) = 0.5 * r^2 * log(r^2), where r = ||pts_1 - pts_2||_2
def kernel_distance(pts_1, pts_2, eps=1e-8):
    N, M = pts_1.size(0), pts_2.size(0)
    d = (pts_1 ** 2).sum(dim=-1).view(N, 1) + (pts_2 ** 2).sum(dim=-1).view(1, M) - 2 * pts_1 @ pts_2.t()
    return 0.5 * d * d.add(eps).log()


class TPSWarp(nn.Module):
    def __init__(self, tgt_height, tgt_width, tgt_pts):
        super().__init__()
        self.tgt_shape = [tgt_height, tgt_width]
        N = tgt_pts.size(0)

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3).float()
        forward_kernel[:N, :N].copy_(kernel_distance(tgt_pts.float(), tgt_pts.float()))
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(tgt_pts.float())
        forward_kernel[-2:, :N].copy_(tgt_pts.float().transpose(0, 1))
        
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target coordinate matrix
        tgt_size = tgt_height * tgt_width
        tgt_grid = get_grid(tgt_height, tgt_width).view(-1, 2)
        tgt_grid_partial_repr = kernel_distance(tgt_grid.float(), tgt_pts.float())
        tgt_grid_repr = torch.cat([tgt_grid_partial_repr, torch.ones(tgt_size, 1).float(), tgt_grid.float()], dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('pad', torch.zeros(3, 2).float())
        self.register_buffer('tgt_grid_repr', tgt_grid_repr)

    def forward(self, src_pts):
        B = src_pts.size(0)
        H, W = self.tgt_shape
        x = torch.cat([src_pts.float(), self.pad.expand(B, 3, 2)], 1)
        mapping = torch.matmul(self.inverse_kernel, x)
        tgt_grid = torch.matmul(self.tgt_grid_repr, mapping)
        return tgt_grid.view(B, H, W, 2).float()


class InverseWarp(nn.Module):
    def __init__(self, src_height, src_width, tgt_height, tgt_width, kernel_size=3, num_perm=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.tgt_shape = [tgt_height, tgt_width]
        self.num_perm = num_perm
        self.register_buffer('kernel', get_gaussian_kernel(kernel_size).view(1, 1, kernel_size, kernel_size))
        self.register_buffer('src_grid', get_grid(src_height, src_width))
        self.register_buffer('tgt_grid', get_grid(tgt_height, tgt_width))
        self.register_buffer('x_grid', torch.arange(tgt_width).view(1, -1).repeat(tgt_height, 1).view(1, -1).float())
        self.register_buffer('y_grid', torch.arange(tgt_height).view(-1, 1).repeat(1, tgt_width).view(1, -1).float())
        self.register_buffer('perm', torch.stack([torch.randperm(tgt_height * tgt_width) for _ in range(num_perm)]))

    def forward(self, src_grid, niter=5, pad=True, erode=True):
        B, Hs, Ws, _ = src_grid.shape
        H, W = self.tgt_shape
        N, P = niter, self.num_perm

        dsrc = src_grid - self.src_grid
        dsrc = F.interpolate(dsrc.permute(0, 3, 1, 2), self.tgt_shape, mode="bilinear")
        dx = dsrc[:, 0].view(B, -1) * W / 2
        dy = dsrc[:, 1].view(B, -1) * H / 2
        y_grid = self.y_grid.expand(B, -1) + dy
        x_grid = self.x_grid.expand(B, -1) + dx
        y_grid = y_grid.round().long()
        x_grid = x_grid.round().long()
        field = y_grid * W + x_grid

        # filter out of bound
        out_of_bound = (y_grid < 0) | (x_grid < 0) | (y_grid > H - 1) | (x_grid > W - 1)
        field[out_of_bound] = -1

        # filter duplicates
        if P > 1:
            field = field + 1 # B H*W
            field = field.unsqueeze(1).expand(-1, P, -1) # B P H*W
            dx = dx.unsqueeze(1).expand(-1, P, -1)  # B P H*W
            dy = dy.unsqueeze(1).expand(-1, P, -1)  # B P H*W
            perm = self.perm.unsqueeze(0).expand(B, -1, -1) # B P H*W
            field = torch.gather(field, 2, perm) # B P H*W
            dx = torch.gather(dx, 2, perm) # B P H*W
            dy = torch.gather(dy, 2, perm) # B P H*W
            field, idx = field.sort(dim=-1)
            field[:, :, 1:] *= ((field[:, :, 1:] - field[:, :, :-1]) != 0).long()
            idx = idx.sort(dim=-1)[1]
            field = torch.gather(field, 2, idx)
            dx = torch.cat([torch.zeros(B, P, 1, device=dx.device), dx], dim=2)
            dy = torch.cat([torch.zeros(B, P, 1, device=dy.device), dy], dim=2)
            field = torch.cat([torch.zeros(B, P, 1, device=field.device).long(), field], dim=2)
            inv_dx = torch.zeros_like(dx).scatter_(2, field, -dx)[:, :, 1:].view(B, P, H, W)
            inv_dx = inv_dx.mean(dim=1)
            inv_dy = torch.zeros_like(dy).scatter_(2, field, -dy)[:, :, 1:].view(B, P, H, W)
            inv_dy = inv_dy.mean(dim=1)
            mask = torch.zeros_like(dx).scatter_(2, field, 1)[:, 0, 1:].view(B, H, W).bool()
        else:
            field = field + 1
            field, idx = field.sort(dim=-1)
            field[:, 1:] *= ((field[:, 1:] - field[:, :-1]) != 0).long()
            idx = idx.sort(dim=-1)[1]
            field = torch.gather(field, 1, idx)
            dx = torch.cat([torch.zeros(B, 1, device=dx.device), dx], dim=1)
            dy = torch.cat([torch.zeros(B, 1, device=dy.device), dy], dim=1)
            field = torch.cat([torch.zeros(B, 1, device=field.device).long(), field], dim=1)
            inv_dx = torch.zeros_like(dx).scatter_(1, field, -dx)[:, 1:].view(B, H, W)
            inv_dy = torch.zeros_like(dy).scatter_(1, field, -dy)[:, 1:].view(B, H, W)
            mask = torch.zeros_like(dx).scatter_(1, field, 1)[:, 1:].view(B, H, W).bool()

        padding = self.kernel_size // 2

        if pad:
            Hp, Wp = H + 2 * (N + 1), W + 2 * (N + 1)
            inv_dx = F.pad(inv_dx, (N + 1, N + 1, N + 1, N + 1))
            inv_dy = F.pad(inv_dy, (N + 1, N + 1, N + 1, N + 1))
            mask = F.pad(mask, (N + 1, N + 1, N + 1, N + 1))
        else:
            Hp, Wp = H, W

        for i in range(niter):
            # dilate mask
            new_mask = torch.zeros_like(mask)
            new_mask[:, 1:] = (~mask[:, 1:] & mask[:, :-1])
            new_mask[:, :-1] = (~mask[:, :-1] & mask[:, 1:]) | new_mask[:, :-1]
            new_mask[:, :, 1:] = (~mask[:, :, 1:] & mask[:, :, :-1]) | new_mask[:, :, 1:]
            new_mask[:, :, :-1] = (~mask[:, :, :-1] & mask[:, :, 1:]) | new_mask[:, :, :-1]
            # compute missing values using kxk mean
            new_inv_dx = F.conv2d(inv_dx.view(B, 1, Hp, Wp), self.kernel, padding=padding).view(B, Hp, Wp)
            new_inv_dy = F.conv2d(inv_dy.view(B, 1, Hp, Wp), self.kernel, padding=padding).view(B, Hp, Wp)
            new_sum = F.conv2d(mask.float().view(B, 1, Hp, Wp), self.kernel, padding=padding).view(B, Hp, Wp)
            inv_dx = inv_dx.clone()
            inv_dy = inv_dy.clone()
            inv_dx[new_mask] = new_inv_dx[new_mask] / new_sum[new_mask]
            inv_dy[new_mask] = new_inv_dy[new_mask] / new_sum[new_mask]
            # update mask
            mask = mask | new_mask

        if erode:
            for i in range(niter):
                # erode mask
                new_mask = torch.zeros_like(mask)
                new_mask[:, 1:] = (mask[:, 1:] & ~mask[:, :-1])
                new_mask[:, :-1] = (mask[:, :-1] & ~mask[:, 1:]) | new_mask[:, :-1]
                new_mask[:, :, 1:] = (mask[:, :, 1:] & ~mask[:, :, :-1]) | new_mask[:, :, 1:]
                new_mask[:, :, :-1] = (mask[:, :, :-1] & ~mask[:, :, 1:]) | new_mask[:, :, :-1]
                # update mask
                mask = mask & ~new_mask

        inv_dx = inv_dx.clone()
        inv_dy = inv_dy.clone()
        inv_dx[~mask] = 2 * W
        inv_dy[~mask] = 2 * H

        inv_dx = inv_dx[:, N + 1:-(N + 1), N + 1:-(N + 1)]
        inv_dy = inv_dy[:, N + 1:-(N + 1), N + 1:-(N + 1)]

        dtgt = torch.stack([inv_dx * 2 / W, inv_dy * 2 / H], dim=3)
        tgt_grid = self.tgt_grid + dtgt
        return tgt_grid
