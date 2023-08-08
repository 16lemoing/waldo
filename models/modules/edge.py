import math
import torch
import torch.nn as nn

from tools.utils import flatten, unflatten


class EdgeExtractor(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        # params
        self.max_edge = math.sqrt(32)

        # kernel
        k = kernel_size
        assert k % 2 == 1
        self.reflect = torch.nn.ReflectionPad2d(k // 2)
        mean = torch.ones(1, 1, k, k) / (k ** 2)
        self.register_buffer('mean_kernel', mean)
        sobel = torch.tensor(list(range(k))) - k // 2
        sobel_x, sobel_y = sobel.view(-1, 1), sobel.view(1, -1)
        sum_xy = sobel_x ** 2 + sobel_y ** 2
        sum_xy[sum_xy == 0] = 1
        sobel_x, sobel_y = sobel_x / sum_xy, sobel_y / sum_xy
        sobel = torch.stack([sobel_x.unsqueeze(0), sobel_y.unsqueeze(0)], dim=0)
        self.register_buffer('sobel_kernel', torch.tensor(sobel))

    def forward(self, vid, eps=1e-6, blur=False):
        flow, shape = flatten(vid, ndim=4)
        B, C, H, W = flow.shape
        flow = flow.reshape(B * C, 1, H, W)
        mean_flow = F.conv2d(self.reflect(flow), self.mean_kernel)
        mean_flow_norm = (mean_flow.view(B, C, H, W) ** 2).sum(dim=1, keepdim=True)
        flow_norm = (flow.view(B, C, H, W) ** 2).sum(dim=1, keepdim=True)
        dominant_flow = (flow_norm > mean_flow_norm).float()
        flow_edge = F.conv2d(self.reflect(flow), self.sobel_kernel)
        flow_edge = ((flow_edge ** 2).sum(dim=1, keepdim=True) + eps).sqrt() / self.max_edge
        flow_edge = 1 - (1 - flow_edge.view(B, C, H, W)).prod(dim=1, keepdim=True)
        flow_edge = flow_edge.reshape(B, 1, H, W)
        return unflatten(flow_edge, shape), unflatten(dominant_flow, shape)