import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import torch
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter

from tools.utils import color_transfer, flatten_vid, unflatten_vid

class Logger():
    def __init__(self, opt):
        self.writer = SummaryWriter(opt.log_path)
        self.log_path = opt.log_path
        self.fps = opt.log_fps
        self.imagenet_norm = opt.imagenet_norm
        if opt.palette is not None:
            assert len(opt.palette) == 3 * opt.num_lyt
            self.palette = np.array([opt.palette[3 * k: 3 * (k + 1)] + [255] for k in range(opt.num_lyt)]).astype(np.float64) / 255

    def is_empty(self, tensors):
        for tensor in tensors:
            if 0 in tensor.size():
                return True
        return False

    def detach(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu()
        return tensor

    def number(self, tensor):
        if isinstance(tensor, torch.Tensor) and tensor.isnan().any():
            return torch.zeros_like(tensor)
        return tensor

    def log_img(self, name, tensor, nrow, global_iter, natural=True, normalize=False, span=None, pad_value=0, grid_div=10):
        if self.is_empty([tensor]):
            return
        tensor = self.detach(tensor)
        with torch.no_grad():
            alpha = tensor[:, 3:].float()
            tensor = tensor[:, :3].float()
            if 0 not in alpha.size():
                assert normalize
                assert span is not None
                tensor = add_alpha_grid(tensor, alpha, span)
            if natural and normalize and self.imagenet_norm:
                # tensor should be in [-1 1]
                tensor *= torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]])
                tensor += torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]])
                tensor = tensor.clamp(0, 1)
                normalize = False
            grid = make_grid(tensor, nrow=nrow, normalize=normalize, range=span, pad_value=pad_value)
            self.writer.add_image(name, grid, global_iter)

    def get_pts(self, obj_pts, bg_pts, height, width, mul, dpi=64):
        obj_pts = self.detach(obj_pts)
        bg_pts = self.detach(bg_pts)
        obj_pts, vid_size = flatten_vid(obj_pts, vid_ndim=5)
        bg_pts, _ = flatten_vid(bg_pts, vid_ndim=5)
        B, No, Lo, _ = obj_pts.shape
        _, _, L, _ = bg_pts.shape
        height = height * mul
        width = width * mul
        img = torch.zeros(B, 3, height, width)
        colormap = cm.get_cmap('jet', No + 2)(np.linspace(0, 1, No + 2)[:No + 1])
        colormap[0, :3] = 0.5
        for i in range(B):
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            fig.patch.set_facecolor('#FFFFFF')
            c = colormap[0]
            plt.scatter(bg_pts[i, 0, :, 0], -bg_pts[i, 0, :, 1], marker="x", color=c, linewidths=mul, s=10 * mul)
            for j in range(No):
                c = colormap[1 + j]
                plt.scatter(obj_pts[i, j, :, 0], -obj_pts[i, j, :, 1], marker="x", color=c, linewidths=mul, s=10 * mul)
            plt.xlim([-1, 1])
            plt.ylim([-1, 1])
            plt.axis('off')
            plt.tight_layout()
            fig.canvas.draw()
            data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img[i] = torch.from_numpy(data).permute(2, 0, 1) / 255 * 2 - 1
            plt.close()

        return unflatten_vid(img, vid_size)

    def get_delta_mot(self, obj_pts, bg_pts, ref_obj_pts, ref_bg_pts, height, width, mul, dpi=64):
        obj_pts = self.detach(obj_pts)
        bg_pts = self.detach(bg_pts)
        ref_obj_pts = self.detach(ref_obj_pts)
        ref_bg_pts = self.detach(ref_bg_pts)
        B, T, No, Lo, _ = obj_pts.shape
        _, _, _, L, _ = bg_pts.shape
        height = height * mul
        width = width * mul
        vid = torch.zeros(B, T, 3, height, width)
        colormap = cm.get_cmap('jet', No + 2)(np.linspace(0, 1, No + 2)[:No + 1])
        colormap[0, :3] = 0.5
        for i in range(B):
            for t in range(T):
                fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
                fig.patch.set_facecolor('#FFFFFF')
                c = colormap[0]
                plt.scatter(ref_bg_pts[i, 0, :, 0], -ref_bg_pts[i, 0, :, 1], marker="x", color=c, linewidths=mul, s=10 * mul)
                x, y = ref_bg_pts[i, 0, :, 0], -ref_bg_pts[i, 0, :, 1]
                u, v = bg_pts[i, t, 0, :, 0] - ref_bg_pts[i, 0, :, 0], (-bg_pts[i, t, 0, :, 1]) - (-ref_bg_pts[i, 0, :, 1])
                plt.quiver(x, y, u, v, color=c, angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=2, headlength=2, headaxislength=2)
                for j in range(No):
                    x, y = ref_obj_pts[i, j, :, 0], -ref_obj_pts[i, j, :, 1]
                    u, v = obj_pts[i, t, j, :, 0] - ref_obj_pts[i, j, :, 0], (-obj_pts[i, t, j, :, 1]) - (-ref_obj_pts[i, j, :, 1])
                    c = colormap[1 + j]
                    plt.scatter(ref_obj_pts[i, j, :, 0], -ref_obj_pts[i, j, :, 1], marker="x", color=c, linewidths=mul, s=10 * mul)
                    plt.quiver(x, y, u, v, color=c, angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=2, headlength=2, headaxislength=2)
                plt.xlim([-1, 1])
                plt.ylim([-1, 1])
                plt.axis('off')
                plt.tight_layout()
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                vid[i, t] = torch.from_numpy(data).permute(2, 0, 1) / 255 * 2 - 1
                plt.close()
        return vid

    def get_mot(self, obj_pts, bg_pts, height, width, mul, dpi=64, forward=True):
        obj_pts = self.detach(obj_pts)
        bg_pts = self.detach(bg_pts)
        B, T, No, Lo, _ = obj_pts.shape
        _, _, _, L, _ = bg_pts.shape
        height = height * mul
        width = width * mul
        vid = torch.zeros(B, T - 1, 3, height, width)
        colormap = cm.get_cmap('jet', No + 2)(np.linspace(0, 1, No + 2)[:No + 1])
        colormap[0, :3] = 0.5
        for i in range(B):
            for t in range(T-1):
                fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
                fig.patch.set_facecolor('#FFFFFF')
                c = colormap[0]
                if forward:
                    x, y = bg_pts[i, t, 0, :, 0], -bg_pts[i, t, 0, :, 1]
                    u, v = bg_pts[i, t + 1, 0, :, 0] - bg_pts[i, t, 0, :, 0], (-bg_pts[i, t + 1, 0, :, 1]) - (-bg_pts[i, t, 0, :, 1])
                else:
                    x, y = bg_pts[i, t + 1, 0, :, 0], -bg_pts[i, t + 1, 0, :, 1]
                    u, v = bg_pts[i, t, 0, :, 0] - bg_pts[i, t + 1, 0, :, 0], (-bg_pts[i, t, 0, :, 1]) - (-bg_pts[i, t + 1, 0, :, 1])
                plt.quiver(x, y, u, v, color=c, angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=2, headlength=2, headaxislength=2)
                for j in range(No):
                    if forward:
                        x, y = obj_pts[i, t, j, :, 0], -obj_pts[i, t, j, :, 1]
                        u, v = obj_pts[i, t + 1, j, :, 0] - obj_pts[i, t, j, :, 0], (-obj_pts[i, t + 1, j, :, 1]) - (-obj_pts[i, t, j, :, 1])
                    else:
                        x, y = obj_pts[i, t + 1, j, :, 0], -obj_pts[i, t + 1, j, :, 1]
                        u, v = obj_pts[i, t, j, :, 0] - obj_pts[i, t + 1, j, :, 0], (-obj_pts[i, t, j, :, 1]) - (-obj_pts[i, t + 1, j, :, 1])
                    c = colormap[1 + j]
                    plt.quiver(x, y, u, v, color=c, angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=2, headlength=2, headaxislength=2)
                plt.xlim([-1, 1])
                plt.ylim([-1, 1])
                plt.axis('off')
                plt.tight_layout()
                fig.canvas.draw()
                data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                vid[i, t] = torch.from_numpy(data).permute(2, 0, 1) / 255 * 2 - 1
                plt.close()

        return vid

    def get_lyt(self, tensor, lyt_size, use_palette=False):
        tensor = self.detach(tensor)
        tensor, vid_size = flatten_vid(tensor, vid_ndim=5)
        tensor = tensor.max(dim=-3)[1]
        if use_palette:
            colormap = self.palette
        else:
            colormap = cm.get_cmap('jet', lyt_size + 1)(np.linspace(0, 1, lyt_size + 1)[:lyt_size])
            colormap[0, :3] = 0.5
        tensor = color_transfer(tensor.unsqueeze(1), colormap)
        return unflatten_vid(tensor, vid_size)

    def log_lyt(self, name, tensor, nrow, global_iter, lyt_size, use_palette=False):
        if self.is_empty([tensor]):
            return
        tensor = self.detach(tensor)
        with torch.no_grad():
            tensor, vid_size = flatten_vid(tensor, vid_ndim=5)
            alpha = tensor[:, lyt_size:].float()
            tensor = tensor[:, :lyt_size].float()
            tensor = tensor.max(dim=-3)[1]
            if use_palette:
                colormap = self.palette
            else:
                colormap = cm.get_cmap('jet', lyt_size + 1)(np.linspace(0, 1, lyt_size + 1)[:lyt_size])
                colormap[0, :3] = 0.5
            tensor = color_transfer(tensor.unsqueeze(1), colormap)
            if 0 not in alpha.size():
                tensor = add_alpha_grid(tensor, alpha, span=(-1, 1))
            tensor = unflatten_vid(tensor, vid_size)
            if vid_size is None:
                self.log_img(name, tensor, nrow, global_iter, normalize=True, span=(-1, 1))
            else:
                self.log_vid(name, tensor, nrow, global_iter, normalize=True, span=(-1, 1))

    def log_seg(self, name, tensor, global_iter, nrow=None, seg_dim=None, is_soft=False):
        if self.is_empty([tensor]):
            return
        tensor = self.detach(tensor)
        with torch.no_grad():
            if is_soft:
                seg_dim = tensor.size(-3)
                tensor = tensor.max(dim=-3)[1]
            else:
                assert seg_dim is not None
            seg, vid_size = flatten_vid(tensor, vid_ndim=4)
            colormap = cm.get_cmap('hsv', seg_dim + 1)(np.linspace(0, 1, seg_dim + 1)[:seg_dim])
            seg = color_transfer(seg.unsqueeze(1), colormap)
            seg = unflatten_vid(seg, vid_size)
            if vid_size is None:
                assert nrow is not None
                self.log_img(name, seg, nrow, global_iter, normalize=True, span=(-1, 1))
            else:
                self.log_vid(name, seg, nrow, global_iter, normalize=True, span=(-1, 1))

    def log_vid(self, name, tensor, nrow, global_iter, natural=True, normalize=False, span=None, ts=None, ctx_mask=None, pad_value=0):
        if self.is_empty([tensor]):
            return
        tensor = self.detach(tensor)
        with torch.no_grad():
            alpha = tensor[:, :, 3:].float()
            tensor = tensor[:, :, :3]
            if natural and normalize and self.imagenet_norm:
                # tensor should be in [-1 1]
                tensor *= torch.tensor([[[[0.229]], [[0.224]], [[0.225]]]])
                tensor += torch.tensor([[[[0.485]], [[0.456]], [[0.406]]]])
                tensor = tensor.clamp(0, 1)
                normalize = False
            if ctx_mask is not None:
                # show synthetic frames with red border
                low_h, low_w = int(0.03 * tensor.size(3)), int(0.03 * tensor.size(4))
                high_h, high_w = tensor.size(3) - low_h, tensor.size(4) - low_w
                red_color = torch.tensor([[[1.]], [[0.]], [[0.]]])
                pred_frames = tensor[~ctx_mask]
                pred_frames[:, :, :low_h] = red_color
                pred_frames[:, :, high_h:] = red_color
                pred_frames[:, :, :, :low_w] = red_color
                pred_frames[:, :, :, high_w:] = red_color
                tensor[~ctx_mask] = pred_frames
            elif ts is not None:
                # show synthetic frames with red border
                low_h, low_w = int(0.03 * tensor.size(3)), int(0.03 * tensor.size(4))
                high_h, high_w = tensor.size(3) - low_h, tensor.size(4) - low_w
                red_color = torch.tensor([[[[1.]], [[0.]], [[0.]]]])
                tensor[:, ts, :, :low_h] = red_color
                tensor[:, ts, :, high_h:] = red_color
                tensor[:, ts, :, :, :low_w] = red_color
                tensor[:, ts, :, :, high_w:] = red_color
            if 0 not in alpha.size():
                for i in range(tensor.size(1)):
                    tensor[:, i] = add_alpha_grid(tensor[:, i], alpha[:, i], span)
            grid = [make_grid(tensor[:, i], nrow=nrow, normalize=normalize, range=span, pad_value=pad_value) for i in range(tensor.size(1))]
            grid = torch.stack(grid, dim=0).unsqueeze(0)
            bgrid = (grid * 255).clamp(0, 255).byte()
            self.writer.add_video(name, bgrid, global_iter, self.fps)

    def get_flow(self, flow, mul=10):
        flow = self.detach(flow)
        if len(flow.shape) == 5:
            bs, t, _, h, w = flow.shape
            flow = flow.permute(0, 1, 3, 4, 2)
            flow_vid = torch.zeros(bs, t, 3, h, w)
            for i in range(t):
                flow_vid[:, i] = self.get_flow_rgb(flow[:, i], mul=mul)
            return flow_vid
        else:
            flow = flow.permute(0, 2, 3, 1)
            return self.get_flow_rgb(flow, mul=mul)

    def log_flow(self, name, flow, nrow, global_iter, mul=10, ctx_mask=None):
        if self.is_empty([flow]):
            return
        flow = self.detach(flow)
        with torch.no_grad():
            if len(flow.shape) == 5:
                bs, t, _, h, w = flow.shape
                flow = flow.permute(0, 1, 3, 4, 2)
                flow_vid = torch.zeros(bs, t, 3, h, w)
                for i in range(t):
                    flow_vid[:, i] = self.get_flow_rgb(flow[:, i], mul=mul)
                self.log_vid(name, flow_vid, nrow, global_iter, ctx_mask=ctx_mask)
            else:
                flow = flow.permute(0, 2, 3, 1)
                self.log_img(name, self.get_flow_rgb(flow, mul=mul), nrow, global_iter)

    def log_scalar(self, name, scalar, global_iter):
        if scalar is not None:
            if type(scalar) == list:
                for i, x in enumerate(scalar):
                    self.log_scalar(f"{name}_{i}", self.number(self.detach(x)), global_iter)
            else:
                self.writer.add_scalar(name, self.number(self.detach(scalar)), global_iter)

    def log_scalars(self, name, scalars, global_iter):
        for s in scalars:
            self.log_scalar(f"{name}/{s}", scalars[s], global_iter)

    def log_text(self, name, texts, global_iter):
        text = '\n'.join([f"[{i}] {text}" for i, text in enumerate(texts)])
        self.writer.add_text(name, text, global_iter)

    def get_flow_rgb(self, flow, mul=10):
        r = (flow ** 2).sum(-1).sqrt() / np.sqrt(2) * mul
        r[r > 1] = 1.
        theta = (1 + torch.atan2(flow.select(-1, -1), flow.select(-1, 0)) / np.pi) / 2
        cmp = cm.get_cmap('hsv', 128)
        flow_rgba = cmp(theta.numpy())
        flow_rgb = torch.tensor(flow_rgba[:, :, :, :3]).float()
        flow_rgb = r.unsqueeze(-1) * flow_rgb
        return flow_rgb.permute(0, 3, 1, 2)

def add_alpha_grid(tensor, alpha, span, grid_div=10):
    _, _, H, W = tensor.shape
    alpha = (alpha - span[0]) / (span[1] - span[0])
    grid_size = H // grid_div
    grid = torch.stack([torch.arange(H).view(-1, 1).expand(-1, W), torch.arange(W).view(1, -1).expand(H, -1)])
    midspan0 = 2 / 3 * span[0] + 1 / 3 * span[1]
    midspan1 = 1 / 3 * span[0] + 2 / 3 * span[1]
    grid = ((grid[:1] // grid_size) % 2 == (grid[1:] // grid_size) % 2).float()
    grid = grid * (midspan1 - midspan0) + midspan0
    tensor = alpha * tensor + (1 - alpha) * grid
    return tensor