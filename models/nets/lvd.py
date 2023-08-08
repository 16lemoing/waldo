import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from models.modules import TPSWarp, InverseWarp, trunc_normal_, init_weights, ConvPatchProj, CustomNorm, MultiBlocks
from tools.utils import flatten_vid, unflatten_vid, get_grid, flatten, unflatten

class LVD(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # params
        self.use_refiner = opt.pe_use_refiner
        self.use_post_refiner = opt.pe_use_post_refiner
        self.use_filter = opt.pe_use_edge_filter
        self.time_dropout = opt.time_dropout
        self.freeze_obj = opt.freeze_obj
        self.remove_obj = opt.remove_obj
        self.use_disocc = opt.use_disocc
        self.include_self = opt.include_self
        self.restrict_to_ctx = opt.restrict_to_ctx

        if opt.pad_obj_alpha > 0:
            Ho = int(opt.obj_shape[0] * opt.patch_size * opt.scale_factor)
            Wo = int(opt.obj_shape[1] * opt.patch_size * opt.scale_factor)
            Po = int(opt.pad_obj_alpha * opt.scale_factor)
            obj_alpha_mask = torch.ones(Ho, Wo)
            obj_alpha_mask[:Po] = 0
            obj_alpha_mask[:, :Po] = 0
            obj_alpha_mask[-Po:] = 0
            obj_alpha_mask[:, -Po:] = 0
            self.register_buffer('obj_alpha_mask', obj_alpha_mask.view(1, 1, 1, *obj_alpha_mask.shape))
        else:
            self.obj_alpha_mask = 1
        bg_alpha = torch.ones(1, 1, opt.dim, int(opt.dim * opt.aspect_ratio))
        if opt.pad_bg_alpha > 0:
            P = int(opt.pad_bg_alpha * opt.scale_factor)
            bg_alpha[:, :, :P] = -1
            bg_alpha[:, :, :, :P] = -1
            bg_alpha[:, :, -P:] = -1
            bg_alpha[:, :, :, -P:] = -1
        self.register_buffer('bg_alpha', bg_alpha)
        self.blend_mode_obj = opt.pe_refiner_blend_mode_obj
        self.blend_mode_bg = opt.pe_refiner_blend_mode_bg

        # occ
        self.register_buffer("diag", torch.eye(opt.num_obj, opt.num_obj)[None, None, :, :])

        # nets
        dtype = ("RGB" if opt.input_rgb else "") + ("L" if opt.input_lyt else "") + ("F" if opt.input_flow else "")
        self.encoder = ImageEncoder(opt, dtype=dtype)
        self.layer_estimator = LayerEstimator(opt, opt.oe_depth, opt.oe_num_timesteps)
        self.pose_estimator = PoseEstimator(opt, opt.pe_depth, opt.pe_pts_mode, init_mode=opt.pe_estimator_init_mode)
        self.warper = Warper(opt, repeat_border=opt.pe_repeat_border)
        self.decoder = ImageDecoder(opt, dtype="A", init_mode=opt.pe_decoder_init_mode, use_prior=opt.pe_decoder_use_prior)

    def compute_occ(self, occ_score, eps=1e-6):
        B, T, No = occ_score.shape
        occ = torch.exp(-occ_score ** 2) + eps
        occ = occ.view(B, T, No, 1) / (occ.view(B, T, No, 1) + occ.view(B, T, 1, No))
        occ = occ - 0.5 * self.diag
        # background is occluded by all objects
        occ = torch.cat([torch.ones(B, T, No, 1, device=occ.device), occ], dim=3)  # B T No No+1
        # background occludes none of the objects
        occ = torch.cat([torch.zeros(B, T, 1, No + 1, device=occ.device), occ], dim=2)  # B T No+1 No+1
        return occ

    def reduce_time(self, obj, bg, occ_obj_alpha, occ_bg_alpha, eps=1e-6):
        B, T, No, _, H, W = occ_obj_alpha.shape

        # reduce time for obj
        occ_obj_score = (occ_obj_alpha + 1) / 2 + eps
        if self.time_dropout:
            t = torch.randint(low=0, high=T, size=[B, 1, 1], device=obj.device).expand(-1, -1, No)
            rd = torch.rand(B, T, No, device=obj.device)
            e = rd.gather(dim=1, index=t)
            mask = (rd >= e).float() # at least one instance should be kept per timestep
            mask = mask.view(B, T, No, 1, 1, 1)
            occ_obj_score = occ_obj_score * mask
        occ_obj_score = F.normalize(occ_obj_score, p=1, dim=1)
        occ_obj = torch.cat([obj, occ_obj_alpha], dim=3)  # B T No C+1 Ho Wo
        obj = (occ_obj * occ_obj_score).sum(dim=1)  # B No C+1 Ho Wo

        # reduce time for bg
        occ_bg_score = (occ_bg_alpha + 1) / 2 + eps
        if self.time_dropout:
            t = torch.randint(low=0, high=T, size=[B, 1], device=obj.device).expand(-1, -1)
            rd = torch.rand(B, T, device=obj.device)
            e = rd.gather(dim=1, index=t)
            mask = (rd >= e).float()  # at least one instance should be kept per timestep
            mask = mask.view(B, T, 1, 1, 1)
            occ_bg_score = occ_bg_score * mask
        occ_bg_score = F.normalize(occ_bg_score, p=1, dim=1)
        occ_bg = torch.cat([bg, occ_bg_alpha], dim=2)  # B T C+1 H W
        bg = (occ_bg * occ_bg_score).sum(dim=1)  # B C+1 H W
        return obj, bg

    def reduce_comp(self, vid, occ, flow):
        B, T = vid.shape[:2]
        No = vid.size(2) - 1
        vid = (vid + 1) / 2  # B T No+1 C+1 H W
        # set background alpha to one
        alpha = torch.cat([torch.ones_like(vid[:, :, :1, -1:]), vid[:, :, 1:, -1:]], dim=2)  # B T No+1 1 H W
        # alpha = vid[:, :, :, -1:]
        # merge occ
        occ = (1 - alpha * occ.view(B, T, No + 1, No + 1, 1, 1)).prod(dim=2).unsqueeze(dim=3)  # B T No+1 1 H W
        # compose
        alpha = occ * alpha
        vid = (alpha * vid[:, :, :, :-1]).sum(dim=2)
        flow = (alpha[:, :-1] * flow).sum(dim=2)
        vid = 2 * vid - 1
        return vid, (2 * alpha - 1).squeeze(3), flow

    def forward(self, input=None, obj_pose=None, bg_pose=None, grid=None, x=None, x_obj=None, x_bg=None, obj_alpha=None, bg_alpha=None, occ=None, occ_score=None, ctx_ts=None, pred_ts=None, cls=None, mode=""):
        if mode == "encode_input":
            x = self.encoder(input)
            return x
        if mode == "estimate_layer":
            x_obj, x_bg, cls = self.layer_estimator(x)
            return x_obj, x_bg, cls
        elif mode == "estimate_pose":
            obj_pose, bg_pose, occ_score, pts_rest_obj, pts_rest_bg, last_obj, last_bg = self.pose_estimator(x, x_obj, x_bg)
            return obj_pose, bg_pose, occ_score, pts_rest_obj, pts_rest_bg, last_obj, last_bg
        elif mode == "estimate_alpha_grid_occ":
            obj_alpha, bg_alpha = self.decoder(x_obj), self.bg_alpha.expand(x_obj.size(0), -1, -1, -1)
            if self.remove_obj:
                obj_alpha = 0 * obj_alpha - 1
            if self.freeze_obj:
                obj_alpha = 0 * obj_alpha + 1
            obj_alpha = self.obj_alpha_mask * obj_alpha + (1 - self.obj_alpha_mask) * (-1.)
            grid = self.warper(obj_pose, bg_pose)
            occ = self.compute_occ(occ_score)
            return occ, obj_alpha, bg_alpha, grid
        elif mode == "decode_layer":
            obj, bg = self.warper.layer_from_input(input, grid)
            occ_obj_alpha, occ_bg_alpha, output_alpha = self.warper.alpha_to_alpha(obj_alpha, bg_alpha, grid, occ)
            obj, bg = self.reduce_time(obj, bg, occ_obj_alpha, occ_bg_alpha)
            return obj, bg, output_alpha
        elif mode == "decode_output":
            if self.restrict_to_ctx:
                flow, alpha_unflt, alpha, alpha_ctx, disocc = self.warper.grid_to_flow_ctx(input, grid, occ, obj_alpha, bg_alpha, cls, ctx_ts, pred_ts)
            else:
                flow, alpha_unflt, alpha, alpha_ctx, disocc = self.warper.grid_to_flow(input, grid, occ, obj_alpha, bg_alpha, cls, ctx_ts, pred_ts)
            output, raw_output = self.warper.input_to_output(input, alpha_ctx, flow, ctx_ts)
            raw_alpha = output[:, :, -1:] if output is not None else None
            if self.use_disocc:
                if self.include_self:
                    disocc = torch.cat([disocc, torch.ones_like(disocc[:, :1])], dim=1)
                raw_output = torch.cat([raw_output, disocc], dim=3)
            output = output[:, :, :-1] if output is not None else None
            return output, flow, alpha_unflt, alpha, raw_alpha, raw_output, alpha_ctx
        else:
            raise ValueError


def get_num_channels(dtype, num_lyt):
    num_channels = 0
    if "A" in dtype:
        num_channels += 1
    if "L" in dtype:
        num_channels += num_lyt
    if "M" in dtype:
        num_channels += 1
    if "S" in dtype:
        num_channels += 2
    if "RGB" in dtype:
        num_channels += 3
    if "F" in dtype:
        num_channels += 2
    return num_channels


def scale(tensor, scale_factor, mode="bilinear"):
    img, shape = flatten(tensor, ndim=4)
    if scale_factor != 1:
        img = F.interpolate(img, scale_factor=scale_factor, mode=mode)
    return unflatten(img, shape)


class ImageEncoder(nn.Module):
    def __init__(self, opt, dtype="RGB"):
        super().__init__()
        # params
        self.scale_factor = opt.load_dim / opt.dim if opt.load_dim > 0 else opt.scale_factor

        # img proj
        num_channels = get_num_channels(dtype, opt.num_lyt)
        self.from_img = ConvPatchProj(opt.patch_size, opt.embed_dim, opt.norm_layer_patch, num_channels, from_patch=True, hr_ratio=opt.hr_ratio, use_hr=opt.use_hr)

        # init weights
        self.apply(init_weights)

    def forward(self, vid, return_list=False):
        img, shape = flatten(vid, ndim=4)
        img = scale(img, 1 / self.scale_factor)
        x = self.from_img(img, return_list=return_list)
        return x if return_list else unflatten(x, shape)


def get_circle(shape, p=1.):
    H, W = shape
    x = torch.arange(W).view(1, -1).expand(H, -1)
    y = torch.arange(H).view(-1, 1).expand(-1, W)
    x = (x - W / 2).abs()
    y = (y - H / 2).abs()
    r = (x ** 2 + y ** 2).sqrt()
    return (r < p * min(H, W) / 2).unsqueeze(0)


class ImageDecoder(nn.Module):
    def __init__(self, opt, dtype="RGB", init_mode="", skip_channels=0, use_prior=False):
        super().__init__()
        # params
        self.has_alpha = "A" in dtype
        latent_obj_size = opt.obj_shape[0] * opt.obj_shape[1]
        latent_size = opt.latent_shape[0] * opt.latent_shape[1]
        self.latent_shape = {latent_size: opt.latent_shape, latent_obj_size: opt.obj_shape}
        num_channels = get_num_channels(dtype, opt.num_lyt)
        self.use_prior = use_prior
        if use_prior:
            Ho, Wo = opt.obj_shape[0] * opt.patch_size, opt.obj_shape[1] * opt.patch_size
            self.register_buffer("circle", get_circle([Ho, Wo], p=0.75).float().view(1, 1, Ho, Wo))
        self.scale_factor = opt.scale_factor

        # img proj
        self.norm = CustomNorm(opt.norm_layer, opt.embed_dim)
        self.to_img = ConvPatchProj(opt.patch_size, opt.embed_dim, opt.norm_layer_patch, num_channels, skip_channels=skip_channels, from_patch=False, hr_ratio=opt.hr_ratio, use_hr=opt.use_hr)

        # init weights
        self.apply(init_weights)
        self.init_bias = 0
        if init_mode in ["zero", "five"]:
            self.to_img.proj.weight.data.zero_()
        if init_mode == "five":
            self.init_bias = 5

    def forward(self, x, x_list=None, fuse_m=None, drop_alpha=False, skip=None):
        x, shape = flatten(x, ndim=3)
        skip, _ = flatten(skip, ndim=4)
        if fuse_m is not None:
            fuse_m, _ = flatten_vid(fuse_m)
        x = self.norm(x)
        img = self.to_img(x, latent_shape=self.latent_shape[x.size(1)], x_list=x_list, fuse_m=fuse_m, skip=skip)
        img = img + self.init_bias
        if self.has_alpha:
            alpha = img[:, -1:].tanh()
            if self.use_prior:
                alpha = self.circle * 1 + (1 - self.circle) * alpha
            img = torch.cat([img[:, :-1], alpha], dim=1)
        if self.has_alpha and drop_alpha:
            img = img[:, :-1]
        img = scale(img, self.scale_factor)
        return unflatten(img, shape)


class PoseEstimator(nn.Module):
    def __init__(self, opt, depth, pts_mode, init_mode="zero"):
        super().__init__()
        # params
        self.latent_obj_size = opt.obj_shape[0] * opt.obj_shape[1]
        self.latent_size = opt.latent_shape[0] * opt.latent_shape[1]
        self.pts_mode = pts_mode
        self.bound_rest = opt.bound_rest
        self.soft_bound_rest = opt.soft_bound_rest
        self.norm_scale = opt.norm_scale
        self.tgt_scale = opt.tgt_scale
        self.has_bg = opt.has_bg
        self.fix_bg = opt.fix_bg
        self.fix_bg1 = opt.fix_bg1
        self.bound_scale = opt.bound_scale
        self.min_scale = opt.min_scale
        self.max_scale = opt.max_scale
        self.use_delta = opt.use_delta
        self.occ_mode = opt.occ_mode
        self.ctx_len = opt.ctx_len
        self.use_last = opt.use_last_pose_decoder

        # embedding
        self.obj_embed = nn.Parameter(torch.randn(1, 1, self.latent_obj_size, opt.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.latent_size, opt.embed_dim))

        # main blocks
        args = {"block_type": "full",
                "depth": depth,
                "dim": opt.embed_dim,
                "num_heads": opt.num_heads,
                "norm_layer": opt.norm_layer,
                "spectral_norm_layer": None,
                "noise": False,
                "dropout": opt.dropout}
        self.blocks = MultiBlocks(**args)

        # pose prediction
        scale = opt.init_scale_obj
        if pts_mode == "head":
            tgt_pts = get_grid(*opt.obj_shape).view(1, 1, self.latent_obj_size, 2).expand(-1, opt.num_obj, -1, -1)
            bias = (tgt_pts.reshape(1, -1, 2) * scale).atanh()
            self.pose_size = 2
        elif pts_mode == "prior":
            self.register_buffer("tgt_pts", get_grid(*opt.obj_shape).view(1, 1, self.latent_obj_size, 2))
            if opt.rd_translate_bias:
                m = opt.translate_bias_mul
                bias = [torch.tensor([[[[0., 0., scale, 0., 0., opt.aspect_ratio * scale, m * np.random.rand(), m * np.random.rand()]]]]) for _ in range(opt.num_obj)]
                bias = torch.cat(bias, dim=1)
            elif opt.circle_translate_bias:
                r = opt.circle_translate_radius
                theta = [i * 2 * math.pi / (opt.num_obj + 1) for i in range(opt.num_obj)]
                tx, ty = [r * math.cos(x) for x in theta], [r * math.sin(x) for x in theta]
                bias = [torch.tensor([[[[0., 0., scale, 0., 0., opt.aspect_ratio * scale, x, y]]]]) for (x, y) in zip(tx, ty)]
                bias = torch.cat(bias, dim=1)
            else:
                bias = torch.tensor([[[[0., 0., scale, 0., 0., opt.aspect_ratio * scale, 0., 0.]]]])
            mul = torch.tensor([[[[opt.mul_delta_obj, opt.mul_delta_obj, opt.mul_scale_obj, opt.mul_scale_obj, opt.mul_scale_obj, opt.mul_scale_obj, 1., 1.]]]])
            self.pose_size = 8
            if opt.bound_rest:
                min_scale = opt.min_scale_bound
                max_scale = opt.max_scale_bound
                max_translate = opt.max_translate_bound
                self.register_buffer("min_bound", torch.tensor([[[0., 0., min_scale, 0., 0., opt.aspect_ratio * min_scale, -max_translate, -max_translate]]]))
                self.register_buffer("max_bound", torch.tensor([[[0., 0., max_scale, 0., 0., opt.aspect_ratio * max_scale, max_translate, max_translate]]]))
            if opt.has_bg:
                bg_bias = torch.tensor([[[[0., 0., 1., 0., 0., 1., 0., 0.]]]])
                self.register_buffer("bg_bias", bg_bias)
                self.bg_mul = opt.bg_mul
                self.register_buffer("tgt_pts_bg", get_grid(*opt.latent_shape).view(1, 1, self.latent_size, 2))
        self.register_buffer("bias", bias)
        self.register_buffer("mul", mul)
        self.scale_size = 1 if opt.bound_scale else 0

        # output projection
        self.occ_size = 1 # opt.num_obj * (opt.num_obj - 1)
        self.register_buffer("occ_bias", torch.tensor([[2. * i for i in range(opt.num_obj)]]))
        self.norm = CustomNorm(opt.norm_layer, opt.embed_dim)
        self.head = nn.Linear(opt.embed_dim, self.pose_size + self.scale_size + self.occ_size)

        # init weights
        trunc_normal_(self.obj_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(init_weights)
        if init_mode == "zero":
            self.head.weight.data.zero_()
            self.head.bias.data.zero_()

    def forward(self, x, x_obj, x_bg, eps=1e-6):
        B, T, L, C = x.shape
        _, No, Lo, _ = x_obj.shape
        P, S = self.pose_size, self.scale_size

        # embedding
        x = x + self.pos_embed
        x_obj = (x_obj + self.obj_embed).view(B, 1, No * Lo, C).expand(-1, T, -1, -1)
        if self.has_bg:
            x_bg = (x_bg + self.pos_embed).view(B, 1, L, C).expand(-1, T, -1, -1)
            x = torch.cat([x_bg, x_obj, x], dim=2)
        else:
            x = torch.cat([x_obj, x], dim=2)

        # main blocks
        x, shape = flatten(x, ndim=3)
        x = self.blocks(x)
        x = x[:, :L + No * Lo] if self.has_bg else x[:, :No * Lo]
        x_for_head = x[:, L:] if self.has_bg and self.fix_bg else x
        out = self.head(self.norm(x_for_head))
        pose, scale, occ = out[:, :, :P], out[:, :, P:P + S], out[:, :, P + S:]
        bg_pose = None
        if self.has_bg and not self.fix_bg:
            bg_pose = pose[:, :L]
            pose = pose[:, -No * Lo:]
            scale = scale[:, -No * Lo:]
            occ = occ[:, -No * Lo:]

        # pose prediction for objects
        if self.pts_mode == "head":
            rest = (pose ** 2).flatten(start_dim=1).mean(-1, keepdim=True)
            pose = (pose + self.bias).tanh()
            pose = pose.view(-1, No, Lo, 2) # B' No Lo 2
        elif self.pts_mode == "prior":
            pose = pose.tanh()
            if self.bound_rest:
                if self.soft_bound_rest:
                    min_mask = (pose < self.min_bound).float()
                    max_mask = (pose > self.max_bound).float()
                    rest = (min_mask * (pose - self.min_bound) ** 2 + max_mask * (pose - self.max_bound) ** 2) #* self.lambda_bound # better to have abs ? (not for delta pts!)
                else:
                    rest = pose ** 2
                    bound_mask = (pose < self.min_bound) | (pose > self.max_bound)
                    rest = rest * bound_mask.float()
            else:
                rest = pose ** 2
            rest = rest.flatten(start_dim=1).mean(-1)
            pose = pose.view(-1, No, Lo, 8) * self.mul + self.bias
            delta_pts = pose[:, :, :, :2].view(-1, No, Lo, 2) # B' No Lo 2
            if not self.use_delta:
                delta_pts = delta_pts * 0
            transform = pose[:, :, :, 2:].view(-1, No, Lo, 3, 2).mean(dim=2) # B' No 3 2
            if self.norm_scale:
                linear = transform[:, :, :2]
                norm = (linear[:, :, 0, 0] * linear[:, :, 1, 1] - linear[:, :, 1, 0] * linear[:, :, 0, 1]).abs() + eps
                linear = linear * self.tgt_scale / torch.sqrt(norm.view(-1, No, 1, 1) + eps)
                transform = torch.cat([linear, transform[:, :, 2:]], dim=2)
            if self.bound_scale:
                scale = scale.tanh()
                scale = (scale + 1) / 2
                scale = scale.view(-1, No, Lo, 1, 1).mean(dim=2)
                scale = self.min_scale + scale * (self.max_scale - self.min_scale)
                linear = transform[:, :, :2]
                norm = (linear[:, :, 0, 0] * linear[:, :, 1, 1] - linear[:, :, 1, 0] * linear[:, :, 0, 1]).abs() + eps
                linear = linear * scale / torch.sqrt(norm.view(-1, No, 1, 1) + eps)
                transform = torch.cat([linear, transform[:, :, 2:]], dim=2)
            last_obj = torch.cat([transform.reshape(B, T, No, 6)[:, self.ctx_len - 1], delta_pts.reshape(B, T, No, Lo * 2)[:, self.ctx_len - 1]], dim=2) if self.use_last else None
            pts = self.tgt_pts.expand(-1, No, -1, -1) # 1 No Lo 2
            pts = pts + delta_pts
            pts = torch.cat([pts, torch.ones_like(pts[:, :, :, :1])], dim=-1) # B' No Lo 3
            pose = pts @ transform
            pose = pose.reshape(-1, No, Lo, 2) # B' No Lo 2
        pose = unflatten(pose, shape)
        rest = unflatten(rest, shape)

        # occ prediction for obj
        Bc = occ.size(0)
        if self.occ_mode == "normalize":
            occ = occ.view(Bc, No, Lo).mean(dim=2)
            min_occ, max_occ = occ.min(dim=1, keepdim=True)[0], occ.max(dim=1, keepdim=True)[0]
            occ_score = (occ - min_occ) / (max_occ - min_occ + eps) * 4 * No
        elif self.occ_mode == "bias":
            occ_score = occ.view(Bc, No, Lo).mean(dim=2) + self.occ_bias
        elif self.occ_mode == "freeze":
            occ_score = torch.ones(Bc, No, device=occ.device)
        else:
            occ_score = occ.view(Bc, No, Lo).mean(dim=2)
        occ_score = unflatten(occ_score, shape)

        # pose prediction for bg
        bg_rest = None
        last_bg = None
        if self.has_bg:
            if not self.fix_bg:
                assert self.pts_mode == "prior"
                bg_pose = bg_pose.tanh()
                bg_rest = bg_pose ** 2
                bg_rest = bg_rest.flatten(start_dim=1).mean(-1)
                bg_pose = bg_pose.view(-1, 1, L, 8) + self.bg_bias
                delta_pts = bg_pose[:, :, :, :2].view(-1, 1, L, 2)  # B' 1 L 2
                transform = bg_pose[:, :, :, 2:].view(-1, 1, L, 3, 2).mean(dim=2)  # B' 1 3 2
                last_bg = torch.cat([transform.reshape(B, T, 1, 6)[:, self.ctx_len - 1], delta_pts.reshape(B, T, 1, L * 2)[:, self.ctx_len - 1]], dim=2) if self.use_last else None
                pts = self.bg_mul * self.tgt_pts_bg + delta_pts # B' 1 L 2
                pts = torch.cat([pts, torch.ones_like(pts[:, :, :, :1])], dim=-1)  # B' 1 L 3
                bg_pose = pts @ transform
                bg_pose = bg_pose.reshape(-1, 1, L, 2)  # B' 1 L 2
                bg_rest = unflatten(bg_rest, shape)
            else:
                bg_pose = self.tgt_pts_bg.expand(B * T, -1, -1, -1)
            bg_pose = unflatten(bg_pose, shape)
            if self.fix_bg1:
                bg_pose1 = self.tgt_pts_bg.unsqueeze(1).expand(B, -1, -1, -1, -1)
                bg_pose = torch.cat([bg_pose1, bg_pose[:, 1:]], dim=1)

        return pose, bg_pose, occ_score, rest, bg_rest, last_obj, last_bg

def gather_time(tensor, ts):
    B, Tc, Tp = ts.shape
    gather_shape = [B, -1] + [1 for _ in tensor.shape[2:]]
    expand_shape = [-1, -1] + list(tensor.shape[2:])
    view_shape = [B, Tc, Tp] + list(tensor.shape[2:])
    return tensor.gather(1, ts.reshape(*gather_shape).expand(*expand_shape)).reshape(view_shape)

class Warper(nn.Module):
    def __init__(self, opt, repeat_border=False):
        super().__init__()
        src_pts = get_grid(*opt.latent_shape).view(-1, 2)
        tgt_pts = get_grid(*opt.obj_shape).view(-1, 2)
        self.time_dropout = opt.time_dropout
        self.num_obj = opt.num_obj
        self.latent_obj_size = opt.obj_shape[0] * opt.obj_shape[1]
        self.latent_size = opt.latent_shape[0] * opt.latent_shape[1]
        self.tgt_shape = [int(opt.obj_shape[0] * opt.patch_size * opt.scale_factor), int(opt.obj_shape[1] * opt.patch_size * opt.scale_factor)]
        self.src_shape = [opt.dim, int(opt.dim * opt.aspect_ratio)]
        self.src_shape_hd = [opt.load_dim, int(opt.load_dim * opt.aspect_ratio)] if opt.load_dim > 0 else self.src_shape
        self.register_buffer('src_pts', src_pts)
        self.register_buffer('tgt_pts', tgt_pts)
        self.register_buffer('src_grid', get_grid(*self.src_shape))
        self.register_buffer('src_grid_hd', get_grid(*self.src_shape_hd))
        self.register_buffer('tgt_grid', get_grid(*self.tgt_shape))
        self.tps_obj = TPSWarp(*self.tgt_shape, tgt_pts)
        self.invert_obj = InverseWarp(*self.tgt_shape, *self.src_shape, num_perm=opt.num_perm_grid)
        self.normalize_alpha = opt.normalize_alpha
        self.use_lyt_filtering = opt.use_lyt_filtering
        self.use_lyt_opacity = opt.use_lyt_opacity
        self.weight_cls = opt.weight_cls
        self.min_cls = opt.min_cls
        self.include_self = opt.include_self
        self.fast = opt.load_dim == 0
        self.scale_hd = opt.load_dim / opt.dim if opt.load_dim > 0 else 1
        self.tps_bg = TPSWarp(*self.src_shape, src_pts)
        self.invert_bg = InverseWarp(*self.src_shape, *self.src_shape, num_perm=opt.num_perm_grid)
        self.no_filter = opt.no_filter
        self.allow_ghost = opt.allow_ghost


    def layer_from_input(self, input, grid):
        obj = self.obj_from_input(input, grid)
        bg = self.bg_from_input(input, grid)
        return obj, bg

    def obj_from_input(self, input, grid):
        tgt_grid_obj, _, _, _ = grid
        B, T = input.shape[:2]
        C = input.size(-3)
        Ho, Wo = self.tgt_shape
        H, W = self.src_shape
        No = self.num_obj

        input_for_obj = input.view(B * T, 1, C, H, W).expand(-1, No, -1, -1, -1) if input.ndim == 5 else input[:, :, 1:]
        input_for_obj = input_for_obj.reshape(B * T * No, C, H, W)
        tgt_grid_obj = tgt_grid_obj.reshape(B * T * No, Ho, Wo, 2)
        obj = F.grid_sample(input_for_obj, tgt_grid_obj).view(B, T, No, C, Ho, Wo)
        return obj

    def bg_from_input(self, input, grid):
        _, _, tgt_grid_bg, _ = grid
        B, T = input.shape[:2]
        C = input.size(-3)
        H, W = self.src_shape

        input_for_bg = input if input.ndim == 5 else input[:, :, :1]
        input_for_bg = input_for_bg.view(B * T, C, H, W)
        tgt_grid_bg = tgt_grid_bg.reshape(B * T, H, W, 2)
        bg = F.grid_sample(input_for_bg, tgt_grid_bg).view(B, T, C, H, W)
        return bg

    def layer_to_output(self, obj, bg, grid, delta_bg=1, delta_obj=1):
        output = self.obj_to_output(obj, grid, delta_obj)
        output = torch.cat([self.bg_to_output(bg, grid, delta_bg), output], dim=2)
        return output

    def obj_to_output(self, obj, grid, delta_obj=1):
        _, src_grid_obj, _, _ = grid
        B, T, No, _, _, _ = src_grid_obj.shape
        C = obj.size(-3) - 1
        Ho, Wo = self.tgt_shape
        H, W = self.src_shape

        obj = obj.view(B, 1, No, C + 1, Ho, Wo).expand(-1, T, -1, -1, -1, -1) if obj.ndim == 5 else obj
        obj = obj.reshape(B * T * No, C + 1, Ho, Wo)
        src_grid_obj = src_grid_obj.reshape(B * T * No, H, W, 2)
        return (F.grid_sample(obj + delta_obj, src_grid_obj) - delta_obj).view(B, T, No, C + 1, H, W)

    def bg_to_output(self, bg, grid, delta_bg=1, eps=1e-6):
        _, _, _, src_grid_bg = grid
        B, T, _, _, _ = src_grid_bg.shape
        C = bg.size(-3) - 1
        H, W = self.src_shape

        bg = bg.view(B, 1, C + 1, H, W).expand(-1, T, -1, -1, -1) if bg.ndim == 4 else bg
        bg = bg.reshape(B * T, C + 1, H, W)
        src_grid_bg = src_grid_bg.reshape(B * T, H, W, 2)
        return (F.grid_sample(bg + delta_bg, src_grid_bg) - delta_bg).view(B, T, 1, C + 1, H, W)

    def alpha_to_alpha(self, obj_alpha, bg_alpha, grid, occ):
        _, src_grid_obj, _, _ = grid
        B, T, No, _, _, _ = src_grid_obj.shape

        obj_alpha = obj_alpha.unsqueeze(1).expand(-1, T, -1, -1, -1, -1) # B T No 1 Ho Wo
        bg_alpha = bg_alpha.unsqueeze(1).expand(-1, T, -1, -1, -1) # B T 1 H W
        output_alpha = self.layer_to_output(obj_alpha, bg_alpha, grid) # B T No+1 1 H W
        output_alpha = (output_alpha + 1) / 2
        occ = (1 - output_alpha * occ.view(B, T, No + 1, No + 1, 1, 1)).prod(dim=2).unsqueeze(dim=3)  # B T No+1 1 H W
        output_alpha = occ * output_alpha
        obj_occ, bg_occ = self.layer_from_input(occ, grid)
        obj_alpha, bg_alpha = obj_occ * (obj_alpha + 1) - 1, bg_occ * (bg_alpha + 1) - 1
        return obj_alpha, bg_alpha, output_alpha

    def grid_to_bg_flow_from_ref_to_pred(self, grid, ctx_len, ref):
        _, _, tgt_grid_bg, src_grid_bg = grid
        bg_flow = tgt_grid_bg[:, [ref]] - tgt_grid_bg[:, ctx_len:] # B T H W 2
        bg_flow = bg_flow.permute(0, 1, 4, 2, 3) # B T 2 H W
        grid = [None, None, None, src_grid_bg[:, ctx_len:]]
        bg_flow = self.bg_to_output(bg_flow, grid, delta_bg=0).squeeze(2) # B T 2 H W
        bg_flow = scale(bg_flow, self.scale_hd)
        return bg_flow.permute(0, 1, 3, 4, 2) # B T H W 2

    def grid_to_obj_flow_from_ref_to_pred(self, grid, ctx_len, ref, obj_id):
        tgt_grid_obj, src_grid_obj, _, _ = grid
        obj_flow = tgt_grid_obj[:, [ref], [obj_id]] - tgt_grid_obj[:, ctx_len:, [obj_id]] # B T 1 Ho Wo 2
        obj_flow = obj_flow.permute(0, 1, 2, 5, 3, 4) # B T 1 2 Ho Wo
        grid = [None, src_grid_obj[:, ctx_len:, [obj_id]], None, None]
        obj_flow = self.obj_to_output(obj_flow, grid, delta_obj=0).squeeze(2) # B T 2 H W
        obj_flow = scale(obj_flow, self.scale_hd)
        return obj_flow.permute(0, 1, 3, 4, 2) # B T H W 2

    def grid_to_bg_flow_from_ctx_to_ref(self, grid, ctx_len, ref):
        _, _, tgt_grid_bg, src_grid_bg = grid
        bg_flow = tgt_grid_bg[:, :ctx_len] - tgt_grid_bg[:, [ref]] # B T H W 2
        bg_flow = bg_flow.permute(0, 1, 4, 2, 3) # B T 2 H W
        grid = [None, None, None, src_grid_bg[:, [ref]].repeat(1, ctx_len, 1, 1, 1)]
        bg_flow = self.bg_to_output(bg_flow, grid, delta_bg=0).squeeze(2) # B T 2 H W
        bg_flow = scale(bg_flow, self.scale_hd)
        return bg_flow.permute(0, 1, 3, 4, 2) # B T H W 2

    def grid_to_flow(self, input, grid, occ, obj_alpha, bg_alpha, cls, ctx_ts, pred_ts):
        tgt_grid_obj, src_grid_obj, tgt_grid_bg, src_grid_bg = grid
        B, _, No, _, _, _ = src_grid_obj.shape
        Tc, Tp, T = ctx_ts.size(1), pred_ts.size(0), input.size(1)
        H, W = self.src_shape
        Hd, Wd = self.src_shape_hd
        Ho, Wo = self.tgt_shape

        hd_input = input # B T C Hd Wd
        input = scale(hd_input, 1 / self.scale_hd) # B T C H W

        to_ctx = lambda tensor: gather_time(tensor, ctx_ts)
        to_pred = lambda tensor: tensor[:, pred_ts]

        # prepare rough obj and bg alpha
        obj_alpha = ((obj_alpha + 1) / 2).unsqueeze(1).expand(-1, T, -1, -1, -1, -1) # B T No 1 Ho Wo
        bg_alpha = ((bg_alpha + 1) / 2).unsqueeze(1).expand(-1, T, -1, -1, -1)  # B T 1 H W

        # project rough alpha
        alpha = self.layer_to_output(obj_alpha, bg_alpha, grid, delta_bg=0, delta_obj=0) # B T No+1 1 H W

        # refine alpha using lyt on input time-window
        if not self.no_filter:
            lyt = input[:, :, 3:]  # B T Nl H W
            hd_lyt = hd_input[:, :, 3:]  # B T Nl Hd Wd
            Nl = lyt.size(2)
            if cls is None or self.weight_cls:
                alpha_win = alpha[:, :, 1:] + 1e-6  # B T No 1 H W
                if self.weight_cls:
                    lyt_alpha = (cls + self.min_cls).view(B, 1, No, Nl, 1, 1) * lyt.unsqueeze(2).softmax(dim=-3)  # B T No Nl H W
                    lyt_alpha = lyt_alpha.sum(dim=-3, keepdim=True)  # B T No 1 H W
                    alpha_win = alpha_win * lyt_alpha  # B T No 1 H W
                sum_alpha_win = alpha_win.sum(dim=(1, 4, 5), keepdim=True)  # B 1 No 1 1 1
                lyt_win = lyt.unsqueeze(2) * alpha_win  # B T No Nl H W
                mean_lyt_win = lyt_win.sum(dim=(1, 4, 5), keepdim=True) / sum_alpha_win  # B 1 No Nl 1 1
                lyt_alpha = (mean_lyt_win.softmax(dim=-3) - hd_lyt.unsqueeze(2).softmax(dim=-3)).abs()  # B T No Nl Hd Wd
                lyt_alpha = 1 - lyt_alpha.sum(dim=-3, keepdim=True) / 2  # B T No 1 Hd Wd
            else:
                lyt_alpha = (cls.view(B, 1, No, Nl, 1, 1) - hd_lyt.unsqueeze(2).softmax(dim=-3)).abs()  # B T No Nl Hd Wd
                lyt_alpha = 1 - lyt_alpha.sum(dim=-3, keepdim=True) / 2  # B T No 1 Hd Wd

        alpha = scale(alpha, self.scale_hd)

        if not self.no_filter:
            obj_alpha = alpha[:, :, 1:] * lyt_alpha  # B T No 1 Hd Wd
            alpha = torch.cat([alpha[:, :, :1], obj_alpha], dim=2)  # B T No+1 1 Hd Wd

        # compute occlusion
        occ = occ.reshape(B, T, No + 1, No + 1, 1, 1)
        alpha_occ = (1 - alpha * occ).prod(dim=2).unsqueeze(dim=3) # B T No+1 1 H W
        alpha = alpha_occ * alpha # B T No+1 1 H W
        alpha_unflt = alpha

        # adapt grid
        src_grid_obj = to_pred(src_grid_obj).unsqueeze(1).expand(-1, Tc, -1, -1, -1, -1, -1)
        src_grid_obj = src_grid_obj.reshape(B * Tc, Tp, No, H, W, 2)
        src_grid_bg = to_pred(src_grid_bg).unsqueeze(1).expand(-1, Tc, -1, -1, -1, -1)
        src_grid_bg = src_grid_bg.reshape(B * Tc, Tp, H, W, 2)
        grid = [None, src_grid_obj, None, src_grid_bg]

        # compute flow in obj and bg referential between ctx and pred timesteps
        obj_flow = to_ctx(tgt_grid_obj) - to_pred(tgt_grid_obj).unsqueeze(1)  # B Tc Tp No Ho Wo 2
        obj_flow = obj_flow.permute(0, 1, 2, 3, 6, 4, 5).reshape(B * Tc, Tp, No, 2, Ho, Wo)
        bg_flow = to_ctx(tgt_grid_bg) - to_pred(tgt_grid_bg).unsqueeze(1)  # B Tc Tp H W 2
        bg_flow = bg_flow.permute(0, 1, 2, 5, 3, 4).reshape(B * Tc, Tp, 2, H, W)

        # warp flow from bg and obj
        flow = self.layer_to_output(obj_flow, bg_flow, grid, delta_bg=0, delta_obj=0)  # B*Tc Tp No+1 2 H W
        flow = flow.view(B, Tc, Tp, No + 1, 2, H, W)
        flow = scale(flow, self.scale_hd) # B Tc Tp No+1 2 Hd Wd
        grid = flow.permute(0, 1, 2, 3, 5, 6, 4)  # B Tc Tp No+1 Hd Wd 2
        grid = self.src_grid_hd + grid.reshape(B * Tc * Tp * (No + 1), Hd, Wd, 2)

        # warp alpha with flow
        alpha_ctx = to_ctx(alpha) # B Tc Tp No+1 1 Hd Wd
        alpha_ctx = alpha_ctx.reshape(B * Tc * Tp * (No + 1), 1, Hd, Wd)
        alpha_ctx = F.grid_sample(alpha_ctx, grid)
        alpha_ctx = alpha_ctx.reshape(B, Tc, Tp, No + 1, 1, Hd, Wd)
        disocc = alpha_ctx.max(dim=3)[0] # B Tc Tp 1 Hd Wd

        # compute tgt occ
        occ = to_pred(occ).view(B, 1, Tp, No + 1, No + 1, 1, 1)
        if self.fast:
            # memory intensive but fast
            alpha_ctx_occ = (1 - alpha_ctx * occ).prod(dim=3).unsqueeze(dim=4)  # B Tc Tp No+1 1 Hd Wd
        else:
            # memory friendly but slow
            alpha_ctx_occ = torch.zeros(B, Tc, Tp, No + 1, 1, Hd, Wd, device=alpha_ctx.device)
            for i in range(No + 1):
                alpha_ctx_occ[:, :, :, i] = (1 - alpha_ctx * occ[:, :, :, :, [i]]).prod(dim=3)
        alpha_ctx = alpha_ctx_occ * alpha_ctx  # B Tc Tp No+1 1 Hd Wd

        # reduce flow
        flow = (alpha_ctx * flow).sum(dim=3) # B Tc Tp 2 Hd Wd

        # prepare out
        alpha_unflt = alpha_unflt.squeeze(-3) * 2 - 1
        alpha = alpha.squeeze(-3) * 2 - 1
        alpha_ctx = alpha_ctx.squeeze(-3) * 2 - 1

        if self.fast:
            return flow, alpha_unflt, alpha, alpha_ctx, disocc
        else:
            return flow, None, alpha, alpha_ctx, disocc

    def grid_to_flow_ctx(self, input, grid, occ, obj_alpha, bg_alpha, cls, ctx_ts, pred_ts):
        tgt_grid_obj, src_grid_obj, tgt_grid_bg, src_grid_bg = grid
        B, _, No, _, _, _ = src_grid_obj.shape
        Tc, Tp, T = ctx_ts.size(1), pred_ts.size(0), input.size(1)
        H, W = self.src_shape
        Hd, Wd = self.src_shape_hd
        Ho, Wo = self.tgt_shape

        hd_input = input  # B T C Hd Wd
        input = scale(hd_input, 1 / self.scale_hd)  # B T C H W

        to_multi_ctx = lambda tensor: gather_time(tensor, ctx_ts)
        to_pred = lambda tensor: tensor[:, pred_ts]
        to_ctx = lambda tensor: tensor[:, :Tc]

        # prepare rough obj and bg alpha
        obj_alpha = ((obj_alpha + 1) / 2).unsqueeze(1).expand(-1, T, -1, -1, -1, -1)  # B T No 1 Ho Wo
        bg_alpha = ((bg_alpha + 1) / 2).unsqueeze(1).expand(-1, T, -1, -1, -1)  # B T 1 H W

        # project rough alpha
        alpha = self.layer_to_output(obj_alpha, bg_alpha, grid, delta_bg=0, delta_obj=0)  # B T No+1 1 H W
        alpha = to_ctx(alpha)  # B Tc No+1 1 H W

        # refine alpha using lyt on input time-window
        lyt = to_ctx(input)[:, :, 3:]  # B Tc Nl H W
        hd_lyt = to_ctx(hd_input)[:, :, 3:]  # B Tc Nl Hd Wd
        Nl = lyt.size(2)
        if cls is None or self.weight_cls:
            alpha_win = alpha[:, :, 1:] + 1e-6  # B Tc No 1 H W
            if self.weight_cls:
                lyt_alpha = (cls + self.min_cls).view(B, 1, No, Nl, 1, 1) * lyt.unsqueeze(2).softmax(dim=-3)  # B Tc No Nl H W
                lyt_alpha = lyt_alpha.sum(dim=-3, keepdim=True)  # B Tc No 1 H W
                alpha_win = alpha_win * lyt_alpha  # B Tc No 1 H W
            sum_alpha_win = alpha_win.sum(dim=(1, 4, 5), keepdim=True)  # B 1 No 1 1 1
            lyt_win = lyt.unsqueeze(2) * alpha_win  # B Tc No Nl H W
            mean_lyt_win = lyt_win.sum(dim=(1, 4, 5), keepdim=True) / sum_alpha_win  # B 1 No Nl 1 1
            ## memory intensive
            lyt_alpha = (mean_lyt_win.softmax(dim=-3) - hd_lyt.unsqueeze(2).softmax(dim=-3)).abs()  # B Tc No Nl Hd Wd
            lyt_alpha = 1 - lyt_alpha.sum(dim=-3, keepdim=True) / 2  # B Tc No 1 Hd Wd
            ## memory friendly
            # lyt_alpha = torch.zeros(B, T, No, 1, H, W, device=lyt.device)
            # for i in range(No):
            #     lyt_alpha[:, :, i] = (mean_lyt_win[:, :, i].softmax(dim=-3) - hd_lyt.softmax(dim=-3)).abs().sum(dim=-3, keepdim=True)
            # lyt_alpha = 1 - lyt_alpha / 2
            ## end
        else:
            lyt_alpha = (cls.view(B, 1, No, Nl, 1, 1) - hd_lyt.unsqueeze(2).softmax(dim=-3)).abs()  # B Tc No Nl Hd Wd
            lyt_alpha = 1 - lyt_alpha.sum(dim=-3, keepdim=True) / 2  # B Tc No 1 Hd Wd
        alpha = scale(alpha, self.scale_hd)
        obj_alpha = alpha[:, :, 1:] * lyt_alpha  # B Tc No 1 Hd Wd
        # obj_alpha = scale(alpha[:, :, 1:], self.scale_hd) * lyt_alpha  # B T No 1 Hd Wd
        # bg_alpha = 1 - obj_alpha.sum(dim=2, keepdim=True)
        # alpha = torch.cat([bg_alpha, obj_alpha], dim=2)  # B T No+1 1 Hd Wd
        alpha = torch.cat([alpha[:, :, :1], obj_alpha], dim=2)  # B Tc No+1 1 Hd Wd

        # compute occlusion
        occ = occ.reshape(B, T, No + 1, No + 1, 1, 1)
        alpha_occ = (1 - alpha * to_ctx(occ)).prod(dim=2).unsqueeze(dim=3)  # B Tc No+1 1 H W
        alpha = alpha_occ * alpha  # B Tc No+1 1 H W
        alpha_unflt = alpha
        # if cls is not None:
        #     alpha_unflt = alpha_unflt + 0 * cls.mean() # trick to enforce all parameters being used in the loss

        # adapt grid
        src_grid_obj = to_pred(src_grid_obj).unsqueeze(1).expand(-1, Tc, -1, -1, -1, -1, -1)
        src_grid_obj = src_grid_obj.reshape(B * Tc, Tp, No, H, W, 2)
        src_grid_bg = to_pred(src_grid_bg).unsqueeze(1).expand(-1, Tc, -1, -1, -1, -1)
        src_grid_bg = src_grid_bg.reshape(B * Tc, Tp, H, W, 2)
        grid = [None, src_grid_obj, None, src_grid_bg]

        # compute flow in obj and bg referential between ctx and pred timesteps
        # print("shape", to_multi_ctx(tgt_grid_obj).shape, to_pred(tgt_grid_obj).unsqueeze(1).shape, ctx_ts.shape, pred_ts.shape)
        obj_flow = to_multi_ctx(tgt_grid_obj) - to_pred(tgt_grid_obj).unsqueeze(1)  # B Tc Tp No Ho Wo 2
        obj_flow = obj_flow.permute(0, 1, 2, 3, 6, 4, 5).reshape(B * Tc, Tp, No, 2, Ho, Wo)
        bg_flow = to_multi_ctx(tgt_grid_bg) - to_pred(tgt_grid_bg).unsqueeze(1)  # B Tc Tp H W 2
        bg_flow = bg_flow.permute(0, 1, 2, 5, 3, 4).reshape(B * Tc, Tp, 2, H, W)

        # warp flow from bg and obj
        if self.allow_ghost:
            is_obj = 1
        else:
            is_obj = self.obj_to_output(torch.ones_like(obj_flow[:, :, :, :1]), grid, delta_obj=0)
            is_obj = (scale(is_obj, self.scale_hd) > 0.9).float()
            is_obj = is_obj.view(B, Tc, Tp, No, 1, Hd, Wd)
            is_obj = torch.cat([torch.ones_like(is_obj[:, :, :, :1]), is_obj], dim=3)  # B Tc Tp No+1 1 Hd Wd
        flow = self.layer_to_output(obj_flow, bg_flow, grid, delta_bg=0, delta_obj=0)  # B*Tc Tp No+1 2 H W
        flow = flow.view(B, Tc, Tp, No + 1, 2, H, W)
        flow = scale(flow, self.scale_hd)  # B Tc Tp No+1 2 Hd Wd
        grid = flow.permute(0, 1, 2, 3, 5, 6, 4)  # B Tc Tp No+1 Hd Wd 2
        grid = self.src_grid_hd + grid.reshape(B * Tc * Tp * (No + 1), Hd, Wd, 2)

        # warp alpha with flow
        alpha_ctx = to_multi_ctx(alpha)  # B Tc Tp No+1 1 Hd Wd
        alpha_ctx = alpha_ctx.reshape(B * Tc * Tp * (No + 1), 1, Hd, Wd)
        alpha_ctx = F.grid_sample(alpha_ctx, grid)
        alpha_ctx = alpha_ctx.reshape(B, Tc, Tp, No + 1, 1, Hd, Wd) * is_obj
        disocc = alpha_ctx.max(dim=3)[0]  # B Tc Tp 1 Hd Wd

        # compute tgt occ
        occ = to_pred(occ).view(B, 1, Tp, No + 1, No + 1, 1, 1)
        if self.fast:
            # memory intensive but fast
            alpha_ctx_occ = (1 - alpha_ctx * occ).prod(dim=3).unsqueeze(dim=4)  # B Tc Tp No+1 1 Hd Wd
        else:
            # memory friendly but slow
            alpha_ctx_occ = torch.zeros(B, Tc, Tp, No + 1, 1, Hd, Wd, device=alpha_ctx.device)
            for i in range(No + 1):
                alpha_ctx_occ[:, :, :, i] = (1 - alpha_ctx * occ[:, :, :, :, i].unsqueeze(4)).prod(dim=3)
        alpha_ctx = alpha_ctx_occ * alpha_ctx  # B Tc Tp No+1 1 Hd Wd

        # reduce flow
        flow = (alpha_ctx * flow).sum(dim=3)  # B Tc Tp 2 Hd Wd

        # prepare out
        alpha_unflt = alpha_unflt.squeeze(-3) * 2 - 1
        alpha = alpha.squeeze(-3) * 2 - 1
        alpha_ctx = alpha_ctx.squeeze(-3) * 2 - 1

        if self.fast:
            return flow, alpha_unflt, alpha, alpha_ctx, disocc
        else:
            return flow, None, alpha, alpha_ctx, disocc

    def input_to_output(self, input, alpha, flow, ctx_ts, eps=1e-6):
        B, Tc, Tp, _, _, _ = flow.shape
        Hd, Wd = self.src_shape_hd
        C = input.size(-3)

        to_ctx = lambda tensor: gather_time(tensor, ctx_ts)

        output = F.grid_sample(to_ctx(input).reshape(B * Tc * Tp, C, Hd, Wd),
                               self.src_grid_hd + flow.permute(0, 1, 2, 4, 5, 3).reshape(B * Tc * Tp, Hd, Wd, 2))
        output = output.reshape(B, Tc, Tp, C, Hd, Wd)

        score = ((alpha + 1) / 2).sum(dim=3, keepdim=True)  # B Tc Tp 1 Hd Wd
        if self.include_self and Tp == input.size(1):
            score = torch.cat([score, torch.ones_like(score[:, :1])], dim=1)
            alpha = torch.cat([alpha, torch.ones_like(alpha[:, :1])], dim=1)
            output = torch.cat([output, input.unsqueeze(1)], dim=1)
        raw_output = torch.cat([output, alpha], dim=3)  # B Tc Tp C+No+1 Hd Wd

        output = torch.cat([output, score * 2 - 1], dim=3) # B Tc Tp C+1 Hd Wd

        score = F.normalize(score + eps, p=1, dim=1)
        output = (output * score).sum(dim=1)  # B Tp C+1 Hd Wd

        return output, raw_output

    def forward(self, obj_pose, bg_pose, invert=True):
        B, T, No, _, _ = obj_pose.shape
        Lo, L = self.latent_obj_size, self.latent_size

        src_pts_obj = obj_pose.view(B * T * No, Lo, 2)  # B*T*No Lo 2
        tgt_grid_obj = self.tps_obj(src_pts_obj)
        src_grid_obj = self.invert_obj(tgt_grid_obj) if invert else None
        tgt_grid_obj = tgt_grid_obj.view(B, T, No, *tgt_grid_obj.shape[1:])
        src_grid_obj = src_grid_obj.view(B, T, No, *src_grid_obj.shape[1:]) if invert else None

        src_pts_bg = bg_pose.view(B * T, L, 2)  # B*T L 2
        tgt_grid_bg = self.tps_bg(src_pts_bg)
        src_grid_bg = self.invert_bg(tgt_grid_bg, erode=False) if invert else None
        tgt_grid_bg = tgt_grid_bg.view(B, T, *tgt_grid_bg.shape[1:])
        src_grid_bg = src_grid_bg.view(B, T, *src_grid_bg.shape[1:]) if invert else None
        return tgt_grid_obj, src_grid_obj, tgt_grid_bg, src_grid_bg


class LayerEstimator(nn.Module):
    def __init__(self, opt, depth, num_timesteps):
        super().__init__()
        # params
        self.num_obj = opt.num_obj
        self.latent_size = opt.latent_shape[0] * opt.latent_shape[1]
        self.latent_obj_size = opt.obj_shape[0] * opt.obj_shape[1]
        self.decompose_embed = opt.decompose_embed_oe
        self.has_bg = opt.has_bg
        self.pred_cls = opt.pred_cls

        # embedding
        if opt.decompose_embed_oe:
            self.obj_spatial_embed = nn.Parameter(torch.randn(1, 1, self.latent_obj_size, opt.embed_dim))
            self.obj_num_embed = nn.Parameter(torch.randn(1, opt.num_obj, 1, opt.embed_dim))
        else:
            self.obj_embed = nn.Parameter(torch.randn(1, opt.num_obj, self.latent_obj_size, opt.embed_dim))
        self.time_embed = nn.Parameter(torch.randn(1, num_timesteps, 1, opt.embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, 1, self.latent_size, opt.embed_dim))

        # main blocks
        self.norm = CustomNorm(opt.norm_layer, opt.embed_dim)
        args = {"block_type": "obj",
                "depth": depth,
                "dim": opt.embed_dim,
                "num_heads": opt.num_heads,
                "norm_layer": opt.norm_layer,
                "spectral_norm_layer": None,
                "noise": False,
                "dropout": opt.dropout}
        self.blocks = MultiBlocks(**args)

        # head
        if opt.pred_cls:
            self.cls_norm = CustomNorm(opt.norm_layer, opt.embed_dim)
            self.cls_head = nn.Linear(opt.embed_dim, opt.num_lyt)

        # init weights
        if opt.decompose_embed_oe:
            trunc_normal_(self.obj_spatial_embed, std=.02)
            trunc_normal_(self.obj_num_embed, std=.02)
        else:
            trunc_normal_(self.obj_embed, std=.02)
        trunc_normal_(self.time_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(init_weights)

    def forward(self, x):
        B, T, L, C = x.shape
        No, Lo = self.num_obj, self.latent_obj_size
        x_bg = None

        # embedding
        x = (x + self.pos_embed + self.time_embed[:, :T]).reshape(B, T, L, C)
        obj_embed = (self.obj_spatial_embed + self.obj_num_embed) if self.decompose_embed else self.obj_embed
        x_obj = obj_embed.expand(B, -1, -1, -1).reshape(B, No * Lo, C)
        if self.has_bg:
            x_bg = self.pos_embed.expand(B, -1, -1, -1).reshape(B, L, C)
            x_obj = torch.cat([x_bg, x_obj], dim=1)

        # main blocks
        x, shape = flatten(x, ndim=3)
        x = self.norm(x)
        x_obj = self.blocks(x_obj, x_ctx=x)
        if self.has_bg:
            x_bg = x_obj[:, :L]
        x_obj = x_obj[:, -No * Lo:]

        # head
        cls = None
        if self.pred_cls:
            x_cls = x_obj.reshape(B, No, Lo, -1).mean(2) # B No C
            cls = self.cls_head(self.cls_norm(x_cls)) # B No Nl
            cls = cls.softmax(dim=-1) # B No Nl

        x_bg = x_bg.reshape(B, L, C)
        x_obj = x_obj.reshape(B, No, Lo, C)
        return x_obj, x_bg, cls