import torch
import torch.nn as nn

from models.modules import trunc_normal_, init_weights, CustomNorm, MultiBlocks, Block
from tools.utils import flatten_vid, unflatten_vid, get_grid, flatten, unflatten, to_ctx, from_ctx


class FLP(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # nets
        self.compress = LatentCompressor(opt, opt.pg_com_depth)
        self.encode = PoseEncoder(opt, opt.pg_enc_depth, opt.pg_num_timesteps, opt.pg_embed_noise)
        self.decode = PoseDecoder(opt, opt.pg_dec_depth, opt.pg_inject_noise, opt.pg_modulate_noise)

    def get_last_layer(self):
        return self.decode.head.weight

    def forward(self, obj_pose, bg_pose, occ_score, x_obj, x_bg, last_obj, last_bg, ctx_mask=None, mode="training"):
        if mode == "training":
            assert ctx_mask is not None
            z_obj = self.compress(x_obj)
            z_bg = self.compress(x_bg.unsqueeze(1))
            z = torch.cat([z_bg, z_obj], dim=1)
            x = self.encode(obj_pose, bg_pose, occ_score, z, ctx_mask)
            obj_pose, bg_pose, occ_score = self.decode(obj_pose, bg_pose, occ_score, x, ctx_mask, last_obj, last_bg)
            return obj_pose, bg_pose, occ_score
        else:
            raise ValueError


class PoseEncoder(nn.Module):
    def __init__(self, opt, depth, num_timesteps, embed_noise):
        super().__init__()
        # params
        self.latent_size = opt.latent_shape[0] * opt.latent_shape[1]
        self.latent_obj_size = opt.obj_shape[0] * opt.obj_shape[1]
        self.embed_noise = embed_noise
        self.cat_z = opt.cat_z

        # embedding
        self.lay_embed = nn.Parameter(torch.randn(1, 1, opt.num_obj + 1, opt.embed_dim))
        self.time_embed = nn.Parameter(torch.randn(1, num_timesteps + 1, 1, opt.embed_dim))
        self.to_obj_emb = nn.Linear(self.latent_obj_size * 2 + 1, opt.embed_dim)
        self.to_bg_emb = nn.Linear(self.latent_size * 2, opt.embed_dim)

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
        self.norm = CustomNorm(opt.norm_layer, opt.embed_dim)

        # init weights
        trunc_normal_(self.lay_embed, std=.02)
        trunc_normal_(self.time_embed, std=.02)
        self.apply(init_weights)

    def forward(self, obj_pose, bg_pose, occ_score, z, ctx_mask):
        C, L = z.size(-1), self.latent_size
        B, T, No, Lo, _ = obj_pose.shape

        # embedding
        obj_pose = to_ctx(obj_pose, ctx_mask) # B' No Lo 2
        bg_pose = to_ctx(bg_pose, ctx_mask) # B' L 2
        occ_score = to_ctx(occ_score, ctx_mask) # B' No
        x_obj = self.to_obj_emb(torch.cat([obj_pose.view(-1, No, Lo * 2), occ_score.view(-1, No, 1)], dim=2)) # B' No C
        x_obj = from_ctx(x_obj, ctx_mask) # B T No C
        x_bg = self.to_bg_emb(bg_pose.view(-1, 1, L * 2)) # B' 1 C
        x_bg = from_ctx(x_bg, ctx_mask)  # B T 1 C
        x = torch.cat([x_bg, x_obj], dim=2) # B T No+1 C
        z = z.view(B, 1, No + 1, C)
        if self.cat_z:
            x = torch.cat([z, x], dim=1) + self.time_embed[:, :(T + 1)] + self.lay_embed # B T+1 No+1 C
        else:
            x = x + self.time_embed[:, :T] + self.lay_embed + z.mean() * 0 # to avoid grad errors

        # pad mask
        if self.cat_z:
            ctx_mask = torch.cat([torch.ones_like(ctx_mask[:, :1]), ctx_mask], dim=1)
        pred_mask = ~ctx_mask

        # main blocks
        x = to_ctx(x, ctx_mask) # B'' No+1 C
        x = self.blocks(x, ctx_mask=ctx_mask)
        x = self.norm(x)
        x = from_ctx(x, ctx_mask) # B T+1 No+1 C
        if self.cat_z:
            x_init = (self.time_embed[:, :(T + 1)] + self.lay_embed).expand(B, -1, -1, -1)
        else:
            x_init = (self.time_embed[:, :T] + self.lay_embed).expand(B, -1, -1, -1)
        if self.embed_noise:
            x_init = x_init + torch.randn(B, 1, 1, C, device=x.device)
        x[pred_mask] = to_ctx(x_init, pred_mask)
        return x


class PoseHead(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # params
        self.latent_size = opt.latent_shape[0] * opt.latent_shape[1]
        self.latent_obj_size = opt.obj_shape[0] * opt.obj_shape[1]
        self.embed_dim = opt.embed_dim
        self.num_obj = opt.num_obj
        self.pred_len = opt.vid_len - opt.ctx_len

        # pose prediction
        if opt.unconstrained_pose_decoder:
            mul_delta_obj, init_scale_obj, mul_scale_obj = 1, 1, 1
        else:
            mul_delta_obj, init_scale_obj, mul_scale_obj = opt.mul_delta_obj, opt.init_scale_obj, opt.mul_scale_obj
        self.mul_delta_obj = mul_delta_obj
        self.register_buffer("tgt_pts_obj", get_grid(*opt.obj_shape).view(1, 1, self.latent_obj_size, 2))
        self.register_buffer("bias_obj", torch.tensor([[[init_scale_obj, 0., 0., opt.aspect_ratio * init_scale_obj, 0., 0.]]]))
        self.register_buffer("mul_obj", torch.tensor([[[mul_scale_obj, mul_scale_obj, mul_scale_obj, mul_scale_obj, 1., 1.]]]))
        self.register_buffer("tgt_pts_bg", get_grid(*opt.latent_shape).view(1, 1, self.latent_size, 2))
        self.register_buffer("bias_bg", torch.tensor([[[1., 0., 0., 1., 0., 0.]]]))
        obj_pose_size = 6 + 2 * self.latent_obj_size
        bg_pose_size = 6 + 2 * self.latent_size
        occ_size = 1
        self.obj_head = nn.Linear(16, obj_pose_size + occ_size)
        self.bg_head = nn.Linear(16, bg_pose_size)

        # init weights
        self.apply(init_weights)
        if opt.zero_init_dec:
            self.obj_head.weight.data.zero_()
            self.obj_head.bias.data.zero_()
            self.bg_head.weight.data.zero_()
            self.bg_head.bias.data.zero_()

    def forward(self, x):
        L, Lo, No, C = self.latent_size, self.latent_obj_size, self.num_obj, self.embed_dim
        Tp = self.pred_len

        # pose prediction
        x = x.view(-1, No + 1, 16)
        x_obj = x[:, 1:]  # B' No C
        x_bg = x[:, :1]  # B' 1 C
        x_obj = self.obj_head(x_obj).view(-1, No, 6 + 2 * Lo + 1)
        x_bg = self.bg_head(x_bg).view(-1, 1, 6 + 2 * L)
        pred_obj_pose, pred_occ_score, pred_bg_pose = x_obj[:, :, :-1].tanh(), x_obj[:, :, -1], x_bg.tanh()

        # apply obj transform
        transform = (self.mul_obj * pred_obj_pose[:, :, :6] + self.bias_obj).view(-1, No, 3, 2)  # B' No 3 2
        delta_pts = (self.mul_delta_obj * pred_obj_pose[:, :, 6:]).view(-1, No, Lo, 2)  # B' No Lo 2
        pts = self.tgt_pts_obj.expand(-1, No, -1, -1)
        pts = pts + delta_pts  # B' No Lo 2
        pts = torch.cat([pts, torch.ones_like(pts[:, :, :, :1])], dim=-1)  # B' No Lo 3
        pred_obj_pose = pts @ transform

        # apply bg transform
        transform = (pred_bg_pose[:, :, :6] + self.bias_bg).view(-1, 1, 3, 2)  # B' 1 3 2
        delta_pts = pred_bg_pose[:, :, 6:].view(-1, 1, L, 2)  # B' 1 L 2
        pts = self.tgt_pts_bg
        pts = pts + delta_pts  # B' 1 L 2
        pts = torch.cat([pts, torch.ones_like(pts[:, :, :, :1])], dim=-1)  # B' 1 L 3
        pred_bg_pose = pts @ transform # B' 1 L 2

        # fill input with pred output
        pred_obj_pose = pred_obj_pose.view(-1, Tp, No, Lo, 2)
        pred_bg_pose = pred_bg_pose.view(-1, Tp, 1, L, 2)
        pred_occ_score = pred_occ_score.view(-1, Tp, No)

        return pred_obj_pose, pred_bg_pose, pred_occ_score


class PoseDecoder(nn.Module):
    def __init__(self, opt, depth, inject_noise, modulate_noise):
        super().__init__()
        # params
        self.latent_size = opt.latent_shape[0] * opt.latent_shape[1]
        self.latent_obj_size = opt.obj_shape[0] * opt.obj_shape[1]
        self.modulate_noise = modulate_noise
        self.embed_dim = opt.embed_dim
        self.num_obj = opt.num_obj
        self.use_last = opt.use_last_pose_decoder
        self.bg_mul = opt.bg_mul_pose_decoder
        self.cat_z = opt.cat_z

        # main blocks
        args = {"block_type": "full_with_cond_norm" if modulate_noise else "full",
                "depth": depth,
                "dim": opt.embed_dim,
                "num_heads": opt.num_heads,
                "norm_layer": "ln_not_affine" if modulate_noise else opt.norm_layer,
                "spectral_norm_layer": None,
                "noise": inject_noise}
        self.self_blocks = nn.ModuleList([Block(**args) for _ in range(args["depth"])])
        args["block_type"] = "cross"
        args["noise"] = False
        args["norm_layer"] = opt.norm_layer
        self.cross_blocks = nn.ModuleList([Block(**args) for _ in range(args["depth"])])

        # pose prediction
        if opt.unconstrained_pose_decoder:
            mul_delta_obj, init_scale_obj, mul_scale_obj = 1, 1, 1
        else:
            mul_delta_obj, init_scale_obj, mul_scale_obj = opt.mul_delta_obj, opt.init_scale_obj, opt.mul_scale_obj
        self.mul_delta_obj = mul_delta_obj
        if opt.use_last_pose_decoder:
            self.bias_obj = 0
            self.bias_bg = 0
        else:
            self.register_buffer("bias_obj", torch.tensor([[[init_scale_obj, 0., 0., opt.aspect_ratio * init_scale_obj, 0., 0.]]]))
            self.register_buffer("bias_bg", torch.tensor([[[1., 0., 0., 1., 0., 0.]]]))
        self.register_buffer("tgt_pts_obj", get_grid(*opt.obj_shape).view(1, 1, self.latent_obj_size, 2))
        self.register_buffer("mul_obj", torch.tensor([[[mul_scale_obj, mul_scale_obj, mul_scale_obj, mul_scale_obj, 1., 1.]]]))
        self.register_buffer("tgt_pts_bg", get_grid(*opt.latent_shape).view(1, 1, self.latent_size, 2))

        obj_pose_size = 6 + 2 * self.latent_obj_size
        bg_pose_size = 6 + 2 * self.latent_size
        occ_size = 1
        self.norm = CustomNorm(opt.norm_layer, opt.embed_dim)
        self.obj_head = nn.Linear(opt.embed_dim, obj_pose_size + occ_size)
        self.bg_head = nn.Linear(opt.embed_dim, bg_pose_size)

        # init weights
        self.apply(init_weights)
        if opt.zero_init_dec:
            self.obj_head.weight.data.zero_()
            self.obj_head.bias.data.zero_()
            self.bg_head.weight.data.zero_()
            self.bg_head.bias.data.zero_()

    def forward(self, obj_pose, bg_pose, occ_score, x, ctx_mask, last_obj, last_bg, eps=1e-6):
        L, Lo, No, C = self.latent_size, self.latent_obj_size, self.num_obj, self.embed_dim

        # pad mask
        if self.cat_z:
            ctx_mask = torch.cat([torch.ones_like(ctx_mask[:, :1]), ctx_mask], dim=1)
        pred_mask = ~ctx_mask

        # main blocks
        x_ctx = to_ctx(x, ctx_mask)
        x_pred = to_ctx(x, pred_mask)
        z_cond = torch.randn([x_pred.size(0), 1, C], device=x.device) if self.modulate_noise else None
        for self_block, cross_block in zip(self.self_blocks, self.cross_blocks):
            x_pred = self_block(x_pred, ctx_mask=pred_mask, z_cond=z_cond)
            x_pred = cross_block(x_pred, x_ctx=x_ctx, ctx_mask=ctx_mask)

        # pose prediction
        x_pred = self.norm(x_pred)
        x_obj = x_pred[:, 1:] # B' No C
        x_bg = x_pred[:, :1] # B' 1 C
        x_obj = self.obj_head(x_obj).view(-1, No, 6 + 2 * Lo + 1)
        x_bg = self.bg_head(x_bg).view(-1, 1, 6 + 2 * L)
        pred_obj_pose, pred_occ_score, pred_bg_pose = x_obj[:, :, :-1].tanh(), x_obj[:, :, -1], x_bg.tanh()
        if self.use_last:
            pred_obj_pose = pred_obj_pose + to_ctx(last_obj.unsqueeze(1).expand(-1, x.size(1), -1, -1), pred_mask)
            pred_bg_pose = pred_bg_pose + to_ctx(last_bg.unsqueeze(1).expand(-1, x.size(1), -1, -1), pred_mask)

        # apply obj transform
        transform = (self.mul_obj * pred_obj_pose[:, :, :6] + self.bias_obj).view(-1, No, 3, 2)  # B' No 3 2
        delta_pts = (self.mul_delta_obj * pred_obj_pose[:, :, 6:]).view(-1, No, Lo, 2)  # B' No Lo 2
        pts = self.tgt_pts_obj.expand(-1, No, -1, -1)
        pts = pts + delta_pts  # B' No Lo 2
        pts = torch.cat([pts, torch.ones_like(pts[:, :, :, :1])], dim=-1)  # B' No Lo 3
        pred_obj_pose = pts @ transform

        # apply bg transform
        transform = (pred_bg_pose[:, :, :6] + self.bias_bg).view(-1, 1, 3, 2)  # B' 1 3 2
        delta_pts = pred_bg_pose[:, :, 6:].view(-1, 1, L, 2)  # B' 1 L 2
        pts = self.bg_mul * self.tgt_pts_bg
        pts = pts + delta_pts  # B' 1 L 2
        pts = torch.cat([pts, torch.ones_like(pts[:, :, :, :1])], dim=-1)  # B' 1 L 3
        pred_bg_pose = pts @ transform # B' 1 L 2

        # fill input with pred output
        obj_pose = obj_pose.clone()
        bg_pose = bg_pose.clone()
        occ_score = occ_score.clone()
        if self.cat_z:
            pred_mask = pred_mask[:, 1:]
        obj_pose[pred_mask] = pred_obj_pose
        bg_pose[pred_mask] = pred_bg_pose
        occ_score[pred_mask] = pred_occ_score

        return obj_pose, bg_pose, occ_score


class LatentCompressor(nn.Module):
    def __init__(self, opt, depth, spectral_norm_layer=None):
        super().__init__()
        # embedding
        self.cls_embed = nn.Parameter(torch.randn(1, 1, opt.embed_dim))

        # main blocks
        args = {"block_type": "cls",
                "depth": depth,
                "dim": opt.embed_dim,
                "num_heads": opt.num_heads,
                "norm_layer": opt.norm_layer,
                "spectral_norm_layer": spectral_norm_layer,
                "dropout": opt.dropout}
        self.norm = CustomNorm(opt.norm_layer, opt.embed_dim)
        self.blocks = MultiBlocks(**args)

        # init weights
        trunc_normal_(self.cls_embed, std=.02)
        self.apply(init_weights)

    def forward(self, x):
        x, vid_size = flatten_vid(x, vid_ndim=4)
        x = self.norm(x)
        z_cls = self.cls_embed.expand(x.size(0), -1, -1)
        z_cls = self.blocks(z_cls, x_ctx=x)
        return unflatten_vid(z_cls, vid_size)