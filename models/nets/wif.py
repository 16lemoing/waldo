import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.path as mpltPath

from models.modules import UNet
from tools.utils import expand, get_grid

class WIF(nn.Module):
    def __init__(self, opt):
        super().__init__()
        # params
        self.score = opt.ii_score
        self.ab = opt.ii_ab
        self.opt = opt

        # net
        scale_hd = opt.load_dim / opt.dim if (opt.load_dim > 0 and opt.ii_ft_hd) else 1
        extra_in = 1 if opt.use_disocc else 0
        if opt.ii_score:
            num_channels_in = 3 + opt.num_lyt + opt.num_obj + 1 + extra_in
            num_channels_out = 5 if opt.ii_ab else 4
            zero_init = opt.ii_ab
        else:
            num_channels_in = (3 + opt.num_lyt + opt.num_obj + 1 + extra_in) * opt.ctx_len
            num_channels_out = 3
            zero_init = False
        self.unet = UNet(num_channels_in, num_channels_out, opt.ii_embed_dim, opt.norm_layer_patch, opt.ii_depth, scale_hd, zero_init, opt.ii_upmode)
        src_shape = [opt.dim, int(opt.dim * opt.aspect_ratio)]
        src_shape_hd = [opt.load_dim, int(opt.load_dim * opt.aspect_ratio)] if opt.load_dim > 0 else src_shape
        self.src_grid_hd = get_grid(*src_shape_hd).cuda()


    def get_last_layer(self):
        return self.unet.from_emb.weight

    def forward(self, vid):
        B, Tc, T, C, H, W = vid.shape
        vid = vid.permute(0, 2, 1, 3, 4, 5)

        # prepare input
        if self.score:
            out = vid.reshape(B * T * Tc, C, H, W)
        else:
            out = vid.reshape(B * T, Tc * C, H, W)

        out = self.unet(out)

        # fuse
        if self.score:
            out = out.reshape(B, T, Tc, -1, H, W)
            vid_beta, vid_score = out[:, :, :, :3], out[:, :, :, 3:4].softmax(dim=2)
            vid_alpha = (vid[:, :, :, 4:5] + 5).sigmoid() if self.ab else 0
            vid = ((vid_alpha * vid[:, :, :, :3] + vid_beta) * vid_score).sum(dim=2)
        else:
            vid = out.reshape(B, T, -1, H, W)
        return vid

    def inpaint(self, inpainter, raw_output, alpha, alpha_ctx, real_vid, pred_flow, ctx_len, warper, grid):
        shadows = []
        if self.opt.use_inpainter:
            if self.opt.ii_last_only:
                mask = 1 - ((alpha_ctx + 1) / 2).sum(dim=3, keepdim=True)[:, -1]  # B Tp 1 H W
            else:
                mask = 1 - ((alpha_ctx + 1) / 2).sum(dim=3, keepdim=True).max(dim=1)[0]  # B Tp 1 H W
            mask_thresh = 0.1
            if self.opt.fix_thresh:
                mask = (mask > mask_thresh).float()
            else:
                mask = (mask > 1 - mask_thresh).float()
            if self.opt.ii_last_only:
                obj_mask = ((alpha_ctx[:, :, :, 1:] + 1) / 2).sum(dim=3, keepdim=True)[:, -1]
            else:
                obj_mask = ((alpha_ctx[:, :, :, 1:] + 1) / 2).sum(dim=3, keepdim=True).max(dim=1)[0]
            obj_mask_thresh = 0.9
            obj_mask = (obj_mask > obj_mask_thresh).float()
            if self.opt.use_expansion:
                mask = expand(mask, num=self.opt.num_expansion)
                mask = mask * (1 - obj_mask)
        if self.opt.loop_ii:
            inp_pred_vid = []
            Tp = raw_output.size(2)
            for t in range(Tp):
                inp_pred_vid.append(self.forward(raw_output[:, :, t:t + 1]))
            if self.opt.use_inpainter:
                assert self.opt.inpaint_obj
                assert self.opt.propagate_unique
                ref_left_mask = None
                ref_right_mask = None
                for t in range(Tp):
                    img = inp_pred_vid[t].squeeze(1)
                    curr_mask = mask[:, t]
                    if t == 0:
                        ref = -1  # hardcoded inpaint the last frame and propagate to previous ones
                        # ref = -Tp
                        ref_to_pred_bg_flow = warper.grid_to_bg_flow_from_ref_to_pred(grid, ctx_len, ref)
                        ctx_to_ref_bg_flow = warper.grid_to_bg_flow_from_ctx_to_ref(grid, ctx_len, ref)
                        ref_img = inp_pred_vid[ref].squeeze(1)
                        obj_mask_ref = obj_mask[:, ref]
                        # gather background from context
                        for t2 in range(ctx_len - 1, -1, -1):
                            ctx_img = real_vid[:, t2]
                            ctx_mask = (alpha[:, t2, :1] > 1 - mask_thresh).float()
                            warped_img = F.grid_sample(ctx_img, ctx_to_ref_bg_flow[:, t2] + self.src_grid_hd)
                            warped_mask = F.grid_sample(ctx_mask, ctx_to_ref_bg_flow[:, t2] + self.src_grid_hd)
                            warped_mask = (warped_mask > 1 - mask_thresh).float()
                            # check shadows
                            if self.opt.use_shadows and t2 == ctx_len - 1:
                                shadow_mask = ((warped_img - ref_img).abs().mean(dim=1, keepdim=True) > .25).float() * warped_mask * (1 - obj_mask_ref)
                                shadows.append(shadow_mask)
                                shadow_mask = 1 - expand(1 - shadow_mask, num=5)
                                shadows.append(shadow_mask)
                                shadow_mask = expand(shadow_mask, num=5)  # filter small regions
                                shadows.append(shadow_mask)
                                shadow_mask[:, :, :int(shadow_mask.size(2) * 0.4)] = 0
                                shadows.append(shadow_mask)
                                shadow_mask = expand(shadow_mask, num=30, soft=self.opt.soft_shadow)
                                shadows.append(shadow_mask)
                            inter_mask = obj_mask_ref * warped_mask
                            ref_img = inter_mask * warped_img + (1 - inter_mask) * ref_img
                            obj_mask_ref = (1 - inter_mask) * obj_mask_ref
                            if self.opt.ii_last_only:
                                break

                        if self.opt.fix_mask:
                            ref_mask = 1 - (1 - mask[:, ref]) * (1 - obj_mask_ref)  # expand(1 - (1 - mask[:, ref]) * (1 - obj_mask_ref), 3)
                            masked_ref_img = (1 - ref_mask) * ref_img
                            ref_img = inpainter(ref_img, ref_mask, is_masked=False)  # , exp=False
                        else:
                            ref_mask = 1 - (1 - mask[:, ref]) * (1 - obj_mask_ref)
                            masked_ref_img = (1 - mask[:, ref]) * (1 - obj_mask_ref) * ref_img
                            ref_img = inpainter(masked_ref_img, ref_mask)

                    if t == 0 and self.opt.propagate_obj:
                        border = 3  # pixels
                        pred_grid = pred_flow[:, -1, -1].permute(0, 2, 3, 1) + self.src_grid_hd  # B H W 2
                        h, w = self.src_grid_hd.shape[1:3]
                        pred_grid[:, :, :, 0] = (pred_grid[:, :, :, 0] * w + w - 1) / 2
                        pred_grid[:, :, :, 1] = (pred_grid[:, :, :, 1] * h + h - 1) / 2
                        orig_grid = self.src_grid_hd.clone()
                        orig_grid[:, :, :, 0] = (orig_grid[:, :, :, 0] * w + w - 1) / 2
                        orig_grid[:, :, :, 1] = (orig_grid[:, :, :, 1] * h + h - 1) / 2
                        is_left_border = pred_grid[:, :, :, 0] < border
                        is_right_border = pred_grid[:, :, :, 0] >= w - border
                        all_obj_mask = (((alpha_ctx[:, :, -1, 1:] + 1) / 2).max(dim=1)[0] > 0.9).float()  # B No H W
                        is_left_obj = is_left_border.float().unsqueeze(1) * all_obj_mask
                        is_right_obj = is_right_border.float().unsqueeze(1) * all_obj_mask
                        # print("any", is_left_obj.sum() > 0, "pred_grid", pred_grid.min(), pred_grid.max())
                        if is_left_obj.sum() > 0:
                            left_obj_id = is_left_obj.flatten(start_dim=2).sum(-1).argmax(dim=1)
                            left_obj_id = int(left_obj_id[0])
                            left_obj = is_left_obj[:, left_obj_id].bool()
                            border_val = pred_grid[left_obj]
                            orig_val = orig_grid[left_obj]
                            left_corners = [(0, float(border_val[:, 1].min())),
                                            (0, float(border_val[:, 1].max())),
                                            (float(orig_val[:, 0].max()), float(orig_val[:, 1].max())),
                                            (float(orig_val[:, 0].max()), float(orig_val[:, 1].min()))]
                            # print(left_corners)
                            ref_left_mask = point_in_polygon(orig_grid, left_corners).float()
                            masked_ref_img = (1 - ref_left_mask) * raw_output[:, -1, -1, :3]  # obj_mask[:, ref] *   inp_pred_vid[ref].squeeze(1)
                            ref_left_obj = inpainter(masked_ref_img, ref_left_mask)  # 1 - obj_mask[:, ref] * (1 - ref_left_mask)
                            ref_to_pred_left_obj_flow = warper.grid_to_obj_flow_from_ref_to_pred(grid, ctx_len, ref, left_obj_id)
                        if is_right_obj.sum() > 0:
                            right_obj_id = is_right_obj.flatten(start_dim=2).sum(-1).argmax(dim=1)
                            right_obj_id = int(right_obj_id[0])
                            right_obj = is_right_obj[:, right_obj_id].bool()
                            border_val = pred_grid[right_obj]
                            orig_val = orig_grid[right_obj]
                            right_corners = [(float(orig_val[:, 0].min()), float(orig_val[:, 1].min())),
                                             (float(orig_val[:, 0].min()), float(orig_val[:, 1].max())),
                                             (w - 1, float(border_val[:, 1].max())),
                                             (w - 1, float(border_val[:, 1].min()))]
                            ref_right_mask = point_in_polygon(orig_grid, right_corners).float()
                            masked_ref_img = (1 - ref_right_mask) * raw_output[:, -1, -1, :3]  # obj_mask[:, ref] * inp_pred_vid[ref].squeeze(1)
                            ref_right_obj = inpainter(masked_ref_img, ref_right_mask)  # 1 - obj_mask[:, ref] * (1 - ref_right_mask)
                            ref_to_pred_right_obj_flow = warper.grid_to_obj_flow_from_ref_to_pred(grid, ctx_len, ref, right_obj_id)

                    warped_img = F.grid_sample(ref_img, ref_to_pred_bg_flow[:, t] + self.src_grid_hd)
                    warped_mask = F.grid_sample(ref_mask, ref_to_pred_bg_flow[:, t] + self.src_grid_hd)
                    warped_mask = (warped_mask > 1 - mask_thresh).float()
                    if ref_left_mask is not None:
                        warped_left_obj = F.grid_sample(ref_left_obj, ref_to_pred_left_obj_flow[:, t] + self.src_grid_hd)
                        warped_left_mask = F.grid_sample(ref_left_mask, ref_to_pred_left_obj_flow[:, t] + self.src_grid_hd)
                        warped_left_mask = (warped_left_mask > 1 - mask_thresh).float()
                        warped_mask = 1 - (1 - warped_mask) * (1 - warped_left_mask)
                        curr_mask = 1 - (1 - curr_mask) * (1 - warped_left_mask)
                        warped_img = (1 - warped_left_mask) * warped_img + warped_left_mask * warped_left_obj
                    if ref_right_mask is not None:
                        warped_right_obj = F.grid_sample(ref_right_obj, ref_to_pred_right_obj_flow[:, t] + self.src_grid_hd)
                        warped_right_mask = F.grid_sample(ref_right_mask, ref_to_pred_right_obj_flow[:, t] + self.src_grid_hd)
                        warped_right_mask = (warped_right_mask > 1 - mask_thresh).float()
                        warped_mask = 1 - (1 - warped_mask) * (1 - warped_right_mask)
                        curr_mask = 1 - (1 - curr_mask) * (1 - warped_right_mask)
                        warped_img = (1 - warped_right_mask) * warped_img + warped_right_mask * warped_right_obj
                    obj_mask_t = obj_mask[:, t]
                    if self.opt.use_shadows:
                        warped_shadow_mask = F.grid_sample(shadow_mask, ref_to_pred_bg_flow[:, t] + self.src_grid_hd)
                        if not self.opt.soft_shadow:
                            warped_shadow_mask = (warped_shadow_mask > 1 - mask_thresh).float()
                        curr_mask = curr_mask * (1 - warped_shadow_mask * (1 - obj_mask_t))
                    inter_mask = curr_mask * warped_mask
                    img = inter_mask * warped_img + (1 - inter_mask) * img
                    curr_mask = (1 - inter_mask) * curr_mask
                    if self.opt.fix_mask:
                        m = expand(1 - (1 - curr_mask) * (1 - obj_mask_t), 3)
                        inp_img = inpainter(img, m, exp=False, is_masked=False)
                    else:
                        masked_img = (1 - curr_mask) * (1 - obj_mask_t) * img
                        inp_img = inpainter(masked_img, 1 - (1 - curr_mask) * (1 - obj_mask_t))
                    inp_pred_vid[t] = ((1 - curr_mask) * img + curr_mask * inp_img).unsqueeze(1)
            inp_pred_vid = torch.cat(inp_pred_vid, dim=1)
        else:
            inp_pred_vid = self.forward(raw_output)
            if self.opt.use_inpainter:
                for t in range(inp_pred_vid.size(1)):
                    if self.opt.inpaint_obj:
                        masked_img = (1 - mask[:, t]) * (1 - obj_mask[:, t]) * inp_pred_vid[:, t]
                        inp_img = self.inpainter(masked_img, 1 - (1 - mask[:, t]) * (1 - obj_mask[:, t]))
                        inp_pred_vid[:, t] = (1 - mask[:, t]) * inp_pred_vid[:, t] + mask[:, t] * inp_img
                    else:
                        masked_img = (1 - mask[:, t]) * inp_pred_vid[:, t]
                        inp_pred_vid[:, t] = self.inpainter(masked_img, mask[:, t])
        inp_pred_vid = torch.cat([real_vid[:, :ctx_len], inp_pred_vid], dim=1)
        return inp_pred_vid


def point_in_polygon(pts, corners):
    device = pts.device
    B, H, W, _ = pts.shape
    assert B == 1
    pts = pts.view(-1, 2).cpu().numpy()
    path = mpltPath.Path(corners)
    mask = path.contains_points(pts)
    return torch.from_numpy(mask).view(B, 1, H, W).to(device)

