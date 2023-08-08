import os
import math
import lpips

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision.transforms import GaussianBlur

from models.nets.lvd import LVD
from models.nets.flp import FLP
from models.nets.wif import WIF
from models.modules import VGGLoss, get_gan_loss, EdgeExtractor, MatInpainter
from models import load_network, save_network, print_network
from tools.utils import mkdir, flatten, unflatten, to_cuda, DummyOpt, DummyDecorator, DummyScaler, mkdir, dump_image, dump_video, flatten_vid, unflatten_vid, to_ctx, from_ctx


class Synthesizer(torch.nn.Module):
    def __init__(self, opt, engine, is_train=True, is_main=True, logger=None):
        super().__init__()
        self.opt = opt
        self.engine = engine
        self.is_main = is_main

        self.initialize_networks(is_train)
        self.parallelize_networks()
        self.load_networks()
        if self.is_main:
            self.print_networks()

        if is_train:
            self.create_optimizers()
            self.initialize_losses()

        self.logger = logger if self.is_main else None
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.lpips_alex = None
        self.lpips_vgg = None
        self.nancount = 0

        if opt.use_inpainter:
            self.inpainter = MatInpainter(opt).cuda()


    def forward(self, data, mode='', log=False, global_iter=None, tmp_iter=None, is_eval=False, dump=False):
        real_img, real_vid, real_lyt, real_flow = self.preprocess_input(data)

        if mode == 'vid_object_extractor':
            return self.vid_object_extractor(real_vid, real_lyt, real_flow, log, global_iter, tmp_iter, is_eval, dump)

        elif mode == 'vid_inpainting':
            return self.vid_inpainting(real_vid, real_lyt, real_flow, log, global_iter, tmp_iter, is_eval, dump)

        elif mode == 'vid_pose_generator':
            return self.vid_pose_generator(real_vid, real_lyt, real_flow, log, global_iter, tmp_iter, is_eval, dump)

        elif mode == 'img_object_extractor':
            return self.img_object_extractor(real_img, real_lyt, real_flow, log, global_iter, tmp_iter, is_eval, dump)

        elif mode == 'vid_prediction':
            return self.vid_prediction(real_vid, real_lyt, real_flow, log, global_iter, tmp_iter, dump)

        else:
            raise ValueError(f"mode '{mode}' is invalid")


    def requires_data(self, mode):
        return True
        # return mode not in ['img_generator']


    def preprocess_input(self, data, is_fake=False):
        data = {} if data is None else data
        data["img"] = to_cuda(data, "img")
        data["vid"] = to_cuda(data, "vid")
        data["lyt"] = to_cuda(data, "lyt")
        data["flow"] = to_cuda(data, "flow")
        return data["img"], data["vid"], data["lyt"], data["flow"]


    def initialize_networks(self, is_train):
        self.net_pe = LVD(self.opt).cuda() if self.opt.use_pe else None
        self.net_pg = FLP(self.opt).cuda() if self.opt.use_pg else None
        self.net_ii = WIF(self.opt).cuda() if self.opt.use_ii else None


    def parallelize_networks(self):
        self.net_pe = self.engine.data_parallel(self.net_pe, broadcast_buffers=False)
        self.net_pg = self.engine.data_parallel(self.net_pg)
        self.net_ii = self.engine.data_parallel(self.net_ii)


    def load_networks(self):
        map_location = {'cuda:%d' % 0: 'cuda:%d' % self.engine.local_rank}
        self.net_pe = load_network(self.net_pe, "pe", self.opt, map_location=map_location, verbose=self.is_main, required=False)
        self.net_pg = load_network(self.net_pg, "pg", self.opt, map_location=map_location, verbose=self.is_main, required=False, load_path=self.opt.pg_load_path, iter=self.opt.pg_iter)
        self.net_ii = load_network(self.net_ii, "ii", self.opt, map_location=map_location, verbose=self.is_main, required=False, load_path=self.opt.ii_load_path, iter=self.opt.ii_iter)


    def print_networks(self):
        print_network(self.net_pe)
        print_network(self.net_pg)
        print_network(self.net_ii)


    def save_networks(self, global_iter, name=None):
        save_network(self.net_pe, "pe", global_iter, self.opt, name)
        save_network(self.net_pg, "pg", global_iter, self.opt, name)
        save_network(self.net_ii, "ii", global_iter, self.opt, name)


    def create_optimizers(self):
        def filter_params(params):
            return params
            # return filter(lambda p: p.requires_grad, params)

        def get_adam_opt(net, lr, b1, b2, wd, freeze=False):
            if net is None or freeze or len(list(net.parameters())) == 0:
                return DummyOpt()
            else:
                return torch.optim.Adam(filter_params(net.parameters()), lr=lr, betas=(b1, b2)) #, weight_decay=wd)

        def get_adamw_opt(net, lr, b1, b2, wd, freeze=False):
            if net is None or freeze or len(list(net.parameters())) == 0:
                return DummyOpt()
            else:
                split_params = hasattr(net.module, 'no_weight_decay') if self.engine.distributed else hasattr(net, 'no_weight_decay')
                if split_params:
                    skip = net.module.no_weight_decay() if self.engine.distributed else net.no_weight_decay()
                    params = add_weight_decay(net, wd, skip)
                    wd = 0.
                    print("Not applying weight decay for these parameters: ", skip)
                else:
                    params = net.parameters()
                return torch.optim.AdamW(filter_params(params), lr=lr, betas=(b1, b2), weight_decay=wd)

        opt_choices = {"adam": get_adam_opt, "adamw": get_adamw_opt}
        get_opt = opt_choices[self.opt.optimizer]
        self.opt_pe = get_opt(self.net_pe, self.opt.lr, self.opt.beta1, self.opt.beta2, self.opt.wd)
        self.opt_pg = get_opt(self.net_pg, self.opt.lr, self.opt.beta1, self.opt.beta2, self.opt.wd)
        self.opt_ii = get_opt(self.net_ii, self.opt.lr, self.opt.beta1, self.opt.beta2, self.opt.wd)


    def initialize_amp(self):
        if self.opt.use_amp:
            self.amp_decorator = autocast
            Scaler = GradScaler
        else:
            self.amp_decorator = DummyDecorator
            Scaler = DummyScaler
        self.img_object_extractor_scaler = Scaler()
        self.vid_object_extractor_scaler = Scaler()
        self.vid_inpainting_scaler = Scaler()
        self.vid_pose_generator_scaler = Scaler()
        self.img_inpainting_discriminator_scaler = Scaler()
        self.vid_inpainting_discriminator_scaler = Scaler()


    def initialize_losses(self):
        self.vgg_loss = VGGLoss()
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='none')
        gan_loss = get_gan_loss(self.opt)
        self.gan_loss_pd = gan_loss(self.net_pd)
        self.gan_loss_id = gan_loss(self.net_id)
        self.edge = EdgeExtractor(kernel_size=self.opt.edge_size).cuda()


    def save_img(self, real_img, fake_img, tmp_iter, suffix=""):
        real_folder = os.path.join(self.opt.result_path, f"real{suffix}")
        fake_folder = os.path.join(self.opt.result_path, f"fake{suffix}")
        B = real_img.size(0)
        for i in range(B):
            img_per_iter = self.engine.world_size * B
            img_id = img_per_iter * tmp_iter + self.engine.global_rank * B + i
            real_save_path = os.path.join(real_folder, f"img_{img_id:05d}.png")
            fake_save_path = os.path.join(fake_folder, f"img_{img_id:05d}.png")
            if not os.path.exists(real_save_path):
                dump_image(real_img[i], real_save_path)
            dump_image(fake_img[i], fake_save_path)


    def save_vid(self, vid, name, tmp_iter, overwrite=True):
        folder = os.path.join(self.opt.result_path, name)
        mkdir(folder)
        B = vid.size(0)
        for i in range(B):
            vid_per_iter = self.engine.world_size * B
            vid_id = vid_per_iter * tmp_iter + self.engine.global_rank * B + i
            save_path = os.path.join(folder, f"vid_{vid_id:05d}.mp4")
            if overwrite or not os.path.exists(save_path):
                dump_video(vid[i].cpu(), save_path)


    def sync_scalars(self, scalars):
        for s in scalars:
            try:
                scalars[s] = self.engine.all_reduce_tensor(scalars[s])
            except:
                print("Could not synchronise scalar:", s)


    def img_object_extractor(self, real_img, real_lyt, real_flow, log, global_iter, tmp_iter, is_eval, dump):
        mode = "eval" if is_eval else "train"
        log_dic = {"scalar": {}, "img": {}, "vid": {}, "obj_lyt": {}, "sem_lyt": {}, "flow": {}}

        real_vid, real_lyt, real_flow = real_img.unsqueeze(1), real_lyt.unsqueeze(1), real_flow.unsqueeze(1)
        log_dic = self.extract_object(real_vid, real_lyt, real_flow, log_dic, is_eval, "img", global_iter, log)

        # log images
        if self.logger and log:
            for name, tensor in log_dic["img"].items():
                self.logger.log_img(f"img_object_extractor/{mode}/{name}", tensor[:4], 2, global_iter, normalize=True, span=(-1, 1))
            for name, tensor in log_dic["vid"].items():
                self.logger.log_vid(f"img_object_extractor/{mode}/{name}", tensor[:4], 2, global_iter, normalize=True, span=(-1, 1))
            for name, tensor in log_dic["obj_lyt"].items():
                self.logger.log_lyt(f"img_object_extractor/{mode}/{name}", tensor[:4], 2, global_iter, self.opt.num_obj + 1, False)
            for name, tensor in log_dic["sem_lyt"].items():
                self.logger.log_lyt(f"img_object_extractor/{mode}/{name}", tensor[:4], 2, global_iter, self.opt.num_lyt, True)
            for name, tensor in log_dic["flow"].items():
                self.logger.log_flow(f"img_object_extractor/{mode}/{name}", tensor[:4], 2, global_iter)

        # log scalars
        if not is_eval and self.logger:
            self.logger.log_scalars("img_object_extractor/train", log_dic["scalar"], global_iter)

        # sync scalars
        if is_eval:
            self.sync_scalars(log_dic["scalar"])

        return log_dic["scalar"]


    def vid_object_extractor(self, real_vid, real_lyt, real_flow, log, global_iter, tmp_iter, is_eval, dump):
        mode = "eval" if is_eval else "train"
        log_dic = {"scalar": {}, "img": {}, "vid": {}, "obj_lyt": {}, "sem_lyt": {}, "flow": {}, "pts": {}}

        log_dic = self.extract_object(real_vid, real_lyt, real_flow, log_dic, is_eval, "vid", global_iter, log)

        # log images
        if self.logger and log:
            for name, tensor in log_dic["img"].items():
                self.logger.log_img(f"vid_object_extractor/{mode}/{name}", tensor[:4], 2, global_iter, normalize=True, span=(-1, 1))
            for name, tensor in log_dic["vid"].items():
                self.logger.log_vid(f"vid_object_extractor/{mode}/{name}", tensor[:4], 2, global_iter, normalize=True, span=(-1, 1))
            for name, tensor in log_dic["obj_lyt"].items():
                self.logger.log_lyt(f"vid_object_extractor/{mode}/{name}", tensor[:4], 2, global_iter, self.opt.num_obj + 1, False)
            for name, tensor in log_dic["sem_lyt"].items():
                self.logger.log_lyt(f"vid_object_extractor/{mode}/{name}", tensor[:4], 2, global_iter, self.opt.num_lyt, True)
            for name, tensor in log_dic["flow"].items():
                self.logger.log_flow(f"vid_object_extractor/{mode}/{name}", tensor[:4], 2, global_iter)

        # viz
        if is_eval and self.opt.viz:
            real_flow = self.logger.get_flow(real_flow) * 2 - 1
            rec_flow = self.logger.get_flow(log_dic["flow"]["rec_flow"]) * 2 - 1
            found_obj = -(2 * log_dic["vid"]["found_obj"] + 1).expand(-1, -1, 3, -1, -1)
            mov_obj = log_dic["vid"]["mov_obj"].expand(-1, -1, 3, -1, -1)
            obj_lyt = self.logger.get_lyt(log_dic["obj_lyt"]["obj_lyt"], self.opt.num_obj + 1, use_palette=False)
            sem_lyt = self.logger.get_lyt(real_lyt, self.opt.num_lyt, use_palette=True)
            obj_pts = log_dic["pts"]["obj_pts"]
            bg_pts = log_dic["pts"]["bg_pts"]
            pts = self.logger.get_pts(obj_pts, bg_pts, self.opt.dim, int(self.opt.dim * self.opt.aspect_ratio), 4)
            mot = self.logger.get_mot(obj_pts, bg_pts, self.opt.dim, int(self.opt.dim * self.opt.aspect_ratio), 4)
            obj_area = ((log_dic["obj_lyt"]["obj_lyt"][:, :, 1:] + 1) / 2).sum(dim=(1, 3, 4)).unsqueeze(1).expand(-1, real_vid.size(1), -1)
            thresh_obj_pts = obj_pts.clone()
            thresh = 10
            thresh_obj_pts[obj_area < thresh] = thresh_obj_pts[obj_area < thresh] * 0 -2
            thresh_pts = self.logger.get_pts(thresh_obj_pts, bg_pts, self.opt.dim, int(self.opt.dim * self.opt.aspect_ratio), 4)
            thresh_mot = self.logger.get_mot(thresh_obj_pts, bg_pts, self.opt.dim, int(self.opt.dim * self.opt.aspect_ratio), 4)
            self.save_vid(real_vid, "real_vid", tmp_iter)
            self.save_vid(real_flow, "real_flow", tmp_iter)
            self.save_vid(rec_flow, "rec_flow", tmp_iter)
            self.save_vid(mov_obj, "mov_obj", tmp_iter)
            self.save_vid(obj_lyt, "obj_lyt", tmp_iter)
            self.save_vid(sem_lyt, "sem_lyt", tmp_iter)
            self.save_vid(found_obj, "found_obj", tmp_iter)
            self.save_vid(pts, "pts", tmp_iter)
            self.save_vid(mot, "mot", tmp_iter)
            self.save_vid(thresh_pts, "thresh_pts", tmp_iter)
            self.save_vid(thresh_mot, "thresh_mot", tmp_iter)

        # log scalars
        if not is_eval and self.logger:
            self.logger.log_scalars("vid_object_extractor/train", log_dic["scalar"], global_iter)

        # sync scalars
        if is_eval:
            self.sync_scalars(log_dic["scalar"])

        return log_dic["scalar"]


    def vid_inpainting(self, real_vid, real_lyt, real_flow, log, global_iter, tmp_iter, is_eval, dump):
        mode = "eval" if is_eval else "train"
        log_dic = {"scalar": {}, "img": {}, "vid": {}, "obj_lyt": {}, "sem_lyt": {}, "flow": {}}

        real_img, inp_vid, log_dic = self.inpaint(real_vid, real_lyt, real_flow, log_dic, is_eval, "vid", global_iter)
        if "dis" in self.opt.vid_inpainting_losses:
            log_dic = self.discriminate_inpainting(real_img, inp_vid, log_dic, is_eval, "vid", global_iter)

        # log images
        if self.logger and log:
            for name, tensor in log_dic["img"].items():
                self.logger.log_img(f"vid_inpainting/{mode}/{name}", tensor[:4], 2, global_iter, normalize=True, span=(-1, 1))
            for name, tensor in log_dic["vid"].items():
                self.logger.log_vid(f"vid_inpainting/{mode}/{name}", tensor[:4], 2, global_iter, normalize=True, span=(-1, 1))
            for name, tensor in log_dic["obj_lyt"].items():
                self.logger.log_lyt(f"vid_inpainting/{mode}/{name}", tensor[:4], 2, global_iter, self.opt.num_obj + 1, False)
            for name, tensor in log_dic["sem_lyt"].items():
                self.logger.log_lyt(f"vid_inpainting/{mode}/{name}", tensor[:4], 2, global_iter, self.opt.num_lyt, True)
            for name, tensor in log_dic["flow"].items():
                self.logger.log_flow(f"vid_inpainting/{mode}/{name}", tensor[:4], 2, global_iter)

        # log scalars
        if not is_eval and self.logger:
            self.logger.log_scalars("vid_inpainting/train", log_dic["scalar"], global_iter)

        # sync scalars
        if is_eval:
            self.sync_scalars(log_dic["scalar"])

        return log_dic["scalar"]


    def vid_pose_generator(self, real_vid, real_lyt, real_flow, log, global_iter, tmp_iter, is_eval, dump):
        mode = "eval" if is_eval else "train"
        log_dic = {"scalar": {}, "img": {}, "vid": {}, "obj_lyt": {}, "sem_lyt": {}, "flow": {}, "ctx_mask":{}, "pts":{}}

        log_dic = self.generate_pose(real_vid, real_lyt, real_flow, log_dic, is_eval, "vid", log)

        # log images
        if self.logger and log:
            for name, tensor in log_dic["img"].items():
                self.logger.log_img(f"vid_pose_generator/{mode}/{name}", tensor[:4], 2, global_iter, normalize=True, span=(-1, 1))
            for name, tensor in log_dic["vid"].items():
                ctx_mask = log_dic["ctx_mask"][name][:4] if name in log_dic["ctx_mask"] else None
                self.logger.log_vid(f"vid_pose_generator/{mode}/{name}", tensor[:4], 2, global_iter, normalize=True, span=(-1, 1), ctx_mask=ctx_mask)
            for name, tensor in log_dic["obj_lyt"].items():
                self.logger.log_lyt(f"vid_pose_generator/{mode}/{name}", tensor[:4], 2, global_iter, self.opt.num_obj + 1, False)
            for name, tensor in log_dic["sem_lyt"].items():
                self.logger.log_lyt(f"vid_pose_generator/{mode}/{name}", tensor[:4], 2, global_iter, self.opt.num_lyt, True)
            for name, tensor in log_dic["flow"].items():
                self.logger.log_flow(f"vid_pose_generator/{mode}/{name}", tensor[:4], 2, global_iter)

        # viz
        if is_eval and self.opt.viz:
            real_obj_pts = log_dic["pts"]["real_obj_pts"]
            real_bg_pts = log_dic["pts"]["real_bg_pts"]
            pred_obj_pts = log_dic["pts"]["pred_obj_pts"]
            pred_bg_pts = log_dic["pts"]["pred_bg_pts"]
            ref_obj_pts = real_obj_pts[:, self.opt.ctx_len - 1]
            ref_bg_pts = real_bg_pts[:, self.opt.ctx_len - 1]
            real_obj_pts = real_obj_pts[:, self.opt.ctx_len:]
            real_bg_pts = real_bg_pts[:, self.opt.ctx_len:]
            pred_obj_pts = pred_obj_pts[:, self.opt.ctx_len:]
            pred_bg_pts = pred_bg_pts[:, self.opt.ctx_len:]
            obj_area = ((log_dic["obj_lyt"]["rec_obj_lyt"][:, :, 1:] + 1) / 2).sum(dim=(1, 3, 4)).unsqueeze(1).expand(-1, real_vid.size(1) - self.opt.ctx_len, -1)
            thresh_real_obj_pts = real_obj_pts.clone()
            thresh_pred_obj_pts = pred_obj_pts.clone()
            thresh = 10
            ref_obj_pts[obj_area[:, 0] < thresh] = ref_obj_pts[obj_area[:, 0] < thresh] * 0 - 2
            thresh_real_obj_pts[obj_area < thresh] = thresh_real_obj_pts[obj_area < thresh] * 0 - 2
            thresh_pred_obj_pts[obj_area < thresh] = thresh_pred_obj_pts[obj_area < thresh] * 0 - 2
            real_delta_mot = self.logger.get_delta_mot(thresh_real_obj_pts, real_bg_pts, ref_obj_pts, ref_bg_pts, self.opt.dim, int(self.opt.dim * self.opt.aspect_ratio), 4)
            pred_delta_mot = self.logger.get_delta_mot(thresh_pred_obj_pts, pred_bg_pts, ref_obj_pts, ref_bg_pts, self.opt.dim, int(self.opt.dim * self.opt.aspect_ratio), 4)
            self.save_vid(real_vid, "real_vid", tmp_iter)
            self.save_vid(real_delta_mot, "real_delta_mot", tmp_iter)
            self.save_vid(pred_delta_mot, "pred_delta_mot", tmp_iter)

        # log scalars
        if not is_eval and self.logger:
            self.logger.log_scalars("vid_pose_generator/train", log_dic["scalar"], global_iter)

        # sync scalars
        if is_eval:
            self.sync_scalars(log_dic["scalar"])

        return log_dic["scalar"]


    def vid_prediction(self, real_vid, real_lyt, real_flow, log, global_iter, tmp_iter, dump):
        mode = "test"
        log_dic = {"scalar": {}, "img": {}, "vid": {}, "obj_lyt": {}, "sem_lyt": {}, "flow": {}}
        is_eval = True

        rec_vid, inp_rec_vid, pred_vid, inp_pred_vid = self.predict(real_vid, real_lyt, real_flow, log_dic, is_eval, "vid", global_iter)

        # log images
        if self.logger and log:
            for name, tensor in log_dic["img"].items():
                self.logger.log_img(f"vid_prediction/{mode}/{name}", tensor[:4], 2, global_iter, normalize=True, span=(-1, 1))
            for name, tensor in log_dic["vid"].items():
                self.logger.log_vid(f"vid_prediction/{mode}/{name}", tensor[:4], 2, global_iter, normalize=True, span=(-1, 1))
            for name, tensor in log_dic["obj_lyt"].items():
                self.logger.log_lyt(f"vid_prediction/{mode}/{name}", tensor[:4], 2, global_iter, self.opt.num_obj + 1, False)
            for name, tensor in log_dic["sem_lyt"].items():
                self.logger.log_lyt(f"vid_prediction/{mode}/{name}", tensor[:4], 2, global_iter, self.opt.num_lyt, True)
            for name, tensor in log_dic["flow"].items():
                self.logger.log_flow(f"vid_prediction/{mode}/{name}", tensor[:4], 2, global_iter)

        # dump videos
        if dump:
            self.save_vid(real_vid, "real_vid", tmp_iter)
            self.save_vid(rec_vid, "rec_vid", tmp_iter)
            self.save_vid(pred_vid, "pred_vid", tmp_iter) if pred_vid is not None else None
            self.save_vid(inp_rec_vid, "inp_rec_vid", tmp_iter)
            self.save_vid(inp_pred_vid, "inp_pred_vid", tmp_iter) if inp_pred_vid is not None else None
            self.save_vid(log_dic["vid"]["pred_disocc"].expand(-1, -1, 3, -1, -1), "pred_disocc", tmp_iter)
            self.save_vid(log_dic["vid"]["rec_disocc"].expand(-1, -1, 3, -1, -1), "rec_disocc", tmp_iter)

        return log_dic["scalar"]


    def predict(self, real_vid, real_lyt, real_flow, log_dic, is_eval, dtype, global_iter):
        ctx_len = 1 if dtype == "img" else self.opt.ctx_len

        with torch.no_grad():
            # merge modalities into a single input
            real_input = torch.cat([real_vid if self.opt.input_rgb else real_vid[:, :, :0],
                                    real_lyt if self.opt.input_lyt else real_vid[:, :, :0],
                                    real_flow if self.opt.input_flow else real_vid[:, :, :0]], dim=2)

            # encode input to feature space
            x = self.net_pe(input=real_input, mode="encode_input") # B T L C

            # extract layer features from context video features
            x_ctx = x[:, :ctx_len]
            x_obj, x_bg, cls = self.net_pe(x=x_ctx, mode="estimate_layer")

            # extract pose from layer features and video features
            obj_pose, bg_pose, occ_score, pts_rest_obj, pts_rest_bg, last_obj, last_bg = self.net_pe(x=x, x_obj=x_obj, x_bg=x_bg, mode="estimate_pose")
            occ, obj_alpha, bg_alpha, grid = self.net_pe(x_obj=x_obj, obj_pose=obj_pose, bg_pose=bg_pose, occ_score=occ_score, mode="estimate_alpha_grid_occ")

            # reconstruct video
            ctx_ts = torch.arange(ctx_len, device=x.device, dtype=torch.int64)
            ctx_ts = ctx_ts.view(1, -1, 1).expand(x.size(0), -1, x.size(1))
            if self.opt.last_n_ctx > 0:
                # take n penultimate frames for context
                ctx_ts = ctx_ts[:, -self.opt.last_n_ctx:].contiguous()

            pred_ts = torch.arange(real_vid.size(1), device=x.device, dtype=torch.int64)
            real_input = torch.cat([real_vid, real_lyt], dim=2)
            rec_output, _, _, _, _, raw_output, alpha_ctx = self.net_pe(input=real_input, grid=grid, occ=occ, obj_alpha=obj_alpha, bg_alpha=bg_alpha, ctx_ts=ctx_ts, pred_ts=pred_ts, cls=cls, mode="decode_output")
            rec_vid = rec_output[:, :, :3]

            disocc_max = alpha_ctx.max(dim=3)[0].max(dim=1)[0]
            disocc_min = alpha_ctx.max(dim=3)[0].min(dim=1)[0]
            disocc_max[disocc_max - disocc_min > 1] = 0
            log_dic["vid"]["rec_disocc"] = disocc_max.unsqueeze(2)

            # inpaint reconstructed video
            if self.opt.loop_ii:
                inp_rec_vid = []
                for t in range(raw_output.size(1)):
                    inp_rec_vid.append(self.net_ii(raw_output[:, t:t+1]))
                inp_rec_vid = torch.cat(inp_rec_vid, dim=1)
            else:
                inp_rec_vid = self.net_ii(raw_output)

            # predict video
            pred_vid, inp_pred_vid = None, None
            if not self.opt.no_future:
                ctx_mask = torch.arange(0, real_vid.size(1), device=real_vid.device).view(1, -1).expand(real_vid.size(0), -1) < ctx_len
                pred_obj_pose, pred_bg_pose, pred_occ_score = self.net_pg(obj_pose, bg_pose, occ_score, x_obj, x_bg, last_obj, last_bg, ctx_mask=ctx_mask)
                pred_ts = torch.arange(ctx_len, real_vid.size(1), device=x.device, dtype=torch.int64)
                ctx_ts = torch.arange(ctx_len, device=x.device, dtype=torch.int64)
                ctx_ts = ctx_ts.view(1, -1, 1).expand(x.size(0), -1, pred_ts.size(0))
                occ, obj_alpha, bg_alpha, grid = self.net_pe(x_obj=x_obj, obj_pose=pred_obj_pose, bg_pose=pred_bg_pose,
                                                             occ_score=pred_occ_score, mode="estimate_alpha_grid_occ")
                pred_output, pred_flow, _, alpha, _, raw_output, alpha_ctx = self.net_pe(input=real_input, grid=grid, occ=occ, obj_alpha=obj_alpha, bg_alpha=bg_alpha, ctx_ts=ctx_ts, pred_ts=pred_ts, cls=cls, mode="decode_output")
                ##
                disocc_max = alpha_ctx.max(dim=3)[0].max(dim=1)[0]
                disocc_min = alpha_ctx.max(dim=3)[0].min(dim=1)[0]
                disocc_max[disocc_max - disocc_min > 1] = 0
                log_dic["vid"]["pred_disocc"] = disocc_max.unsqueeze(2)

                pred_vid = pred_output[:, :, :3]
                pred_vid = torch.cat([real_vid[:, :ctx_len], pred_vid], dim=1)

                # inpaint predicted vid
                warper = self.net_pe.module.warper
                inp_pred_vid = self.net_ii.module.inpaint(self.inpainter, raw_output, alpha, alpha_ctx, real_vid, pred_flow, ctx_len, warper, grid)
                # if self.opt.loop_ii:
                #     inp_pred_vid = []
                #     for t in range(raw_output.size(1)):
                #         inp_pred_vid.append(self.net_ii(raw_output[:, t:t+1]))
                #     inp_pred_vid = torch.cat(inp_pred_vid, dim=1)
                # else:
                #     inp_pred_vid = self.net_ii(raw_output)
                # inp_pred_vid = torch.cat([real_vid[:, :ctx_len], inp_pred_vid], dim=1)


        log_dic["vid"]["real_vid"] = real_vid
        log_dic["vid"]["rec_vid"] = rec_vid
        log_dic["vid"]["inp_rec_vid"] = inp_rec_vid
        if not self.opt.no_future:
            log_dic["vid"]["pred_vid"] = pred_vid
            log_dic["vid"]["inp_pred_vid"] = inp_pred_vid

        return rec_vid, inp_rec_vid, pred_vid, inp_pred_vid


    def inpaint(self, real_vid, real_lyt, real_flow, log_dic, is_eval, dtype, global_iter):
        nll_loss = torch.tensor(0., requires_grad=True).cuda()
        adv_loss = torch.tensor(0., requires_grad=True).cuda()
        inpainting_scaler = self.img_inpainting_scaler if dtype == "img" else self.vid_inpainting_scaler
        losses = self.opt.img_inpainting_losses if dtype == "img" else self.opt.vid_inpainting_losses
        ctx_len = 1 if dtype == "img" else self.opt.ctx_len

        if not is_eval:
            self.opt_ii.zero_grad()

        with self.amp_decorator(), torch.set_grad_enabled(not is_eval):

            with torch.no_grad():
                # merge modalities into a single input
                real_input = torch.cat([real_vid if self.opt.input_rgb else real_vid[:, :, :0],
                                        real_lyt if self.opt.input_lyt else real_vid[:, :, :0],
                                        real_flow if self.opt.input_flow else real_vid[:, :, :0]], dim=2)

                # encode input to feature space
                x = self.net_pe(input=real_input, mode="encode_input") # B T L C

                # extract layer features from context video features
                x_ctx = x[:, :ctx_len]
                x_obj, x_bg, cls = self.net_pe(x=x_ctx, mode="estimate_layer")

                # extract pose from layer features and video features
                obj_pose, bg_pose, occ_score, pts_rest_obj, pts_rest_bg, last_obj, last_bg = self.net_pe(x=x, x_obj=x_obj, x_bg=x_bg, mode="estimate_pose")
                occ, obj_alpha, bg_alpha, grid = self.net_pe(x_obj=x_obj, obj_pose=obj_pose, bg_pose=bg_pose, occ_score=occ_score, mode="estimate_alpha_grid_occ")

                # reconstruct video
                real_input = torch.cat([real_vid, real_lyt], dim=2)

            multi_optim = False
            if multi_optim:
                inp_vid = torch.zeros_like(real_vid[:, 1:])
                alpha_ctx = torch.zeros_like(real_vid[:, 1:, :1])
                for i in range(1, real_vid.size(1)):
                    self.opt_ii.zero_grad()
                    loss = torch.tensor(0., requires_grad=True).cuda()
                    with torch.no_grad():
                        ctx_ts = torch.arange(i, device=x.device, dtype=torch.int64)
                        ctx_ts = ctx_ts.view(1, -1, 1).expand(x.size(0), -1, 1)
                        pred_ts = torch.arange(i, i + 1, device=x.device, dtype=torch.int64)

                        rec_output, _, _, _, _, raw_output, alpha_ctxi = self.net_pe(input=real_input, grid=grid, occ=occ, obj_alpha=obj_alpha,
                                                                                    bg_alpha=bg_alpha, ctx_ts=ctx_ts, pred_ts=pred_ts, cls=cls,
                                                                                    mode="decode_output")
                        alpha_ctx[:, [i - 1]] = (((alpha_ctxi + 1) / 2).sum(dim=3, keepdim=True).max(dim=1)[0] * 2 - 1).detach()
                    inp_img = self.net_ii(raw_output)
                    if "sharp_vid" in losses:
                        loss += (inp_img - real_vid[:, [i]]).abs().mean()
                    if "lpips_vid" in losses:
                        if self.lpips_vgg is None:
                            self.lpips_vgg = lpips.LPIPS(net='vgg').cuda()
                        loss += self.lpips_vgg(inp_img[:, 0], real_vid[:, i]).mean()
                    if not is_eval:
                        inpainting_scaler.scale(loss).backward()
                        grad_clip(self.net_ii, self.opt.clip_value)
                        self.opt_ii.step()
                    inp_vid[:, [i - 1]] = inp_img.detach()
                real_vid = real_vid[:, 1:]
                rec_vid = None
            else:
                with torch.no_grad():
                    ctx_ts = torch.arange(ctx_len, device=x.device, dtype=torch.int64)
                    ctx_ts = ctx_ts.view(1, -1, 1).expand(x.size(0), -1, x.size(1) - ctx_len)
                    pred_ts = torch.arange(ctx_len, real_vid.size(1), device=x.device, dtype=torch.int64)

                    rec_output, _, _, _, _, raw_output, alpha_ctx = self.net_pe(input=real_input, grid=grid, occ=occ, obj_alpha=obj_alpha, bg_alpha=bg_alpha, ctx_ts=ctx_ts, pred_ts=pred_ts, cls=cls, mode="decode_output")
                    alpha_ctx = ((alpha_ctx + 1) / 2).sum(dim=3, keepdim=True).max(dim=1)[0] * 2 - 1
                rec_vid = rec_output[:, :, :3] if rec_output is not None else None
                inp_output = self.net_ii(raw_output)
                real_vid, inp_vid = real_vid[:, ctx_len:], inp_output[:, :, :3]

            log_dic["vid"]["rec_vid"] = rec_vid
            log_dic["vid"]["inp_vid"] = inp_vid
            log_dic["vid"]["real_vid"] = real_vid
            log_dic["vid"]["alpha_ctx"] = alpha_ctx
            if self.opt.use_disocc:
                log_dic["vid"]["disocc"] = raw_output[:, :, :, -1:].max(dim=1)[0] * 2 - 1

            # sharp vid
            log_dic["scalar"]["sharp_vid"] = (inp_vid - real_vid).abs().mean()
            log_dic["scalar"]["sharp_rec"] = (rec_vid - real_vid).abs().mean() if rec_vid is not None else None
            if "sharp_vid" in losses:
                nll_loss += log_dic["scalar"]["sharp_vid"] * self.opt.lambda_sharp_vid
            log_dic["scalar"]["sharp_delta"] = log_dic["scalar"]["sharp_vid"] - log_dic["scalar"]["sharp_rec"] if rec_vid is not None else None

            # lpips vid
            if "lpips_vid" in losses:
                if self.lpips_vgg is None:
                    self.lpips_vgg = lpips.LPIPS(net='vgg').cuda()
                inp_img, _ = flatten(inp_vid, ndim=4)
                real_img, _ = flatten(real_vid, ndim=4)
                log_dic["scalar"]["lpips_vid"] = self.lpips_vgg(inp_img, real_img).mean()
                nll_loss += log_dic["scalar"]["lpips_vid"] * self.opt.lambda_lpips_vid

            # adv vid
            real_img, inp_img = real_vid[:, 0], inp_vid[:, 0]
            if "adv" in losses:
                requires_grad(self.net_id, False)
                fake_score = self.net_id(inp_img)
                log_dic["scalar"]["adv"] = self.gan_loss_id.generator_loss_logits(fake_score).sum()
                adv_loss += log_dic["scalar"]["adv"]
                if self.opt.use_adaptive_lambda and not is_eval:
                    lambda_adv = get_adaptive_lambda(nll_loss, adv_loss, self.net_ii.module.get_last_layer())
                else:
                    lambda_adv = self.opt.lambda_adv
                adv_loss = adv_loss * lambda_adv

            loss = nll_loss + adv_loss
            log_dic["scalar"]["loss"] = loss

            if not multi_optim:
                nanloss = torch.stack(self.engine.all_gather_tensor(loss.isnan())).any()
                if nanloss:
                    self.nancount += 1
                    print(f"[{global_iter}] skipping because loss is nan")
                    if self.nancount > 10:
                        print(log_dic["scalar"])
                        raise ValueError
                else:
                    self.nancount = 0

                # optim step
                if not nanloss and not is_eval:
                    inpainting_scaler.scale(loss).backward()
                    grad_clip(self.net_ii, self.opt.clip_value)
                    self.opt_ii.step()

        # detach tensors from computational graph
        log_dic = {k1: {k2: v.detach() for k2, v in d.items() if v is not None and 0 not in v.size()} for k1, d in log_dic.items()}
        return real_img, inp_img.detach(), log_dic


    def discriminate_inpainting(self, real_img, inp_vid, log_dic, is_eval, dtype, global_iter):
        loss = torch.tensor(0., requires_grad=True).cuda()
        inpainting_discriminator_scaler = self.img_inpainting_discriminator_scaler if dtype == "img" else self.vid_inpainting_discriminator_scaler

        if not is_eval:
            self.opt_id.zero_grad()

        with self.amp_decorator(), torch.set_grad_enabled(not is_eval):
            requires_grad(self.net_id, True)
            fake_score = self.net_id(inp_vid)
            real_score = self.net_id(real_img)

            if not is_eval:
                log_dic["scalar"]["dis"] = self.gan_loss_id.discriminator_loss_logits(real_score, fake_score, real_img, inp_vid)
                loss = log_dic["scalar"]["dis"] * self.opt.lambda_dis
            else:
                log_dic["scalar"]["real_score"] = torch.mean(real_score)
                log_dic["scalar"]["fake_score"] = torch.mean(fake_score)

            log_dic["scalar"]["dis_loss"] = loss

            nanloss = torch.stack(self.engine.all_gather_tensor(loss.isnan())).any()
            if nanloss:
                self.nancount += 1
                print(f"[{global_iter}] skipping because loss is nan")
                if self.nancount > 10:
                    print(log_dic["scalar"])
                    raise ValueError
            else:
                self.nancount = 0
                # optim step
                if not nanloss and not is_eval:
                    inpainting_discriminator_scaler.scale(loss).backward()
                    grad_clip(self.net_id, self.opt.clip_value)
                    self.opt_id.step()

        # detach tensors from computational graph
        log_dic = {k1: {k2: v.detach() for k2, v in d.items() if v is not None and 0 not in v.size()} for k1, d in log_dic.items()}
        return log_dic


    def generate_pose(self, real_vid, real_lyt, real_flow, log_dic, is_eval, dtype, log):
        nll_loss = torch.tensor(0., requires_grad=True).cuda()
        adv_loss = torch.tensor(0., requires_grad=True).cuda()
        pose_generator_scaler = self.img_pose_generator_scaler if dtype == "img" else self.vid_pose_generator_scaler
        min_ctx_length = self.opt.min_ctx_length_img if dtype == "img" else self.opt.min_ctx_length_vid
        max_ctx_length = self.opt.max_ctx_length_img if dtype == "img" else self.opt.max_ctx_length_vid
        ctx_len = 1 if dtype == "img" else self.opt.ctx_len
        losses = self.opt.img_pose_generator_losses if dtype == "img" else self.opt.vid_pose_generator_losses

        if not is_eval:
            self.opt_pg.zero_grad()

        with self.amp_decorator(), torch.set_grad_enabled(not is_eval):
            # create ctx vid
            B, T, _, _, _ = real_vid.shape
            ctx_size = torch.randint(low=min_ctx_length, high=max_ctx_length + 1, size=(B, 1), device=real_vid.device)
            ctx_mask = torch.arange(0, T, device=real_vid.device).view(1, T).expand(B, -1) < ctx_size

            with torch.no_grad():
                # merge modalities into a single input
                real_input = torch.cat([real_vid if self.opt.input_rgb else real_vid[:, :, :0],
                                        real_lyt if self.opt.input_lyt else real_vid[:, :, :0],
                                        real_flow if self.opt.input_flow else real_vid[:, :, :0]], dim=2)

                # encode input to feature space
                x = self.net_pe(input=real_input, mode="encode_input")  # B T L C

                # extract layer features from context video features
                x_ctx = x[:, :ctx_len]
                x_obj, x_bg, cls = self.net_pe(x=x_ctx, mode="estimate_layer")

                # extract pose from layer features and video features
                real_obj_pose, real_bg_pose, real_occ_score, _, _, last_obj, last_bg = self.net_pe(x=x, x_obj=x_obj, x_bg=x_bg, mode="estimate_pose")

            # predict pose from ctx
            pred_obj_pose, pred_bg_pose, pred_occ_score = self.net_pg(real_obj_pose, real_bg_pose, real_occ_score, x_obj, x_bg, last_obj, last_bg, ctx_mask=ctx_mask)

            # reconstruct video
            if log or (is_eval and self.opt.viz):
                with torch.no_grad():
                    # real_input = real_input[:, :, :-2]
                    real_input = torch.cat([real_vid, real_lyt], dim=2)
                    ctx_ts = torch.arange(ctx_len, device=x.device, dtype=torch.int64)
                    ctx_ts = ctx_ts.view(1, -1, 1).expand(x.size(0), -1, x.size(1))
                    pred_ts = torch.arange(real_vid.size(1), device=x.device, dtype=torch.int64)
                    occ, obj_alpha, bg_alpha, grid = self.net_pe(x_obj=x_obj, obj_pose=real_obj_pose, bg_pose=real_bg_pose,
                                                                 occ_score=real_occ_score, mode="estimate_alpha_grid_occ")
                    rec_output, _, rec_output_alpha, rec_output_alpha_flt, _, _, _ = self.net_pe(input=real_input, grid=grid, occ=occ, obj_alpha=obj_alpha,
                                                                                              bg_alpha=bg_alpha, ctx_ts=ctx_ts, pred_ts=pred_ts, cls=cls,
                                                                                              mode="decode_output")
                    occ, obj_alpha, bg_alpha, grid = self.net_pe(x_obj=x_obj, obj_pose=pred_obj_pose, bg_pose=pred_bg_pose,
                                                                 occ_score=pred_occ_score, mode="estimate_alpha_grid_occ")
                    pred_output, _, pred_output_alpha, pred_output_alpha_flt, _, _, _ = self.net_pe(input=real_input, grid=grid, occ=occ, obj_alpha=obj_alpha,
                                                                                                 bg_alpha=bg_alpha, ctx_ts=ctx_ts, pred_ts=pred_ts, cls=cls,
                                                                                                 mode="decode_output")
                    rec_vid = rec_output[:, :, :3]
                    pred_vid = pred_output[:, :, :3]
                    log_dic["vid"]["rec_vid"] = rec_vid
                    log_dic["vid"]["pred_vid"] = pred_vid
                    log_dic["ctx_mask"]["pred_vid"] = ctx_mask
                    log_dic["vid"]["real_vid"] = real_vid
                    log_dic["obj_lyt"]["rec_obj_lyt"] = rec_output_alpha
                    log_dic["obj_lyt"]["rec_obj_lyt_flt"] = rec_output_alpha_flt
                    log_dic["obj_lyt"]["pred_obj_lyt"] = pred_output_alpha
                    log_dic["obj_lyt"]["pred_obj_lyt_flt"] = pred_output_alpha_flt

            log_dic["pts"]["real_obj_pts"] = real_obj_pose
            log_dic["pts"]["real_bg_pts"] = real_bg_pose
            log_dic["pts"]["pred_obj_pts"] = pred_obj_pose
            log_dic["pts"]["pred_bg_pts"] = pred_bg_pose

            log_dic["scalar"]["rec_obj_pose"] = (to_ctx(real_obj_pose, ~ctx_mask) - to_ctx(pred_obj_pose, ~ctx_mask)).abs().mean()
            log_dic["scalar"]["rec_bg_pose"] = (to_ctx(real_bg_pose, ~ctx_mask) - to_ctx(pred_bg_pose, ~ctx_mask)).abs().mean()
            log_dic["scalar"]["rec_occ_score"] = (to_ctx(real_occ_score, ~ctx_mask) - to_ctx(pred_occ_score, ~ctx_mask)).abs().mean()
            if "rec_obj_pose" in losses:
                nll_loss += log_dic["scalar"]["rec_obj_pose"] * self.opt.lambda_rec_obj_pose
            if "rec_bg_pose" in losses:
                nll_loss += log_dic["scalar"]["rec_bg_pose"] * self.opt.lambda_rec_bg_pose
            if "rec_occ_score" in losses:
                nll_loss += log_dic["scalar"]["rec_occ_score"] * self.opt.lambda_rec_occ_score

            loss = nll_loss + adv_loss
            log_dic["scalar"]["loss"] = loss

            nanloss = torch.stack(self.engine.all_gather_tensor(loss.isnan())).any()
            if nanloss:
                self.nancount += 1
                print(f"[iter] skipping because loss is nan")
                if self.nancount > 10:
                    print(log_dic["scalar"])
                    raise ValueError
            else:
                self.nancount = 0

            # optim step
            if not nanloss and not is_eval:
                pose_generator_scaler.scale(loss).backward()
                grad_clip(self.net_pg, self.opt.clip_value)
                self.opt_pg.step()

        # detach tensors from computational graph
        log_dic = {k1: {k2: v.detach() for k2, v in d.items() if v is not None and 0 not in v.size()} for k1, d in log_dic.items()}
        return log_dic


    def extract_object(self, real_vid, real_lyt, real_flow, log_dic, is_eval, dtype, global_iter, log):
        nll_loss = torch.tensor(0., requires_grad=True).cuda()
        object_extractor_scaler = self.img_object_extractor_scaler if dtype == "img" else self.vid_object_extractor_scaler
        losses = self.opt.img_object_extractor_losses if dtype == "img" else self.opt.vid_object_extractor_losses
        ctx_len = 1 if dtype == "img" else self.opt.ctx_len

        if not is_eval:
            self.opt_pe.zero_grad()

        with self.amp_decorator(), torch.set_grad_enabled(not is_eval):
            # merge modalities into a single input
            if not is_eval and self.opt.drop_input_p > 0:
                B, T = real_vid.shape[:2]
                mul_rgb = torch.rand(B, T, device=real_vid.device) > self.opt.drop_input_p
                mul_lyt = torch.rand(B, T, device=real_vid.device) > self.opt.drop_input_p
                mul_flow = torch.rand(B, T, device=real_vid.device) > self.opt.drop_input_p
                if self.opt.input_rgb:
                    mul_rgb = (~ mul_flow) & (~ mul_lyt) & (~ mul_rgb) | mul_rgb # guarantee that at least one input is not masked
                elif self.opt.input_flow:
                    mul_flow = (~ mul_flow) & (~ mul_lyt) | mul_flow  # guarantee that at least one input is not masked
                reshape = lambda x: x.view(B, T, 1, 1, 1).float()
                mul_rgb, mul_lyt, mul_flow = reshape(mul_rgb), reshape(mul_lyt), reshape(mul_flow)
            else:
                mul_rgb, mul_lyt, mul_flow = 1., 1., 1.
            real_input = torch.cat([real_vid * mul_rgb if self.opt.input_rgb else real_vid[:, :, :0],
                                    real_lyt * mul_lyt if self.opt.input_lyt else real_vid[:, :, :0],
                                    real_flow * mul_flow if self.opt.input_flow else real_vid[:, :, :0]], dim=2)

            # encode input to feature space
            x = self.net_pe(input=real_input, mode="encode_input") # B T L C

            # extract layer features from context video features
            x_ctx = x[:, :ctx_len]
            x_obj, x_bg, cls = self.net_pe(x=x_ctx, mode="estimate_layer")

            # extract pose from layer features and video features
            obj_pose, bg_pose, occ_score, pts_rest_obj, pts_rest_bg, last_obj, last_bg = self.net_pe(x=x, x_obj=x_obj, x_bg=x_bg, mode="estimate_pose")
            occ, obj_alpha, bg_alpha, grid = self.net_pe(x_obj=x_obj, obj_pose=obj_pose, bg_pose=bg_pose, occ_score=occ_score, mode="estimate_alpha_grid_occ")

            log_dic["vid"]["occ"] = occ.unsqueeze(2) * 2 - 1

            # reconstruct video
            real_input = torch.cat([real_vid, real_lyt], dim=2)

            if self.opt.ctx_mode == "full":
                ctx_ts = torch.arange(x.size(1), device=x.device, dtype=torch.int64)
                ctx_ts = ctx_ts.view(1, -1, 1).expand(x.size(0), -1, x.size(1))
            if self.opt.ctx_mode == "prev" or self.opt.ctx_mode == "prev_rd":
                ctx_ts = torch.roll(torch.arange(x.size(1), device=x.device, dtype=torch.int64), shifts=1, dims=0)
                ctx_ts = ctx_ts.view(1, 1, -1).expand(x.size(0), -1, -1)
            if self.opt.ctx_mode == "prev_rd":
                rd_ts = torch.randint(low=0, high=x.size(1), size=[x.size(0), self.opt.rd_ctx_num, x.size(1)], device=x.device)
                ctx_ts = torch.cat([ctx_ts, rd_ts], dim=1)
            pred_ts = torch.arange(real_vid.size(1), device=x.device, dtype=torch.int64)

            rec_output, rec_flow, rec_output_alpha, output_alpha_flt, raw_alpha, raw_output, _ = self.net_pe(input=real_input, grid=grid, occ=occ, obj_alpha=obj_alpha, bg_alpha=bg_alpha, ctx_ts=ctx_ts, pred_ts=pred_ts, cls=cls, mode="decode_output")

            if self.opt.ctx_mode == "full":
                rec_flow = rec_flow[:, :, 1:]
                idx = torch.arange(x.size(1) - 1, device=x.device, dtype=torch.int64)
                idx = idx.view(1, 1, -1, 1, 1, 1).expand(x.size(0), -1, -1, *rec_flow.shape[-3:])
                rec_flow = rec_flow.gather(dim=1, index=idx)[:, 0]
            else:
                rec_flow = rec_flow[:, 0, 1:]

            rec_vid, rec_lyt = rec_output[:, :, :3], rec_output[:, :, 3:]
            log_dic["vid"]["rec_vid"] = rec_vid
            log_dic["vid"]["real_vid"] = real_vid
            log_dic["sem_lyt"]["rec_lyt"] = rec_lyt
            log_dic["sem_lyt"]["real_lyt"] = real_lyt
            log_dic["obj_lyt"]["obj_lyt"] = rec_output_alpha
            log_dic["obj_lyt"]["obj_lyt_flt"] = output_alpha_flt
            if self.opt.use_disocc:
                log_dic["vid"]["disocc"] = raw_output[:, :, :, -1:].max(dim=1)[0] * 2 - 1

            if self.opt.swap_flt:
                rec_output_alpha = output_alpha_flt

            # same mean motion in layers
            a = (rec_output_alpha[:, :, 1:] + 1) / 2 + 1e-6 # B T No H W
            sum_a = a.sum(dim=3, keepdim=True).sum(dim=4, keepdim=True) # B T No 1 1
            mean_flow = (real_flow.unsqueeze(2) * a.unsqueeze(3)).sum(dim=4, keepdim=True).sum(dim=5, keepdim=True) / sum_a.unsqueeze(3) # B T No 2 1 1
            log_dic["scalar"]["obj_flow"] = (a * (real_flow.unsqueeze(2) - mean_flow).abs().sum(dim=3)).mean()
            if "obj_flow" in losses:
                nll_loss += log_dic["scalar"]["obj_flow"] * self.opt.lambda_obj_flow

            # maintain active cluster
            cluster_size = ((rec_output_alpha[:, :, 1:] + 1) / 2).permute(0, 1, 3, 4, 2) # B T H W No
            log_dic["scalar"]["activity"] = (-cluster_size.reshape(-1, self.opt.num_obj).mean(0)).topk(dim=0, k=self.opt.num_obj // 4)[0].mean()
            log_dic["scalar"]["topactivity"] = (-cluster_size.reshape(cluster_size.size(0), -1, self.opt.num_obj).mean(1).topk(dim=0, k=cluster_size.size(0) // 4)[0]).topk(dim=1, k=self.opt.num_obj // 4)[0].mean()
            if "activity" in losses:
                img_mul = self.opt.img_mul_act_reg if dtype == "img" else 1.
                nll_loss += log_dic["scalar"]["activity"] * self.opt.lambda_activity * img_mul
            if "topactivity" in losses:
                img_mul = self.opt.img_mul_act_reg if dtype == "img" else 1.
                nll_loss += log_dic["scalar"]["topactivity"] * self.opt.lambda_activity * img_mul

            # entropy
            entropy = (rec_output_alpha + 1) / 2  # B T No+1 H W
            entropy = F.normalize(entropy + 1e-6, p=1, dim=2)
            entropy = -torch.sum(torch.mul(entropy, torch.log(entropy + 1e-6)), dim=2, keepdim=True) / 0.37
            log_dic["vid"]["entropy"] = entropy * 2 - 1
            entropy_flt = (output_alpha_flt + 1) / 2  # B T No+1 H W
            entropy_flt = F.normalize(entropy_flt + 1e-6, p=1, dim=2)
            entropy_flt = -torch.sum(torch.mul(entropy_flt, torch.log(entropy_flt + 1e-6)), dim=2, keepdim=True) / 0.37
            thresh = 0.999
            lyt_edge_mask = (blur(real_lyt / 10 + 1 / 2, sigma=2, kernel_size=3).max(dim=2, keepdim=True)[0] > thresh).float()
            entropy_flt_edge = entropy_flt * lyt_edge_mask
            log_dic["vid"]["entropy_flt"] = entropy_flt * 2 - 1
            log_dic["vid"]["entropy_flt_edge"] = entropy_flt_edge * 2 - 1
            log_dic["vid"]["lyt_edge_mask"] = lyt_edge_mask * 2 - 1
            log_dic["scalar"]["ent"] = entropy.mean()
            log_dic["scalar"]["ent_flt"] = entropy_flt.mean()
            log_dic["scalar"]["ent_flt_edge"] = entropy_flt_edge.mean()
            if "ent" in losses:
                nll_loss += log_dic["scalar"]["ent"] * self.opt.lambda_ent
            if "ent_flt" in losses:
                nll_loss += log_dic["scalar"]["ent_flt"] * self.opt.lambda_ent_flt
            if "ent_flt_edge" in losses:
                nll_loss += log_dic["scalar"]["ent_flt_edge"] * self.opt.lambda_ent_flt_edge

            # extract flow edges
            flow_edge, dominant_flow = self.edge(real_flow, blur=False) # B T 1 H W
            flow_edge = (flow_edge > self.opt.flow_thresh).float()

            # extract moving object
            fg_prop = (real_lyt[:, :, self.opt.fg_idx] / 10 + 1 / 2).sum(dim=2, keepdim=True)  # B T 1 H W
            nofg_prop = 1 - fg_prop
            bg_prop = (real_lyt[:, :, self.opt.bg_idx] / 10 + 1 / 2).sum(dim=2, keepdim=True)  # B T 1 H W
            nobg_prop = 1 - bg_prop
            nofg_flow = blur(torch.cat([nofg_prop, nofg_prop * real_flow], dim=2), self.opt.blur_sigma)
            sum_nofg_flow = nofg_flow[:, :, :1] + (nofg_flow[:, :, :1] == 0).float()
            mean_nofg_flow = nofg_flow[:, :, 1:] / sum_nofg_flow
            mean_bg_flow = mean_nofg_flow
            log_dic["flow"]["mean_bg_flow"] = mean_bg_flow
            delta_flow = fg_prop * (real_flow - mean_bg_flow).abs().sum(dim=2, keepdim=True)
            mov_obj_mask = delta_flow > self.opt.mov_obj_thresh # 0.075
            if self.opt.use_dominant_flow_other:
                other_prop = (real_lyt[:, :, self.opt.other_idx] / 10 + 1 / 2).sum(dim=2, keepdim=True)  # B T 1 H W
                mov_obj_mask = torch.max(mov_obj_mask.float(), other_prop * dominant_flow * flow_edge)
            if self.opt.use_flow_nobg:
                thresh = 0.1
                flow_mask = (flow_edge > thresh) & (nobg_prop > 0)
                mov_obj_mask = mov_obj_mask | flow_mask
            mov_obj_mask = mov_obj_mask.float()
            fg_mask = ((rec_output_alpha[:, :, 1:] + 1) / 2).sum(dim=2, keepdim=True)
            found_obj = -fg_mask # B T 1 H W
            found_obj_flt = -((output_alpha_flt[:, :, 1:] + 1) / 2).sum(dim=2, keepdim=True)
            mov_obj = mov_obj_mask * 2 - 1
            mov_obj[mov_obj < 0] *= self.opt.reg_bg_mul
            if self.opt.use_fg:
                mov_obj[(mov_obj < 0) & (fg_prop > 0)] = 0
            if self.opt.use_nobg:
                mov_obj[(mov_obj < 0) & (nobg_prop > 0)] = 0
            if self.opt.use_nobg_edge:
                thresh = 0.1
                mov_obj[(mov_obj < 0) & (nobg_prop > 0) & (flow_edge > thresh)] = self.opt.nobg_edge_mul

            found_obj = blur(found_obj, self.opt.blur_sigma) if self.opt.blur_alpha else found_obj
            mov_obj = blur(mov_obj, self.opt.blur_sigma) if self.opt.blur_alpha else mov_obj
            log_dic["vid"]["mov_obj"] = mov_obj
            log_dic["vid"]["fg_prop"] = fg_prop
            log_dic["vid"]["found_obj"] = found_obj
            log_dic["vid"]["found_obj_flt"] = found_obj_flt
            log_dic["scalar"]["abs_mov"] = (mov_obj_mask - fg_mask).abs().mean()
            log_dic["scalar"]["reg_mov"] = (mov_obj * found_obj).mean()
            log_dic["scalar"]["reg_fg"] = (-found_obj * (1 - fg_prop)).mean()
            if "abs_mov" in losses:
                nll_loss += log_dic["scalar"]["abs_mov"] * self.opt.lambda_abs_mov
            if "reg_mov" in losses:
                warmup_mul, warmup_iter = self.opt.warmup_reg_mov_mul, self.opt.warmup_reg_mov_iter
                mul = max(1, warmup_mul * (1 - global_iter / warmup_iter)) if warmup_iter > 0 else 1
                img_mul = self.opt.img_mul_act_reg if dtype == "img" else 1.
                nll_loss += log_dic["scalar"]["reg_mov"] * self.opt.lambda_reg_mov * mul * img_mul
            if "reg_fg" in losses:
                nll_loss += log_dic["scalar"]["reg_fg"] * self.opt.lambda_reg_fg
            if "ce_mean_lyt_obj" in losses:
                nll_loss += log_dic["scalar"]["ce_mean_lyt_obj"] * self.opt.lambda_ce_mean_lyt_obj

            grid = self.net_pe.module.warper.src_grid # 1 H W 2
            B, T, Lo, No, _ = obj_pose.shape
            obj_grid = obj_pose.view(*obj_pose.shape[:3], *self.opt.obj_shape, 2) # B T No ho wo 2
            obj_cell = (obj_grid[:, :, :, 1:, 1:] + obj_grid[:, :, :, 1:, :-1] + obj_grid[:, :, :, :-1, 1:] + obj_grid[:, :, :, :-1, :-1]) / 4
            obj_center = obj_grid.view(*obj_pose.shape[:3], -1, 2).mean(dim=3) # B T No 2
            obj_cell_dis = (grid ** 2).sum(dim=-1).view(1, -1) + (obj_cell ** 2).sum(dim=-1).view(-1, 1) - 2 * obj_cell.view(-1, 2) @ grid.view(-1, 2).t()
            obj_cell_dis = obj_cell_dis.view(*obj_grid.shape[:2], self.opt.num_obj, -1, *grid.shape[1:3]) # B T No (ho-1)*(wo-1) H W
            obj_cell_dis = obj_cell_dis.sum(dim=3) # B T No H W
            obj_center_dis = (grid ** 2).sum(dim=-1).view(1, -1) + (obj_center ** 2).sum(dim=-1).view(-1, 1) - 2 * obj_center.view(-1, 2) @ grid.view(-1, 2).t()
            obj_center_dis = obj_center_dis.view(*obj_grid.shape[:2], self.opt.num_obj, *grid.shape[1:3])  # B T No H W
            mov_obj_mask = blur(mov_obj_mask, self.opt.blur_sigma) if self.opt.blur_alpha else mov_obj_mask
            fg_mask = blur(fg_mask, self.opt.blur_sigma) if self.opt.blur_alpha else fg_mask
            log_dic["scalar"]["cell_dis"] = ((mov_obj_mask + self.opt.cell_dis_eps) * (1 - fg_mask) * obj_cell_dis).min(dim=2)[0].mean()

            log_dic["scalar"]["center_dis"] = (mov_obj_mask * obj_center_dis).min(dim=2)[0].mean()

            if "cell_dis" in losses:
                nll_loss += (log_dic["scalar"]["cell_dis"]) * self.opt.lambda_cell_dis
            if "center_dis" in losses:
                nll_loss += log_dic["scalar"]["center_dis"] * self.opt.lambda_center_dis

            # reconstruct flow
            log_dic["flow"]["real_flow"] = real_flow
            log_dic["flow"]["rec_flow"] = rec_flow
            log_dic["scalar"]["l1_flow"] = (real_flow[:, 1:] - rec_flow).abs().mean()
            if "l1_flow" in losses:
                warmup_mul, warmup_iter = self.opt.warmup_l1_flow_mul, self.opt.warmup_l1_flow_iter
                mul = min(warmup_mul, 1 + (warmup_mul - 1) * (global_iter / warmup_iter)) if warmup_iter > 0 else 1
                nll_loss += log_dic["scalar"]["l1_flow"] * self.opt.lambda_l1_flow * mul
            if "l1_flow_mov_obj" in losses:
                nll_loss += log_dic["scalar"]["l1_flow_mov_obj"] * self.opt.lambda_l1_flow_mov_obj
            if "l1_flow_other" in losses:
                nll_loss += log_dic["scalar"]["l1_flow_other"] * self.opt.lambda_l1_flow_other

            # compute cross entropy between real and reconstructed layouts
            log_dic["scalar"]["ce_lyt"] = self.ce_loss(flatten(rec_lyt, ndim=4)[0], flatten(real_lyt, ndim=4)[0].max(dim=-3)[1]).mean()
            log_dic["scalar"]["ce_lyt_obj"] = (self.ce_loss(flatten(fg_mask * rec_lyt, ndim=4)[0], flatten(real_lyt, ndim=4)[0].max(dim=-3)[1]) * flatten(mov_obj_mask, ndim=3)[0]).mean()
            log_dic["scalar"]["soft_ce_lyt"] = self.ce_loss(flatten(rec_lyt, ndim=4)[0], flatten(real_lyt / 10 + 1 / 2, ndim=4)[0]).mean()
            if "ce_lyt" in losses:
                nll_loss += log_dic["scalar"]["ce_lyt"] * self.opt.lambda_ce_lyt
            if "ce_lyt_obj" in losses:
                nll_loss += log_dic["scalar"]["ce_lyt_obj"] * self.opt.lambda_ce_lyt_obj
            if "soft_ce_lyt" in losses:
                nll_loss += log_dic["scalar"]["soft_ce_lyt"] * self.opt.lambda_soft_ce_lyt

            # match vid
            log_dic["scalar"]["sharp_vid"] = (rec_vid - real_vid).abs().mean()
            real_vid = blur(real_vid, self.opt.blur_sigma) if self.opt.blur_pxl else real_vid
            rec_vid = blur(rec_vid, self.opt.blur_sigma) if self.opt.blur_pxl else rec_vid
            pxl = real_vid - rec_vid
            pxl = pxl.abs().flatten(start_dim=1).mean(-1) if self.opt.l1_pxl else (pxl ** 2).flatten(start_dim=1).mean(-1)
            log_dic["scalar"]["pxl_vid"] = pxl.mean()
            if "pxl_vid" in losses:
                mul = min(1, global_iter / self.opt.warmup_pxl_vid_iter) if self.opt.warmup_pxl_vid_iter > 0 else 1
                mul = math.sin(mul * math.pi / 2) if self.opt.cosine_warmup_pxl_vid else mul
                nll_loss += log_dic["scalar"]["pxl_vid"] * self.opt.lambda_pxl_vid * mul
            if "sharp_vid" in losses:
                mul = min(1, global_iter / self.opt.warmup_sharp_vid_iter) if self.opt.warmup_sharp_vid_iter > 0 else 1
                nll_loss += log_dic["scalar"]["sharp_vid"] * self.opt.lambda_sharp_vid * mul

            # regularize grid
            log_dic["pts"]["obj_pts"] = obj_pose
            log_dic["pts"]["bg_pts"] = bg_pose
            log_dic["scalar"]["pts_reg_obj"] = compute_pts_regularization(obj_pose, *self.opt.obj_shape)
            if "pts_reg_obj" in losses:
                nll_loss += log_dic["scalar"]["pts_reg_obj"] * self.opt.lambda_pts_reg
            if self.opt.has_bg:
                log_dic["scalar"]["pts_reg_bg"] = compute_pts_regularization(bg_pose, *self.opt.latent_shape)
            if "pts_reg_bg" in losses:
                nll_loss += log_dic["scalar"]["pts_reg_bg"] * self.opt.lambda_pts_reg

            # restore to origin
            if self.opt.ada_pts_rest:
                log_dic["scalar"]["pts_rest_obj"] = (pts_rest_obj * pxl.view(-1, 1)).mean()
                if self.opt.has_bg and not self.opt.fix_bg:
                    log_dic["scalar"]["pts_rest_bg"] = (pts_rest_bg * pxl.view(-1, 1)).mean()
            elif self.opt.ada_pts_rest_detach:
                log_dic["scalar"]["pts_rest_obj"] = (pts_rest_obj * pxl.detach().view(-1, 1)).mean()
                if self.opt.has_bg and not self.opt.fix_bg:
                    log_dic["scalar"]["pts_rest_bg"] = (pts_rest_bg * pxl.detach().view(-1, 1)).mean()
            else:
                log_dic["scalar"]["pts_rest_obj"] = pts_rest_obj.mean()
                if self.opt.has_bg and not self.opt.fix_bg:
                    log_dic["scalar"]["pts_rest_bg"] = pts_rest_bg.mean()
            if "pts_rest_obj" in losses:
                nll_loss += log_dic["scalar"]["pts_rest_obj"] * self.opt.lambda_pts_rest
            if "pts_rest_bg" in losses:
                nll_loss += log_dic["scalar"]["pts_rest_bg"] * self.opt.lambda_pts_rest

            loss = nll_loss
            log_dic["scalar"]["loss"] = loss

            nanloss = torch.stack(self.engine.all_gather_tensor(loss.isnan())).any()
            if nanloss:
                self.nancount += 1
                print(f"[{global_iter}] skipping because loss is nan")
                # raise ValueError
                if self.nancount > 10:
                    print(log_dic["scalar"])
                    raise ValueError
            else:
                self.nancount = 0

            # optim step
            if not nanloss and not is_eval:
                object_extractor_scaler.scale(loss).backward()
                grad_clip(self.net_pe, self.opt.clip_value)
                self.opt_pe.step()

        # detach tensors from computational graph
        log_dic = {k1: {k2: v.detach() for k2, v in d.items() if v is not None and 0 not in v.size()} for k1, d in log_dic.items()}
        return log_dic


def requires_grad(net, flag=True):
    if net is not None:
        for p in net.parameters():
            p.requires_grad = flag


def grad_clip(net, value):
    if value > 0:
        torch.nn.utils.clip_grad_norm_(net.parameters(), value)


# Adapted from https://github.com/rwightman/pytorch-image-models
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay, no_decay = [], []
    decay_name, no_decay_name = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name.replace("module.", "") in skip_list:
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            decay.append(param)
            decay_name.append(name)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def get_adaptive_lambda(nll_loss, adv_loss, last_layer):
    adv_grads = torch.autograd.grad(adv_loss, last_layer, retain_graph=True)[0]
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    adaptive_lambda = torch.norm(nll_grads) / (torch.norm(adv_grads) + 1e-4)
    adaptive_lambda = torch.clamp(adaptive_lambda, 0.0, 1e4).detach()
    return adaptive_lambda


def blur(vid, sigma=3.0, kernel_size=23):
    img = vid.view(-1, *vid.shape[-3:])
    img = GaussianBlur(kernel_size=kernel_size, sigma=sigma)(img)
    vid = img.view_as(vid)
    return vid


def compute_pts_regularization(pose, num_pts_h, num_pts_w):
    pts = pose.view(-1, num_pts_h, num_pts_w, 2)
    pts_reg_h = ((pts[:, 1:-1] - 0.5 * (pts[:, 2:] + pts[:, :-2])) ** 2).mean()
    pts_reg_w = ((pts[:, :, 1:-1] - 0.5 * (pts[:, :, 2:] + pts[:, :, :-2])) ** 2).mean()
    return pts_reg_h + pts_reg_w


def nanprint(name, tensor):
    if isinstance(tensor, list) or isinstance(tensor, tuple):
        for i, t in enumerate(tensor):
            nanprint(f"{name}_{i}", t)
    elif tensor is None:
        print(f"[{name}] is None")
    else:
        print(f"[{name}] is NaN: {tensor.isnan().any()}")