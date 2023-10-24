import torch
from torch.distributed.elastic.multiprocessing.errors import record
import random

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(current_dir)
sys.path.insert(0, code_dir)

import warnings
warnings.filterwarnings("ignore")

from tools.options import Options
from tools.engine import Engine
from tools.logger import Logger
from tools.utils import mkdir, dappend
from models.synthesizer import Synthesizer
from helpers import Helper

class SynthesizerTrainer(Helper):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt["synthesizer"]
        
    def step_model(self, dtype, global_iter, log):
        modes = self.opt.img_modes if dtype == "img" else self.opt.vid_modes
        if 'None' not in modes:
            train_data_info = self.train_img_data_info if dtype == "img" else self.train_vid_data_info
            for mode in modes:
                data = self.next_batch(train_data_info)
                self.synthesizer(data, mode=mode, log=log, global_iter=global_iter)

    def eval_model(self, dtype, data, global_iter, tmp_iter, dump):
        metrics = {}
        modes = self.opt.img_modes if dtype == "img" else self.opt.vid_modes
        if 'None' not in modes:
            for i, mode in enumerate(modes):
                m = self.synthesizer(data, mode=mode, log=tmp_iter == 0, global_iter=global_iter, tmp_iter=tmp_iter, is_eval=True, dump=(dump and i == 0))
                metrics.update(m)
        return metrics

    def run(self):
        with Engine(self.opt) as engine:
            self.engine = engine
            fold_train, fold_valid = self.opt.init_fold_train, self.opt.init_fold_valid
            fold_train = random.randrange(self.opt.num_folds_train) if (self.opt.num_folds_train and self.opt.random_fold_train) else fold_train
            if 'None' not in self.opt.img_modes:
                self.train_img_data_info = self.get_data_info("train", "img", fold=fold_train, num_folds=self.opt.num_folds_train)
                self.valid_img_data_info = self.get_data_info(self.opt.eval_phase, "img", fold=fold_valid, num_folds=self.opt.num_folds_valid) if self.opt.num_iter_eval is not None else None
            if 'None' not in self.opt.vid_modes:
                self.train_vid_data_info = self.get_data_info("train", "vid", fold=fold_train, num_folds=self.opt.num_folds_train)
                self.valid_vid_data_info = self.get_data_info(self.opt.eval_phase, "vid", fold=fold_valid, num_folds=self.opt.num_folds_valid) if self.opt.num_iter_eval is not None else None

            is_main = self.engine.is_main
            logger = Logger(self.opt) if is_main else None

            self.synthesizer = Synthesizer(self.opt, self.engine, is_train=True, is_main=is_main, logger=logger)
            self.synthesizer.train()

            if is_main:
                if 'None' not in self.opt.img_modes:
                    mkdir(os.path.join(self.opt.result_path, f"real_{self.opt.img_modes[0]}"))
                    mkdir(os.path.join(self.opt.result_path, f"fake_{self.opt.img_modes[0]}"))

            best_eval_score_img = None
            best_eval_score_vid = None

            start_iter = int(self.opt.which_iter) + 1 if self.opt.cont_train else 0
            for global_iter in range(start_iter, self.opt.num_iter):

                # train
                log = global_iter % self.opt.log_freq == 0 or global_iter < 10 or (global_iter < 1000 and global_iter % 100 == 0)# and is_main
                self.step_model("img", global_iter, log)
                self.step_model("vid", global_iter, log)
                if global_iter % self.opt.ema_freq:
                    self.synthesizer.accum_ema()

                # eval
                if self.opt.num_iter_eval is not None and global_iter % self.opt.num_iter_eval == 0:
                    self.synthesizer.eval()

                    img_metrics = {}
                    if 'None' not in self.opt.img_modes:
                        dump = self.opt.compute_fid
                        for tmp_iter, img_data in enumerate(self.valid_img_data_info["loader_iter"]):
                            dappend(img_metrics, self.eval_model("img", img_data, global_iter, tmp_iter, dump))
                            if self.opt.max_batch_eval_img is not None and tmp_iter + 1 >= self.opt.max_batch_eval_img:
                                break
                        self.reinit_batches(self.valid_img_data_info)
                        img_metrics = {k: torch.stack(v).mean() for k, v in img_metrics.items()}

                    vid_metrics = {}
                    if 'None' not in self.opt.vid_modes:
                        dump = False
                        for tmp_iter, vid_data in enumerate(self.valid_vid_data_info["loader_iter"]):
                            dappend(vid_metrics, self.eval_model("vid", vid_data, global_iter, tmp_iter, dump))
                            if self.opt.max_batch_eval_vid is not None and tmp_iter + 1 >= self.opt.max_batch_eval_vid:
                                break
                        self.reinit_batches(self.valid_vid_data_info)
                        vid_metrics = {k: torch.stack(v).mean() for k, v in vid_metrics.items()}

                    if is_main:
                        for metric in img_metrics:
                            logger.log_scalar(f"img/eval/{metric}", img_metrics[metric], global_iter)
                        for metric in vid_metrics:
                            logger.log_scalar(f"vid/eval/{metric}", vid_metrics[metric], global_iter)
                        if 'None' not in self.opt.img_modes and self.opt.img_metric in img_metrics:
                            eval_score_img = img_metrics[self.opt.img_metric]
                            is_better = best_eval_score_img is None or eval_score_img < best_eval_score_img
                            if is_better:
                                best_eval_score_img = eval_score_img
                                self.synthesizer.save_networks(global_iter, name="best_img")
                        if 'None' not in self.opt.vid_modes and self.opt.vid_metric in vid_metrics:
                            eval_score_vid = vid_metrics[self.opt.vid_metric]
                            is_better = best_eval_score_vid is None or eval_score_vid < best_eval_score_vid
                            if is_better:
                                best_eval_score_vid = eval_score_vid
                                # print("Saving network")
                                self.synthesizer.save_networks(global_iter, name="best_vid")
                                # print("Done saving network")
                        print(f"[EVAL] Iteration {global_iter:05d}/{self.opt.num_iter:05d}")

                    if self.engine.distributed:
                        self.engine.barrier()
                    self.synthesizer.train()

                if log and is_main:
                    log_string = f"Img epoch {self.train_img_data_info['epoch']:07.2f} " if "None" not in self.opt.img_modes else ""
                    log_string += f"fold {self.train_img_data_info['fold']}, " if "None" not in self.opt.img_modes and self.train_img_data_info['fold'] is not None else ""
                    log_string += f"Vid epoch {self.train_vid_data_info['epoch']:07.2f} " if "None" not in self.opt.vid_modes else ""
                    log_string += f"fold {self.train_vid_data_info['fold']}, " if "None" not in self.opt.vid_modes and self.train_vid_data_info['fold'] is not None else ""
                    log_string += f"Iteration {global_iter:05d}/{self.opt.num_iter:05d}"
                    print(log_string)

                # checkpoint
                if self.opt.save_freq > 0 and global_iter % self.opt.save_freq == 0 and is_main:
                    self.synthesizer.save_networks(global_iter)
                if self.opt.save_latest_freq > 0 and global_iter % self.opt.save_latest_freq == 0 and is_main:
                    self.synthesizer.save_networks(global_iter, name="latest")

            if is_main:
                self.synthesizer.save_networks(self.opt.num_iter - 1, name="latest")
                print('Training was successfully finished.')


@record
def main(opt):
    print("Error file: ", os.environ["TORCHELASTIC_ERROR_FILE"])
    SynthesizerTrainer(opt).run()


if __name__ == "__main__":
    opt = Options().parse(load_synthesizer=True, save=True)
    main(opt)
