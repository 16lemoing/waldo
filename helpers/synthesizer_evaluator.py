from tqdm import tqdm

import torch
from torch.distributed.elastic.multiprocessing.errors import record

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

class SynthesizerEvaluator(Helper):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt["synthesizer"]

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
            if 'None' not in self.opt.img_modes:
                self.eval_img_data_info = self.get_data_info(self.opt.eval_phase, "img")
            if 'None' not in self.opt.vid_modes:
                self.eval_vid_data_info = self.get_data_info(self.opt.eval_phase, "vid")

            is_main = self.engine.is_main
            logger = Logger(self.opt) if is_main else None
            global_iter = 0

            self.synthesizer = Synthesizer(self.opt, self.engine, is_train=False, is_main=is_main, logger=logger)
            self.synthesizer.eval()

            img_metrics = {}
            if 'None' not in self.opt.img_modes:
                dump = False
                for tmp_iter, img_data in tqdm(enumerate(self.eval_img_data_info["loader_iter"])):
                    dappend(img_metrics, self.eval_model("img", img_data, global_iter, tmp_iter, dump))
                    if self.opt.max_batch_eval_img is not None and tmp_iter + 1 >= self.opt.max_batch_eval_img:
                        break
                img_metrics = {k: torch.stack(v).mean() for k, v in img_metrics.items()}

            for metric in img_metrics:
                logger.log_scalar(f"img/test/{metric}", img_metrics[metric], global_iter)
                print(f"img/test/{metric}: ", img_metrics[metric])

            vid_metrics = {}
            if 'None' not in self.opt.vid_modes:
                dump = True
                for tmp_iter, vid_data in enumerate(self.eval_vid_data_info["loader_iter"]):
                    dappend(vid_metrics, self.eval_model("vid", vid_data, global_iter, tmp_iter, dump))
                    if self.opt.max_batch_eval_vid is not None and tmp_iter + 1 >= self.opt.max_batch_eval_vid:
                        break
                vid_metrics = {k: torch.stack(v).mean() for k, v in vid_metrics.items()}

            for metric in vid_metrics:
                logger.log_scalar(f"vid/test/{metric}", vid_metrics[metric], global_iter)
                print(f"vid/test/{metric}: ", vid_metrics[metric])

            if is_main:
                print('Evaluation was successfully finished.')


@record
def main(opt):
    print("Error file: ", os.environ["TORCHELASTIC_ERROR_FILE"])
    SynthesizerEvaluator(opt).run()


if __name__ == "__main__":
    opt = Options().parse(load_synthesizer=True, save=True)
    main(opt)
