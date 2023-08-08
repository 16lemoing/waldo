from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf

import argparse
from glob import glob
from joblib import Parallel, delayed
import numpy as np
import cv2
from tqdm import tqdm

import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, code_dir)

from tools.eval.lpips_tensorflow import lpips_tf
tf.compat.v1.disable_eager_execution()


def load_video(file, compress, resize):
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    frames = []
    while success:
        if compress is not None:
            h, w = compress
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        if resize is not None:
            h, w = resize
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        frames.append(image)
        success, image = vidcap.read()
    return np.stack(frames) / 255.


def get_video_files(folder):
    return sorted(glob(os.path.join(folder, "*.mp4")))


def load_videos(video_files, compress, resize, num_workers):
    videos = Parallel(n_jobs=num_workers)(delayed(load_video)(file, compress, resize) for file in video_files)
    return np.stack(videos)


def get_folder(exp_tag):
    all_folders = glob(f"results/*{exp_tag}")
    assert len(all_folders) == 1, f"Too many possibilities for this tag {exp_tag}:\n{all_folders}"
    return all_folders[0]


class TFMetrics:
    def __init__(self, metrics):
        config = tf.compat.v1.ConfigProto() # log_device_placement=True
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config)
        self.image_0 = tf.compat.v1.placeholder(tf.float32)
        self.image_1 = tf.compat.v1.placeholder(tf.float32)
        self.metrics = []
        for m in metrics:
            if m == "lpips":
                self.metrics.append(lpips_tf.lpips(self.image_0, self.image_1, model='net-lin', net='alex'))
            elif m == "ssim":
                self.metrics.append(tf.compat.v1.image.ssim(self.image_0, self.image_1, max_val=1))
            elif m == "psnr":
                self.metrics.append(tf.compat.v1.image.psnr(self.image_0, self.image_1, max_val=1))
            elif m == "msssim":
                self.metrics.append(tf.compat.v1.image.ssim_multiscale( self.image_0, self.image_1, max_val=1))
            else:
                raise ValueError


    def measure_batch(self, pred_image, target_image):
        return self.sess.run(
            self.metrics,
            feed_dict={
                self.image_0: target_image,
                self.image_1: pred_image
            }
        )


def main(args):
    tf_metrics = TFMetrics(args.metrics)
    root = get_folder(args.vid_tag)
    real_video_files = get_video_files(os.path.join(root, args.real_folder))
    fake_video_files = get_video_files(os.path.join(root, args.fake_folder))
    assert len(real_video_files) == len(fake_video_files)
    total_size = len(real_video_files)

    metrics = [[[] for _ in range(args.vid_length)] for _ in range(len(args.metrics))]

    for i in tqdm(range(total_size // args.batch_size)): #range(4): #
        start = i * args.batch_size
        end = min(start + args.batch_size, total_size)
        real_videos_np = load_videos([real_video_files[i] for i in range(start, end)], args.compress, args.resize, args.num_workers)
        fake_videos_np = load_videos([fake_video_files[i] for i in range(start, end)], args.compress, args.resize, args.num_workers)
        for t in range(args.vid_length):
            with tf.device("GPU"):
                metrics_v = tf_metrics.measure_batch(fake_videos_np[:, t], real_videos_np[:, t])
                for j, v in enumerate(metrics_v):
                    metrics[j][t] += v.tolist()
                    print(f"[{args.metrics[j]}:{t}] : {np.mean(metrics[j][t])}")

    for t in range(args.vid_length):
        for j in range(len(args.metrics)):
            print(f"[{args.metrics[j]}:{t}] : {np.mean(metrics[j][t]), np.std(metrics[j][t])}")
            if t >= args.vid_context:
                print(f"[cum {args.metrics[j]}:{t}] : {np.mean(metrics[j][args.vid_context:t+1]), np.std(metrics[j][args.vid_context:t + 1])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('vid_tag', type=str)
    parser.add_argument('vid_length', type=int)
    parser.add_argument('vid_context', type=int)
    parser.add_argument('--real_folder', type=str, default="real_vid")
    parser.add_argument('--fake_folder', type=str, default="inp_pred_vid")
    parser.add_argument('--compress', type=int, nargs="+", default=None)
    parser.add_argument('--resize', type=int, nargs="+", default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--metrics', type=str, nargs="+", default=["lpips", "msssim"])
    args = parser.parse_args()
    main(args)