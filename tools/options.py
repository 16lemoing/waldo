import os
import pickle
import argparse
from argparse import Namespace

from tools import utils


CITYSCAPES_PALETTE = [0, 0, 0, 128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156,
                      190, 153, 153, 153, 153, 153, 250, 170, 30, 220, 220, 0,
                      107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
                      255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100,
                      0, 80, 100, 0, 0, 230, 119, 11, 32]

KITTI_PALETTE = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156,
                 190, 153, 153, 153, 153, 153, 250, 170, 30, 220, 220, 0,
                 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60,
                 255, 0, 0, 0, 0, 142, 0, 0, 70, 0, 60, 100,
                 0, 80, 100, 0, 0, 230, 119, 11, 32]

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options():
    def __init__(self):
        pass

    def initialize(self, parser):
        parser = self.initialize_base(parser)
        parser = self.initialize_synthesizer(parser)
        return parser

    def initialize_base(self, parser, prefix=""):
        # experiment specifics
        parser.add_argument(f'--{prefix}name', type=str, required=(prefix == ""), help='name of the experiment')
        parser.add_argument(f'--{prefix}datetime', type=str, required=(prefix == ""), help='datetime of the experiment')
        parser.add_argument(f'--{prefix}gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

        # for mixed precision
        parser.add_argument(f'--{prefix}use_amp', type=str2bool, nargs='?', const=True, default=False, help='if specified, use apex mixed precision')

        # for input / output sizes
        parser.add_argument(f'--{prefix}batch_size_img', type=int, default=1, help='input batch size for images')
        parser.add_argument(f'--{prefix}batch_size_vid', type=int, default=1, help='input batch size for videos')
        parser.add_argument(f'--{prefix}true_dim', type=int, default=1024, help='resolution of images after loading')
        parser.add_argument(f'--{prefix}dim', type=int, default=512, help='target resolution of images')
        parser.add_argument(f'--{prefix}load_dim', type=int, default=0, help='target resolution of images')
        parser.add_argument(f'--{prefix}flow_dim', type=int, default=0, help='target resolution of flow')
        parser.add_argument(f'--{prefix}input_ratio', type=float, default=1.0, help='ratio width/height of input images, final width will be dim * aspect_ratio')
        parser.add_argument(f'--{prefix}aspect_ratio', type=float, default=1.0, help='target width/height ratio')
        parser.add_argument(f'--{prefix}imagenet_norm', type=str2bool, nargs='?', const=True, default=False, help='normalize images with imagenet statistics')
        parser.add_argument(f'--{prefix}colorjitter', type=float, default=None, help='randomly change the brightness, contrast and saturation of images')
        parser.add_argument(f'--{prefix}colorjitter_no_contrast', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument(f'--{prefix}rotate', type=float, default=0, help='randomly rotate images')

        # for setting inputs
        parser.add_argument(f'--{prefix}dataroot', type=str, default='./datasets/bair/')
        parser.add_argument(f'--{prefix}dataset', type=str, default='bair')
        parser.add_argument(f'--{prefix}num_folds_train', type=int, default=None, help='if specified, only load data fold by fold')
        parser.add_argument(f'--{prefix}num_folds_valid', type=int, default=None, help='if specified, only load data fold by fold')
        parser.add_argument(f'--{prefix}num_folds_test', type=int, default=None, help='if specified, only load data fold by fold')
        parser.add_argument(f'--{prefix}random_fold_train', type=str2bool, nargs='?', const=True, default=False, help='if specified, use random starting fold')
        parser.add_argument(f'--{prefix}init_fold_train', type=int, default=None, help='if specified, cycle through folds starting at specified fold')
        parser.add_argument(f'--{prefix}init_fold_valid', type=int, default=None, help='if specified, cycle through folds starting at specified fold')
        parser.add_argument(f'--{prefix}init_fold_test', type=int, default=None, help='if specified, cycle through folds starting at specified fold')
        parser.add_argument(f'--{prefix}data_specs', type=str, default=None, help='if specified, string indicating specificities of the data')
        parser.add_argument(f'--{prefix}from_vid', type=str2bool, nargs='?', const=True, default=False, help='if specified, data is stored as video files, otherwise frame by frame')
        parser.add_argument(f'--{prefix}from_animation', type=str2bool, nargs='?', const=True, default=False, help='if specified, data is animated by the dataloader')
        parser.add_argument(f'--{prefix}is_vid', type=str2bool, nargs='?', const=True, default=False, help='if specified, take into account sequential aspect')
        parser.add_argument(f'--{prefix}vid_len', type=int, default=16, help='number of frames in produced videos')
        parser.add_argument(f'--{prefix}load_vid_len', type=int, default=None, help='load video with more frames to do random subsampling')
        parser.add_argument(f'--{prefix}load_n_plus_1', type=str2bool, nargs='?', const=True, default=False, help='if specified, load n contiguous frames and 1 into the future')
        parser.add_argument(f'--{prefix}load_n_rd', type=str2bool, nargs='?', const=True, default=False, help='if specified, load n random frames')
        parser.add_argument(f'--{prefix}load_2_apart', type=str2bool, nargs='?', const=True, default=False, help='if specified, load n random frames')
        parser.add_argument(f'--{prefix}max_vid_step', type=int, default=1000, help='max frames skipped in random subsampling')
        parser.add_argument(f'--{prefix}vid_skip', type=int, default=1, help='number of frames to skip between each clip')
        parser.add_argument(f'--{prefix}categories', type=str, nargs="+", help='labels for the videos')
        parser.add_argument(f'--{prefix}load_data', type=str2bool, nargs='?', const=True, default=False, help='if specified, load data information from file')
        parser.add_argument(f'--{prefix}save_data', type=str2bool, nargs='?', const=True, default=False, help='if specified, save data information so that it does not have to be recomputed everytime')
        parser.add_argument(f'--{prefix}force_compute_metadata', type=str2bool, nargs='?', const=True, default=False, help='if specified, force re-computation of metadata after dataset update')
        parser.add_argument(f'--{prefix}shuffle_valid', type=str2bool, nargs='?', const=True, default=False, help='if specified, both training and validation set are shuffled')
        parser.add_argument(f'--{prefix}no_h_flip', type=str2bool, nargs='?', const=True, default=False, help='if specified, do not horizontally flip the images for data argumentation')
        parser.add_argument(f'--{prefix}no_v_flip', type=str2bool, nargs='?', const=True, default=False, help='if specified, do not vertically flip the images for data argumentation')
        parser.add_argument(f'--{prefix}resize_img', type=int, nargs="+", default=None, help='if specified, resize images to specified h,w once they are loaded')
        parser.add_argument(f'--{prefix}resize_center_crop_img', type=int, default=None, help='if specified, square crop images to specified size once they are loaded')
        parser.add_argument(f'--{prefix}original_size', type=int, nargs="+", default=None, help='if resize, specify original size')
        parser.add_argument(f'--{prefix}min_zoom', type=float, default=1., help='parameter for augmentation method consisting in zooming and cropping')
        parser.add_argument(f'--{prefix}max_zoom', type=float, default=1., help='parameter for augmentation method consisting in zooming and cropping')
        parser.add_argument(f'--{prefix}fixed_crop', type=int, nargs="+", default=None, help='if specified, apply a random crop of the given size')
        parser.add_argument(f'--{prefix}centered_crop', type=str2bool, nargs='?', const=True, default=False, help='if specified, cropped area is centered horizontally and vertically')
        parser.add_argument(f'--{prefix}horizontal_centered_crop', type=str2bool, nargs='?', const=True, default=False, help='if specified, cropped area is centered horizontally and vertically')
        parser.add_argument(f'--{prefix}fixed_top_centered_zoom', type=float, default=None, help='if specified, crop the image to the upper center part')
        parser.add_argument(f'--{prefix}num_workers', default=8, type=int, help='# threads for loading data')
        parser.add_argument(f'--{prefix}num_workers_eval', default=None, type=int, help='# threads for loading data')
        parser.add_argument(f'--{prefix}load_from_opt_file', type=str2bool, nargs='?', const=True, default=False, help='load options from checkpoints and use that as default')
        parser.add_argument(f'--{prefix}load_signature', type=str, default="", help='specifies experiment signature from which to load options')
        parser.add_argument(f'--{prefix}fps', default=10, type=int, help='frames per second')
        parser.add_argument(f'--{prefix}is_tar', type=str2bool, nargs='?', const=True, default=False, help='images are stored within tar files')
        parser.add_argument(f'--{prefix}load_n_from_tar', type=int, default=1, help='load n images from the same archive at once')
        parser.add_argument(f'--{prefix}update_tar_every_n', type=int, default=1, help='update the archive every n steps')
        parser.add_argument(f'--{prefix}one_every_n', type=int, default=1, help='skip frames to decrease fps')
        parser.add_argument(f'--{prefix}num_lyt', default=None, type=int, help='num of layout classes')
        parser.add_argument(f'--{prefix}lyt_model', default="", type=str, help='model used for layout computation')
        parser.add_argument(f'--{prefix}flow_model', default="", type=str, help='model used for optical flow computation')
        parser.add_argument(f'--{prefix}load_lyt', type=str2bool, nargs='?', const=True, default=False, help='load semantic layouts')
        parser.add_argument(f'--{prefix}load_flow', type=str2bool, nargs='?', const=True, default=False, help='load optical flow')
        parser.add_argument(f'--{prefix}load_100', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument(f'--{prefix}load_all', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument(f'--{prefix}palette', type=int, nargs="+", default=None, help='colors for layout')
        parser.add_argument(f'--{prefix}fg_idx', type=int, nargs="+", default=None, help='layout index corresponding to foreground objects')
        parser.add_argument(f'--{prefix}bg_idx', type=int, nargs="+", default=None, help='layout index corresponding to background')
        parser.add_argument(f'--{prefix}other_idx', type=int, nargs="+", default=None, help='layout index corresponding to neither foreground nor background')
        parser.add_argument(f'--{prefix}skip_first', type=str2bool, nargs='?', const=True, default=False, help='skip first frame of video')
        parser.add_argument(f'--{prefix}remap_lyt', type=int, nargs='+', default=[], help='remap some classes to other classes: src1 tgt1 src2 tgt2...')

        # for augmentation
        parser.add_argument(f'--{prefix}use_aug_img', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument(f'--{prefix}aug_rd_fill', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument(f'--{prefix}aug_min_mask', type=float, default=0.0)
        parser.add_argument(f'--{prefix}aug_max_mask', type=float, default=0.0)
        parser.add_argument(f'--{prefix}aug_alpha', type=float, default=2.0)
        parser.add_argument(f'--{prefix}aug_sigma', type=float, default=0.2)
        parser.add_argument(f'--{prefix}aug_max_zoom', type=float, default=1.3)
        parser.add_argument(f'--{prefix}aug_min_translate', type=float, default=-0.3)
        parser.add_argument(f'--{prefix}aug_max_translate', type=float, default=0.3)
        parser.add_argument(f'--{prefix}aug_min_rotate', type=float, default=-0.3)
        parser.add_argument(f'--{prefix}aug_max_rotate', type=float, default=0.3)
        parser.add_argument(f'--{prefix}aug_padding_mode', type=str, default='zeros', choices=["zeros", "reflection"])

        # for tps augmentation
        parser.add_argument(f'--{prefix}use_tps_aug_img', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument(f"--{prefix}tps_aug_simulate_obj", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument(f"--{prefix}tps_aug_max_translate", type=float, default=0.)
        parser.add_argument(f"--{prefix}tps_aug_min_translate", type=float, default=0.)
        parser.add_argument(f"--{prefix}tps_aug_max_scale", type=float, default=.5)
        parser.add_argument(f"--{prefix}tps_aug_min_scale", type=float, default=.5)
        parser.add_argument(f"--{prefix}tps_aug_max_scale_bg", type=float, default=1.)
        parser.add_argument(f"--{prefix}tps_aug_min_scale_bg", type=float, default=1.)
        parser.add_argument(f"--{prefix}tps_aug_min_mask", type=float, default=0.)
        parser.add_argument(f"--{prefix}tps_aug_max_mask", type=float, default=0.)
        parser.add_argument(f"--{prefix}tps_aug_max_delta", type=float, default=0.)
        parser.add_argument(f"--{prefix}tps_aug_max_delta_bg", type=float, default=0.)
        parser.add_argument(f"--{prefix}tps_aug_mask_kernel", type=int, default=13)
        parser.add_argument(f"--{prefix}tps_aug_circle_kernel", type=int, default=5)
        parser.add_argument(f"--{prefix}tps_aug_circle_alpha", type=float, default=0.1)

        # for animating data
        parser.add_argument(f'--{prefix}rd_len', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument(f'--{prefix}min_rd_len', type=int, default=1)
        parser.add_argument(f'--{prefix}max_rd_len', type=int, default=1)# for tpsynthesizer
        parser.add_argument(f'--{prefix}single_digit', type=str2bool, nargs='?', const=True, default=False)

        # for display and checkpointing
        parser.add_argument(f'--{prefix}log_freq', type=int, default=None, help='if specified, frequency at which logger is updated with images')
        parser.add_argument(f'--{prefix}log_fps', type=int, default=4, help='logs videos at specified speed in frames per second')
        parser.add_argument(f'--{prefix}save_freq', type=int, default=-1, help='frequency of saving models, if -1 don\'t save')
        parser.add_argument(f'--{prefix}save_latest_freq', type=int, default=5000, help='frequency of saving the latest model')
        parser.add_argument(f'--{prefix}save_path', type=str, default='./')

        # for loading
        parser.add_argument(f'--{prefix}cont_train', type=str2bool, nargs='?', const=True, default=False, help='continue training with model from which_iter')

        # for training
        parser.add_argument(f'--{prefix}num_iter', type=int, default=1000, help='number of training iterations')
        parser.add_argument(f'--{prefix}img_modes', type=str, default=['None'], nargs='+', help='training mode to process image data')
        parser.add_argument(f'--{prefix}vid_modes', type=str, default=['None'], nargs='+', help='training mode to process video data')
        parser.add_argument(f'--{prefix}img_acc_nums', type=int, default=['None'], nargs='+', help='number of accumulation steps for image mode')
        parser.add_argument(f'--{prefix}vid_acc_nums', type=int, default=['None'], nargs='+', help='number of accumulation steps for video mode')
        parser.add_argument(f'--{prefix}img_skip_nums', type=int, default=['None'], nargs='+', help='number of skipped steps for image mode')
        parser.add_argument(f'--{prefix}vid_skip_nums', type=int, default=['None'], nargs='+', help='number of skipped steps for video mode')
        parser.add_argument(f'--{prefix}vid_step_every', type=int, default=1, help='number of img training steps before each vid training step')

        # for evaluating
        parser.add_argument(f'--{prefix}num_iter_eval', type=int, default=None, help='if specified, number of iterations between each evaluation phase1')
        parser.add_argument(f'--{prefix}max_batch_eval_img', type=int, default=None, help='if specified, max number of eval batches to speed up evaluation')
        parser.add_argument(f'--{prefix}max_batch_eval_vid', type=int, default=None, help='if specified, max number of eval batches to speed up evaluation')
        parser.add_argument(f'--{prefix}compute_fid', type=str2bool, nargs='?', const=True, default=False, help='frechet inception distance')
        parser.add_argument(f'--{prefix}compute_fvd', type=str2bool, nargs='?', const=True, default=False, help='frechet video distance')
        parser.add_argument(f'--{prefix}img_metric', type=str, default='', help='the image metric used for selecting best checkpoint')
        parser.add_argument(f'--{prefix}vid_metric', type=str, default='', help='the video metric used for selecting best checkpoint')
        parser.add_argument(f'--{prefix}eval_phase', type=str, default='valid', choices=['train', 'valid', 'test'])

        return parser

    def initialize_synthesizer(self, parser):
        # for synthesizer model
        parser.add_argument("--s_patch_size", type=int, default=8, help="project image to embeddings patch by patch")
        parser.add_argument('--s_latent_shape', type=int, nargs='+', default=[4, 8], help='spatial shape of latent vectors')
        parser.add_argument("--s_embed_dim", type=int, default=512, help="dimension of embedding vectors")
        parser.add_argument("--s_cap_dim", type=int, default=768, help="dimension of caption embedding vectors")
        parser.add_argument('--s_num_timesteps', type=int, default=16, help='temporal capacity of transformer models')
        parser.add_argument('--s_num_captions', type=int, default=0, help='maximum length of captions')
        parser.add_argument('--s_num_obj', type=int, default=1, help='number of objects')
        parser.add_argument('--s_norm_layer', type=str, default="ln", choices=['pn', 'ln', 'ln_not_affine'])
        parser.add_argument("--s_num_heads", type=int, default=8, help="number of attention heads")
        parser.add_argument("--s_enc_depth", type=int, default=4, help="number of transformer layers")
        parser.add_argument("--s_dec_depth", type=int, default=4, help="number of transformer layers")
        parser.add_argument("--s_gen_depth", type=int, default=8, help="number of transformer layers")
        parser.add_argument('--s_gen_noise', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_gen_noise_modulation', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_gen_mapping', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_gen_random_obj_embed', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_com_depth", type=int, default=1, help="number of transformer layers")
        parser.add_argument("--s_dis_depth", type=int, default=7, help="number of transformer layers")
        parser.add_argument("--s_dis_cls_depth", type=int, default=1, help="number of transformer layers")
        parser.add_argument('--s_dis_spectral_norm_layer', type=str, default="None", choices=["None", "sn", "isn"])
        parser.add_argument("--s_dis_stddev_group", type=int, default=4, help="number of group")
        parser.add_argument('--s_dis_use_stddev', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_dis_spatial_stddev_group", type=int, default=4, help="number of group")
        parser.add_argument('--s_dis_use_spatial_stddev', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_dis_latent_mode', type=str, default='identity', choices=["identity", "compress_space"])
        parser.add_argument('--s_dis_flow_mode', type=str, default='identity', choices=["identity", "compress_space_mlp", "compress_space_cnn", "compress_time"])
        parser.add_argument("--s_unc_temporal_dropout", type=float, default=0, help="proportion of temporal step to be dropped")
        parser.add_argument("--s_unc_drop_obj", type=float, default=0, help="proportion of temporal step to be dropped")
        parser.add_argument('--s_unc_non_trivial', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_unc_init', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_unc_arch', type=str, default='mlp', choices=["mlp", "cnn"])
        parser.add_argument('--s_unc_mode_img', type=str, default='obj_to_1', choices=['rd_aug_obj', 'aug_obj', 'obj_to_1', 'obj_plus_n_to_1'])
        parser.add_argument('--s_unc_mode_vid', type=str, default='obj_plus_n_to_1', choices=['obj_to_1', 'obj_plus_n_to_1', 't_from_random_n'])
        parser.add_argument("--s_unc_n", type=int, default=2, help="number of ctx to reconstruct one frame during training")
        parser.add_argument("--s_min_ctx_length_img", type=int, default=0, help="minimum context length")
        parser.add_argument("--s_max_ctx_length_img", type=int, default=16, help="maximum context length")
        parser.add_argument("--s_min_ctx_length_vid", type=int, default=0, help="minimum context length")
        parser.add_argument("--s_max_ctx_length_vid", type=int, default=16, help="maximum context length")
        parser.add_argument('--s_use_d', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_adaptive_lambda', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_norm_layer_patch', type=str, default="ln2d", choices=['bn2d', 'ln2d'])
        parser.add_argument('--s_freeze_obj', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_remove_obj', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_no_ctx_fake', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_aug_policy", type=str, nargs='+', default=["None"], choices=['color', 'translation', 'translation_1', 'strong_translation', 'cutout', 'erase', 'erase_ratio', 'erase2_ratio', 'rand_erase_ratio', 'rotate', 'cutmix', 'hue', 'filter', 'geo', 'crop', 'stl_erase_ratio'])
        parser.add_argument('--s_dis', type=str, default='temporal', choices=["temporal", "image"])
        parser.add_argument("--s_style_embed_dim", type=int, default=128)
        parser.add_argument("--s_style_embed_mul", type=int, default=[1, 1, 2, 2, 2], nargs="+")
        parser.add_argument("--s_style_latent_shape", type=int, default=[4, 4], nargs="+")
        parser.add_argument('--s_gan', type=str, default='vivit', choices=["vivit", "swish", "hybrid"])
        parser.add_argument('--s_unc_obj_mode', type=str, default='code', choices=["code", "embed"])
        parser.add_argument("--s_drop_quant", type=float, default=0, help="proportion of latent not to be quantized")

        # for vqsynthesizer model
        parser.add_argument("--s_codebook_size", type=int, default=256, help="number of entries in codebook")
        parser.add_argument('--s_use_latent_norm', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_codebook_dim", type=int, default=0, help="if greater than 0, define custom dimension for codebook entries")

        # for keysynthesizer
        parser.add_argument('--s_dis_on_rec', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_obj_shape', type=int, nargs='+', default=[4, 4], help='spatial shape of latent object vectors')
        parser.add_argument('--s_gen_attn_mode', type=str, default='dp', choices=["dp", "nsed"])
        parser.add_argument("--s_swap_p", type=float, default=0)

        # for tpsynthesizer
        parser.add_argument('--s_debug', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_mul_start_iter", type=int, default=0)
        parser.add_argument("--s_mul_end_iter", type=int, default=0)
        parser.add_argument("--s_head_scale", type=float, default=1.)
        parser.add_argument("--s_mul_scale_obj", type=float, default=1.)
        parser.add_argument("--s_mul_delta_obj", type=float, default=1.)
        parser.add_argument("--s_init_scale_obj", type=float, default=1.)
        parser.add_argument('--s_pe_use_scorer', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pe_use_refiner', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pe_use_post_refiner', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pe_use_edge_filter', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pe_repeat_border', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pe_filter_order', type=int, default=1)
        parser.add_argument('--s_pe_filter_blur', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pe_refiner_mode', type=str, default=["comp"], nargs="+", choices=["comp", "bg", "obj"])
        parser.add_argument('--s_pe_decoder_init_mode', type=str, default="", choices=["", "zero", "five"])
        parser.add_argument('--s_pe_refiner_init_mode', type=str, default="", choices=["", "mfive"])
        parser.add_argument('--s_pe_estimator_init_mode', type=str, default="zero", choices=["", "zero"])
        parser.add_argument('--s_pe_refiner_blend_mode_obj', type=str, default="", choices=["", "fusion", "alpha", "mean"])
        parser.add_argument('--s_pe_refiner_blend_mode_bg', type=str, default="", choices=["", "fusion", "alpha", "mean"])
        parser.add_argument('--s_pe_refiner_depth', type=int, default=2)
        parser.add_argument('--s_pe_post_refiner_depth', type=int, default=2)
        parser.add_argument('--s_oe_use_decoder', type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--s_decompose_embed_oe', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_bound_alpha', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pe_decoder_use_prior', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_pe_depth", type=int, default=8, help="number of transformer layers")
        parser.add_argument('--s_pe_pts_mode', type=str, default="prior", choices=["head", "prior"])
        parser.add_argument("--s_oe_num_timesteps", type=int, default=5)
        parser.add_argument("--s_oe_depth", type=int, default=8, help="number of transformer layers")
        parser.add_argument('--s_oe_pts_mode', type=str, default="prior", choices=["head", "prior"])
        parser.add_argument("--s_pg_simple", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_pg_simple_head", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_pg_depth", type=int, default=6, help="number of layers")
        parser.add_argument("--s_pg_com_depth", type=int, default=2, help="number of transformer layers")
        parser.add_argument("--s_pg_enc_depth", type=int, default=4, help="number of transformer layers")
        parser.add_argument("--s_pg_dec_depth", type=int, default=4, help="number of transformer layers")
        parser.add_argument("--s_pg_num_timesteps", type=int, default=5)
        parser.add_argument("--s_pg_batch_size_mul", type=int, default=1, help="augment batch size wih various ctx masking windows")
        parser.add_argument('--s_pg_pts_mode', type=str, default="prior", choices=["head", "prior"])
        parser.add_argument('--s_pg_embed_noise', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pg_inject_noise', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pg_modulate_noise', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pg_load_path', type=str, default=None)
        parser.add_argument('--s_pg_iter', type=str, default=None)
        parser.add_argument("--s_pd_com_depth", type=int, default=2, help="number of transformer layers")
        parser.add_argument("--s_pd_enc_depth", type=int, default=4, help="number of transformer layers")
        parser.add_argument('--s_oe_freeze_iter', type=int, default=0)
        parser.add_argument('--s_oe_init_mode', type=str, default="", choices=["white_sphere", ""])
        parser.add_argument("--s_ii_depth", type=int, default=4, help="number of convolutional layers")
        parser.add_argument("--s_ii_embed_dim", type=int, default=512, help="dimension of embedding vectors")
        parser.add_argument('--s_ii_score', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_ii_ab', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_ii_load_path', type=str, default=None)
        parser.add_argument('--s_ii_iter', type=str, default=None)
        parser.add_argument('--s_ii_upmode', type=str, default="bilinear")
        parser.add_argument('--s_ii_ft_hd', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_progressive_scale', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_bound_rest', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_min_scale_bound", type=float, default=-0.5)
        parser.add_argument("--s_max_scale_bound", type=float, default=0.5)
        parser.add_argument("--s_max_translate_bound", type=float, default=0.5)
        parser.add_argument('--s_soft_bound_rest', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_zero_init_dec', type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--s_norm_scale', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_bound_scale', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_delta', type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--s_rd_translate_bias', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_circle_translate_bias', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_circle_translate_radius', type=float, default=0.25)
        parser.add_argument("--s_translate_bias_mul", type=float, default=1)
        parser.add_argument("--s_pad_obj_alpha", type=int, default=0)
        parser.add_argument("--s_pad_bg_alpha", type=int, default=0)
        parser.add_argument("--s_min_scale", type=float, default=0)
        parser.add_argument("--s_max_scale", type=float, default=2)
        parser.add_argument("--s_max_bg_alpha", type=float, default=1)
        parser.add_argument("--s_min_bg_alpha", type=float, default=0.0001)
        parser.add_argument("--s_mean_obj_alpha", type=float, default=0)
        parser.add_argument("--s_tgt_scale", type=float, default=1)
        parser.add_argument('--s_has_bg', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_fix_bg', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_fix_bg1', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_hr', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_hr_ratio', type=int, default=1)
        parser.add_argument('--s_obj_discovery_mode', type=str, default=[], nargs="+", choices=["warp", "refine", "pyramid"])
        parser.add_argument("--s_warmup_bg_iter", type=int, default=0)
        parser.add_argument("--s_warmup_bg_score_iter", type=int, default=0)
        parser.add_argument("--s_warmup_obj_score_iter", type=int, default=0)
        parser.add_argument('--s_use_layout', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_soft_bg', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_bg_color', type=str, default="black", choices=["black", "grey", "white"])
        parser.add_argument("--s_min_conf", type=float, default=0)
        parser.add_argument('--s_occ_mode', type=str, default="", choices=["", "bias", "normalize", "freeze"])
        parser.add_argument('--s_normalize_alpha', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_reg_bg_mul", type=float, default=0.25)
        parser.add_argument("--s_num_perm_grid", type=int, default=1)
        parser.add_argument('--s_filter_alpha', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_interpolate_grid', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_min_obj_shift", type=float, default=0.0)
        parser.add_argument("--s_max_obj_shift", type=float, default=0.04)
        parser.add_argument('--s_shuffle_bg', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_alpha_norm', type=float, default=0)
        parser.add_argument("--s_ctx_len", type=int, default=10)
        parser.add_argument('--s_input_lyt', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_input_flow', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_input_rgb', type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--s_use_flow_nobg', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_dominant_flow_other', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_nobg', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_nobg_edge', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_nobg_edge_mul", type=float, default=0)
        parser.add_argument('--s_use_fg', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_lyt_filtering', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_lyt_opacity', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_pred_cls', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_weight_cls', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_swap_flt', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_scale_factor', type=int, default=1)
        parser.add_argument("--s_mov_obj_thresh", type=float, default=0.02)
        parser.add_argument("--s_flow_thresh", type=float, default=0.01)
        parser.add_argument("--s_min_cls", type=float, default=0.001)
        parser.add_argument('--s_rd_ctx_num', type=int, default=1)
        parser.add_argument('--s_ctx_mode', type=str, default="full", choices=["full", "prev", "prev_rd"])
        parser.add_argument('--s_include_self', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_edge_size', type=int, default=7)
        parser.add_argument('--s_use_last_pose_decoder', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_unconstrained_pose_decoder', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_bg_mul", type=float, default=1.)
        parser.add_argument("--s_bg_mul_pose_decoder", type=float, default=1.)
        parser.add_argument("--s_img_mul_act_reg", type=float, default=1.)
        parser.add_argument("--s_cell_dis_eps", type=float, default=0.)
        parser.add_argument('--s_fill_mask', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_disocc', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_split_pred_ts', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_restrict_to_ctx', type=str2bool, nargs='?', const=True, default=False) # need ctx to be first frames and same ctx for all pred ts, need a better fix in the future
        parser.add_argument("--s_dropout", type=float, default=0)
        parser.add_argument('--s_no_filter', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_cat_z', type=str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--s_loop_ii', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_drop_input_p", type=float, default=0.)
        parser.add_argument('--s_no_future', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_last_n_ctx', type=int, default=0)
        parser.add_argument('--s_viz', type=str2bool, nargs='?', const=True, default=False)

        # for inpainting
        parser.add_argument('--s_inpaint_obj', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_propagate_unique', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_shadows', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_expansion', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_soft_shadow', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_propagate_obj', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_inpainter', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_mat_inpainter', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_inpainter_path', type=str, default=None)
        parser.add_argument('--s_ii_last_only', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_fix_thresh', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_fix_mask', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_num_expansion', type=int, default=2)
        parser.add_argument('--s_allow_ghost', type=str2bool, nargs='?', const=True, default=False)

        # for model construction
        parser.add_argument('--s_use_pe', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_pg', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_pd', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_oe', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_og', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_od', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_te', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_ii', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_use_id', type=str2bool, nargs='?', const=True, default=False)

        # for learning strategy
        parser.add_argument('--s_vid_autoencoder_losses', type=str, nargs='+', default=["pxl_ctx", "pxl_gen"], choices=["pxl_ctx", "pxl_gen", "vgg_ctx", "vgg_gen", "warped_latent_ctx", "pred_latent_ctx"])
        parser.add_argument('--s_vid_generator_losses', type=str, nargs='+', default=["pxl_ctx", "pxl_gen"], choices=["pxl_ctx", "pxl_gen", "vgg_ctx", "vgg_gen", "adv", "dis", "collapse", "adv_collapse", "cls_obj_ctx", "cls_obj_gen", "reg_map_ctx", "reg_map_gen", "reg_obj_ctx", "reg_obj_gen", "pxl_warp_ctx", "triv_flow", "pxl_aug_ctx", "pts_reg", "pts_dis", "pxl_obj", "warp", "pts_rest", "pxl_obj_alpha"])
        parser.add_argument('--s_img_generator_losses', type=str, nargs='+', default=["pxl_ctx", "pxl_gen"], choices=["pxl_ctx", "pxl_gen", "vgg_ctx", "vgg_gen", "adv", "dis", "collapse", "adv_collapse", "reg", "flow", "cls_obj_ctx", "cls_obj_gen", "reg_map_ctx", "reg_map_gen", "reg_obj_ctx", "reg_obj_gen"])
        parser.add_argument('--s_img_autoencoder_losses', type=str, nargs='+', default=["pxl", "qnt"], choices=["pxl", "vgg", "adv", "dis", "reg", "qnt"])
        parser.add_argument('--s_img_pose_extractor_losses', type=str, nargs='+', default=["pxl_ctx"], choices=["sharp_pxl_conf_ctx", "dummy_conf", "conf", "pxl_conf_ctx", "delta_pose", "delta_bg_pose", "pxl_ctx", "vgg_ctx", "pts_reg", "bg_pts_reg", "pxl_obj", "pts_rest", "bg_pts_rest", "pxl_obj_alpha"])
        parser.add_argument('--s_vid_pose_extractor_losses', type=str, nargs='+', default=["pxl_ctx"], choices=["sharp_pxl_conf_ctx", "dummy_conf", "conf", "pxl_conf_ctx", "delta_pose", "delta_bg_pose", "pxl_ctx", "vgg_ctx", "pts_reg", "bg_pts_reg", "pxl_obj", "pts_rest", "bg_pts_rest", "pxl_obj_alpha"])
        parser.add_argument('--s_vid_object_extractor_losses', type=str, nargs='+', default=["pxl_vid"], choices=["obj_flow", "topactivity", "activity", "cluster_dis", "ent", "ent_flt_edge", "ent_flt", "reg_mov_flt", "center_dis", "cell_dis", "abs_mov", "ce_mean_lyt_obj", "ce_lyt_obj", "l1_flow_mov_obj", "l1_flow_other", "fg_tube_ent", "reg_fg", "tube_ent", "reg_mov", "rec_edge", "l1_flow", "soft_ce_mean_lyt", "ce_mean_lyt", "soft_ce_lyt", "ce_lyt", "color_diversity", "pxl_mean_vid", "reg_kcenters2", "reg_kcenters", "rec_edge_vid", "rec_edge_vid2", "rec_edge_vid3", "rec_edge_vid2s", "rec_edge_obj2", "reg_surface", "inter_obj", "expansion", "shift_penalty", "rec_edge_obj", "rec_alpha_obj", "reg_raw_vid_plus3", "reg_raw_vid_plus_mean", "reg_obj_pose", "reg_raw_vid_plus2", "reg_raw_vid_plus", "reg_alpha", "reg_color", "reg_edge", "iou_obj", "pxl_inter_obj", "pxl_obj", "pxl_bg", "sharp_vid", "push_obj_pose", "push_raw_vid", "reg_raw_vid", "pxl_comp", "entropy", "reg_raw_bg", "pxl_raw_obj", "pxl_raw_fg", "pxl_raw_bg", "pxl_vid", "pts_reg_bg", "pts_reg_obj", "pts_rest_obj", "pts_rest_bg", "pts_reg_bg_self", "pts_reg_obj_self", "pts_rest_obj_self", "pts_rest_bg_self"])
        parser.add_argument('--s_img_object_extractor_losses', type=str, nargs='+', default=["pxl_vid"], choices=["obj_flow", "topactivity", "activity", "cluster_dis", "ent", "ent_flt_edge", "ent_flt", "reg_mov_flt", "center_dis", "cell_dis", "abs_mov", "ce_mean_lyt_obj", "ce_lyt_obj", "l1_flow_mov_obj", "l1_flow_other", "fg_tube_ent", "reg_fg", "tube_ent", "reg_mov", "rec_edge", "l1_flow", "soft_ce_mean_lyt", "ce_mean_lyt", "soft_ce_lyt", "ce_lyt", "color_diversity", "pxl_mean_vid", "reg_kcenters2", "reg_kcenters", "rec_edge_vid", "rec_edge_vid2", "rec_edge_vid3", "rec_edge_vid2s", "rec_edge_obj2", "reg_surface", "inter_obj", "expansion", "shift_penalty", "rec_edge_obj", "rec_alpha_obj", "reg_raw_vid_plus3", "reg_raw_vid_plus_mean", "reg_obj_pose", "reg_raw_vid_plus2", "reg_raw_vid_plus", "reg_alpha", "reg_color", "reg_edge", "iou_obj", "pxl_inter_obj", "pxl_obj", "pxl_bg", "sharp_vid", "push_obj_pose", "push_raw_vid", "reg_raw_vid", "pxl_comp", "entropy", "reg_raw_bg", "pxl_raw_obj", "pxl_raw_fg", "pxl_raw_bg", "pxl_vid", "pts_reg_bg", "pts_reg_obj", "pts_rest_obj", "pts_rest_bg", "pts_reg_bg_self", "pts_reg_obj_self", "pts_rest_obj_self", "pts_rest_bg_self"])
        parser.add_argument('--s_vid_pose_generator_losses', type=str, nargs='+', default=["rec_obj_pose"], choices=["rec_obj_pose", "rec_bg_pose", "rec_occ_score"])
        parser.add_argument('--s_vid_inpainting_losses', type=str, nargs='+', default=["sharp_vid"], choices=["dis", "adv", "sharp_vid", "lpips_vid"])
        parser.add_argument("--s_lambda_obj_flow", type=float, default=1)
        parser.add_argument("--s_lambda_lpips_vid", type=float, default=1)
        parser.add_argument("--s_lambda_activity", type=float, default=1)
        parser.add_argument("--s_lambda_cluster_dis", type=float, default=1)
        parser.add_argument("--s_lambda_rec_obj_pose", type=float, default=1)
        parser.add_argument("--s_lambda_rec_bg_pose", type=float, default=1)
        parser.add_argument("--s_lambda_rec_occ_score", type=float, default=1)
        parser.add_argument("--s_warmup_reg_mov_iter", type=int, default=0)
        parser.add_argument("--s_warmup_reg_mov_mul", type=int, default=100)
        parser.add_argument("--s_warmup_l1_flow_iter", type=int, default=0)
        parser.add_argument("--s_warmup_l1_flow_mul", type=int, default=100)
        parser.add_argument("--s_lambda_center_dis", type=float, default=1)
        parser.add_argument("--s_lambda_cell_dis", type=float, default=1)
        parser.add_argument("--s_lambda_abs_mov", type=float, default=1)
        parser.add_argument("--s_lambda_fg_tube_ent", type=float, default=1)
        parser.add_argument("--s_lambda_tube_ent", type=float, default=1)
        parser.add_argument("--s_lambda_rec_edge", type=float, default=1)
        parser.add_argument("--s_lambda_reg_mov", type=float, default=1)
        parser.add_argument("--s_lambda_ent_flt", type=float, default=1)
        parser.add_argument("--s_lambda_ent_flt_edge", type=float, default=1)
        parser.add_argument("--s_lambda_ent", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_mean_vid", type=float, default=1)
        parser.add_argument("--s_lambda_soft_ce_lyt", type=float, default=1)
        parser.add_argument("--s_lambda_ce_lyt", type=float, default=1)
        parser.add_argument("--s_lambda_ce_lyt_obj", type=float, default=1)
        parser.add_argument("--s_lambda_reg_fg", type=float, default=1)
        parser.add_argument("--s_lambda_soft_ce_mean_lyt", type=float, default=1)
        parser.add_argument("--s_lambda_ce_mean_lyt", type=float, default=1)
        parser.add_argument("--s_lambda_ce_mean_lyt_obj", type=float, default=1)
        parser.add_argument("--s_lambda_l1_flow", type=float, default=1)
        parser.add_argument("--s_lambda_l1_flow_mov_obj", type=float, default=1)
        parser.add_argument("--s_lambda_l1_flow_other", type=float, default=1)
        parser.add_argument('--s_mean_vid_bg_mode', type=str, nargs='+', default=["norm"], choices=["norm", "raw"])
        parser.add_argument("--s_post_refine_mean_vid", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_lambda_color_diversity", type=float, default=1)
        parser.add_argument("--s_lambda_rec_alpha_obj", type=float, default=1)
        parser.add_argument("--s_lambda_reg_kcenters", type=float, default=1)
        parser.add_argument("--s_lambda_reg_surface", type=float, default=1)
        parser.add_argument("--s_lambda_inter_obj", type=float, default=1)
        parser.add_argument("--s_lambda_expansion", type=float, default=1)
        parser.add_argument("--s_lambda_shift_penalty", type=float, default=1)
        parser.add_argument("--s_lambda_rec_edge_vid", type=float, default=1)
        parser.add_argument("--s_lambda_rec_edge_obj", type=float, default=1)
        parser.add_argument("--s_lambda_reg_edge", type=float, default=1)
        parser.add_argument("--s_lambda_reg_obj_pose", type=float, default=1)
        parser.add_argument("--s_lambda_reg_alpha", type=float, default=1)
        parser.add_argument("--s_lambda_reg_color", type=float, default=1)
        parser.add_argument("--s_warmup_sharp_vid_iter", type=int, default=0)
        parser.add_argument("--s_warmup_pxl_vid_iter", type=int, default=0)
        parser.add_argument("--s_cosine_warmup_pxl_vid", type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_warmup_pxl_bg_iter", type=int, default=0)
        parser.add_argument("--s_warmup_pxl_obj_iter", type=int, default=0)
        parser.add_argument("--s_warmup_reg_raw_vid_iter", type=int, default=0)
        parser.add_argument("--s_warmup_reg_raw_vid_mul", type=int, default=100)
        parser.add_argument("--s_reg_raw_vid_subidx", type=int, nargs="+", default=[0, 2, 6])
        parser.add_argument("--s_warmup_reg_raw_vid_plus_iter", type=int, default=0)
        parser.add_argument("--s_lambda_sharp_vid", type=float, default=1)
        parser.add_argument("--s_lambda_iou_obj", type=float, default=1)
        parser.add_argument("--s_lambda_push_raw_vid", type=float, default=1)
        parser.add_argument("--s_lambda_push_obj_pose", type=float, default=1)
        parser.add_argument("--s_lambda_reg_raw_bg", type=float, default=1)
        parser.add_argument("--s_lambda_reg_raw_vid", type=float, default=1)
        parser.add_argument("--s_lambda_reg_raw_vid_plus", type=float, default=1)
        parser.add_argument("--s_lambda_reg_raw_vid_plus2", type=float, default=1)
        parser.add_argument("--s_lambda_reg_raw_vid_plus_mean", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_inter_obj", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_raw_bg", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_comp", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_raw_fg", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_raw_obj", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_vid", type=float, default=1)
        parser.add_argument('--s_img_reg_every', type=int, default=16, help='interval for applying r1 regularization')
        parser.add_argument("--s_lambda_pxl", type=float, default=1)
        parser.add_argument("--s_lambda_qnt", type=float, default=1)
        parser.add_argument("--s_lambda_cls_obj_ctx", type=float, default=1)
        parser.add_argument("--s_lambda_cls_obj_gen", type=float, default=1)
        parser.add_argument("--s_lambda_reg_obj_ctx", type=float, default=1)
        parser.add_argument("--s_lambda_reg_obj_gen", type=float, default=1)
        parser.add_argument("--s_lambda_reg_map_ctx", type=float, default=1)
        parser.add_argument("--s_lambda_reg_map_gen", type=float, default=1)
        parser.add_argument("--s_lambda_conf", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_conf_ctx", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_ctx", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_gen", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_swap", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_aug_ctx", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_warp_ctx", type=float, default=1)
        parser.add_argument("--s_lambda_rec_pred", type=float, default=1)
        parser.add_argument("--s_lambda_dis_pred", type=float, default=1)
        parser.add_argument("--s_lambda_vgg", type=float, default=10)
        parser.add_argument("--s_lambda_vgg_ctx", type=float, default=10)
        parser.add_argument("--s_lambda_vgg_gen", type=float, default=10)
        parser.add_argument("--s_lambda_rec_swap_pose", type=float, default=1)
        parser.add_argument("--s_lambda_adv", type=float, default=1)
        parser.add_argument("--s_lambda_delta", type=float, default=1)
        parser.add_argument("--s_lambda_dis", type=float, default=1)
        parser.add_argument("--s_lambda_collapse", type=float, default=1)
        parser.add_argument("--s_lambda_match_bg_alpha_pre", type=float, default=1)
        parser.add_argument("--s_lambda_match_bg_alpha_post", type=float, default=1)
        parser.add_argument("--s_lambda_r1", type=float, default=10)
        parser.add_argument("--s_lambda_flow", type=float, default=1)
        parser.add_argument("--s_lambda_entropy", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_inter_bg", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_reg_bg", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_reg_fg", type=float, default=1)
        parser.add_argument("--s_lambda_extent", type=float, default=1)
        parser.add_argument("--s_lambda_spread", type=float, default=1)
        parser.add_argument("--s_lambda_bound_spread", type=float, default=1)
        parser.add_argument("--s_lambda_warped_latent_ctx", type=float, default=1)
        parser.add_argument("--s_lambda_pred_latent_ctx", type=float, default=1)
        parser.add_argument("--s_lambda_triv_flow", type=float, default=1)
        parser.add_argument("--s_lambda_pts_reg", type=float, default=1)
        parser.add_argument("--s_lambda_scale", type=float, default=1)
        parser.add_argument("--s_lambda_pts_dis", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_obj", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_bg", type=float, default=1)
        parser.add_argument("--s_lambda_pts_rest", type=float, default=1)
        parser.add_argument("--s_lambda_delta_bg_pose", type=float, default=1)
        parser.add_argument("--s_lambda_delta_pose", type=float, default=1)
        parser.add_argument("--s_lambda_pxl_obj_alpha", type=float, default=1)
        parser.add_argument("--s_lambda0_pxl_obj_alpha", type=float, default=0.1)
        parser.add_argument('--s_ada_pts_rest', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_ada_pts_rest_detach', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_mask_input', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_commit_latent', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_blur_pxl', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_blur_alpha', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_blur_edge', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_blur_delta', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_l1_pxl', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_blur_in', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument("--s_blur_sigma", type=float, default=3.0)
        parser.add_argument("--s_min_spread", type=float, default=-0.5)
        parser.add_argument("--s_max_spread", type=float, default=0.5)
        parser.add_argument("--s_min_delta_match", type=float, default=0.1)
        parser.add_argument("--s_max_delta_match", type=float, default=0.15)
        parser.add_argument('--s_swap', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_swap2', type=str2bool, nargs='?', const=True, default=False)
        parser.add_argument('--s_time_dropout', type=str2bool, nargs='?', const=True, default=False)

        # for training
        parser.add_argument('--s_optimizer', type=str, default='adam', choices=["adam", "adamw"])
        parser.add_argument('--s_beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--s_beta2', type=float, default=0.99, help='momentum term of adam')
        parser.add_argument('--s_lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--s_wd', type=float, default=0.0, help='weight decay for adamw')
        parser.add_argument('--s_gan_loss', type=str, default="hinge", choices=['original', 'hinge', 'wgan', 'logistic', 'wgan-eps'])
        parser.add_argument('--s_clip_value', type=float, default=0, help='value for gradient clipping, no clipping if 0')
        parser.add_argument('--s_ema_networks', type=str, default=[], nargs="+", help='id of networks for which to compute ema')
        parser.add_argument('--s_ema_beta', type=float, default=0.995, help='beta for ema of network weights')
        parser.add_argument('--s_ema_freq', type=int, default=1, help='frequency at which to update ema weights')

        # for loading
        parser.add_argument('--s_load_path', type=str, default=None, help='load model from which_iter at specified folder')
        parser.add_argument('--s_which_iter', type=str, default=0, help='load the model from specified iteration, or string')
        parser.add_argument('--s_not_strict', type=str2bool, nargs='?', const=True, default=False, help='whether checkpoint exactly matches network architecture')
        parser.add_argument('--s_from_multi', type=str2bool, nargs='?', const=True, default=False, help='load checkpoints trained on multi gpu mode to single gpu')

        # for text encoder model
        parser.add_argument('--s_use_b', type=str2bool, nargs='?', const=True, default=False)

        return parser

    def update_defaults(self, opt, parser):
        if "None" in opt.img_skip_nums:
            parser.set_defaults(img_skip_nums=[1 for _ in opt.img_modes])
        if "None" in opt.img_acc_nums:
            parser.set_defaults(img_acc_nums=[1 for _ in opt.img_modes])
        if "None" in opt.vid_skip_nums:
            parser.set_defaults(vid_skip_nums=[1 for _ in opt.vid_modes])
        if "None" in opt.vid_acc_nums:
            parser.set_defaults(vid_acc_nums=[1 for _ in opt.vid_modes])
        for prefix in [""]:
            if getattr(opt, f"{prefix}dataset") == "cityscapes":
                parser.set_defaults(**{f"{prefix}dataroot": "datasets/cityscapes"})
                parser.set_defaults(**{f"{prefix}true_ratio": 2})
                parser.set_defaults(**{f"{prefix}aspect_ratio": 2})
                parser.set_defaults(**{f"{prefix}true_dim": 1024})
                parser.set_defaults(**{f"{prefix}categories": None})
                parser.set_defaults(**{f"{prefix}no_h_flip": True})
                parser.set_defaults(**{f"{prefix}no_v_flip": True})
                parser.set_defaults(**{f"{prefix}from_vid": False})
                parser.set_defaults(**{f"{prefix}num_lyt": 20})
                parser.set_defaults(**{f"{prefix}lyt_model": "deeplabv3"})
                parser.set_defaults(**{f"{prefix}flow_model": "raft"})
                parser.set_defaults(**{f"{prefix}palette": CITYSCAPES_PALETTE})
                parser.set_defaults(**{f"{prefix}fg_idx": [0, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18, 19]}) # 9
                parser.set_defaults(**{f"{prefix}bg_idx": [1, 2, 3, 10, 11]}) # 9
                parser.set_defaults(**{f"{prefix}other_idx": [9]})  # 9
            if getattr(opt, f"{prefix}dataset") == "kitti":
                parser.set_defaults(**{f"{prefix}dataroot": "datasets/kitti"})
                parser.set_defaults(**{f"{prefix}true_ratio": 3.25})
                parser.set_defaults(**{f"{prefix}aspect_ratio": 3.25})
                parser.set_defaults(**{f"{prefix}true_dim": 375})
                parser.set_defaults(**{f"{prefix}categories": None})
                parser.set_defaults(**{f"{prefix}no_h_flip": True})
                parser.set_defaults(**{f"{prefix}no_v_flip": True})
                parser.set_defaults(**{f"{prefix}from_vid": False})
                parser.set_defaults(**{f"{prefix}num_lyt": 19})
                parser.set_defaults(**{f"{prefix}lyt_model": "deeplabv3"})
                parser.set_defaults(**{f"{prefix}flow_model": "raft"})
                parser.set_defaults(**{f"{prefix}palette": KITTI_PALETTE})
                parser.set_defaults(**{f"{prefix}fg_idx": [3, 4, 5, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18]}) # 8
                parser.set_defaults(**{f"{prefix}bg_idx": [0, 1, 2, 9, 10]}) # 8
                parser.set_defaults(**{f"{prefix}other_idx": [8]})  # 8
        return parser

    def gather_options(self):
        # initialize parser with basic
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        # get options
        opt = parser.parse_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)
            # get options
            opt = parser.parse_args()

        # modify some defaults based on parser
        parser = self.update_defaults(opt, parser)
        opt = parser.parse_args()

        self.parser = parser
        return opt

    def print_options(self, opt, opt_type, opt_prefix=""):
        def dash_pad(s, length=50):
            num_dash = max(length - len(s) // 2, 0)
            return '-' * num_dash
        opt_str = opt_type + " Options"
        message = ''
        message += dash_pad(opt_str) + ' ' + opt_str + ' ' + dash_pad(opt_str) + '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(opt_prefix + k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        end_str = opt_type + " End"
        message += dash_pad(end_str) + ' ' + end_str + ' ' + dash_pad(end_str) + '\n'
        print(message)

    def option_file_path(self, opt, signature, makedir=False):
        expr_dir = os.path.join(opt.save_path, "checkpoints", signature)
        if makedir:
            utils.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt, signature):
        file_name = self.option_file_path(opt, signature, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, opt.load_signature, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def split_options(self, opt):
        base_opt = Namespace()
        synthesizer_opt = Namespace()
        for k, v in sorted(vars(opt).items()):
            if k.startswith("s_"):
                setattr(synthesizer_opt, k[2:], v)
            else:
                setattr(base_opt, k, v)
        return base_opt, synthesizer_opt

    def copy_options(self, target_options, source_options, new_only=False):
        for k, v in sorted(vars(source_options).items()):
            if not (new_only and k in target_options):
                setattr(target_options, k, v)

    def process_base(self, base_opt, signature):
        # set gpu ids
        str_ids = base_opt.gpu_ids.split(',')
        base_opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                base_opt.gpu_ids.append(id)

        # set additional paths
        base_opt.checkpoint_path = os.path.join(base_opt.save_path, "checkpoints", signature)
        base_opt.log_path = os.path.join(base_opt.save_path, "logs", signature)
        base_opt.result_path = os.path.join(base_opt.save_path, "results", signature)

        assert (base_opt.dim & (base_opt.dim - 1)) == 0, f"Dim {base_opt.dim} must be power of two."

        # set width size
        if base_opt.fixed_crop is None:
            base_opt.width_size = int(base_opt.dim * base_opt.aspect_ratio)
            base_opt.height_size = int(base_opt.width_size / base_opt.aspect_ratio)
        else:
            base_opt.height_size, base_opt.width_size = base_opt.fixed_crop

        # set resize factor
        if base_opt.resize_img is not None:
            print("Resize img to", base_opt.resize_img)
            assert base_opt.original_size is not None
            base_opt.resize_factor_h = base_opt.true_dim / base_opt.original_size[0]
            base_opt.resize_factor_w = base_opt.true_dim * base_opt.true_ratio / base_opt.original_size[1]
        else:
            base_opt.resize_factor_h = 1
            base_opt.resize_factor_w = 1

        # set signature
        base_opt.signature = signature

    def parse(self, load_synthesizer=False, save=False):
        opt = self.gather_options()
        signature = opt.datetime + "-" + opt.name

        base_opt, synthesizer_opt = self.split_options(opt)

        if 'SLURM_JOB_NUM_NODES' in os.environ and int(os.environ['SLURM_JOB_NUM_NODES']) >= 1:
            is_main = int(os.environ['SLURM_NODEID']) == 0 and int(os.environ['LOCAL_RANK']) == 0
        else:
            is_main = int(os.environ['LOCAL_RANK']) == 0

        if is_main:
            if save:
                self.save_options(opt, signature)
            self.print_options(base_opt, "Base")
            if load_synthesizer:
                self.print_options(synthesizer_opt, "Synthesizer", "s_")

        self.process_base(base_opt, signature)

        self.copy_options(synthesizer_opt, base_opt)

        self.base_opt = base_opt
        # self.extra_base_opt = extra_base_opt if load_extra_base else None
        self.synthesizer_opt = synthesizer_opt if load_synthesizer else None

        self.opt = {"base": self.base_opt,
                    "synthesizer": self.synthesizer_opt}

        return self.opt