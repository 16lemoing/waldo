#!/bin/bash

LVD_TAG=$1
GPU_IDS="0,1,2,3"
NUM_GPUS=4
NUM_NODES=1
DATETIME=`date "+%Y-%m-%d-%H:%M:%S"`
TORCHRUN="torch.distributed.run --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS} --rdzv_backend=c10d"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m ${TORCHRUN} helpers/synthesizer_trainer.py \
 --name train_wif_cityscapes --datetime ${DATETIME} --gpu_ids ${GPU_IDS} \
 --dataset "cityscapes" --dim 128 --load_dim 512 --true_dim 512 --flow_dim 128 --vid_len 5 --load_vid_len 14 --load_n_plus_1 --num_workers 8 --num_workers_eval 1 --shuffle_valid \
 --num_iter 1000000 --num_iter_eval 10000 --save_latest_freq 1000 --log_freq 10000 \
 --s_patch_size 16 --s_latent_shape 8 16 --s_embed_dim 512 --s_num_obj 16 --s_num_timesteps 5 \
 --s_use_pe --s_use_ii \
 --vid_modes 'vid_inpainting' --s_vid_inpainting_losses "sharp_vid" "lpips_vid" \
 --s_blur_pxl --s_blur_sigma 2.0 \
 --batch_size_vid 8 --max_batch_eval_vid 8 \
 --s_lambda_pts_rest 20 \
 --s_bound_alpha --s_l1_pxl \
 --s_oe_depth 2 --s_pe_depth 2 --s_bound_rest --s_soft_bound_rest \
 --s_pe_use_scorer --s_oe_use_decoder "False" --s_pe_decoder_init_mode "five" \
 --s_has_bg \
 --s_pe_refiner_init_mode "mfive" \
 --s_pe_estimator_init_mode "zero" --s_pad_obj_alpha 3 --s_pad_bg_alpha 3 \
 --s_init_scale_obj 0.25 --s_mul_scale_obj 0.25 --s_mul_delta_obj 0.2 \
 --s_circle_translate_bias --s_circle_translate_radius 0.2 --s_num_perm_grid 1 \
 --skip_first --s_ctx_len 4 --load_lyt --load_flow --s_input_lyt --s_input_flow \
 --s_blur_edge --remap_lyt 13 19 18 19 7 6 8 6 --s_reg_bg_mul 0.25  \
 --s_lambda_cell_dis 10 --s_lambda_l1_flow 100 \
 --max_zoom 1.3 --no_v_flip "False" --colorjitter 0.5 --colorjitter_no_contrast \
 --s_lambda_reg_mov 10 --s_use_lyt_filtering --s_use_fg --s_use_lyt_filtering \
 --s_use_lyt_opacity --s_swap_flt --s_mov_obj_thresh 0.005 --s_use_dominant_flow_other \
 --s_pred_cls --s_weight_cls --s_min_cls 0.1 \
 --s_not_strict --s_which_iter "latest" --s_load_path "checkpoints/"${LVD_TAG} \
 --s_ctx_mode "prev" --s_ii_score --s_ii_ab \
 --s_edge_size 15 --s_flow_thresh 0.02 --s_bg_mul 1.2 \
 --s_ii_depth 6 --vid_metric "loss" --s_input_rgb "False"