#!/bin/bash

LVD_TAG=$1
FLP_TAG=$2
WIF_TAG=$3
GPU_IDS="0"
NUM_GPUS=1
NUM_NODES=1
DATETIME=`date "+%Y-%m-%d-%H:%M:%S"`
TORCHRUN="torch.distributed.run --nnodes=${NUM_NODES} --nproc_per_node=${NUM_GPUS} --rdzv_backend=c10d"

CUDA_VISIBLE_DEVICES=${GPU_IDS} python -m ${TORCHRUN} helpers/synthesizer_evaluator.py \
 --name test_kitti --datetime ${DATETIME} --gpu_ids ${GPU_IDS} \
 --dataset "kitti" --load_all --dim 128 --load_dim 256 --true_dim 256 --flow_dim 128 --vid_len 10 --num_workers 8 --num_workers_eval 1 \
 --num_iter 1000000 --num_iter_eval 10000 --save_latest_freq 1000 --log_freq 10000 \
 --s_patch_size 16 --s_latent_shape 8 26 --s_embed_dim 512 --s_num_obj 16 --s_num_timesteps 5 \
 --s_use_pe --s_use_ii --s_use_pg \
 --vid_modes 'vid_prediction' \
 --s_blur_pxl --s_blur_sigma 2.0 \
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
 --s_blur_edge --remap_lyt 12 18 17 18 6 5 7 5 --s_reg_bg_mul 0.25  \
 --s_lambda_cell_dis 10 --s_lambda_l1_flow 100 \
 --max_zoom 1.3 --no_v_flip "False" --colorjitter 0.5 --colorjitter_no_contrast \
 --s_lambda_reg_mov 10 --s_use_lyt_filtering --s_use_fg --s_use_lyt_filtering \
 --s_use_lyt_opacity --s_swap_flt --s_mov_obj_thresh 0.005 --s_use_dominant_flow_other \
 --s_pred_cls --s_weight_cls --s_min_cls 0.1 \
 --s_not_strict --s_which_iter "latest" --s_load_path "checkpoints/"${LVD_TAG} \
 --s_ctx_mode "prev" --s_ii_score --s_ii_ab \
 --s_edge_size 15 --s_flow_thresh 0.02 --s_bg_mul 1.2 \
 --s_unconstrained_pose_decoder --s_lambda_rec_occ_score 0.01 \
 --s_oe_num_timesteps 5 --s_pg_num_timesteps 10 \
 --s_ii_depth 6 \
 --s_ii_iter "latest" --s_ii_load_path "checkpoints/"${WIF_TAG} \
 --s_pg_iter "latest" --s_pg_load_path "checkpoints/"${FLP_TAG} \
 --batch_size_vid 1 --eval_phase "test" --s_input_rgb "False" \
 --s_use_last_pose_decoder --s_bg_mul_pose_decoder 1.2  --s_restrict_to_ctx \

