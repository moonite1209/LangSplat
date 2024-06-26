#!/bin/bash
CASE_NAME="waldo_kitchen"

# path to lerf_ovs/label
gt_folder="../data/lerf_ovs/label"

root_path="../"

python evaluate_iou_loc.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/output \
        --ae_ckpt_dir ${root_path}/ckpt \
        --output_dir ${root_path}/eval_result \
        --json_folder ${gt_folder} \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 
        