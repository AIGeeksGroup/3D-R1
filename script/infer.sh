#!/bin/bash

# Inference script for 3D-R1
# Edit paths/flags below as needed.

python main.py \
    --test_only \
    --test_ckpt ./checkpoints/checkpoint_rl.pth \
    --dataset scannet \
    --vocab qwen/Qwen2.5-7B \
    --qformer_vocab google-bert/bert-base-uncased \
    --checkpoint_dir ./results \
    --use_color --use_normal \
    --detector point_encoder \
    --captioner 3dr1 \
    --use_additional_encoders \
    --use_depth \
    --use_image \
    --depth_encoder_dim 256 \
    --image_encoder_dim 256 \
    --enable_dynamic_views \
    --view_selection_weight 0.1 \
    --use_pytorch3d_rendering \
    --use_multimodal_model


