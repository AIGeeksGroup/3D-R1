#!/bin/bash

# RL Training Script with Unified Datasets for 3D-R1
# This script uses unified datasets that include target_text for reward computation

export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    main_rl.py \
    --checkpoint_dir ./checkpoints/rl_unified_training \
    --dataset scanqa \
    --vocab "qwen/Qwen2.5-7B" \
    --qformer_vocab "bert-base-uncased" \
    --captioner 3dr1 \
    --detector point_encoder \
    --max_des_len 256 \
    --max_gen_len 64 \
    --batchsize_per_gpu 2 \
    --base_lr 1e-5 \
    --final_lr 1e-7 \
    --weight_decay 0.01 \
    --clip_gradient 1.0 \
    --warm_lr 1e-7 \
    --warm_lr_epochs 5 \
    --max_epoch 50 \
    --log_every 10 \
    --save_every 1000 \
    --eval_every_iteration 2000 \
    --start_eval_after 1000 \
    --criterion "CiDEr" \
    --use_color \
    --use_height \
    --use_multiview \
    --use_additional_encoders \
    --use_depth \
    --use_image \
    --max_multiview_images 4 \
    --max_multiview_depth 4 \
    --depth_encoder_dim 256 \
    --image_encoder_dim 256 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --enable_dynamic_views \
    --view_selection_weight 0.05 \
    --use_unified_dataset \
    --use_rl_training \
    --grpo_beta 0.1 \
    --grpo_lambda 0.95 \
    --grpo_kl_penalty 0.1 \
    --grpo_max_grad_norm 1.0 \
    --seed 42

