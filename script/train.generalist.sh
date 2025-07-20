export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0

# LoRA training script for 3D-R1
# This script enables LoRA for efficient fine-tuning

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    main.py \
    --checkpoint_dir ./checkpoints/fast_eval \
    --dataset scenecold \
    --vocab "qwen/Qwen2.5-7B" \
    --qformer_vocab "bert-base-uncased" \
    --captioner 3dr1 \
    --detector point_encoder \
    --max_des_len 256 \
    --max_gen_len 64 \
    --batchsize_per_gpu 4 \
    --base_lr 1e-4 \
    --final_lr 1e-6 \
    --weight_decay 0.01 \
    --clip_gradient 1.0 \
    --warm_lr 1e-7 \
    --warm_lr_epochs 10 \
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
    --eval_batch_size 8 \
    --eval_max_samples 1000 \
    --eval_use_fp16 \
    --eval_skip_metrics \
    --seed 42