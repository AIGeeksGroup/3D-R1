export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0

# RL training script for 3D-R1
# This script performs RL training using GRPO with SFT weights as initialization

python main_rl.py \
    --use_color --use_normal \
    --detector point_encoder \
    --captioner 3dr1 \
    --sft_checkpoint ./ckpts-sft-5/qwen2-7b/ll3da-generalist/checkpoint_best.pth \
    --pretrained_weights ./pretrained/scannet_point.pth \
    --dataset scenecold_dataset \
    --vocab qwen/Qwen2.5-7B \
    --qformer_vocab bert-base-large \
    --checkpoint_dir ./ckpts-rl/qwen2-7b/ll3da-generalist-rl \
    --max_epoch 10 \
    --dist_url tcp://localhost:12345 \
    --eval_every_iteration 1000 \
    --save_every 500 \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 2 --ngpus 4 \
    --max_des_len 512 \
    --max_prompt 1 \
    --rl_beta 0.1 \
    --rl_lr 1e-5 \
    --rl_num_epochs 4 \
    --rl_max_grad_norm 1.0 \
    --enable_dynamic_views \
    --view_selection_weight 0.1 \
    --use_lora \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.1
