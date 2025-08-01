export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NVIDIA_TF32_OVERRIDE=0

python main.py \
    --use_color --use_normal \
    --detector detector_Vote2Cap_DETR \
    --captioner 3dr1 \
    --pretrained_weights ./pretrained/vote2cap-detr/scannet_vote2cap_detr_XYZ_COLOR_NORMAL.pth \
    --warm_lr_epochs 1 \
    --dataset scenecold_dataset \
    --vocab qwen/Qwen2.5-7B \
    --qformer_vocab bert-base-large \
    --checkpoint_dir ./ckpts-sft-5/qwen2-7b/ll3da-generalist \
    --max_epoch 2 \
    --dist_url tcp://localhost:12345 \
    --eval_every_iteration 3000 \
    --start_eval_after -1 \
    --save_every 1000 \
    --criterion 'CiDEr' \
    --freeze_detector --freeze_llm \
    --batchsize_per_gpu 4 --ngpus 4 --base_lr 1e-2 --final_lr 1e-6 \
    --max_des_len 512 \
    --max_prompt 1
