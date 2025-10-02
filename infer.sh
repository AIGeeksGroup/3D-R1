# Full multimodal inference
python main.py \
    --test_only \
    --test_ckpt ./checkpoints/checkpoint_rl.pth \
    --dataset scannnet \
    --vocab qwen/Qwen2.5-7B \
    --qformer_vocab bert-base-large \
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