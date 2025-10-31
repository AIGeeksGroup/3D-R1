#!/bin/bash

# Usage:
#   bash script/visualize.sh /path/to/scene.ply scene0000_00 outputs/preds npz
# Args:
#   $1: point cloud path (default: /path/to/scene.ply)
#   $2: scan name (default: scene0000_00)
#   $3: output dir for predictions (default: outputs/preds)
#   $4: format: npz|json (default: npz)

POINT_CLOUD=${1:-/path/to/scene.ply}
SCAN_NAME=${2:-scene0000_00}
OUTPUT_DIR=${3:-outputs/preds}
FORMAT=${4:-npz}

# 1) Export predictions first
python visualization/export_predictions.py \
  --output_dir "$OUTPUT_DIR" \
  --format "$FORMAT" \
  --max_scenes 1 \
  --test_only \
  --test_ckpt ./checkpoints/checkpoint_rl.pth \
  --dataset scannet \
  --vocab qwen/Qwen2.5-7B \
  --qformer_vocab google-bert/bert-base-uncased \
  --checkpoint_dir ./results \
  --use_color --use_normal \
  --detector point_encoder \
  --captioner 3dr1 \
  --use_additional_encoders --use_depth --use_image \
  --enable_dynamic_views --use_pytorch3d_rendering --use_multimodal_model

PRED_PATH="$OUTPUT_DIR/$SCAN_NAME.$FORMAT"

# 2) Visualize
python visualization/bbox_visualization.py \
  --point_cloud "$POINT_CLOUD" \
  --predictions "$PRED_PATH" \
  --radius 0.01 \
  --max_points 500000 \
  --spawn