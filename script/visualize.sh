#!/bin/bash
python visualization/bbox_visualization.py \
  --point_cloud /path/to/scene.ply \
  --predictions /path/to/preds.npz \
  --radius 0.01 \
  --max_points 500000 \
  --spawn