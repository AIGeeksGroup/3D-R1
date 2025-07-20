## 3D-R1 Visualization

This folder provides a rerun-based 3D visualizer for rendering point clouds alongside 3D-R1 detection boxes — including a reference workflow to go from a custom RGB video to a point-cloud with bounding boxes, and record a demo video.

### Requirements
- Python 3.9
- Required: `rerun-sdk`
- For point cloud loading (at least one):
  - Recommended: `trimesh`
  - Alternative: `plyfile`

Install:
```bash
pip install rerun-sdk trimesh plyfile
```

### Inputs
- Point cloud file: `.ply` is recommended. Other mesh/point formats supported by `trimesh` may work.
- Predictions file: `.npz` or `.json`.

`.npz` options:
- With `corners` shaped `[N, 8, 3]`: internally converted to center/half_size with identity rotation.
- Or with `centers`([N,3]), `sizes`([N,3]), `headings`([N]): uses Z-axis rotation; `half_size = sizes / 2`.

Optional fields: `labels`([N]), `scores`([N]).

`.json` prediction list item schema:
```json
{
  "center": [x, y, z],
  "scale": [sx, sy, sz],
  "rotation": [[...],[...],[...]],  // optional 3x3; identity if missing
  "label": "chair",               // optional
  "score": 0.92                    // optional
}
```

### Scripts and Entrypoints
- Main visualizer: `visualization/bbox_visualization.py`
- Convenience script: `script/visualize.sh`

#### Run directly (use `--spawn` to open rerun in a separate window)
```bash
python visualization/bbox_visualization.py \
  --point_cloud /path/to/scene.ply \
  --predictions /path/to/preds.npz \
  --radius 0.01 \
  --max_points 500000 \
  --spawn
```

Alternatively, via the convenience script:
```bash
bash script/visualize.sh
```
Update the placeholder paths in the script before running.

### Common Arguments
- `--point_cloud, -p`: path to the point cloud file (required)
- `--predictions, -d`: path to predictions (.npz or .json, required)
- `--radius`: point radius, default `0.01`
- `--max_points`: max number of points to render, default `1_000_000` (random subsample if exceeded)
- Rerun common args: e.g., `--spawn`, `--save`, etc. (see rerun help)

Show full help:
```bash
python visualization/bbox_visualization.py --help
```

### Behavior and UI
- Right-handed coordinates, Z-up.
- Points rendered with the given radius; predicted boxes appear progressively along a time axis for demo-style playback.
- Use the rerun UI to rotate/zoom, toggle views and entities.

### Troubleshooting
- `rerun is required`: install with `pip install rerun-sdk`.
- Point cloud fails to load: install `trimesh`; if still failing, try `plyfile` and ensure `.ply` has `x,y,z` (optional `red,green,blue`).
- `.npz` keys mismatch: ensure you follow one of the supported key layouts above.

### Minimal examples
```bash
# .npz with centers/sizes/headings
python visualization/bbox_visualization.py \
  -p data/scannet/scene0000_00.ply \
  -d outputs/scene0000_00_preds.npz \
  --spawn
```

```bash
# .json list
python visualization/bbox_visualization.py \
  -p data/scannet/scene0000_00.ply \
  -d outputs/scene0000_00_preds.json \
  --radius 0.008 --max_points 800000 --spawn
```

---

## End-to-end: From a custom video to a bounding-box video

Below is a reference workflow for producing a point-cloud with 3D-R1 detections and recording a short demo video.

### 1) Reconstruct a point cloud from your video (SLAM)

Use any recent RGB SLAM/reconstruction system to obtain a dense colored point cloud from your video, e.g. SLAM3R or MASt3R-SLAM. Follow their README to run on your video and export a point cloud.

- Tips for cleaner point clouds:
  - Downsample frames to reduce noise and memory.
  - Raise confidence thresholds for triangulation.
  - Apply simple denoising when exporting each keyframe point cloud, such as removing statistical outliers (example in Open3D):

```python
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(valid_pts)
pcd.colors = o3d.utility.Vector3dVector(valid_colors)
pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=10, std_ratio=1.5)
points = np.asarray(pcd.points)
color = (np.asarray(pcd.colors) * 255.0).astype(np.uint8)
```

### 2) Align the point cloud (Z-up, Manhattan-style if possible)

3D-R1 expects Z-up right-handed coordinates. Align your reconstruction to ScanNet-style orientation (Z-up; major walls aligned with XY planes) when possible. You can estimate a Manhattan frame from images, video, or surface normals, or perform manual alignment in DCC tools (e.g., Blender).

### 3) Ensure a roughly metric scale

3D-R1 is trained with metric scale (1 unit ≈ 1 meter). If your SLAM export is scale-ambiguous, rescale the point cloud to a plausible room height (e.g., ~2.5m for indoor scenes):

```python
import open3d as o3d
import numpy as np

pc = o3d.io.read_point_cloud(point_cloud_file)
pts = np.asarray(pc.points)
min_z, max_z = np.min(pts[:, 2]), np.max(pts[:, 2])
height = max_z - min_z
scale = 2.5 / max(height, 1e-6)
pc.points = o3d.utility.Vector3dVector(pts * scale)
o3d.io.write_point_cloud("scaled_point_cloud.ply", pc)
```

### 4) Run 3D-R1 to obtain 3D detections

Produce detection predictions for your scene point cloud and save them as one of the supported formats (`.npz` or `.json`) expected by `bbox_visualization.py`.

- `.npz` (recommended): provide either `corners` ([N,8,3]) or `centers`/`sizes`/`headings` with optional `labels`/`scores`.
- `.json`: list of boxes with `center`, `scale`, optional `rotation` (3×3), `label`, `score`.

If you already have an evaluation or inference script producing detections, point its output here. Otherwise, adapt your pipeline to emit one of the supported formats.

### 5) Visualize and record a short video

Use the rerun visualizer to play back detections over time and record:

```bash
python visualization/bbox_visualization.py \
  -p scaled_point_cloud.ply \
  -d outputs/scene_preds.npz \
  --spawn --save outputs/scene_demo.rrd
```

Notes on recording:
- `--save` writes a `.rrd` recording; open it later via `rerun outputs/scene_demo.rrd` and screen-record the playback.
- For scripted video export, capture a sequence of screenshots (e.g., OS-level or tooling), then assemble with `ffmpeg`:

```bash
ffmpeg -framerate 30 -i frames/frame_%05d.png -pix_fmt yuv420p outputs/scene_demo.mp4
```

This yields a short demo video showing your reconstructed scene with 3D bounding boxes evolving over time.


