import argparse
import sys
import os
import json
import numpy as np

try:
    import rerun as rr
    import rerun.blueprint as rrb
except Exception as e:
    raise RuntimeError("rerun is required: pip install rerun-sdk") from e


def _load_point_cloud(pcd_path: str):
    try:
        import trimesh

        mesh = trimesh.load(pcd_path, force="mesh")
        if hasattr(mesh, "vertices"):
            points = np.asarray(mesh.vertices, dtype=np.float32)
            colors = None
            if hasattr(mesh, "visual") and hasattr(mesh.visual, "vertex_colors"):
                vc = np.asarray(mesh.visual.vertex_colors)
                if vc.shape[-1] >= 3:
                    colors = vc[:, :3]
            if colors is None:
                colors = np.full_like(points, 200, dtype=np.uint8)
            return points, colors
    except Exception:
        pass

    try:
        from plyfile import PlyData

        with open(pcd_path, "rb") as f:
            plydata = PlyData.read(f)
        v = plydata["vertex"]
        points = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)
        if {"red", "green", "blue"}.issubset(v.data.dtype.names):
            colors = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(
                np.uint8
            )
        else:
            colors = np.full_like(points, 200, dtype=np.uint8)
        return points, colors
    except Exception as e:
        raise RuntimeError(
            f"Failed to load point cloud {pcd_path}. Install spatiallm or provide a PLY readable by plyfile/trimesh."
        ) from e


def _load_predictions(pred_path: str):
    ext = os.path.splitext(pred_path)[1].lower()
    if ext in [".npz"]:
        data = np.load(pred_path, allow_pickle=True)
        if "corners" in data:
            corners = data["corners"]
            N = corners.shape[0]
            centers = corners.mean(axis=1)
            half_sizes = 0.5 * (corners.max(axis=1) - corners.min(axis=1))
            rotations = np.tile(np.eye(3)[None, ...], (N, 1, 1))
            labels = data["labels"] if "labels" in data else None
            scores = data["scores"] if "scores" in data else None
            return centers, half_sizes, rotations, labels, scores
        elif all(k in data for k in ["centers", "sizes", "headings"]):
            centers = data["centers"].astype(np.float32)
            sizes = data["sizes"].astype(np.float32)
            headings = data["headings"].astype(np.float32)
            half_sizes = 0.5 * sizes
            cosv = np.cos(headings)
            sinv = np.sin(headings)
            rotations = np.stack(
                [
                    np.stack([cosv, -sinv, np.zeros_like(cosv)], axis=-1),
                    np.stack([sinv,  cosv, np.zeros_like(cosv)], axis=-1),
                    np.stack([np.zeros_like(cosv), np.zeros_like(cosv), np.ones_like(cosv)], axis=-1),
                ],
                axis=-2,
            )
            labels = data["labels"] if "labels" in data else None
            scores = data["scores"] if "scores" in data else None
            return centers, half_sizes, rotations, labels, scores
        else:
            raise RuntimeError("Unsupported .npz keys. Expect corners or centers/sizes/headings.")
    elif ext in [".json"]:
        with open(pred_path, "r") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            centers = []
            half_sizes = []
            rotations = []
            labels = []
            scores = []
            for it in obj:
                centers.append(it["center"])  # 3
                half_sizes.append(0.5 * np.asarray(it["scale"]))
                rotations.append(it.get("rotation", np.eye(3).tolist()))
                labels.append(it.get("label"))
                scores.append(it.get("score"))
            centers = np.asarray(centers, dtype=np.float32)
            half_sizes = np.asarray(half_sizes, dtype=np.float32)
            rotations = np.asarray(rotations, dtype=np.float32)
            labels = np.asarray(labels) if any(x is not None for x in labels) else None
            scores = np.asarray(scores) if any(x is not None for x in scores) else None
            return centers, half_sizes, rotations, labels, scores
        else:
            raise RuntimeError("JSON predictions must be a list of box dicts.")
    else:
        raise RuntimeError("Unsupported prediction file format. Use .npz or .json")


def main():
    parser = argparse.ArgumentParser("3D-R1 Bounding Box Visualization (rerun)")
    parser.add_argument(
        "-p",
        "--point_cloud",
        type=str,
        required=True,
        help="Path to the input point cloud file (.ply recommended)")
    parser.add_argument(
        "-d",
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions (.npz or .json) from 3D-R1 det")
    parser.add_argument("--radius", type=float, default=0.01, help="Point radius for visualization")
    parser.add_argument("--max_points", type=int, default=1000000, help="Maximum number of points to visualize")

    rr.script_add_args(parser)
    args = parser.parse_args()

    centers, half_sizes, rotations, labels, scores = _load_predictions(args.predictions)
    points, colors = _load_point_cloud(args.point_cloud)

    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(name="3D", origin="/world", background=[255, 255, 255]),
        collapse_panels=True,
    )
    rr.script_setup(args, "rerun_3dr1_bboxes", default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    if points.shape[0] > args.max_points:
        idx = np.random.permutation(points.shape[0])[: args.max_points]
        points = points[idx]
        colors = colors[idx]

    rr.log(
        "world/points",
        rr.Points3D(positions=points, colors=colors, radii=args.radius),
        static=True,
    )

    num_entities = centers.shape[0]
    seconds = 0.5
    for ti in range(num_entities + 1):
        rr.set_time_seconds("time_sec", ti * seconds)
        for bi in range(ti):
            uid = str(bi)
            group = "pred"
            label = None if labels is None else (str(labels[bi]) if not np.isscalar(labels[bi]) else str(int(labels[bi])))
            center = centers[bi]
            half = half_sizes[bi]
            rotation = rotations[bi]
            rr.log(
                f"world/{group}/{uid}",
                rr.Boxes3D(centers=center, half_sizes=half, labels=label),
                rr.InstancePoses3D(mat3x3=rotation),
                static=False,
            )

    rr.script_teardown(args)


if __name__ == "__main__":
    main()


