import os
import json
import argparse
import numpy as np
import sys
import torch

from typing import Dict, Any

# Reuse training/eval plumbing
from models.model_general import CaptionNet
from main import make_args_parser, build_dataset


def _to_numpy(x: torch.Tensor):
    return x.detach().cpu().numpy()


def _corners_to_center_scale(corners: np.ndarray):
    # corners: [K, 8, 3]
    centers = corners.mean(axis=1)  # [K,3]
    mins = corners.min(axis=1)
    maxs = corners.max(axis=1)
    scales = maxs - mins
    return centers, scales


def save_npz(path: str, boxes: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **boxes)


def save_json(path: str, corners: np.ndarray, labels: np.ndarray | None, scores: np.ndarray | None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    centers, scales = _corners_to_center_scale(corners)
    items = []
    for i in range(centers.shape[0]):
        item = {
            "center": centers[i].tolist(),
            "scale": scales[i].tolist(),
        }
        if labels is not None:
            item["label"] = int(labels[i])
        if scores is not None:
            item["score"] = float(scores[i])
        items.append(item)
    with open(path, "w") as f:
        json.dump(items, f)


def main():
    # First parse exporter-specific args
    exp_parser = argparse.ArgumentParser(add_help=True)
    exp_parser.add_argument("--output_dir", required=True, type=str, help="Directory to write predictions")
    exp_parser.add_argument("--format", default="npz", choices=["npz", "json"], help="Output format")
    exp_parser.add_argument("--eval_split", default="val", choices=["val", "test"], help="Dataset split to export")
    exp_parser.add_argument("--max_scenes", default=None, type=int, help="Export at most N scenes")
    known, _ = exp_parser.parse_known_args()

    # Strip exporter-specific flags from sys.argv before using project arg parser
    original_argv = sys.argv[:]
    strip_keys = {"--output_dir", "--format", "--eval_split", "--max_scenes"}
    filtered = []
    skip_next = False
    for i, tok in enumerate(original_argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if tok in strip_keys:
            skip_next = True  # assume flag followed by a value
            continue
        filtered.append(tok)
    sys.argv = [original_argv[0]] + filtered

    # Reuse project args
    args = make_args_parser()

    # Restore argv
    sys.argv = original_argv
    # Ensure eval mode
    args.test_only = True

    # Build datasets/dataloaders
    dataset_config, datasets, dataloaders = build_dataset(args)

    # Build model
    model = CaptionNet(args, dataset_config, datasets['train'])
    if args.test_ckpt:
        ckpt = torch.load(args.test_ckpt, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model = model.cuda().eval()

    # Choose split loaders
    loaders = dataloaders['test']
    total_written = 0
    for loader in loaders:
        # Map split
        # Skip non-target split if needed (our builder uses 'val' for test loaders)
        for batch in loader:
            # device transfer
            for k in batch:
                batch[k] = batch[k].cuda(non_blocking=True) if isinstance(batch[k], torch.Tensor) else batch[k]

            model_input = {
                'point_clouds':         batch['point_clouds'],
                'point_clouds_color':   batch['pcl_color'],
                'point_cloud_dims_min': batch['point_cloud_dims_min'],
                'point_cloud_dims_max': batch['point_cloud_dims_max'],
                'qformer_input_ids':     batch['qformer_input_ids'],
                'qformer_attention_mask':batch['qformer_attention_mask'],
                'instruction':           batch['instruction'],
                'instruction_mask':      batch['instruction_mask'],
            }
            with torch.no_grad():
                outputs = model(model_input, is_eval=True)

            corners_b = _to_numpy(outputs["box_corners"])              # [B, K, 8, 3]
            sem_prob_b = _to_numpy(outputs.get("sem_cls_prob", None))   # [B, K, C] or None
            obj_prob_b = _to_numpy(outputs.get("objectness_prob", None))# [B, K] or [B,K,2] or None
            idxs = batch['scan_idx'].detach().cpu().numpy()              # [B]

            B = corners_b.shape[0]
            for bi in range(B):
                scan_idx = int(idxs[bi])
                # Retrieve scan name if available
                try:
                    scan_name = loader.dataset.scan_names[scan_idx]
                except Exception:
                    scan_name = f"scan_{scan_idx:05d}"

                corners = corners_b[bi]

                labels = None
                if sem_prob_b is not None:
                    labels = np.argmax(sem_prob_b[bi], axis=-1).astype(np.int64)

                scores = None
                if obj_prob_b is not None:
                    # If objectness is (B,K) use that; if (B,K,2) take prob of positive class
                    arr = obj_prob_b[bi]
                    if arr.ndim == 2 and arr.shape[-1] == 2:
                        scores = arr[..., 1]
                    else:
                        scores = arr.squeeze(-1)
                
                out_path = os.path.join(known.output_dir, f"{scan_name}.{known.format}")
                if known.format == "npz":
                    to_save = {"corners": corners}
                    if labels is not None:
                        to_save["labels"] = labels
                    if scores is not None:
                        to_save["scores"] = scores
                    save_npz(out_path, to_save)
                else:
                    save_json(out_path, corners, labels, scores)

                total_written += 1
                if known.max_scenes is not None and total_written >= known.max_scenes:
                    print(f"Wrote {total_written} predictions to {known.output_dir}")
                    return

    print(f"Wrote {total_written} predictions to {known.output_dir}")


if __name__ == "__main__":
    main()


