# Copyright (c) Facebook, Inc. and its affiliates.

""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import h5py
import numpy as np
import torch
import multiprocessing as mp
from types import SimpleNamespace

import utils.pc_util as pc_util
from torch.utils.data import Dataset
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)
from utils.pc_util import scale_points, shift_scale_points
from utils.random_cuboid import RandomCuboid

IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
BASE = "."  ## Replace with path to dataset
DATASET_ROOT_DIR = os.path.join(BASE, "data", "scannet", "scannet_data")
DATASET_METADATA_DIR = os.path.join(BASE, "data", "scannet", "meta_data")


class DatasetConfig(object):
    def __init__(self):
        self.num_semcls = 18
        self.num_angle_bin = 1
        self.max_num_obj = 128

        self.type2class = {
            'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 
            'curtain':11, 'refrigerator':12, 'shower curtain':13, 'toilet':14, 
            'sink':15, 'bathtub':16, 'others':17
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.nyu40ids = np.array([
                3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 
                32, 33, 34, 35, 36, 37, 38, 39, 40
             ])
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }
        self.nyu40id2class = self._get_nyu40id2class()

    def _get_nyu40id2class(self):
        lines = [line.rstrip() for line in open(os.path.join(DATASET_METADATA_DIR, 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        nyu40ids2class = {}
        for i in range(len(lines)):
            label_classes_set = set(self.type2class.keys())
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in self.nyu40ids:
                if nyu40_name not in label_classes_set:
                    nyu40ids2class[nyu40_id] = self.type2class["others"]
                else:
                    nyu40ids2class[nyu40_id] = self.type2class[nyu40_name]
        return nyu40ids2class


    def angle2class(self, angle):
        raise ValueError("ScanNet does not have rotated bounding boxes.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)


class ScanNetBaseDataset(Dataset):
    def __init__(
        self,
        args,
        dataset_config,
        split_set="train",
        num_points=40000,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        use_height=False,
        augment=False,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
        use_additional_encoders=False,
    ):

        self.args = args if args is not None else SimpleNamespace()
        self.dataset_config = dataset_config
        # assert split_set in ["train", "val"]
        
        root_dir = DATASET_ROOT_DIR
        meta_data_dir = DATASET_METADATA_DIR

        self.data_path = root_dir
        self.split_set = split_set
        self.num_points = num_points
        self.use_color = use_color
        self.use_normal = use_normal
        self.use_multiview = use_multiview
        self.use_height = use_height
        self.augment = augment
        self.use_random_cuboid = use_random_cuboid
        self.use_additional_encoders = use_additional_encoders
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        
        # Load scan names
        self.scan_names = self._load_scan_names()
        
        self.multiview_data = {}

        # Initialize additional encoders data storage
        if self.use_additional_encoders:
            self.image_data = {}
            self.depth_data = {}
    
    def _load_scan_names(self):
        """Load scan names based on split_set"""
        all_scan_names = list(
            set(
                [
                    os.path.basename(x)[0:12]
                    for x in os.listdir(self.data_path)
                    if x.startswith("scene")
                ]
            )
        )
        
        if self.split_set == "all":
            return all_scan_names
        elif self.split_set in ["train", "val", "test"]:
            split_filenames = os.path.join(DATASET_METADATA_DIR, f"scannetv2_{self.split_set}.txt")
            
            with open(split_filenames, "r") as f:
                scan_names = f.read().splitlines()
            # remove unavailable scans
            num_scans = len(scan_names)
            scan_names = [
                sname for sname in scan_names if sname in all_scan_names
            ]
            print(f"kept {len(scan_names)} scans out of {num_scans}")
            return scan_names
        else:
            raise ValueError(f"Unknown split name {self.split_set}")

    def __len__(self):
        return len(self.scan_names)

    def _get_scan_data(self, scan_name):
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + "_aligned_vert.npy")
        instance_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_ins_label.npy"
        )
        semantic_labels = np.load(
            os.path.join(self.data_path, scan_name) + "_sem_label.npy"
        )
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + "_aligned_bbox.npy")

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]
        
        if self.use_normal:
            normals = mesh_vertices[:,6:9]
            point_cloud = np.concatenate([point_cloud, normals], 1)
        
        args_cfg = getattr(self, "args", SimpleNamespace())

        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(
                    os.path.join(self.data_path, 'enet_feats_maxpool.hdf5'), 
                    'r', libver='latest'
                )
            multiview = self.multiview_data[pid][scan_name]
            point_cloud = np.concatenate([point_cloud, multiview], 1)
            
        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # ------------------------------- LABELS ------------------------------
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        object_ids = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        
        if self.augment and self.use_random_cuboid:
            (
                point_cloud,
                instance_bboxes,
                per_point_labels,
            ) = self.random_cuboid_augmentor(
                point_cloud, instance_bboxes, [instance_labels, semantic_labels]
            )
            instance_labels = per_point_labels[0]
            semantic_labels = per_point_labels[1]
        
        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )
        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        pcl_color = pcl_color[choices]

        target_bboxes_mask[0 : instance_bboxes.shape[0]] = 1
        target_bboxes[0 : instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:

            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = self.dataset_config.rotate_aligned_boxes(
                target_bboxes, rot_mat
            )

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud[..., :3].min(axis=0)
        point_cloud_dims_max = point_cloud[..., :3].max(axis=0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]
        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)
        object_ids[:instance_bboxes.shape[0]] = instance_bboxes[:, -1]
        
        
        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        ret_dict["gt_angle_class_label"] = angle_classes.astype(np.int64)
        ret_dict["gt_angle_residual_label"] = angle_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0 : instance_bboxes.shape[0]] = [
            self.dataset_config.nyu40id2class.get(x, -1)
            for x in instance_bboxes[:, -2][0 : instance_bboxes.shape[0]]
        ]
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["pcl_color"] = pcl_color
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
        ret_dict["gt_object_ids"] = object_ids.astype(np.int64)
        
        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label. 
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in self.dataset_config.nyu40ids:
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
        
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        
        # Load additional modal data if enabled
        if self.use_additional_encoders:
            # Load multiple images if available
            image_path = os.path.join(self.data_path, scan_name, "images")
            if os.path.exists(image_path):
                try:
                    image_files = [f for f in os.listdir(image_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                    if image_files:
                        import cv2
                        # Sort files to ensure consistent ordering
                        image_files.sort()
                        
                        # Load up to max_multiview_images (configurable)
                        max_images = min(getattr(args_cfg, 'max_multiview_images', 8), len(image_files))
                        images = []
                        
                        for i in range(max_images):
                            img = cv2.imread(os.path.join(image_path, image_files[i]))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            img = cv2.resize(img, (224, 224))  # Resize to standard size
                            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
                            images.append(img.transpose(2, 0, 1))  # HWC to CHW
                        
                        # Stack images: (num_images, 3, 224, 224)
                        ret_dict["images"] = np.stack(images, axis=0)
                        
                        # If we have fewer than max_images, pad with zeros
                        if len(images) < max_images:
                            padding = np.zeros((max_images - len(images), 3, 224, 224), dtype=np.float32)
                            ret_dict["images"] = np.concatenate([ret_dict["images"], padding], axis=0)
                    else:
                        max_images = getattr(args_cfg, 'max_multiview_images', 8)
                        ret_dict["images"] = np.zeros((max_images, 3, 224, 224), dtype=np.float32)
                except Exception as e:
                    print(f"Warning: Failed to load images for {scan_name}: {e}")
                    max_images = getattr(args_cfg, 'max_multiview_images', 8)
                    ret_dict["images"] = np.zeros((max_images, 3, 224, 224), dtype=np.float32)
            else:
                max_images = getattr(args_cfg, 'max_multiview_images', 8)
                ret_dict["images"] = np.zeros((max_images, 3, 224, 224), dtype=np.float32)
            
            # Load multiple depth maps if available
            depth_path = os.path.join(self.data_path, scan_name, "depth")
            if os.path.exists(depth_path):
                try:
                    depth_files = [f for f in os.listdir(depth_path) if f.endswith(('.png', '.npy'))]
                    if depth_files:
                        # Sort files to ensure consistent ordering
                        depth_files.sort()
                        
                        # Load up to max_multiview_depth (configurable)
                        max_depth_maps = min(getattr(args_cfg, 'max_multiview_depth', 8), len(depth_files))
                        depth_maps = []
                        
                        for i in range(max_depth_maps):
                            if depth_files[i].endswith('.npy'):
                                depth = np.load(os.path.join(depth_path, depth_files[i]))
                            else:
                                import cv2
                                depth = cv2.imread(os.path.join(depth_path, depth_files[i]), cv2.IMREAD_ANYDEPTH)
                            
                            # Normalize depth to [0, 1] range
                            if depth.max() > 0:
                                depth = depth / depth.max()
                            depth = cv2.resize(depth, (224, 224))
                            depth_maps.append(depth.astype(np.float32))
                        
                        # Stack depth maps: (num_depth_maps, 224, 224)
                        ret_dict["depth_maps"] = np.stack(depth_maps, axis=0)
                        
                        # If we have fewer than max_depth_maps, pad with zeros
                        if len(depth_maps) < max_depth_maps:
                            padding = np.zeros((max_depth_maps - len(depth_maps), 224, 224), dtype=np.float32)
                            ret_dict["depth_maps"] = np.concatenate([ret_dict["depth_maps"], padding], axis=0)
                    else:
                        max_depth_maps = getattr(args_cfg, 'max_multiview_depth', 8)
                        ret_dict["depth_maps"] = np.zeros((max_depth_maps, 224, 224), dtype=np.float32)
                except Exception as e:
                    print(f"Warning: Failed to load depth maps for {scan_name}: {e}")
                    max_depth_maps = getattr(args_cfg, 'max_multiview_depth', 8)
                    ret_dict["depth_maps"] = np.zeros((max_depth_maps, 224, 224), dtype=np.float32)
            else:
                max_depth_maps = getattr(args_cfg, 'max_multiview_depth', 8)

        # Ensure downstream code always sees instruction/qformer placeholders
        if "instruction" not in ret_dict or "instruction_mask" not in ret_dict:
            inst_len = max(1, int(getattr(args_cfg, "eval_instruction_len", 1)))
            ret_dict.setdefault(
                "instruction",
                np.zeros((inst_len,), dtype=np.int64)
            )
            ret_dict.setdefault(
                "instruction_mask",
                np.zeros((inst_len,), dtype=np.float32)
            )
        if "qformer_input_ids" not in ret_dict or "qformer_attention_mask" not in ret_dict:
            q_len = max(1, int(getattr(args_cfg, "eval_qformer_len", 1)))
            ret_dict.setdefault(
                "qformer_input_ids",
                np.zeros((q_len,), dtype=np.int64)
            )
            ret_dict.setdefault(
                "qformer_attention_mask",
                np.zeros((q_len,), dtype=np.float32)
            )
                ret_dict["depth_maps"] = np.zeros((max_depth_maps, 224, 224), dtype=np.float32)
        
        return ret_dict


    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        ret_dict = self._get_scan_data(scan_name)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        return ret_dict
