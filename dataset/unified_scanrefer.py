import os, json, random
import torch
import numpy as np
import random
from copy import deepcopy
from typing import Dict, List
from dataset.scannet_base_dataset import BASE, DatasetConfig, ScanNetBaseDataset
from transformers import AutoTokenizer
from eval_utils.evaluate_densecap import evaluate
from dataset.task_prompts import TASK_PROPMT, BOX_FORMAT


class UnifiedScanReferDataset(ScanNetBaseDataset):
    """
    Unified ScanRefer dataset that supports both standard training and RL training
    Includes target_text for reward computation in GRPO stage
    """
    
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
        use_rl_training=False,  # New flag for RL training
    ):
        super().__init__(
            args,
            dataset_config,
            split_set=split_set,
            num_points=num_points,
            use_color=use_color,
            use_normal=use_normal,
            use_multiview=use_multiview,
            use_height=use_height,
            augment=augment,
            use_random_cuboid=False,
            random_cuboid_min_points=None,
        )
        self.task_name = 'scanrefer'
        self.grid_size_3d = args.grid_size_3d
        self.max_prompts = args.max_prompts
        self.split = split_set
        self.dataset_config = dataset_config
        self.max_des_len = args.max_des_len
        self.eval_func = evaluate
        self.use_rl_training = use_rl_training  # Flag for RL training mode
        
        ## initialize tokenizer and set tokenizer's `padding token` to `eos token`
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab, add_bos_token=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'right'
        self.qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
        self.qtokenizer.pad_token = self.tokenizer.eos_token
        self.qtokenizer.padding_side = 'right'
        
        ## load annotations
        assert split_set in ["train", "val"]
        
        self.scanrefer = json.load(
            open(os.path.join(BASE, 'data', 'ScanRefer', f'ScanRefer_filtered_{split_set}.json'), 'r')
        )
        
        with open(os.path.join(BASE, 'data', 'ScanRefer', f'ScanRefer_filtered_{split_set}.txt'), 'r') as f:
            self.scan_names = f.read().splitlines()
        
        self.annotations = self.scanrefer
        if self.split != 'train':
            self.annotations = [{'scene_id': scene_id} for scene_id in self.scan_names]
        self._tag_dataset(self.annotations, 'densecap')
        
        ## super configuration
        self.tokenizer_config = dict(
            max_length=self.max_des_len, 
            padding='max_length', 
            truncation='longest_first', 
            return_tensors='np'
        )
        print(f"kept {len(self.annotations)} annotations in {len(self.scan_names)} scans...")
        print(f"RL training mode: {self.use_rl_training}")

    
    def _tag_dataset(self, corpus, task_name): 
        for anno in corpus:
            anno['task_name'] = task_name
        return 
    
    def _encode_box_coords(self, annotation_mask, ret_dict):
        center_normalized = ret_dict['gt_box_centers_normalized']
        size_normalized = ret_dict['gt_box_sizes_normalized']
        box_normalized = np.hstack((center_normalized, size_normalized))    # (-1, 6)
        # <cx, cy, cz, w, h, l>
        box_normalized = box_normalized[annotation_mask == 1]
        box_normalized = (box_normalized * self.grid_size_3d).astype(np.int64)
        return ' '.join(BOX_FORMAT.format(*box) for box in box_normalized)
    
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        
        scan_name = self.annotations[idx]['scene_id']
        task_name = self.annotations[idx]['task_name']
        ret_dict = self._get_scan_data(scan_name)
        
        if self.split == 'train':
            prompt = deepcopy(random.choice(TASK_PROPMT[task_name]))
        else:
            prompt = deepcopy(TASK_PROPMT[task_name][0])
            
        prompt_inputs = self.tokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        qformer_inputs = self.qtokenizer.batch_encode_plus([prompt['instruction']], **self.tokenizer_config)
        
        if self.split == 'train':
            
            target_obj_id = int(self.annotations[idx]['object_id'])
            caption = ' '.join(self.annotations[idx]['token'])
            
            ## reference object
            match_mask = (ret_dict["gt_object_ids"] == target_obj_id).astype(np.float32)
            match_mask = match_mask * ret_dict["gt_box_present"]
            
            boxes = self._encode_box_coords(match_mask, ret_dict)
            response = prompt['answer'].format(locations=boxes, caption=caption)
                
            ## input_ids as labels for LLM
            llm_inputs = self.tokenizer.batch_encode_plus(
                [' '.join((prompt['instruction'], response, self.tokenizer.eos_token))],
                **self.tokenizer_config
            )
            
            box_query = np.zeros((self.max_prompts, 8, 3))
            box_mask = np.zeros((self.max_prompts,))
            click_query = np.zeros((self.max_prompts, 3))
            click_mask = np.zeros((self.max_prompts,))
            
            if random.random() > 0.5:
                # use box to identify an object
                ref_gt_box_corner = \
                    ret_dict["gt_box_corners"][match_mask == 1].reshape(8, 3).astype(np.float32)
                box_query[0] = ref_gt_box_corner
                box_mask[0] = 1
            else:
                # use click to identify an object
                try:
                    point_clouds = ret_dict["point_clouds"][:, :3]  # x, y, z
                    object_points = point_clouds[ret_dict["instance_labels"] == (target_obj_id + 1)]    # npt x 3
                    click_query[0] = random.choice(object_points)
                except:
                    click_query[0] = ret_dict["gt_box_centers"][match_mask == 1].reshape(3,).astype(np.float32)
                click_mask[0] = 1
            
            ret_dict['box_query'] = box_query.astype(np.float32)
            ret_dict['box_mask'] = box_mask.astype(np.float32)
            ret_dict['click_query'] = click_query.astype(np.float32)
            ret_dict['click_mask'] = click_mask.astype(np.float32)
            
            if not self.use_rl_training:
                # Standard training mode - include input_ids and attention_mask
                ret_dict['input_ids'] = llm_inputs['input_ids'][0].astype(np.int64)
                ret_dict['attention_mask'] = llm_inputs['attention_mask'][0].astype(np.float32)
                ret_dict['gradient_mask'] = \
                    (llm_inputs['attention_mask'][0] - prompt_inputs['attention_mask'][0]).astype(np.float32)
            else:
                # RL training mode - include target_text for reward computation
                ret_dict['target_text'] = caption  # Target caption for reward computation
                ret_dict['target_response'] = response  # Full target response
                ret_dict['object_id'] = target_obj_id  # Object ID for reference
                ret_dict['description'] = ' '.join(self.annotations[idx]['token'])  # Description for context
        
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['instruction'] = prompt_inputs['input_ids'][0].astype(np.int64)
        ret_dict['instruction_mask'] = prompt_inputs['attention_mask'][0].astype(np.float32)
        ret_dict['qformer_input_ids'] = qformer_inputs['input_ids'][0].astype(np.int64)
        ret_dict['qformer_attention_mask'] = qformer_inputs['attention_mask'][0].astype(np.float32)
        
        return ret_dict


# For backward compatibility, create a Dataset class that uses the unified version
class Dataset(UnifiedScanReferDataset):
    def __init__(self, *args, **kwargs):
        # Default to standard training mode for backward compatibility
        kwargs.setdefault('use_rl_training', False)
        super().__init__(*args, **kwargs)

