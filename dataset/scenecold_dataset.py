# -*- coding: utf-8 -*-
import os, json, random, numpy as np, torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from eval_utils.evaluate_cold import evaluate

from dataset.scannet_base_dataset import ScanNetBaseDataset, DatasetConfig, BASE

SPECIAL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]

class Dataset(ScanNetBaseDataset):
    def __init__(
        self,
        args,
        dataset_config: DatasetConfig,
        split_set="all",
        num_points=40000,
        use_color=False,
        use_normal=False,
        use_multiview=False,
        use_height=False,
        augment=False,
        use_additional_encoders=False,
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
            use_additional_encoders=use_additional_encoders,
        )

        self.task_name = 'cold-start'
        self.split = split_set
        self.eval_func = evaluate
        self.max_prompts = 1
        assert split_set in ["train", "val"]
        ann_path = os.path.join(
            BASE,"data","Scene30K",f"Scene-30K.jsonl",
        )
        scene_path = os.path.join(BASE, "data", "scene_caption.json")
        self.scene_description = json.load(open(scene_path,'r'))
        self.annotations = [json.loads(l) for l in open(ann_path)]
        self.scan_names  = sorted({a["scene_id"] for a in self.annotations})
        print(f"[SceneR1Cold-QA-PC]: "
              f"{len(self.annotations)} Q&A  from {len(self.scan_names)} scans")

        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        
        if not all(tok in self.tokenizer.vocab for tok in SPECIAL_TOKENS):
            self.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

        self.qtokenizer = AutoTokenizer.from_pretrained(args.qformer_vocab)
        
        if not all(tok in self.qtokenizer.vocab for tok in SPECIAL_TOKENS):
            self.qtokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)

        self.cfg_q = dict(max_length=512,
                padding='max_length',
                truncation="longest_first", return_tensors="np")

        self.cfg = dict(
            max_length=args.max_des_len,
            padding='max_length',
            truncation="longest_first",
            return_tensors="np",
        )

    # ------------------------------------------------------------------
    def __len__(self): return len(self.annotations)

    def __getitem__(self, idx):
        sample   = self.annotations[idx]
        scene_id = sample["scene_id"]

        
        ret = self._get_scan_data(scene_id)       
        description = random.choice(self.scene_description[scene_id]['captions'])

        q, cot = sample["question"].strip(), sample["cot"].strip()

        source_txt  = f"given the 3D scene and description:{description}, think step by step and answer the following question:{q}."
        target_txt  = f"{source_txt} Output format:<think>...reasoning...</think><answer>..final answer...</answer> {cot} "
        enc_source  = self.tokenizer.batch_encode_plus([source_txt], **self.cfg)
        enc_target  = self.tokenizer.batch_encode_plus([target_txt], **self.cfg)


        qformer_ids = self.qtokenizer.batch_encode_plus([f"given the 3D scene and description:{description}, think step by step and answer the following question:{q}"], **self.cfg_q)

        box_query = np.zeros((self.max_prompts, 8, 3))
        box_mask = np.zeros((self.max_prompts,))
        click_query = np.zeros((self.max_prompts, 3))
        click_mask = np.zeros((self.max_prompts,))
        
        
        ret["box_query"]   = box_query.astype(np.float32)
        ret["box_mask"]    = box_mask.astype(np.float32)
        ret["click_query"] = click_query.astype(np.float32)
        ret["click_mask"]  = click_mask.astype(np.float32)

       
        ret["input_ids"]        = enc_target["input_ids"][0].astype(np.int64)
        ret["attention_mask"]   = enc_target["attention_mask"][0].astype(np.float32)
        
        # Improved gradient mask computation to prevent extreme values
        gradient_mask = enc_target['attention_mask'][0] - enc_source['attention_mask'].astype(np.float32)
        # Ensure gradient mask is non-negative and properly normalized
        gradient_mask = np.maximum(gradient_mask, 0.0)
        ret["gradient_mask"] = gradient_mask.astype(np.float32)
        ret['scan_idx'] = np.array(idx).astype(np.int64)
        ret["instruction"] = enc_source['input_ids'][0].astype(np.int64)
        ret["instruction_mask"] = enc_source['attention_mask'][0].astype(np.float32)

        ret["qformer_input_ids"]      = qformer_ids["input_ids"][0].astype(np.int64)
        ret["qformer_attention_mask"] = qformer_ids["attention_mask"][0].astype(np.float32)

        
        return ret