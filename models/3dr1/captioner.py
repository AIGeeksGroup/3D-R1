import copy, math, importlib
import torch
import torch.nn.functional as nnf
from torch import nn, Tensor
from typing import Dict
import re
from collections import OrderedDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    InstructBlipQFormerModel,
    InstructBlipQFormerConfig
)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("Warning: PEFT not available. LoRA will not be used.")
from models.3dr1.generation_utils import generation
from models.3dr1.position_embedding import PositionEmbeddingCoordsSine

from utils.box_util import box3d_iou_batch_tensor

SPECIAL_TOKENS = ["<think>", "</think>", "<answer>", "</answer>"]

def proposal_dimension_select(features: Tensor, indices: Tensor) -> Tensor:
    '''
    
    Parameters
    ----------
    features : Tensor, with size [batch x nsrc x ...]
        Data bank, from which to gather information.
    indices : Tensor, with size [batch x ntgt]
        Indices for gathering information from data bank.

    Returns
    -------
    Tensor, with size [batch x ntgt x ...]
        Gathers features in proposal dimension.
    
    '''
    return torch.gather(
        features, 1, 
        indices.reshape(
            *(indices.shape + tuple(1 for _ in features.shape[2:]))
        ).repeat(
            *((1, 1) + features.shape[2:])
        )
    )


def select_proposal_feature(
    prop_features: Tensor, prop_box_corners: Tensor, prop_sem_mask: Tensor, box_query: Tensor
) -> Tensor:
    '''
    
    Parameters
    ----------
    prop_features : Tensor, with size [batch x nproposal x n_embd]
    prop_box_corners : Tensor, with size [batch x nproposal x 8 x 3]
    prop_sem_mask : Tensor, with size [batch x nproposal], 0 for background
    box_query : Tensor, with size [batch x nquery x 8 x 3]

    Returns
    -------
    Tensor, with size [batch x nquery x n_embd]
        Gathers features in proposal dimension.
    
    '''
    # prop_features
    batch_size, nproposal, _, _ = prop_box_corners.shape
    nquery = box_query.shape[1]
    
    matched_box_iou = box3d_iou_batch_tensor(
        prop_box_corners.unsqueeze(1).repeat(1, nquery, 1, 1, 1).reshape(-1, 8, 3),
        box_query.unsqueeze(2).repeat(1, 1, nproposal, 1, 1).reshape(-1, 8, 3)
    )
    matched_box_iou = matched_box_iou.reshape(batch_size, nquery, nproposal)
    matched_box_iou = matched_box_iou * prop_sem_mask.unsqueeze(1)
    
    matched_indices = matched_box_iou.argmax(-1)    # batch x nquery
    return proposal_dimension_select(prop_features, matched_indices)


class captioner(nn.Module):
    
    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_llm is True and not self.use_lora:
            self.transformer.eval()
            for param in self.transformer.parameters():
                param.requires_grad = False
        elif self.freeze_llm is True and self.use_lora:
            # When using LoRA, we want to freeze the base model but train LoRA adapters
            self.transformer.eval()
            for name, param in self.transformer.named_parameters():
                if 'lora_' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        return self
    
    def __init__(self, args, train_dataset):
        super(captioner, self).__init__()
        
        self.encoder_hidden_size = 256
        self.dtype = torch.bfloat16
        self.visual_nquery = 8
        self.nlatent_query = 32
        self.freeze_llm = args.freeze_llm
        
        ## initialize tokenizer for batch decoding
        self.tokenizer = AutoTokenizer.from_pretrained(args.vocab)
        self.nvocabs = len(self.tokenizer)
        
        ## Model type detection and initialization
        self.use_multimodal_model = getattr(args, 'use_multimodal_model', False)
        
        if self.use_multimodal_model:
            # Use Qwen2.5-VL-7B as a multimodal model
            try:
                self.transformer = AutoModelForVision2Seq.from_pretrained(
                    args.vocab,
                    torch_dtype=self.dtype,
                    trust_remote_code=True
                )
                print(f"Loaded multimodal model: {args.vocab}")
            except Exception as e:
                print(f"Failed to load as multimodal model: {e}")
                print("Falling back to causal LM mode")
                self.use_multimodal_model = False
                self.transformer = AutoModelForCausalLM.from_pretrained(
                    args.vocab,
                    torch_dtype=self.dtype,
                    trust_remote_code=True
                )
        else:
            # Use as causal language model (original approach)
            self.transformer = AutoModelForCausalLM.from_pretrained(
                args.vocab,
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            print(f"Loaded causal LM: {args.vocab}")
        
        self.n_embd = self.transformer.config.hidden_size
        self._llm_total_layers = getattr(self.transformer.config, "num_hidden_layers", None)
        
        # Add special tokens if not already present
        if not all(t in self.tokenizer.get_vocab() for t in SPECIAL_TOKENS):
            self.tokenizer.add_tokens(SPECIAL_TOKENS, special_tokens=True)
            self.transformer.resize_token_embeddings(len(self.tokenizer))
        
        # Apply LoRA if enabled and PEFT is available
        self.use_lora = getattr(args, 'use_lora', False)
        if self.use_lora and PEFT_AVAILABLE:
            # Choose task type based on model type
            if self.use_multimodal_model:
                task_type = TaskType.VISION_2_SEQ
                print("Using LoRA for multimodal model (Vision2Seq)")
            else:
                task_type = TaskType.CAUSAL_LM
                print("Using LoRA for causal language model")
            
            lora_config = LoraConfig(
                task_type=task_type,
                inference_mode=False,
                r=getattr(args, 'lora_r', 16),
                lora_alpha=getattr(args, 'lora_alpha', 32),
                lora_dropout=getattr(args, 'lora_dropout', 0.1),
                target_modules=getattr(args, 'lora_target_modules', ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            )
            self.transformer = get_peft_model(self.transformer, lora_config)
            print(f"Applied LoRA with config: r={lora_config.r}, alpha={lora_config.lora_alpha}, dropout={lora_config.lora_dropout}")
            print(f"Target modules: {lora_config.target_modules}")
        elif self.use_lora and not PEFT_AVAILABLE:
            print("Warning: LoRA requested but PEFT not available. Using full model.")
        ## Multi-modality Transformer
        qformer_config = InstructBlipQFormerConfig(
            num_hidden_layers=4,
            encoder_hidden_size=self.encoder_hidden_size
        )
        self.qformer = InstructBlipQFormerModel.from_pretrained(
            args.qformer_vocab, 
            config=qformer_config
        )
        self.qformer_hidden_size = qformer_config.hidden_size
        
        
        ## for prompt feature projection
        self.encoder_to_qformer_projection = nn.Sequential(
            nn.Linear(self.encoder_hidden_size, qformer_config.encoder_hidden_size),
            nn.ReLU(),
            nn.Linear(qformer_config.encoder_hidden_size, qformer_config.encoder_hidden_size),
            nn.ReLU(),
        )
        self.latent_query = nn.Embedding(self.nlatent_query, self.qformer_hidden_size)
        self.qformer_to_language_projection = nn.Linear(self.qformer_hidden_size, self.n_embd)
        
        # Dynamic View Selection
        self.enable_dynamic_views = getattr(args, 'enable_dynamic_views', False)
        if self.enable_dynamic_views:
            from .view_selection import DynamicViewSelection
            self.view_selection = DynamicViewSelection(device="cuda", num_views=4)
            self.view_selection_weight = getattr(args, 'view_selection_weight', 0.1)
        else:
            self.view_selection = None
        
        # Additional Encoders (Point Transformer v3, Depth, Image)
        self.use_additional_encoders = getattr(args, 'use_additional_encoders', False)
        if self.use_additional_encoders:
            from models.encoders import DepthAnythingV2Encoder, SigLIP2ImageEncoder
            # Depth encoder
            self.use_depth = getattr(args, 'use_depth', True)
            if self.use_depth:
                self.depth_encoder = DepthAnythingV2Encoder(
                    model_name="LiheYoung/depth_anything_vitl14",
                    feature_dim=1024,
                    output_dim=getattr(args, 'depth_encoder_dim', 256),
                    device="cuda"
                )
            
            # Image encoder
            self.use_image = getattr(args, 'use_image', True)
            if self.use_image:
                self.image_encoder = SigLIP2ImageEncoder(
                    model_name="google/siglip-vit-large-patch14-384",
                    feature_dim=1024,
                    output_dim=getattr(args, 'image_encoder_dim', 256),
                    device="cuda"
                )
            
            # Feature fusion for additional encoders
            total_additional_dim = 0
            if self.use_depth:
                total_additional_dim += getattr(args, 'depth_encoder_dim', 256)
            if self.use_image:
                total_additional_dim += getattr(args, 'image_encoder_dim', 256)
            
            if total_additional_dim > 0:
                self.additional_feature_fusion = nn.Sequential(
                    nn.Linear(total_additional_dim, self.encoder_hidden_size),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(self.encoder_hidden_size, self.encoder_hidden_size),
                    nn.LayerNorm(self.encoder_hidden_size)
                )
        else:
            self.depth_encoder = None
            self.image_encoder = None
            self.additional_feature_fusion = None
        
        self.max_gen_per_iter = 8
        ## ---- super parameters for evaluation
        self.caption_config = {
            'early_stopping': True,
            'eos_token_id': self.tokenizer.convert_tokens_to_ids("</answer>"),
            # 'eos_token_id': self.tokenizer.eos_token_id,
            # 'num_beams': 4 if args.use_beam_search is True else None,
        }
        self.train()
        # self.set_transformer_trainable(4 if self.freeze_llm else self._llm_total_layers)
        
        # Print parameter statistics
        if self.use_lora and PEFT_AVAILABLE:
            trainable_params, all_params = self.get_trainable_parameters()
            print(f"[captioner] LoRA enabled - Trainable params: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")
            print(f"[captioner] All params: {all_params:,}")
    
    @staticmethod
    def _is_norm_or_embed(name: str) -> bool:
        name = name.lower()
        return any(k in name for k in ["layernorm", "norm", "ln_f", "embedding"])
    
    def get_trainable_parameters(self):
        """Get the number of trainable parameters"""
        if self.use_lora and PEFT_AVAILABLE:
            trainable_params = 0
            all_param = 0
            for _, param in self.transformer.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            return trainable_params, all_param
        else:
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            all_param = sum(p.numel() for p in self.parameters())
            return trainable_params, all_param
    def set_transformer_trainable(self, num_trainable_layers: int):
        """Unfreeze last `num_trainable_layers` LLM layers; others remain frozen."""
        if self.use_lora and PEFT_AVAILABLE:
            print("[captioner] LoRA is enabled, skipping layer-wise unfreezing")
            return
            
        if self._llm_total_layers is None:
            print("[captioner] unknown LLM depth – skip")
            return
        for n, p in self.transformer.named_parameters():
            if ".layers." in n or ".h." in n:
                lid = int(re.search(r"\.(?:layers|h)\.(\d+)\.", n).group(1))
                p.requires_grad = lid >= self._llm_total_layers - num_trainable_layers
            else:
                # keep norms & embedding trainable; others (e.g., final_ln, lm_head) follow same rule
                p.requires_grad = self._is_norm_or_embed(n)
        trainables = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            f"[captioner] LLM: unfroze last {num_trainable_layers} layers – trainable params {trainables/1e6:.1f}M"
        )
    
    
    def _get_instruction_response(self, 
            detector_output: dict, 
            inputs: dict, 
            box_query: Tensor=None,
            box_qmask: Tensor=None,
            click_query: Tensor=None,
            click_qmask: Tensor=None
        ) -> dict:
        
        point_cloud_dims = [
            inputs["point_cloud_dims_min"],
            inputs["point_cloud_dims_max"],
        ]
        net_device = inputs["point_clouds"].device
        batch_size = inputs["point_clouds"].shape[0]
        encoder_hidden_states = detector_output['enc_features']
        
        # Additional Encoders (Point Transformer v3, Depth, Image)
        additional_features = []
        if self.use_additional_encoders:
            # Depth encoder
            if self.depth_encoder is not None and "depth_maps" in inputs:
                depth_features = self.depth_encoder(inputs["depth_maps"])
                additional_features.append(depth_features)
            
            # Image encoder
            if self.image_encoder is not None and "images" in inputs:
                image_features = self.image_encoder(inputs["images"])
                additional_features.append(image_features)
            
            # Fuse additional features
            if additional_features:
                # Concatenate all additional features
                concatenated_features = torch.cat(additional_features, dim=-1)
                
                # Fuse to match encoder dimension
                fused_features = self.additional_feature_fusion(concatenated_features)
                
                # Expand to match point cloud features
                num_points = encoder_hidden_states.shape[1]
                expanded_features = fused_features.unsqueeze(1).expand(-1, num_points, -1)
                
                # Add to encoder hidden states
                encoder_hidden_states = encoder_hidden_states + expanded_features
        
        # Dynamic View Selection
        if self.view_selection is not None and hasattr(inputs, 'instruction_text') and inputs.get('instruction_text'):
            # Apply dynamic view selection
            point_cloud = inputs["point_clouds"][0]  # Use first sample for view selection
            instruction_text = inputs['instruction_text'][0] if isinstance(inputs['instruction_text'], list) else inputs['instruction_text']
            
            selected_view_features, view_indices = self.view_selection(point_cloud, instruction_text)
            
            # Add view selection features to encoder hidden states
            # This is a simplified integration - in practice you might want more sophisticated fusion
            if selected_view_features is not None:
                # Project view features to match encoder dimension
                view_projection = nn.Linear(512, encoder_hidden_states.shape[-1]).to(net_device)
                projected_views = view_projection(selected_view_features)
                
                # Concatenate with existing features (simplified approach)
                # In practice, you might want to use attention or other fusion mechanisms
                encoder_hidden_states = torch.cat([encoder_hidden_states, projected_views.unsqueeze(0).repeat(batch_size, 1, 1)], dim=1)
        
        # Create empty prompt features and masks
        batch_size = encoder_hidden_states.shape[0]
        prompt_feature = torch.zeros(batch_size, 0, self.qformer_hidden_size).to(net_device)
        prompt_mask = torch.zeros(batch_size, 0).to(net_device)
        
        ## gather query feature for qformer: batch x (n_query + n_tokens) x n_embd
        query_tokens = self.latent_query.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        query_tokens = torch.cat((query_tokens, prompt_feature), dim=1)
        query_attention_mask = torch.cat(
            (torch.ones(batch_size, self.nlatent_query).to(net_device), prompt_mask), dim=1)
        
        # prepare qformer inputs: batch x ntoken x n_embd
        query_attention_mask = torch.cat((query_attention_mask, inputs['qformer_attention_mask']), dim=1)
        
        query_outputs = self.qformer(
            input_ids=inputs['qformer_input_ids'],
            attention_mask=query_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=self.encoder_to_qformer_projection(encoder_hidden_states),
        )
        query_outputs = query_outputs[0][:, : self.nlatent_query, :]
        prefix_feature = self.qformer_to_language_projection(query_outputs)
        
        return prefix_feature
        
    
    def forward(self, detector_output: dict, inputs: dict, is_eval: bool=False, task_name: str='qa') -> dict:
        
        if is_eval is False:
            return self.forward_training(detector_output, inputs)
        
        response_config = {
            'ov-det': 64,
            'dense-cap': 48,
            'qa': 512,
            'chat': 512,
        }
        max_gen_length = response_config[task_name]
        
        if task_name in {'ov-det', 'dense-cap'}:
            return self.predict_densecap(detector_output, inputs, task_name, max_gen_length=max_gen_length)
        elif task_name == 'qa':
            return self.predict_answer(detector_output, inputs, max_gen_length=max_gen_length)
        else:
            return self.predict_chat(detector_output, inputs, max_gen_length=max_gen_length)
    
    
    def forward_training(self, detector_output: Dict, inputs: Dict) -> Dict:
        # get word embeddings, NOTE: captioner does not predict <bos> token
        input_ids = inputs['input_ids']         # batch x ntokens
        input_mask = inputs['attention_mask']   # batch x ntokens
        gradient_mask = inputs['gradient_mask'] # batch x ntokens
        
        box_query = inputs.get('box_query', None)       # batch x nquery x 8 x 3
        box_qmask = inputs.get('box_mask', None)        # batch x nquery
        click_query = inputs.get('click_query', None)   # batch x nquery x 3
        click_qmask = inputs.get('click_mask', None)    # batch x nquery
        
        embedding_layer = self.transformer.get_input_embeddings()
        
        # ---- batch x ntoken x n_embd
        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output, 
            inputs=inputs, 
            box_query=box_query,
            box_qmask=box_qmask,
            click_query=click_query,
            click_qmask=click_qmask
        )
        prefix_mask = torch.ones_like(prefix_tokens[..., 0])
        # ---- batch x (ntoken + nword) x n_embd
        inputs_embeds = torch.cat((prefix_tokens, embedding_layer(input_ids)), dim=1)
        attention_mask = torch.cat((prefix_mask, input_mask), dim=1)
        
        # ---- calculate transformer loss
        if self.use_multimodal_model:
            # For multimodal models, we need to handle visual inputs differently
            # Extract visual features from the prefix tokens
            visual_features = prefix_tokens[:, :self.nlatent_query, :]  # Use latent queries as visual features
            
            outputs = self.transformer(
                input_ids=input_ids,
                attention_mask=input_mask.to(self.dtype),
                pixel_values=visual_features.to(self.dtype),  # Use as pseudo pixel values
                return_dict=True
            )
        else:
            # Original causal LM approach
            outputs = self.transformer(
                inputs_embeds=inputs_embeds.to(self.dtype),
                attention_mask=attention_mask.to(self.dtype),
            )
        
        # Main caption loss
        caption_loss = self.loss_caption(
            logits = outputs.logits[:, prefix_tokens.shape[1] - 1: -1],
            target = input_ids,
            mask = gradient_mask.to(self.dtype),
        )
        detector_output['loss'] += caption_loss
        
        # Dynamic View Selection regularization loss
        if self.view_selection is not None:
            view_reg_loss = self.view_selection.get_reg_loss()
            detector_output['loss'] += self.view_selection_weight * view_reg_loss
        
        # Additional encoders regularization loss (optional)
        if self.use_additional_encoders:
            # Add small regularization to prevent overfitting
            additional_reg_loss = 0.0
            if self.depth_encoder is not None:
                for param in self.depth_encoder.parameters():
                    additional_reg_loss += torch.norm(param, p=2)
            if self.image_encoder is not None:
                for param in self.image_encoder.parameters():
                    additional_reg_loss += torch.norm(param, p=2)
            
            # Add regularization loss with small weight
            detector_output['loss'] += 1e-5 * additional_reg_loss
        
        return detector_output

    def loss_caption(self, logits: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        loss_per_word = nnf.cross_entropy(
            logits.permute(0, 2, 1).contiguous(),
            target,
            reduction='none',
        )
        final_loss = torch.sum(loss_per_word * mask) / torch.sum(mask + 1e-6)
        # parameter activation for multi-gpu training
        for param in self.parameters():
            if param.requires_grad:
                final_loss += 0 * torch.sum(param.to(final_loss.dtype) ** 2)
        return final_loss
    
    def save_lora_weights(self, path: str):
        """Save LoRA weights"""
        if self.use_lora and PEFT_AVAILABLE:
            self.transformer.save_pretrained(path)
            print(f"LoRA weights saved to {path}")
        else:
            print("LoRA not enabled, cannot save LoRA weights")
    
    def load_lora_weights(self, path: str):
        """Load LoRA weights"""
        if self.use_lora and PEFT_AVAILABLE:
            from peft import PeftModel
            self.transformer = PeftModel.from_pretrained(self.transformer, path)
            print(f"LoRA weights loaded from {path}")
        else:
            print("LoRA not enabled, cannot load LoRA weights")
    
    def predict_densecap(self, detector_output: Dict, inputs: Dict, task_name: str, max_gen_length: int=64) -> Dict:
        
        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        batch_size, nproposals, _, _ = detector_output['box_corners'].shape
        # ---- to store llm outputs
        output_ids = torch.ones(batch_size, nproposals, max_gen_length).long().to(net_device)
        output_ids = output_ids * self.tokenizer.eos_token_id
        
        # ---- llm input preparation
        instruction = inputs['instruction'][0]              # ntoken
        instruction_mask = inputs['instruction_mask'][0]    # ntoken
        instruction_id = instruction[instruction_mask == 1] # ntoken
        instruction_id = instruction_id[None, :].repeat(batch_size, 1)
        instruction_embedding = embedding_layer(instruction_id) # batch x ntoken x n_embd
        
        prefix_tokens = []
        for proposal_id in range(nproposals):
            box_query=detector_output['box_corners'][:, [proposal_id]]  # batch x 1 x 8 x 3

            click_query=None
            if task_name == 'ov-det':
                click_query=detector_output['query_xyz'][:, [proposal_id]]  # batch x 1 x 3
            
            instruct_prefix_feature=self._get_instruction_response(     # batch x ntoken x n_embd
                detector_output=detector_output,
                inputs=inputs,
                box_query=box_query,        # batch x 1 x 8 x 3
                click_query=click_query,
            )
            instruct_prefix_feature = torch.cat((instruct_prefix_feature, instruction_embedding), dim=1)
            prefix_tokens.append(instruct_prefix_feature.unsqueeze(1))
        # batch x nproposal x 1 x n_embd
        prefix_tokens = torch.cat(prefix_tokens, dim=1).to(self.dtype)
        
        ## filter and rank the queries
        sem_cls_logits = detector_output["sem_cls_logits"]
        objectness_mask = sem_cls_logits.argmax(-1) != (sem_cls_logits.shape[-1] - 1)
        
        ## limit the proposals for generating captions
        candidate_prefix = prefix_tokens[objectness_mask].to(self.dtype)

        gather_output_ids = []
        for start_idx in range(0, candidate_prefix.shape[0], self.max_gen_per_iter):
            prefix = candidate_prefix[start_idx: start_idx + self.max_gen_per_iter]
            scene_cap_output = generation(
                self.transformer, 
                inputs_embeds=prefix,
                max_length=max_gen_length,
                **self.caption_config
            )
            gather_output_ids.append(scene_cap_output['output_ids'])
        gather_output_ids = torch.cat(gather_output_ids, dim=0)
        
        output_ids[objectness_mask] = gather_output_ids
        detector_output['output_ids'] = output_ids
        
        return detector_output
    
    
    def predict_answer(self, detector_output: Dict, inputs: Dict, max_gen_length: int=8) -> Dict:
        
        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        # ---- to store llm outputs
        output_ids = []
        
        # ---- llm input preparation
        instruction = inputs['instruction']             # ntoken
        instruction_mask = inputs['instruction_mask']   # ntoken

        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
        )
        prefix_tokens = prefix_tokens.to(self.dtype)
        
        for batch_id in range(prefix_tokens.shape[0]):
            sample_instruction = instruction[batch_id]     
            sample_mask = instruction_mask[batch_id]     # ntoken
            
            output = generation(
                self.transformer, 
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[batch_id].unsqueeze(0),   # 1 x nprefix x n_embd
                        embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                max_length=max_gen_length,
                **self.caption_config
            )
            output_ids.append(output['output_ids'])
        
        output_ids = torch.cat(output_ids, dim=0)
        detector_output['output_ids'] = output_ids
        
        return detector_output
    
    def predict_chat(self, detector_output: Dict, inputs: Dict, max_gen_length: int=512) -> Dict:
        
        # ---- necessary elements
        embedding_layer = self.transformer.get_input_embeddings()
        net_device = next(self.parameters()).device
        # ---- to store llm outputs
        output_ids = []
        
        # ---- llm input preparation
        instruction = inputs['instruction']             # ntoken
        instruction_mask = inputs['instruction_mask']   # ntoken

        prefix_tokens = self._get_instruction_response(
            detector_output=detector_output,
            inputs=inputs,
        )
        prefix_tokens = prefix_tokens.to(self.dtype)
        
        for batch_id in range(prefix_tokens.shape[0]):
            sample_instruction = instruction[batch_id]     
            sample_mask = instruction_mask[batch_id]     # ntoken
            
            output = self.transformer.generate(
                inputs_embeds=torch.cat(
                    [
                        prefix_tokens[batch_id].unsqueeze(0),   # 1 x nprefix x n_embd
                        embedding_layer(sample_instruction[sample_mask == 1]).unsqueeze(0)
                    ],
                    dim=1
                ),
                max_new_tokens=max_gen_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                no_repeat_ngram_size=4,
                num_return_sequences=1,
            )   # 1 x max_gen_length
            output = output.squeeze(0)
            placeholder = torch.ones(max_gen_length).to(net_device) * self.tokenizer.eos_token_id
            output = output[:min(max_gen_length, output.shape[0])]
            placeholder[:output.shape[0]] = output
            
            output_ids.append(placeholder.unsqueeze(0).long())
        
        output_ids = torch.cat(output_ids, dim=0)
        detector_output['output_ids'] = output_ids
        
        return detector_output
    
