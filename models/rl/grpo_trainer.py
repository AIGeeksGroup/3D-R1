import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
import clip
from PIL import Image
import re
from .reward_functions import RewardFunctions

class GRPOTrainer:
    """
    GRPO (Group Relative Policy Optimization) Trainer for 3D-R1 RL training
    """
    
    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        device,
        grpo_config: Dict = None
    ):
        self.model = model
        self.ref_model = ref_model  # SFT model as reference
        self.tokenizer = tokenizer
        self.device = device
        
        # Initialize reward functions
        self.reward_functions = RewardFunctions(device)
        
        # GRPO hyperparameters
        self.config = {
            'beta': 0.1,  # KL penalty coefficient
            'gamma': 0.99,  # discount factor
            'gae_lambda': 0.95,  # GAE lambda
            'clip_ratio': 0.2,  # PPO clip ratio
            'target_kl': 0.01,  # target KL divergence
            'max_grad_norm': 1.0,  # gradient clipping
            'lr': 1e-5,  # learning rate
            'batch_size': 4,
            'num_epochs': 4,
            'group_size': 8,  # group size for GRPO
        }
        if grpo_config:
            self.config.update(grpo_config)
            
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config['lr']
        )
        
    def generate_responses(self, batch_data: Dict) -> Tuple[List[str], torch.Tensor]:
        """
        Generate responses using the current model
        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batch_data, is_eval=True)
            
            # Extract generated text and logits
            lang_cap_raw = outputs.get('lang_cap', [])
            logits = outputs.get('logits', None)
            
            # Flatten lang_cap to list of strings for GRPO training
            generated_texts = []
            if lang_cap_raw:
                for batch_captions in lang_cap_raw:
                    if isinstance(batch_captions, list):
                        # For dense captioning, take the first non-empty caption
                        for caption in batch_captions:
                            if caption and caption.strip():
                                generated_texts.append(caption.strip())
                                break
                        else:
                            # If no non-empty caption found, use empty string
                            generated_texts.append("")
                    else:
                        # If it's already a string
                        generated_texts.append(str(batch_captions))
            
        return generated_texts, logits
    
    def compute_kl_divergence(self, logits_current: torch.Tensor, 
                            logits_ref: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between current and reference model
        """
        # Handle case where logits might be None
        if logits_current is None or logits_ref is None:
            return torch.tensor(0.0, device=self.device)
        
        # Ensure logits have compatible shapes
        if logits_current.shape != logits_ref.shape:
            # If shapes don't match, return zero KL divergence
            return torch.tensor(0.0, device=self.device)
        
        probs_current = F.softmax(logits_current, dim=-1)
        probs_ref = F.softmax(logits_ref, dim=-1)
        
        kl_div = F.kl_div(
            probs_current.log(), 
            probs_ref, 
            reduction='batchmean'
        )
        
        return kl_div
    
    def train_step(self, batch_data: Dict, ground_truth: List[str]) -> Dict:
        """
        Single training step for GRPO
        """
        self.model.train()
        
        # Generate responses with current model
        generated_texts, current_logits = self.generate_responses(batch_data)
        
        # Generate responses with reference model
        with torch.no_grad():
            self.ref_model.eval()
            ref_outputs = self.ref_model(batch_data, is_eval=True)
            ref_logits = ref_outputs.get('logits', None)
        
        # Compute rewards
        if generated_texts:
            rewards = self.reward_functions.compute_total_reward(
                batch_data, generated_texts, ground_truth
            )
        else:
            # If no generated texts, create zero rewards
            rewards = torch.zeros(1, device=self.device)
        
        # Compute KL divergence
        kl_div = self.compute_kl_divergence(current_logits, ref_logits)
        
        # Compute policy loss with KL penalty
        policy_loss = -torch.mean(rewards) + self.config['beta'] * kl_div
        
        # Backward pass
        self.optimizer.zero_grad()
        policy_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config['max_grad_norm']
        )
        
        self.optimizer.step()
        
        # Compute individual reward components safely
        if generated_texts and len(generated_texts) > 0:
            format_reward = self.reward_functions.compute_format_reward(generated_texts[0])
            perception_reward = self.reward_functions.compute_perception_reward(batch_data, 0, generated_texts[0])
            semantic_reward = self.reward_functions.compute_semantic_similarity_reward(generated_texts[0], ground_truth[0] if ground_truth else "")
        else:
            format_reward = 0.0
            perception_reward = 0.0
            semantic_reward = 0.0
        
        return {
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item(),
            'avg_reward': rewards.mean().item(),
            'format_reward': format_reward,
            'perception_reward': perception_reward,
            'semantic_reward': semantic_reward
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        """
        Train for one epoch
        """
        epoch_stats = {
            'policy_loss': [],
            'kl_div': [],
            'avg_reward': [],
            'format_reward': [],
            'perception_reward': [],
            'semantic_reward': []
        }
        
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="GRPO Training")):
            # Move data to device
            for key in batch_data:
                if isinstance(batch_data[key], torch.Tensor):
                    batch_data[key] = batch_data[key].to(self.device)
            
            # Extract ground truth
            ground_truth = batch_data.get('target_text', [])
            
            # Training step
            step_stats = self.train_step(batch_data, ground_truth)
            
            # Collect statistics
            for key in epoch_stats:
                if key in step_stats:
                    epoch_stats[key].append(step_stats[key])
        
        # Compute averages
        avg_stats = {}
        for key in epoch_stats:
            if epoch_stats[key]:
                avg_stats[key] = np.mean(epoch_stats[key])
            else:
                avg_stats[key] = 0.0
                
        return avg_stats
    
    def save_checkpoint(self, path: str, epoch: int, stats: Dict):
        """
        Save training checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'stats': stats
        }
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['stats']
