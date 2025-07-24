import os, sys, time, math, json, importlib
import torch
import datetime
from collections import defaultdict, OrderedDict
import numpy as np
from tqdm import tqdm

from models.rl.grpo_trainer import GRPOTrainer
from utils.io import save_checkpoint
from utils.misc import SmoothedValue
from utils.dist import (
    init_distributed, 
    is_distributed, 
    is_primary, 
    get_rank,
    barrier,
    all_reduce_average,
    all_gather_dict
)

class Logger:
    def __init__(self, args):
        exp_name = os.path.split(args.checkpoint_dir)[-1]
        self.logger = open(os.path.join(args.checkpoint_dir, f'{exp_name}-rl-logger.log'), 'a')
    def __call__(self, info_str):
        self.logger.write(info_str + "\n")
        self.logger.flush()
        print(info_str)

def do_rl_train(
    args,
    model,
    ref_model,
    tokenizer,
    dataset_config,
    dataloaders,
    best_val_metrics=dict()
):
    """
    RL training using GRPO
    """
    
    logout = Logger(args)
    
    if is_primary():
        logout(f"Starting RL training with args: {args}")
        logout(f"Model: {model}")
    
    device = next(model.parameters()).device
    
    # Initialize GRPO trainer
    grpo_config = {
        'beta': args.rl_beta,
        'lr': args.rl_lr,
        'batch_size': args.batchsize_per_gpu,
        'num_epochs': args.rl_num_epochs,
        'max_grad_norm': args.rl_max_grad_norm,
    }
    
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        device=device,
        grpo_config=grpo_config
    )
    
    # Training loop
    curr_epoch = args.start_epoch
    max_epochs = args.max_epoch
    
    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    
    model.train()
    barrier()
    
    for epoch in range(curr_epoch, max_epochs):
        
        if is_distributed():
            dataloaders["train_sampler"].set_epoch(epoch)
        
        epoch_start_time = time.time()
        
        # Train one epoch
        epoch_stats = trainer.train_epoch(dataloaders['train'])
        
        epoch_time = time.time() - epoch_start_time
        
        # Log statistics
        if is_primary():
            logout(f"Epoch {epoch}/{max_epochs} completed in {epoch_time:.2f}s")
            logout(f"Policy Loss: {epoch_stats['policy_loss']:.4f}")
            logout(f"KL Divergence: {epoch_stats['kl_div']:.4f}")
            logout(f"Average Reward: {epoch_stats['avg_reward']:.4f}")
            logout(f"Format Reward: {epoch_stats['format_reward']:.4f}")
            logout(f"Perception Reward: {epoch_stats['perception_reward']:.4f}")
            logout(f"Semantic Reward: {epoch_stats['semantic_reward']:.4f}")
        
        # Save checkpoint
        if is_primary() and (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, 
                f"rl_checkpoint_epoch_{epoch}.pth"
            )
            trainer.save_checkpoint(checkpoint_path, epoch, epoch_stats)
            logout(f"Saved checkpoint to {checkpoint_path}")
        
        # Evaluation
        if (epoch + 1) % args.eval_every_iteration == 0:
            model.eval()
            for test_loader in dataloaders['test']:
                test_loader.dataset.eval_func(
                    args,
                    epoch,
                    model,
                    dataset_config,
                    test_loader
                )
            model.train()
        
        barrier()
    
    logout("RL training completed!")

def load_sft_model(args, model):
    """
    Load SFT model weights for RL training
    """
    if args.sft_checkpoint is not None:
        checkpoint = torch.load(args.sft_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded SFT checkpoint from {args.sft_checkpoint}")
    else:
        print("Warning: No SFT checkpoint provided for RL training")
    
    return model
