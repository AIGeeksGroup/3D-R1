import os, argparse, importlib
import numpy as np
import torch

from collections import OrderedDict

from engine_rl import do_rl_train, load_sft_model
from models.model_general import CaptionNet
from dataset.scannet_base_dataset import DatasetConfig
from torch.multiprocessing import set_start_method

from utils.io import resume_if_possible
from utils.misc import my_worker_init_fn
from utils.dist import init_distributed, is_distributed, is_primary, get_rank, barrier


def make_args_parser():
    parser = argparse.ArgumentParser("3D-R1 RL Training", add_help=False)

    ##### RL Training #####
    parser.add_argument("--rl_mode", default=True, action="store_true", 
                       help="Enable RL training mode")
    parser.add_argument("--sft_checkpoint", default=None, type=str,
                       help="Path to SFT checkpoint for RL training")
    parser.add_argument("--rl_beta", default=0.1, type=float,
                       help="KL penalty coefficient for RL")
    parser.add_argument("--rl_lr", default=1e-5, type=float,
                       help="Learning rate for RL training")
    parser.add_argument("--rl_num_epochs", default=4, type=int,
                       help="Number of epochs per RL update")
    parser.add_argument("--rl_max_grad_norm", default=1.0, type=float,
                       help="Max gradient norm for RL training")
    
    ##### Model #####
    parser.add_argument("--use_color", default=False, action="store_true")
    parser.add_argument("--use_normal", default=False, action="store_true")
    parser.add_argument("--no_height", default=False, action="store_true")
    parser.add_argument("--use_multiview", default=False, action="store_true")
    
    parser.add_argument("--detector", default="detector_Vote2Cap_DETR", 
                       help="folder of the detector")
    parser.add_argument("--captioner", default=None, type=str, 
                       help="folder of the captioner")
    parser.add_argument("--freeze_detector", default=False, action='store_true')
    parser.add_argument("--freeze_llm", default=False, action='store_true')
    
    parser.add_argument("--max_des_len", default=128, type=int)
    parser.add_argument("--max_gen_len", default=32, type=int)
    
    ##### Dataset #####
    parser.add_argument("--max_prompts", default=16, type=int)
    parser.add_argument("--dataset", default='scannet', help="dataset list split by ','")
    parser.add_argument("--grid_size_3d", default=255, type=int)
    parser.add_argument('--vocab', default="llama-hf/7B", type=str)
    parser.add_argument('--qformer_vocab', default="bert-base-uncased", type=str)
    
    parser.add_argument("--dataset_num_workers", default=16, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=1080, type=int)
    parser.add_argument("--eval_every_iteration", default=4000, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--save_every", default=4000, type=int)
    parser.add_argument("--pretrained_weights", default=None, type=str)

    ##### Dynamic View Selection #####
    parser.add_argument("--enable_dynamic_views", default=False, action="store_true",
                       help="Enable Dynamic View Selection")
    parser.add_argument("--view_selection_weight", default=0.1, type=float,
                       help="Weight for view selection regularization loss")
    
    ##### Additional Encoders (Point Transformer v3, Depth, Image) #####
    parser.add_argument("--use_additional_encoders", default=False, action="store_true",
                       help="Enable additional encoders in captioner (Depth, Image)")
    parser.add_argument("--use_depth", default=True, action="store_true",
                       help="Use depth encoder (Depth-Anything v2)")
    parser.add_argument("--use_image", default=True, action="store_true",
                       help="Use image encoder (SigLIP-2)")
    parser.add_argument("--depth_encoder_dim", default=256, type=int,
                       help="Depth encoder output dimension")
    parser.add_argument("--image_encoder_dim", default=256, type=int,
                       help="Image encoder output dimension")
    
    ##### LoRA Configuration #####
    parser.add_argument("--use_lora", default=False, action="store_true",
                       help="Enable LoRA for efficient fine-tuning")
    parser.add_argument("--lora_r", default=16, type=int,
                       help="LoRA rank (r)")
    parser.add_argument("--lora_alpha", default=32, type=int,
                       help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", default=0.1, type=float,
                       help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", default=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
                       nargs='+', help="Target modules for LoRA")
    
    ##### Distributed #####
    parser.add_argument("--ngpus", default=1, type=int, help='number of gpus')
    parser.add_argument("--dist_url", default='tcp://localhost:12345', type=str)
    
    args = parser.parse_args()
    args.use_height = not args.no_height
    
    return args


def build_dataloader_func(args, dataset, split):
    if is_distributed():
        sampler = torch.utils.data.DistributedSampler(
            dataset, shuffle=(split=='train')
        )
    else:
        if split == "train":
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batchsize_per_gpu,
        num_workers=args.dataset_num_workers,
        worker_init_fn=my_worker_init_fn,
    )
    return sampler, dataloader


def build_dataset(args):
    dataset_config = DatasetConfig()
    datasets = {'train': None, 'test': []}
    
    train_datasets = []
    for dataset in args.dataset.split(','):
        dataset_module = importlib.import_module(f'dataset.{dataset}')
        train_datasets.append(
            dataset_module.Dataset(
                args, dataset_config, split_set="train", 
                use_color=args.use_color, use_normal=args.use_normal,
                use_multiview=args.use_multiview, use_height=args.use_height,
                augment=True
            )
        )
        datasets['test'].append(
            dataset_module.Dataset(
                args, dataset_config, split_set="val", 
                use_color=args.use_color, use_normal=args.use_normal,
                use_multiview=args.use_multiview, use_height=args.use_height,
                augment=False
            )
        )
    datasets['train'] = torch.utils.data.ConcatDataset(train_datasets)
    
    train_sampler, train_loader = build_dataloader_func(args, datasets['train'], split='train')
    dataloaders = {
        'train': train_loader,
        'test': [],
        'train_sampler': train_sampler,
    }
    for dataset in datasets['test']:
        _, test_loader = build_dataloader_func(args, dataset, split='test')
        dataloaders['test'].append(test_loader)
    
    return dataset_config, datasets, dataloaders    


def main(local_rank, args):
    if args.ngpus > 1:
        init_distributed(
            local_rank, global_rank=local_rank, world_size=args.ngpus,
            dist_url=args.dist_url, dist_backend="nccl",
        )
    
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed + get_rank())
    
    if args.checkpoint_dir is None:
        raise AssertionError('Please specify checkpoint_dir!')
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Build datasets and dataloaders
    dataset_config, datasets, dataloaders = build_dataset(args)
    
    # Create main model for RL training
    model = CaptionNet(args, dataset_config, datasets['train'])
    
    # Create reference model (SFT model)
    ref_model = CaptionNet(args, dataset_config, datasets['train'])
    
    # Load SFT weights for reference model
    if args.sft_checkpoint is not None:
        checkpoint = torch.load(args.sft_checkpoint, map_location="cpu")
        ref_model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded SFT checkpoint for reference model: {args.sft_checkpoint}")
    
    # Load pretrained weights for main model if specified
    if args.pretrained_weights is not None:
        checkpoint = torch.load(args.pretrained_weights, map_location="cpu")
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Loaded pretrained weights: {args.pretrained_weights}")
    
    # Move models to GPU
    model_no_ddp = model.cuda()
    model = model.cuda(local_rank)
    ref_model = ref_model.cuda(local_rank)
    
    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )
        ref_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(ref_model)
        ref_model = torch.nn.parallel.DistributedDataParallel(
            ref_model, device_ids=[local_rank]
        )
    
    # Initialize tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.vocab)
    
    # Start RL training
    do_rl_train(
        args, model, ref_model, tokenizer, dataset_config, dataloaders, {}
    )


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))

if __name__ == "__main__":
    args = make_args_parser()
    os.environ['PYTHONWARNINGS']='ignore:semaphore_tracker:UserWarning'
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
