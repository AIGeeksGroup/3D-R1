# 3D-R1 RL Training Guide

This guide explains how to use the unified datasets for RL training with GRPO to resolve the reward gradient issue.

## Problem Solved

The original error `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn` occurred because:

1. **Missing Target Text**: GRPO reward functions needed `target_text` for semantic similarity computation
2. **No Gradient Flow**: Without target text, reward functions couldn't compute gradients properly
3. **Dataset Mismatch**: Standard datasets didn't include target text needed for RL training

## Solution: Unified Datasets

We created unified dataset files that support both standard SFT training and RL training modes:

### Available Unified Datasets

| Task | Dataset | File | Target Text |
|------|---------|------|-------------|
| 3D-QA | ScanQA | `unified_scanqa.py` | Answer text |
| 3D-DC | ScanRefer | `unified_scanrefer.py` | Caption text |
| 3D-DC | Nr3D | `unified_nr3d.py` | Caption text |
| Dialogue | 3D-LLM | `unified_dialogue.py` | Response text |
| Planning | 3D-LLM | `unified_planning.py` | Plan text |

### Key Features

1. **Dual Mode Support**: Each dataset supports both SFT and RL training
2. **Target Text Inclusion**: RL mode includes `target_text` for reward computation
3. **Backward Compatibility**: Standard mode works with existing SFT training
4. **Task-Specific**: Each task has appropriate target text format

## Usage

### 1. RL Training with Unified Datasets

```bash
# Basic RL training
bash script/train.rl_complete.sh

# Custom task training
python main_rl.py \
    --dataset scanqa \
    --use_unified_dataset \
    --use_rl_training \
    --grpo_beta 0.1 \
    --grpo_lambda 0.95 \
    --grpo_kl_penalty 0.1
```

### 2. Task-Specific Training

```bash
# 3D-QA (ScanQA)
python main_rl.py --dataset scanqa --use_unified_dataset --use_rl_training

# 3D-DC (ScanRefer)
python main_rl.py --dataset scanrefer --use_unified_dataset --use_rl_training

# 3D-DC (Nr3D)
python main_rl.py --dataset nr3d --use_unified_dataset --use_rl_training

# Dialogue
python main_rl.py --dataset dialogue --use_unified_dataset --use_rl_training

# Planning
python main_rl.py --dataset planning --use_unified_dataset --use_rl_training
```

### 3. Standard SFT Training (Backward Compatible)

```bash
# Standard training still works
python main.py --dataset scanqa  # Uses standard dataset
```

## Configuration Parameters

### New Arguments

- `--use_unified_dataset`: Enable unified dataset mode
- `--use_rl_training`: Enable RL training mode with target text
- `--grpo_beta`: GRPO beta parameter (default: 0.1)
- `--grpo_lambda`: GRPO lambda parameter (default: 0.95)
- `--grpo_kl_penalty`: KL divergence penalty (default: 0.1)
- `--grpo_max_grad_norm`: Maximum gradient norm (default: 1.0)

### Evaluation Optimizations

- `--eval_batch_size`: Batch size for evaluation (default: same as training)
- `--eval_max_samples`: Limit samples for quick testing
- `--eval_use_fp16`: Use mixed precision for evaluation
- `--eval_skip_metrics`: Skip expensive metric computations

## Dataset Structure

### RL Training Mode Output

```python
ret_dict = {
    # Standard fields
    'point_clouds': ...,
    'instruction': ...,
    'instruction_mask': ...,
    'qformer_input_ids': ...,
    'qformer_attention_mask': ...,
    
    # RL-specific fields
    'target_text': answer,           # Target text for reward computation
    'target_response': response,     # Full target response
    'question': question,           # Question for context
    'object_ids': target_obj_id,    # Object IDs for reference
}
```

### SFT Training Mode Output

```python
ret_dict = {
    # Standard fields
    'point_clouds': ...,
    'instruction': ...,
    'instruction_mask': ...,
    'qformer_input_ids': ...,
    'qformer_attention_mask': ...,
    
    # SFT-specific fields
    'input_ids': ...,              # Input IDs for training
    'attention_mask': ...,         # Attention mask
    'gradient_mask': ...,          # Gradient mask
}
```

## Reward Functions

The reward functions now properly handle target text:

1. **Format Reward**: Ensures proper structure with tags
2. **Perception Reward**: IoU-based object detection accuracy
3. **Semantic Similarity Reward**: CLIP-based text similarity (requires target_text)

## Files Created

### Dataset Files
- `dataset/unified_scanqa.py` - Unified ScanQA dataset
- `dataset/unified_scanrefer.py` - Unified ScanRefer dataset
- `dataset/unified_nr3d.py` - Unified Nr3D dataset
- `dataset/unified_dialogue.py` - Unified Dialogue dataset
- `dataset/unified_planning.py` - Unified Planning dataset

### Supporting Files
- `dataset/task_prompts.py` - Task prompt templates
- `eval_utils/evaluate_qa.py` - QA evaluation
- `eval_utils/evaluate_densecap.py` - Dense captioning evaluation
- `eval_utils/evaluate_dialogue.py` - Dialogue evaluation

### Training Scripts
- `script/train.rl_complete.sh` - Complete RL training script
- `script/train.rl_unified.sh` - Basic RL training script

## Expected Results

- **Before**: `RuntimeError: element 0 of tensors does not require grad`
- **After**: Successful GRPO training with proper gradient flow
- **Reward Computation**: Semantic similarity rewards work correctly
- **Training Stability**: No more gradient-related crashes

## Troubleshooting

### Common Issues

1. **Missing target_text**: Ensure `--use_unified_dataset` and `--use_rl_training` are set
2. **Import errors**: Check that all dataset files are in the correct location
3. **Memory issues**: Reduce batch size or use `--eval_use_fp16`

### Performance Tips

1. **Quick Testing**: Use `--eval_max_samples 100` for fast evaluation
2. **Memory Optimization**: Use `--eval_use_fp16` for mixed precision
3. **Speed Optimization**: Use `--eval_skip_metrics` during training iterations

## Next Steps

1. **Test RL Training**: Run the unified RL training script
2. **Monitor Rewards**: Check that semantic similarity rewards are computed correctly
3. **Adjust Hyperparameters**: Tune GRPO parameters for optimal performance
4. **Evaluate Results**: Compare RL-trained models with SFT models

The unified dataset approach ensures that GRPO training has access to the necessary target text for reward computation while maintaining backward compatibility with existing SFT training pipelines.

