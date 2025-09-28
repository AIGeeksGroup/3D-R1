# -*- coding: utf-8 -*-

import os, re, time, json, torch
from collections import OrderedDict, Counter
from tqdm import tqdm

import utils.capeval.bleu.bleu   as capblue
import utils.capeval.cider.cider as capcider
import utils.capeval.rouge.rouge as caprouge
import utils.capeval.meteor.meteor as capmeteor

from utils.misc import SmoothedValue
from utils.dist import (
    is_primary, barrier, all_gather_dict,
)

# ----------------------------------------------------------------------
TAG_RE   = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>",  re.I | re.S)

def _extract_ans(txt: str) -> str:
    """Extract answer from text with <answer> tags"""
    txt = THINK_RE.sub("", txt)
    m = TAG_RE.search(txt)
    ans = m.group(1) if m else txt
    return " ".join(ans.strip().lower().split())

def _extract_think(txt: str) -> str:
    """Extract thinking from text with <think> tags"""
    m = THINK_RE.search(txt)
    if not m:
        return ""
    return " ".join(m.group(1).strip().lower().split())

def _f1(pred, gold):
    """Compute F1 score between predicted and gold text"""
    pc, gc = pred.split(), gold.split()
    common = Counter(pc) & Counter(gc)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pc)
    r = num_same / len(gc)
    return 2 * p * r / (p + r)

def _score_corpus(refs: dict, hyps: dict):
    """Compute corpus-level scores using multiple metrics"""
    bleu      = capblue.Bleu(4).compute_score(refs, hyps)      # (list, per-sent)
    cider     = capcider.Cider().compute_score(refs, hyps)     # (float, per-sent)
    rouge_l   = caprouge.Rouge().compute_score(refs, hyps)     # (float, per-sent)
    meteor    = capmeteor.Meteor().compute_score(refs, hyps)   # (float, per-sent)

    summary = OrderedDict(
        BLEU1   = bleu[0][0],
        BLEU4   = bleu[0][3],
        CiDEr   = cider[0],
        ROUGE_L = rouge_l[0],
        METEOR  = meteor[0],
    )
    return summary

# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset_loader,
    logout=print,
    curr_train_iter=-1,
):
    """
    Evaluate QA model performance
    """
    device     = next(model.parameters()).device
    tokenizer  = dataset_loader.dataset.tokenizer
    annotations = dataset_loader.dataset.annotations
    num_batches = len(dataset_loader)
    time_delta  = SmoothedValue(10)

    corpus, cand = {}, {}
    think_corpus, think_cand = {}, {}
    em_total, f1_total, n_samples = 0, 0, 0

    model.eval(); barrier()
    epoch_tag = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch >= 0 else ""

    for bi, batch in enumerate(dataset_loader):
        tic = time.time()
        
        # Optimized data loading - move to device in one go
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        model_inp = {
            'point_clouds':          batch['point_clouds'],
            'point_cloud_dims_min':  batch['point_cloud_dims_min'],
            'point_cloud_dims_max':  batch['point_cloud_dims_max'],
            'qformer_input_ids':     batch['qformer_input_ids'],
            'qformer_attention_mask':batch['qformer_attention_mask'],
            'instruction':           batch['instruction'],
            'instruction_mask':      batch['instruction_mask'],
        }
        
        # Clear cache before model forward pass
        torch.cuda.empty_cache()
        
        # Use mixed precision if enabled
        if getattr(args, 'eval_use_fp16', False):
            with torch.cuda.amp.autocast():
                outputs = model(model_inp, is_eval=True, task_name="qa")
        else:
            outputs = model(model_inp, is_eval=True, task_name="qa")
        outputs = all_gather_dict(dict(output_ids=outputs["output_ids"]))
        batch   = all_gather_dict(batch)

        # Optimized batch decoding
        dec_txt = tokenizer.batch_decode(
            outputs['output_ids'],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )

        # Vectorized processing for better performance
        batch_global_indices = batch['scan_idx'].cpu().numpy()
        batch_keys = []
        batch_pred_ans = []
        batch_gold_ans = []
        
        for i, txt in enumerate(dec_txt):
            global_idx = batch_global_indices[i]
            anno = annotations[global_idx]
            key = f"{anno['scene_id']}-{global_idx}"

            pred_ans = _extract_ans(txt)
            gold_ans = _extract_think(anno.get('cot', '')) + _extract_ans(anno.get('cot', ''))

            batch_keys.append(key)
            batch_pred_ans.append(pred_ans)
            batch_gold_ans.append(gold_ans)

        # Batch update metrics
        for i in range(len(batch_keys)):
            em_total += int(batch_pred_ans[i] == batch_gold_ans[i])
            f1_total += _f1(batch_pred_ans[i], batch_gold_ans[i])
            n_samples += 1

            cand[batch_keys[i]] = [batch_pred_ans[i]]
            corpus[batch_keys[i]] = [batch_gold_ans[i]]

        # Clear cache after processing
        torch.cuda.empty_cache()

        # --- log ---
        time_delta.update(time.time() - tic)
        if is_primary() and bi % args.log_every == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**2
            logout(
                f"Eval {epoch_tag} Batch [{bi}/{num_batches}] "
                f"Iter {curr_train_iter}; "
                f"t {time_delta.avg:.2f}s; mem {mem:.1f} MB"
            )
        barrier()
    
    # Skip expensive metric computations if requested
    if getattr(args, 'eval_skip_metrics', False):
        sent_scores = {
            'BLEU1': 0.0, 'BLEU4': 0.0, 'CiDEr': 0.0, 
            'ROUGE_L': 0.0, 'METEOR': 0.0
        }
    else:
        sent_scores = _score_corpus(corpus, cand)
    
    metrics = OrderedDict(
        EM   = round(em_total / n_samples * 100, 2),
        F1   = round(f1_total / n_samples * 100, 2),
        **{k: round(v * 100, 2) for k, v in sent_scores.items()},
    )

    if is_primary():
        logout("\n---------------------- QA Evaluation ----------------------")
        for k, v in metrics.items():
            logout(f"{k:<7}: {v:.2f}")

        with open(os.path.join(args.checkpoint_dir, "qa_pred.json"), "w") as f:
            json.dump(cand, f, indent=2)
        with open(os.path.join(args.checkpoint_dir, "qa_gt.json"), "w") as f:
            json.dump(corpus, f, indent=2)

    return metrics

