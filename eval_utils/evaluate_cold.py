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
    txt = THINK_RE.sub("", txt)
    m = TAG_RE.search(txt)
    ans = m.group(1) if m else txt
    return " ".join(ans.strip().lower().split())
def _extract_think(txt: str) -> str:
    m = THINK_RE.search(txt)
    if not m:
        return ""
    return " ".join(m.group(1).strip().lower().split())

def _f1(pred, gold):
    pc, gc = pred.split(), gold.split()
    common = Counter(pc) & Counter(gc)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    p = num_same / len(pc)
    r = num_same / len(gc)
    return 2 * p * r / (p + r)

def _score_corpus(refs: dict, hyps: dict):
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
        for k in batch: batch[k] = batch[k].to(device)

        model_inp = {
            'point_clouds':          batch['point_clouds'],
            'point_cloud_dims_min':  batch['point_cloud_dims_min'],
            'point_cloud_dims_max':  batch['point_cloud_dims_max'],
            'qformer_input_ids':     batch['qformer_input_ids'],
            'qformer_attention_mask':batch['qformer_attention_mask'],
            'instruction':           batch['instruction'],
            'instruction_mask':      batch['instruction_mask'],
        }
        outputs = model(model_inp, is_eval=True, task_name="qa")
        outputs = all_gather_dict(dict(output_ids=outputs["output_ids"]))
        batch   = all_gather_dict(batch)

        dec_txt = tokenizer.batch_decode(
            outputs['output_ids'],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )

        for i, txt in enumerate(dec_txt):
            global_idx = batch['scan_idx'][i].item()
            anno = annotations[global_idx]
            key  = f"{anno['scene_id']}-{global_idx}"

            pred_ans = _extract_ans(txt)
            gold_ans = _extract_think(anno['cot'])+ _extract_ans(anno['cot']) 


            # —— EM / F1 —— #
            em_total += int(pred_ans == gold_ans)
            f1_total += _f1(pred_ans, gold_ans)
            n_samples += 1

            cand[key]  = [pred_ans]
            corpus[key] = [gold_ans]


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
    # sent_scores = _score_corpus(corpus, cand)
    sent_scores = _score_corpus(corpus,      cand)
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
