#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import json
import time
import glob
import base64
import random
import argparse
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from tqdm import tqdm

# Prefer orjson for speed; fall back to std json
try:
    import orjson as fastjson
    def dumps(obj): return fastjson.dumps(obj).decode("utf-8")
except Exception:
    def dumps(obj): return json.dumps(obj, ensure_ascii=False)

# ---------------------------
# Data loading and indexing
# ---------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def index_scanrefer(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group ScanRefer descriptions by scene_id."""
    idx = defaultdict(list)
    for r in records:
        idx[r["scene_id"]].append({
            "ann_id": r.get("ann_id"),
            "object_id": r.get("object_id"),
            "object_name": r.get("object_name"),
            "description": r.get("description"),
        })
    return idx

def index_nr3d(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group Nr3D descriptions by scene_id."""
    idx = defaultdict(list)
    for r in records:
        idx[r["scene_id"]].append({
            "ann_id": r.get("ann_id"),
            "object_id": r.get("object_id"),
            "object_name": r.get("object_name"),
            "description": r.get("description"),
        })
    return idx

def index_3dllm(caps: Dict[str, Any]) -> Dict[str, List[str]]:
    """3D-LLM: {scene_id: {captions:[...]}} -> {scene_id: [captions]}"""
    out = {}
    for sid, val in caps.items():
        out[sid] = val.get("captions", [])
    return out

def index_scanqa(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Index ScanQA (user-provided format). Keep the first element of "answers" as
    reference gold answer if present.
    """
    idx = defaultdict(list)
    for r in records:
        sid = r.get("scene_id")
        if not sid:
            continue
        idx[sid].append({
            "question_id": r.get("question_id"),
            "question": r.get("question"),
            "choices": r.get("choices"),               # optional
            "answer": (r.get("answers") or [None])[0], # gold (optional)
            "object_ids": r.get("object_ids"),
            "object_names": r.get("object_names"),
        })
    return idx

# ---------------------------
# Multi-view image handling
# ---------------------------

def list_all_images(scene_id: str, scenes_root: str) -> List[str]:
    """
    Return ALL JPG images for the scene under:
      {scenes_root}/{scene_id}/color/*.jpg
    Sorted lexicographically by file name.
    """
    color_dir = os.path.join(scenes_root, scene_id, "color")
    if not os.path.isdir(color_dir):
        return []
    return sorted(glob.glob(os.path.join(color_dir, "*.jpg")))

def chunk_list(seq: List[str], chunk_size: int) -> List[List[str]]:
    """Split a list into chunks of size chunk_size (last chunk may be smaller)."""
    if chunk_size <= 0:
        return [seq]
    return [seq[i:i+chunk_size] for i in range(0, len(seq), chunk_size)]

def encode_image(path: str) -> Dict[str, Any]:
    """Read and base64-encode a JPEG image for Gemini inline upload."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    return {"mime_type": "image/jpeg", "data": data}

# ---------------------------
# Prompt construction
# ---------------------------

PROMPT_TEMPLATE = """You are an AI visual assistant in a 3D scene. Each scene contains a piece of description and multiview images.

## Scene ID
{scene_id}

## Scene Descriptions
- ScanRefer (up to 5):
{scanrefer_block}

- Nr3D (up to 5):
{nr3d_block}

- 3D-LLM (up to 3):
{threedllm_block}

## Question
{question}

## Global Instructions
You will receive the multiview images in several batches.
- For batches 1 to N-1: analyze the images and reply exactly with "ACK {i}" (no other text).
- For the final batch N: synthesize all information from previous batches and output ONLY:
<think>your step-by-step reasoning in 3-8 short sentences</think><answer>the final short answer span</answer>
No extra text, markdown, or JSON.
"""

def make_block(items: List[Dict[str, Any]], cap: int = 5) -> str:
    """Format a bullet list of object-level descriptions."""
    if not items:
        return "  (none)"
    items = items[:cap]
    lines = []
    for it in items:
        obj = it.get("object_name", "object")
        ann = it.get("ann_id")
        desc = (it.get("description") or "").strip().replace("\n", " ")
        lines.append(f'  • [{ann}] {obj}: "{desc}"')
    return "\n".join(lines)

def make_block_caps(caps: List[str], cap: int = 3) -> str:
    """Format a bullet list of scene-level captions."""
    if not caps:
        return "  (none)"
    caps = caps[:cap]
    lines = []
    for i, c in enumerate(caps, 1):
        lines.append(f"  • ({i}) {' '.join(c.split())}")
    return "\n".join(lines)

# ---------------------------
# Gemini querying (BATCHED CHAT)
# ---------------------------

def query_gemini_batched(prompt_text: str, image_paths: List[str], model_name: str, batch_size: int) -> str:
    """
    Start a chat; send initial context; then send images in batches.
    - Batches 1..N-1: ask for 'ACK i'
    - Batch N: ask to output the strict <think>...</think><answer>...</answer> format
    Returns the raw text from the final response.
    """
    import google.generativeai as genai

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    # Start a chat session with the initial context
    chat = model.start_chat(history=[])
    chat.send_message([{"text": prompt_text}])  # context only, no images yet

    batches = chunk_list(image_paths, batch_size if batch_size > 0 else len(image_paths))
    total = len(batches)

    # Send intermediate batches
    for i, batch in enumerate(batches, start=1):
        parts = []
        # Instruction for this batch
        if i < total:
            instr = f"Batch {i}/{total}. Analyze these views. Reply EXACTLY with: ACK {i}"
        else:
            instr = (
                f"Batch {i}/{total} (FINAL). Synthesize ALL batches and output ONLY:\n"
                f"<think>your step-by-step reasoning in 3-8 short sentences</think>"
                f"<answer>the final short answer span</answer>"
            )
        parts.append({"text": instr})
        for p in batch:
            parts.append({"inline_data": encode_image(p)})

        # Retry with exponential backoff per batch
        last_err = None
        for attempt in range(5):
            try:
                resp = chat.send_message(parts, safety_settings=None)
                text = (resp.text or "").strip()
                if i < total:
                    # Intermediate batches expect ACK; continue regardless
                    break
                else:
                    return text
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt + random.random())
        if i < total and last_err is not None:
            # Continue even if an intermediate ACK batch failed
            pass

    raise RuntimeError("No final response received from Gemini.")

# ---------------------------
# Consistency check with LLM
# ---------------------------

CONSISTENCY_PROMPT = """Answer the question concisely using ONLY the provided reasoning.
Return ONLY the final answer span, no explanations.

Question:
{question}

Reasoning:
{think}
"""

def query_consistency_llm(think: str, question: str, model_name: str) -> str:
    """Ask a (possibly different) model to answer using ONLY the think content."""
    import google.generativeai as genai
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    prompt = CONSISTENCY_PROMPT.format(question=question, think=think)
    last_err = None
    for attempt in range(5):
        try:
            resp = model.generate_content([{"text": prompt}], safety_settings=None)
            if hasattr(resp, "text") and resp.text:
                return resp.text.strip()
            if resp.candidates and resp.candidates[0].content.parts:
                txt = resp.candidates[0].content.parts[0].text
                return txt.strip()
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"Consistency LLM failed after retries: {last_err}")

# ---------------------------
# Rule-based filtering
# ---------------------------

STEP_CUE_RE = re.compile(
    r'\b('
    r'(?:Step\s*\d+)|First|Firstly|Second|Third|Next|Then|Afterward|Afterwards|Finally|Last|Conclusion'
    r')\b',
    flags=re.IGNORECASE
)

def word_count(text: str) -> int:
    return len([w for w in re.findall(r"[A-Za-z0-9']+", text) if w])

def extract_think_answer(text: str) -> Dict[str, str]:
    """
    Extract <think>...</think><answer>...</answer> from model output.
    Perform minimal cleanup/fallback if not perfectly formatted.
    """
    m1 = re.search(r"<think>(.*?)</think>", text, flags=re.S)
    m2 = re.search(r"<answer>(.*?)</answer>", text, flags=re.S)
    if m1 and m2:
        return {"think": m1.group(1).strip(), "answer": m2.group(1).strip()}
    cleaned = text.strip().strip("`")
    m1 = re.search(r"<think>(.*?)</think>", cleaned, flags=re.S)
    m2 = re.search(r"<answer>(.*?)</answer>", cleaned, flags=re.S)
    return {
        "think": (m1.group(1).strip() if m1 else ""),
        "answer": (m2.group(1).strip() if m2 else cleaned)
    }

def last_step_snippet(think: str) -> str:
    """
    Return the text from the last step cue (e.g., 'Finally', 'Conclusion', 'Step n', 'Last')
    to the end, as a proxy for the final step content.
    """
    matches = list(STEP_CUE_RE.finditer(think))
    if not matches:
        return think.strip()
    last = matches[-1]
    return think[last.start():].strip()

def contains_question_keyword(text: str, question: str) -> bool:
    """
    Heuristic check: the final step must reference the target entity in the question.
    We require at least one non-trivial word (>=3 letters, alnum) from the question
    to appear in the final step snippet.
    """
    q_words = [w.lower() for w in re.findall(r"[A-Za-z0-9]+", question) if len(w) >= 3]
    t_lower = text.lower()
    return any(w in t_lower for w in q_words)

def passes_format_and_length(think: str, answer: str, min_think_words: int, min_answer_words: int) -> Tuple[bool, str]:
    if not think or not answer:
        return False, "format_missing"
    if word_count(think) < min_think_words:
        return False, "think_too_short"
    if word_count(answer) < min_answer_words:
        return False, "answer_too_short"
    return True, "ok"

def passes_multistep(think: str, question: str) -> Tuple[bool, str]:
    cues = list(STEP_CUE_RE.finditer(think))
    if len(cues) < 3:
        return False, "insufficient_steps"
    final_snip = last_step_snippet(think)
    if not contains_question_keyword(final_snip, question):
        return False, "final_step_no_target_ref"
    return True, "ok"

# ---------------------------
# Levenshtein similarity
# ---------------------------

def levenshtein(a: str, b: str) -> int:
    """
    Classic DP implementation of Levenshtein (edit) distance.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0: return lb
    if lb == 0: return la
    # Ensure the first dimension is the shorter string to save memory
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    curr = [0] * (la + 1)
    for j in range(1, lb + 1):
        curr[0] = j
        bj = b[j-1]
        for i in range(1, la + 1):
            cost = 0 if a[i-1] == bj else 1
            curr[i] = min(
                prev[i] + 1,      # deletion
                curr[i-1] + 1,    # insertion
                prev[i-1] + cost  # substitution
            )
        prev, curr = curr, prev
    return prev[la]

def normalized_lev_similarity(a: str, b: str) -> float:
    a_norm = a.strip()
    b_norm = b.strip()
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    d = levenshtein(a_norm, b_norm)
    denom = max(len(a_norm), len(b_norm))
    return 1.0 - (d / denom)

# ---------------------------
# Main
# ---------------------------

def main(args):
    # Load sources
    scanrefer = load_json(args.scanrefer) if args.scanrefer else []
    nr3d = load_json(args.nr3d) if args.nr3d else []
    threedllm = load_json(args.threedllm) if args.threedllm else {}
    scanqa = load_json(args.scanqa)

    # Build indices by scene
    idx_sr = index_scanrefer(scanrefer)
    idx_nr = index_nr3d(nr3d)
    idx_3d = index_3dllm(threedllm)
    idx_qa = index_scanqa(scanqa)

    scene_ids = sorted(idx_qa.keys())
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.reject_output) or ".", exist_ok=True)

    kept, rejected = 0, 0
    with open(args.output, "w", encoding="utf-8") as fout_keep, \
         open(args.reject_output, "w", encoding="utf-8") as fout_rej:

        for sid in tqdm(scene_ids, desc="Scenes"):
            # Build description blocks once per scene
            s_sr = make_block(idx_sr.get(sid, []), cap=5)
            s_nr = make_block(idx_nr.get(sid, []), cap=5)
            s_caps = make_block_caps(idx_3d.get(sid, []), cap=3)

            # Collect ALL images and split into batches
            views = list_all_images(sid, args.scenes_root)
            batches = chunk_list(views, args.batch_size if args.batch_size > 0 else len(views))

            for qa in idx_qa[sid]:
                qid = qa["question_id"]
                q = qa["question"] or ""
                if qa.get("choices"):
                    q = q.strip() + " Choices: " + " ; ".join(qa["choices"])

                prompt = PROMPT_TEMPLATE.format(
                    scene_id=sid,
                    scanrefer_block=s_sr,
                    nr3d_block=s_nr,
                    threedllm_block=s_caps,
                    question=q.replace("\n", " "),
                )

                if args.dry_run:
                    rec = {
                        "scene_id": sid,
                        "question_id": qid,
                        "question": qa["question"],
                        "gold_answer": qa.get("answer"),
                        "prompt_preview": prompt[:1200],
                        "num_images": len(views),
                        "num_batches": len(batches),
                        "batch_size": args.batch_size
                    }
                    fout_keep.write(dumps(rec) + "\n")
                    kept += 1
                    continue

                # 1) Generate with batched images
                try:
                    final_text = query_gemini_batched(
                        prompt_text=prompt,
                        image_paths=views,
                        model_name=args.model,
                        batch_size=args.batch_size
                    )
                except Exception as e:
                    rec = {
                        "scene_id": sid, "question_id": qid, "question": qa["question"],
                        "gold_answer": qa.get("answer"),
                        "error": f"gen_error: {str(e)}",
                        "meta": {"num_images": len(views), "num_batches": len(batches)}
                    }
                    fout_rej.write(dumps(rec) + "\n")
                    rejected += 1
                    continue

                pair = extract_think_answer(final_text)
                think_text = pair.get("think", "")
                ans_text = pair.get("answer", "")

                # 2) Rule-based checks: format & length
                ok_fmt, reason_fmt = passes_format_and_length(
                    think_text, ans_text, args.min_think_words, args.min_answer_words
                )
                if not ok_fmt:
                    rec = {
                        "scene_id": sid, "question_id": qid, "question": qa["question"],
                        "gold_answer": qa.get("answer"),
                        "output": f"<think>{think_text}</think><answer>{ans_text}</answer>",
                        "reject_reason": reason_fmt
                    }
                    fout_rej.write(dumps(rec) + "\n")
                    rejected += 1
                    continue

                # 3) Multi-step reasoning check
                ok_steps, reason_steps = passes_multistep(think_text, qa["question"] or "")
                if not ok_steps:
                    rec = {
                        "scene_id": sid, "question_id": qid, "question": qa["question"],
                        "gold_answer": qa.get("answer"),
                        "output": f"<think>{think_text}</think><answer>{ans_text}</answer>",
                        "reject_reason": reason_steps
                    }
                    fout_rej.write(dumps(rec) + "\n")
                    rejected += 1
                    continue

                # 4) Consistency check with independent LLM answer
                try:
                    a_hat = query_consistency_llm(
                        think=think_text,
                        question=qa["question"] or "",
                        model_name=args.consistency_model
                    )
                except Exception as e:
                    rec = {
                        "scene_id": sid, "question_id": qid, "question": qa["question"],
                        "gold_answer": qa.get("answer"),
                        "output": f"<think>{think_text}</think><answer>{ans_text}</answer>",
                        "reject_reason": f"consistency_llm_error: {str(e)}"
                    }
                    fout_rej.write(dumps(rec) + "\n")
                    rejected += 1
                    continue

                sim = normalized_lev_similarity(a_hat, ans_text)
                if sim < args.similarity_threshold:
                    rec = {
                        "scene_id": sid, "question_id": qid, "question": qa["question"],
                        "gold_answer": qa.get("answer"),
                        "output": f"<think>{think_text}</think><answer>{ans_text}</answer>",
                        "consistency": {"ahat": a_hat, "similarity": sim},
                        "reject_reason": "low_similarity"
                    }
                    fout_rej.write(dumps(rec) + "\n")
                    rejected += 1
                    continue

                # Passed all checks -> keep
                rec = {
                    "scene_id": sid,
                    "question_id": qid,
                    "question": qa["question"],
                    "gold_answer": qa.get("answer"),
                    "output": f"<think>{think_text}</think><answer>{ans_text}</answer>",
                    "consistency": {"ahat": a_hat, "similarity": sim},
                    "meta": {
                        "model": args.model,
                        "consistency_model": args.consistency_model,
                        "time": int(time.time()),
                        "num_images": len(views),
                        "num_batches": len(batches),
                        "batch_size": args.batch_size
                    }
                }
                fout_keep.write(dumps(rec) + "\n")
                kept += 1

    print(f"[DONE] kept={kept} rejected={rejected} -> {args.output} / {args.reject_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanrefer", type=str, required=True, help="ScanRefer JSON path")
    parser.add_argument("--nr3d", type=str, required=True, help="Nr3D JSON path")
    parser.add_argument("--threedllm", type=str, required=True, help="3D-LLM captions JSON path")
    parser.add_argument("--scanqa", type=str, required=True, help="ScanQA JSON path")
    parser.add_argument("--scenes_root", type=str, required=True, help="ScanNet root (contains scene_id/color/*.jpg)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL path (kept samples)")
    parser.add_argument("--reject-output", type=str, required=True, help="Rejects JSONL path (filtered samples)")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro", help="Gemini model for generation")
    parser.add_argument("--consistency-model", type=str, default="gemini-2.5-pro", help="Model used for consistency check (independent answer from think)")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of images per batch (final batch may be smaller). Use 0 to send all at once.")
    parser.add_argument("--min-think-words", type=int, default=30, help="Minimum number of words required in <think>")
    parser.add_argument("--min-answer-words", type=int, default=20, help="Minimum number of words required in <answer>")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, help="Normalized Levenshtein similarity threshold")
    parser.add_argument("--dry-run", action="store_true", help="Do not call the model; only write prompt previews and batch stats")
    args = parser.parse_args()
    main(args)