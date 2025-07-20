export GOOGLE_API_KEY="YOUR_KEY"
python syntheticcot/synthesize_scene30k.py \
    --scanrefer /path/scanrefer_train.json \
    --nr3d /path/nr3d.json \
    --threedllm /path/3dllm_captions.json \
    --scanqa /path/scanqa_train.json \
    --scenes_root /path/scannet/scans \
    --output /path/scene30k_cot_batched_filtered.jsonl \
    --reject-output /path/scene30k_cot_batched_rejects.jsonl \
    --model gemini-2.5-pro \
    --consistency-model gemini-2.5-pro \
    --batch-size 16 \
    --min-think-words 30 \
    --min-answer-words 20 \
    --similarity-threshold 0.8