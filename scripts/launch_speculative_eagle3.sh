#!/usr/bin/env bash
# Speculative decoding with EAGLE3 draft head (lightweight, ~500MB weights)
set -euo pipefail

docker run --gpus all \
  --shm-size 32g \
  --ipc=host \
  -p 30000:30000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
    --model-path Qwen/Qwen2.5-32B-Instruct-AWQ \
    --speculative-draft-model-path ruipeterpan/Qwen2.5-32B-Instruct_EAGLE3_UltraChat \
    --speculative-algorithm EAGLE3 \
    --speculative-num-steps 8 \
    --speculative-eagle-topk 10 \
    --tp 2 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --enable-metrics \
    --host 0.0.0.0 \
    --port 30000
