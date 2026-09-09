#!/usr/bin/env bash
# Non-speculative baseline on 2x RTX 4090 with Tensor Parallelism (TP=2)
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
    --tp 2 \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --enable-metrics \
    --host 0.0.0.0 \
    --port 30000
