# SGLang Qwen2.5-32B Benchmarking & Speculative Decoding Plan (2x RTX 4090)

## Context
The goal is to establish a rigorous, reproducible LLM inference benchmarking recipe for SGLang serving `Qwen/Qwen2.5-32B-Instruct-AWQ` on dual NVIDIA GeForce RTX 4090 GPUs (48GB total VRAM, PCIe Gen4, no NVLink) under Tensor Parallelism (`--tp 2`) with speculative decoding. The existing guide in `task/README-advanced-2xRTX4090.md` outlines the setup but contains operational flaws: missing required speculative algorithm flags causing server crashes, reliance on manual curl loops and visual log scraping, lack of TTFT/TPOT latency decomposition, unmanaged RadixAttention prefix caching confounders, and no standardized prompt test suite. This plan establishes the structured workspace layout (`plans/`, `dataset/`, `scripts/`, `results/`), provides a curated 90-prompt multi-domain dataset, builds an automated client that scrapes Prometheus metrics, and delivers an error-free lab recipe.

## Approach

### 1. Workspace Directory & Structure Initialization
Create dedicated project directories to separate benchmarking specs, datasets, executable harnesses, and output logs:
- `plans/`: Stores the refined advanced recipe (`plans/README-advanced-2xRTX4090-refined.md`) and the lab report recipe template.
- `dataset/`: Stores categorized benchmark prompts (`prompts_code.jsonl`, `prompts_prose.jsonl`, `prompts_json.jsonl`, `combined_benchmark_dataset.jsonl`) and the reproducible generation script `generate_dataset.py`.
- `scripts/`: Stores Docker launch scripts (`launch_baseline_tp2.sh`, `launch_speculative_standalone.sh`, `launch_speculative_eagle3.sh`), the GPU monitoring daemon (`monitor_gpu.sh`), and the automated benchmark client (`run_benchmark.py`).
- `results/`: Stores raw CSV runs (`results/benchmark_runs.csv`), server launch logs, and generated markdown summary lab reports.

### 2. Dataset Preparation & Deployment
Deploy the standardized 90-prompt benchmark dataset into `dataset/` spanning three distinct inference workloads with predictable token acceptance characteristics:
- `dataset/prompts_code.jsonl` (30 prompts): Python algorithms, async I/O, data structures, and system utilities. High syntactic regularity and docstrings; expected speculative accept rate 70%–85%. Target output: 512 tokens.
- `dataset/prompts_prose.jsonl` (30 prompts): Deep technical explanations, architecture trade-offs, systems post-mortems, and distributed inference concepts. Natural language entropy; expected speculative accept rate 55%–70%. Target output: 512 tokens.
- `dataset/prompts_json.jsonl` (30 prompts): Strict JSON schema extraction, server log parsing, and OpenAPI object generation. Highly repetitive key-value syntax and structural tokens; expected speculative accept rate 75%–90%. Target output: 384 tokens.
- `dataset/combined_benchmark_dataset.jsonl` (90 prompts): Consolidated multi-domain evaluation suite.
- `dataset/generate_dataset.py`: Standalone CLI script allowing re-generation or parameter expansion of the benchmark datasets. Copied directly from verified local artifact `local://generate_dataset.py`.

### 3. Automated Benchmark Client Implementation
Deploy `scripts/run_benchmark.py` (copied from `local://run_benchmark.py`), replacing the manual curl instructions.
- Connection: Connects to SGLang OpenAI-compatible endpoint (`http://localhost:30000/v1/chat/completions`) using HTTP streaming (`stream: true`, `stream_options: {"include_usage": true}`).
- Latency & Throughput Metrics:
  - Measures client-side Time-To-First-Token (TTFT): $t_{\text{first\_token}} - t_{\text{start}}$ (milliseconds).
  - Measures pure Decode Time: $t_{\text{end}} - t_{\text{first\_token}}$ (seconds).
  - Calculates Decode Throughput: $\text{completion\_tokens} / t_{\text{decode}}$ (tokens/second).
  - Calculates End-to-End Throughput: $\text{completion\_tokens} / t_{\text{total}}$ (tokens/second).
- Server Metric Extraction: Automatically scrapes `http://localhost:30000/metrics` via Prometheus HTTP handler before and after each request, reading exact engine counters:
  - `sglang:spec_accept_rate`: Proportion of proposed draft tokens accepted by Qwen2.5-32B target verification.
  - `sglang:spec_accept_length`: Average count of tokens accepted per speculative step.
- Persisted Output:
  - Real-time terminal progress logging per prompt.
  - Appends every trial into `results/benchmark_runs.csv` with fields: `timestamp`, `config_name`, `model`, `prompt_id`, `category`, `prompt_tokens`, `completion_tokens`, `ttft_s`, `decode_time_s`, `total_time_s`, `decode_tok_per_sec`, `e2e_tok_per_sec`, `spec_accept_rate`, `spec_accept_length`.
  - Statistical summary table upon completion: $n$, Min, Median, Mean, P90, Max, StdDev, TTFT Median/Mean, Speculative Accept Rate.

### 4. Server Execution Scripts with Engine Refinements
Deploy executable shell scripts in `scripts/` fixing the critical bugs and parameter omissions identified in `task/README-advanced-2xRTX4090.md`:

#### 4a. Standalone Speculative Decoding (`scripts/launch_speculative_standalone.sh`)
Fixes the missing algorithm flag bug. When using standard model weights like `Qwen/Qwen2.5-1.5B-Instruct` as a draft model, SGLang crashes unless `--speculative-algorithm STANDALONE` is explicitly declared:
```bash
#!/usr/bin/env bash
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
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-num-steps 5 \
    --tp 2 \
    --mem-fraction-static 0.82 \
    --attention-backend flashinfer \
    --enable-metrics \
    --host 0.0.0.0 \
    --port 30000
```
- `--mem-fraction-static 0.82`: Prevents CUDA OOM on 24GB RTX 4090s by reserving headroom for the standalone draft model weights (~3.1GB) and NCCL communication buffers over PCIe.
- `--attention-backend flashinfer`: Standardized high-performance attention backend for Ada Lovelace (SM 8.9).
- `--enable-metrics`: Exposes Prometheus endpoint `/metrics` for automated scraping.

#### 4b. Non-Speculative Baseline (`scripts/launch_baseline_tp2.sh`)
Provides the control configuration for calculating exact speedup ratios ($S = \text{tok/s}_{\text{spec}} / \text{tok/s}_{\text{base}}$):
```bash
#!/usr/bin/env bash
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
```

#### 4c. EAGLE3 Speculative Decoding (`scripts/launch_speculative_eagle3.sh`)
Adds the specialized EAGLE-3 speculation path using `ruipeterpan/Qwen2.5-32B-Instruct_EAGLE3_UltraChat` as an alternative high-efficiency draft head:
- Requires only ~500MB VRAM/disk (vs 3.1GB for 1.5B standalone).
- Uses tree verification (`--speculative-algorithm EAGLE3 --speculative-num-steps 8 --speculative-eagle-topk 10`).

#### 4d. Hardware Monitor Daemon (`scripts/monitor_gpu.sh`)
Logs VRAM allocations and compute utilization per GPU every 1 second via `nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv -l 1` into `results/gpu_telemetry.csv`.

### 5. Production Lab Guide Refinement
Author `plans/README-advanced-2xRTX4090-refined.md` refining the initial task guide:
- Corrects all CLI flags and flags rationale.
- Details the PCIe Gen4 x16 communication dynamics without NVLink.
- Documents RadixAttention handling: Explains how running identical prompts back-to-back measures 100% prefix-cached decode, whereas running `prompts_code.jsonl` measures real-world uncached generation.
- Provides the structured lab notebook markdown template with pinned image digest (`docker inspect --format='{{index .RepoDigests 0}}' lmsysorg/sglang:latest`), attention backend verification, and statistical comparisons.

## Critical Files & Anchors
- `local://benchmark-dataset.jsonl`: Pre-generated 90-prompt dataset artifact ready for copying into `dataset/`.
- `local://run_benchmark.py`: Complete streaming benchmark client with Prometheus scraper ready for copying into `scripts/run_benchmark.py`.
- `local://generate_dataset.py`: Reproducible generator script ready for copying into `dataset/generate_dataset.py`.
- `task/README-advanced-2xRTX4090.md`: Baseline reference guide containing original specifications and target requirements.
- `.env`: Environment file containing `HF_TOKEN` needed for gated model downloads.

## Verification
1. File & Directory Layout Check: Verify directories `plans/`, `dataset/`, `scripts/`, `results/` are created and contain non-empty files.
2. Dataset Validation Check: Run `python3 -c "import json; [json.loads(l) for l in open('dataset/combined_benchmark_dataset.jsonl')]"` verifying all 90 items parse with valid `id`, `category`, `prompt`, and `max_tokens`.
3. Generator Script Test: Execute `python3 dataset/generate_dataset.py --output-dir dataset` and verify exit code 0 and file byte counts match.
4. Benchmark Script CLI Dry-Run: Run `python3 scripts/run_benchmark.py --help` confirming argument parser, options (`--endpoint`, `--dataset`, `--model`, `--output-csv`, `--config-name`, `--warmup`, `--limit`), and dependencies run without syntax error on standard Python 3.
5. Shell Scripts Executable & Syntax Check: Run `bash -n scripts/launch_baseline_tp2.sh` and `bash -n scripts/launch_speculative_standalone.sh` confirming zero syntax errors and valid executable flags (`chmod +x`).

## Assumptions & Contingencies
- **Draft Model Selection**: Defaults to `Qwen/Qwen2.5-1.5B-Instruct` with `--speculative-algorithm STANDALONE`. If host disk space is under 60GB during pod deployment, the user can substitute `Qwen/Qwen2.5-0.5B-Instruct` (~1GB) by changing the draft path argument without altering the benchmark client.
- **RunPod Disk Size**: Assumes the pod is deployed with $\ge 80$GB container disk (or attached Network Volume mounted at `~/.cache/huggingface`). If only 40GB is available, the launcher scripts support pointing cache to an attached network volume path (`/workspace/cache`).
- **Prefix Cache Isolation**: Benchmark runs default to natural multi-prompt execution across the 30-item category files to reflect realistic decode throughput. If the user wishes to isolate single-prompt repeat speedup, passing `--limit 1` or repeating prompt ID tests pure decode with 100% prefix hit rate.
