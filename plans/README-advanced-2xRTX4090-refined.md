# Advanced SGLang Recipe: Qwen2.5-32B-Instruct-AWQ + Speculative Decoding
## Hardware Target: 2x NVIDIA GeForce RTX 4090 (48GB Total VRAM, PCIe Gen4 x16, No NVLink)

This document is the refined, production-grade lab guide and reference recipe for serving and benchmarking `Qwen/Qwen2.5-32B-Instruct-AWQ` on dual RTX 4090 GPUs with SGLang. It resolves operational bugs present in initial drafts, standardizes inference telemetry, explains hardware communication trade-offs, and provides a reproducible lab notebook template.

---

## 1. Executive Hardware & Memory Budget

### Instance Allocation Prerequisites
* **GPUs**: 2x NVIDIA GeForce RTX 4090 (24,564 MiB VRAM each).
* **Interconnect**: PCIe Gen4 x16 (~31.5 GB/s bidirectional). **No NVLink bridge available**.
* **System RAM**: $\ge 62$ GB.
* **Disk Volume**: **Minimum 80 GB** container/volume disk.
  * SGLang Docker Image (`lmsysorg/sglang:latest`): ~15–16 GB.
  * Target Model (`Qwen/Qwen2.5-32B-Instruct-AWQ`): ~19.5 GB.
  * Draft Model (`Qwen/Qwen2.5-1.5B-Instruct` or EAGLE3 head): ~3.1 GB (Standalone) / ~0.5 GB (EAGLE3).
  * OS, Hugging Face caching, runtime scratchpad: ~15–20 GB.

### VRAM Static Allocation Budget (Per GPU, TP=2)
Dual RTX 4090 GPUs provide $2 \times 24$ GB VRAM. Under Tensor Parallelism (`--tp 2`):
1. **Target Model Weights**: ~19.5 GB total $\rightarrow$ ~9.75 GB per GPU.
2. **Draft Model Weights** (Standalone 1.5B): ~3.1 GB total $\rightarrow$ duplicated or partitioned across ranks.
3. **KV Cache & CUDA Graphs (`--mem-fraction-static 0.82`)**:
   * $24\text{ GB} \times 0.82 \approx 19.68\text{ GB}$ dedicated to weights + static KV buffer + execution graphs.
   * Leaves $\sim 4.3\text{ GB}$ dynamic headroom per GPU for NCCL communication buffers, PyTorch peak activation buffers, and draft forward tensors.

---

## 2. Critical Engine Flags & Operational Corrections

1. **`--speculative-algorithm STANDALONE`**:
   * *Critical Fix*: Standard Hugging Face transformer draft models (such as `Qwen/Qwen2.5-1.5B-Instruct`) require explicit declaration of the standalone speculative algorithm. Omitting this flag triggers server initialization failure.
2. **`--mem-fraction-static 0.82`**:
   * In non-speculative mode, `0.85` is optimal. In standalone speculative mode, running at `0.85` risks CUDA OOM during draft model initialization or multi-step KV allocation. `0.82` guarantees stability on 24GB cards.
3. **`--attention-backend flashinfer`**:
   * FlashInfer provides optimized page-attention kernels for Ada Lovelace (SM 8.9), outperforming standard Triton backends for decode phases on consumer GeForce architectures.
4. **`--enable-metrics`**:
   * Exposes Prometheus metrics at `http://localhost:30000/metrics`. Essential for programmatically tracking `sglang:spec_accept_rate` and `sglang:spec_accept_length` without parsing stdout logs.
5. **IPC & Shared Memory**:
   * Always pass `--ipc=host` and `--shm-size 32g` to Docker to allow high-throughput inter-process tensor parallel synchronization.

---

## 3. Hardware Dynamics & Confounders

### PCIe Gen4 x16 Interconnect Bottleneck
Unlike data-center SXM/NVLink topologies (up to 900 GB/s inter-GPU bandwidth), dual PCIe RTX 4090 cards communicate over PCIe Gen4 x16 (theoretical limit ~31.5 GB/s, practical NCCL all-reduce ~22–25 GB/s).
* **Impact on Speculative Decoding**:
  * In standard TP=2 decode, every single token generation incurs AllReduce latency across PCIe.
  * In speculative decoding with $K=5$ steps, the draft model generates 5 candidate tokens (minimal communication overhead), and the target model verifies all 5 tokens in a **single batched forward pass** with a single AllReduce sync phase.
  * High acceptance rates ($>70\%$) overcome PCIe transfer latency and yield substantial wall-clock speedups ($1.4\times$–$1.8\times$).

### RadixAttention Prefix Caching Nuance
SGLang defaults to active RadixAttention tree-based prefix caching.
* **Cold Benchmark**: The 90-prompt benchmark dataset (`dataset/combined_benchmark_dataset.jsonl`) contains distinct system prompts and contexts to evaluate uncached end-to-end decode throughput.
* **Warm / Prefix-Hit Evaluation**: Repeating identical queries measures 100% prefix-cached TTFT (<15ms) and pure token generation rate.

---

## 4. Execution Workflow

### Step 1: Baseline Server Launch (Control)
```bash
bash scripts/launch_baseline_tp2.sh
```
Verify readiness:
```bash
curl -s http://localhost:30000/health
curl -s http://localhost:30000/v1/models
```

### Step 2: Telemetry & Baseline Benchmark
In a separate terminal or background job:
```bash
bash scripts/monitor_gpu.sh results/gpu_telemetry_baseline.csv &
MONITOR_PID=$!

python3 scripts/run_benchmark.py \
  --config-name baseline_tp2 \
  --dataset dataset/combined_benchmark_dataset.jsonl \
  --output-csv results/benchmark_runs.csv

kill -9 $MONITOR_PID
```

### Step 3: Speculative Server Launch (Standalone 1.5B)
Stop the baseline container (`docker stop <container_id>`) and launch:
```bash
bash scripts/launch_speculative_standalone.sh
```
Wait for server readiness, then run:
```bash
bash scripts/monitor_gpu.sh results/gpu_telemetry_spec_standalone.csv &
MONITOR_PID=$!

python3 scripts/run_benchmark.py \
  --config-name spec_standalone_1.5b \
  --dataset dataset/combined_benchmark_dataset.jsonl \
  --output-csv results/benchmark_runs.csv

kill -9 $MONITOR_PID
```

### Step 4: Optional EAGLE-3 Speculative Test
For maximum efficiency with a lightweight speculation head (~500MB):
```bash
bash scripts/launch_speculative_eagle3.sh
```
Execute benchmark client with `--config-name spec_eagle3`.

---

## 5. Lab Report Template

Record final metrics in `results/lab_report.md`:

```markdown
# Benchmark Lab Report: Qwen2.5-32B-Instruct-AWQ on 2x RTX 4090

## Environment Snapshot
* **Date**: YYYY-MM-DD
* **Docker Image**: `docker inspect --format='{{index .RepoDigests 0}}' lmsysorg/sglang:latest`
* **Driver Version**: `nvidia-smi --query-gpu=driver_version --format=csv,noheader`
* **CUDA Version**: 12.x
* **Target Model**: `Qwen/Qwen2.5-32B-Instruct-AWQ` (TP=2)
* **Attention Backend**: FlashInfer

## Aggregate Results

| Configuration | Median Tok/s | Mean Tok/s | P90 Tok/s | Mean TTFT (ms) | Accept Rate | Speedup Ratio |
|---|---|---|---|---|---|---|
| Baseline TP=2 | — | — | — | — | N/A | 1.00x |
| Speculative (1.5B Standalone) | — | — | — | — | —% | —x |
| Speculative (EAGLE3) | — | — | — | — | —% | —x |

## Domain Breakdown (Speculative Standalone 1.5B)
* **Code (`dataset/prompts_code.jsonl`)**: Mean Accept Rate: __%, Speedup: __x
* **Prose (`dataset/prompts_prose.jsonl`)**: Mean Accept Rate: __%, Speedup: __x
* **JSON (`dataset/prompts_json.jsonl`)**: Mean Accept Rate: __%, Speedup: __x

## Hardware Observations
* **Peak VRAM GPU 0**: ___ MiB / 24,564 MiB
* **Peak VRAM GPU 1**: ___ MiB / 24,564 MiB
* **Average Power Draw**: GPU 0: ___ W, GPU 1: ___ W
```
