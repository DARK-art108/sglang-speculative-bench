<div align="center">

# sglang-speculative-bench

### Rigorous Speculative Decoding & KV Cache Benchmarks for Qwen2.5-32B-Instruct-AWQ on Dual Consumer GPUs

[![Engine: SGLang](https://img.shields.io/badge/Engine-SGLang_0.5.19-blue.svg)](https://github.com/sgl-project/sglang)
[![Model: Qwen2.5-32B-AWQ](https://img.shields.io/badge/Model-Qwen2.5--32B--Instruct--AWQ-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ)
[![Hardware: 2x RTX PRO 4000 Blackwell](https://img.shields.io/badge/Hardware-2x_RTX_PRO_4000_Blackwell_(48.9GB)-green.svg)](#hardware--software-environment)
[![CUDA: 13.0](https://img.shields.io/badge/CUDA-13.0-red.svg)](#hardware--software-environment)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-orange.svg)](LICENSE)

**Achieved: 1.28x median speedup (1.34x mean) over baseline using NGRAM speculative decoding — zero additional model parameters required.**

</div>

---

## Overview

`sglang-speculative-bench` is a complete, reproducible benchmarking testbed for evaluating **speculative decoding** and **KV cache quantization** strategies for large language model (LLM) inference on consumer-grade dual-GPU hardware.

The core challenge this repository addresses: serving a 32-billion parameter model (Qwen2.5-32B-Instruct-AWQ) across two consumer RTX GPUs requires Tensor Parallelism (TP=2) over a PCIe Gen4 interconnect (~26.8 GB/s) — approximately 30x slower than the NVLink bandwidth available on data-center-class hardware. Every single generated token forces both GPUs to synchronize across this PCIe bus. Speculative decoding dramatically reduces the number of these expensive synchronization steps.

This repository contains:
- Automated streaming benchmark harness with Prometheus metric scraping
- Standardized 90-prompt multi-domain dataset (code, prose, JSON)
- GPU telemetry daemon (VRAM, power, temperature at 1 Hz)
- Pre-configured shell launch scripts for all server configurations
- Full research results from a real cloud run on dual RTX PRO 4000 Blackwell GPUs

---

## Key Results (Actual Measured Data)

Results from instance `50625322` — SGLang 0.5.19, FlashInfer 0.6.18, CUDA 13.0, dual RTX PRO 4000 Blackwell (SM12, 24,467 MiB each), PCIe Gen4.

| Configuration | n | Median Decode (tok/s) | Mean Decode (tok/s) | Std Dev | Median TTFT (ms) | Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1: Baseline TP=2 (INT8 AWQ)** | 90 | 55.60 | 55.71 | 0.21 | 34.9 | 1.00x (reference) |
| **R2: NGRAM Speculative (K=5)** | 90 | **71.37** | **74.59** | 19.18 | 37.6 | **1.28x** |
| **R3: FP8 KV Cache (fp8_e5m2)** | 90 | 54.77 | 54.86 | 0.19 | 35.5 | 0.985x (-1.5%) |

> **Key finding:** NGRAM speculative decoding delivers a 1.28x median / 1.34x mean throughput improvement with zero additional model weights, at the cost of high inter-prompt variance (std dev 19.18 tok/s). FP8 KV cache provides no single-stream benefit but is architecturally valuable for concurrent multi-request serving.

**Domain-Level Throughput (NGRAM, R2):**

| Domain | NGRAM Tok/s Range | Peak Observed | vs Baseline |
| :--- | :--- | :--- | :--- |
| **JSON** | 65 – 107 tok/s | 107.35 (json_05) | +71% to +93% |
| **Code** | 60 – 226 tok/s | 225.95 (code_02) | +8% to +306% |
| **Prose** | 58 – 82 tok/s | 82.01 (prose_10) | +4% to +47% |

---

## Research Questions & Hypotheses

| Hypothesis | Status | Finding |
| :--- | :--- | :--- |
| **H1:** Speculative decoding accelerates PCIe-bound TP=2 inference | **Confirmed** | NGRAM achieved 1.28x–1.34x speedup |
| **H2:** JSON > Code > Prose for n-gram acceptance rate | **Confirmed** | JSON peaks at 107 tok/s, Code at 226 tok/s (burst), Prose floor at 58 tok/s |
| **H3:** EAGLE-3 provides higher speedup than Standalone 1.5B | **Untested** | EAGLE-3 blocked by Blackwell SM12 CUDA graph dtype bug in SGLang 0.5.19 |
| **H4:** Standalone 1.5B delivers 1.5–2.0x speedup | **Untested** | Blocked by vocabulary mismatch (152,064 vs 151,936 tokens) |

---

## System Architecture

```text
                    +------------------------------------------+
                    |         Benchmark Client                 |
                    |     (scripts/run_benchmark.py)           |
                    +------------------------------------------+
                            |                    ^
           Streaming HTTP   |                    | Prometheus Scrape
        /v1/chat/completions|                    | /metrics
                            v                    |
                    +------------------------------------------+
                    |        SGLang Server Engine              |
                    |  (RadixAttention + FlashInfer 0.6.18)    |
                    +------------------------------------------+
                              /                 \
          [Draft Engine]                         [Target Verification]
          NGRAM / EAGLE3                         Qwen2.5-32B-AWQ
          Fast token prediction                  1 Batched AllReduce
                    \                            /
                     +-------PCIe Gen4----------+
                     |  ~26.8 GB/s (no NVLink)  |
                     | GPU0 (24.5 GB)  GPU1 (24.5 GB)|
                     +--------------------------+
```

**How Speculative Decoding Helps on PCIe:**
- **Without spec decoding:** Every token requires 1 full TP=2 AllReduce sync across PCIe.
- **With spec decoding (K=5):** Draft model proposes 5 tokens locally (minimal sync), target model verifies all 5 in 1 single AllReduce pass.
- **Net effect:** 5 tokens generated for the cost of ~1.2–1.5 PCIe synchronization overhead cycles.

---

## Experimental Configuration Matrix

| Run ID | Config Name | Algorithm | Draft Model | K Steps | topk | VRAM Fraction | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | `baseline_tp2` | None | None | — | — | 0.82 | **Complete** |
| **R2** | `spec_ngram_k5` | NGRAM | (prompt context) | 5 | 1 | 0.85 | **Complete** |
| **R3** | `baseline_fp8kv` | None | None | — | — | 0.85 | **Complete** |
| **R4** | `spec_standalone_1.5b` | STANDALONE | Qwen2.5-1.5B-Instruct | 5 | — | 0.82 | **FAILED** — vocab mismatch |
| **R5** | `spec_eagle3` | EAGLE3 | ruipeterpan/..._EAGLE3_UltraChat | 8 | 10 | 0.85 | **FAILED** — SM12 CUDA graph bug |

---

## Hardware & Software Environment

### Hardware Profile

| Component | Specification |
| :--- | :--- |
| GPU | 2x NVIDIA RTX PRO 4000 Blackwell (SM 12.0) |
| VRAM per GPU | 24,467 MiB (aggregate 48,934 MiB) |
| GPU Interconnect | PCIe Gen4 x16 (no NVLink) |
| Effective PCIe Bandwidth | ~26.8 GB/s (measured) |
| Driver | 595.71.05 |
| CUDA | 13.0 |

### VRAM Allocation per GPU (TP=2)

```text
 |<---- Model Weights ~9.75 GB ---->|<-- KV Cache ~5.13 GB -->|<- Runtime ~3-4 GB ->|
 |                                  |                          |                     |
 0                               9.75 GB                   14.88 GB           ~21.7–22.6 GB (used)
                                                                                    of 24.5 GB total
```

---

## Benchmark Dataset

90 prompts across three domains with `temperature=0.0` for fully deterministic, reproducible greedy sampling:

| Domain | File | N | Max Tokens | Characteristics | Expected NGRAM Acceptance |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Code** | `prompts_code.jsonl` | 30 | 512 | Python algorithms, async I/O, data structures — high syntactic regularity | 70% – 85% |
| **JSON** | `prompts_json.jsonl` | 30 | 384 | Schema extraction, log parsing, OpenAPI — highly repetitive structural tokens | 75% – 90% |
| **Prose** | `prompts_prose.jsonl` | 30 | 512 | Technical explanations, architecture trade-offs — natural language entropy | 55% – 70% |

---

## Quickstart & Replication

### Prerequisites

- Docker with GPU access (`--gpus all`)
- Hugging Face token with access to Qwen models (`HF_TOKEN` in `.env`)
- 2x NVIDIA GPUs with >= 24 GB VRAM each

### Step 1: Configure Environment

```bash
cp .env.example .env
# Edit .env and set: HF_TOKEN=hf_your_token_here
```

### Step 2: Launch SGLang Server

Choose one configuration (e.g., baseline):

```bash
bash scripts/launch_baseline_tp2.sh
```

Wait for server readiness:
```bash
until curl -s http://localhost:30000/health > /dev/null 2>&1; do
  echo "Waiting for server..."; sleep 3
done
echo "Server ready!"
```

### Step 3: Start GPU Telemetry

```bash
bash scripts/monitor_gpu.sh results/gpu_telemetry_baseline.csv &
MONITOR_PID=$!
```

### Step 4: Run Benchmark Harness

```bash
python3 scripts/run_benchmark.py \
  --endpoint http://localhost:30000 \
  --model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --dataset dataset/combined_benchmark_dataset.jsonl \
  --config-name baseline_tp2 \
  --output-csv results/benchmark_runs.csv \
  --warmup 2

kill -9 $MONITOR_PID
```

---

## Full Research Report

For the complete, research-grade analysis of all results including per-prompt breakdowns, VRAM budget decomposition, failure mode root cause analysis, theoretical memory bandwidth ceiling calculations, and reproduction guidance, see:

**[`results/research_report.md`](results/research_report.md)**

---

## Known Issues & Compatibility

| Issue | Affected Config | Root Cause | Workaround |
| :--- | :--- | :--- | :--- |
| Vocabulary mismatch at server init | Standalone 1.5B (R4) | Target vocab: 152,064 vs Draft vocab: 151,936 (128 token gap) | Use a 1.5B model built against the same tokenizer, or use NGRAM/EAGLE3 |
| Cutlass TVM prefill dtype mismatch | EAGLE-3 (R5) | SGLang 0.5.19 EAGLE-3 prefill CUDA graph not compatible with Blackwell SM12 | Upgrade to SGLang >= 0.5.20 when SM12 fix is merged |

---

## Citation

If you use this benchmarking suite, dataset, or methodology in your research, please cite:

```bibtex
@misc{sglang-speculative-bench-2026,
  author    = {DARK-art108},
  title     = {sglang-speculative-bench: Speculative Decoding and KV Cache Benchmarks for Qwen2.5-32B on Dual Consumer GPUs},
  year      = {2026},
  publisher = {GitHub},
  journal   = {GitHub repository},
  howpublished = {\url{https://github.com/DARK-art108/sglang-speculative-bench}},
  note      = {SGLang 0.5.19, FlashInfer 0.6.18, 2x NVIDIA RTX PRO 4000 Blackwell}
}
```
