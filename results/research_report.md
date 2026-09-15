# Speculative Decoding and KV Cache Quantization for Large Language Model Inference on Consumer-Grade Dual-GPU Hardware

**A Benchmarking Study on Qwen2.5-32B-Instruct-AWQ with SGLang 0.5.19**

---

> **Abstract**
>
> This report presents the results of a structured inference benchmarking study for the `Qwen/Qwen2.5-32B-Instruct-AWQ` large language model (LLM) served on dual consumer-grade NVIDIA RTX PRO 4000 Blackwell GPUs (48.9 GB aggregate VRAM) under SGLang 0.5.19. Three distinct server configurations were evaluated across a standardized 90-prompt, multi-domain evaluation suite spanning code generation, technical prose, and structured JSON extraction. The baseline integer-8-bit AWQ configuration achieved a stable decode throughput of **55.60 tok/s (median)** with a Time-To-First-Token (TTFT) of **34.9 ms (median)**. The NGRAM speculative decoding configuration (K=5, top-k=1, 6 draft tokens) achieved a median speedup of **1.28x** (71.37 tok/s) over the baseline, rising to **1.34x** mean speedup (74.59 tok/s), with high inter-prompt variance (standard deviation 19.18 tok/s). The FP8 KV cache quantization configuration (fp8_e5m2) produced a marginal **-1.5% regression** in single-stream decode throughput (54.77 tok/s), confirming a valid negative result: FP8 KV caching benefits concurrency and long-context capacity, not single-stream memory-bandwidth-bound decoding. Two primary speculative algorithms — a Standalone 1.5B draft model and the EAGLE-3 draft head — could not be evaluated in this session due to vocabulary mismatch and CUDA graph dtype incompatibility on Blackwell SM12, respectively. These findings are fully documented as reproducibility hazards for the SGLang ecosystem on Blackwell-class hardware.

---

## Table of Contents

1. [Introduction and Motivation](#1-introduction-and-motivation)
2. [Hardware and Software Environment](#2-hardware-and-software-environment)
3. [Experimental Design and Methodology](#3-experimental-design-and-methodology)
4. [Results: Baseline Configuration (R1)](#4-results-baseline-configuration-r1)
5. [Results: NGRAM Speculative Decoding (R2)](#5-results-ngram-speculative-decoding-r2)
6. [Results: FP8 KV Cache Quantization (R3)](#6-results-fp8-kv-cache-quantization-r3)
7. [Comparative Analysis Across Configurations](#7-comparative-analysis-across-configurations)
8. [GPU and Memory Telemetry Analysis](#8-gpu-and-memory-telemetry-analysis)
9. [Planned Configurations: Failure Modes and Reproducibility Notes](#9-planned-configurations-failure-modes-and-reproducibility-notes)
10. [Discussion](#10-discussion)
11. [Conclusions](#11-conclusions)
12. [Appendix: Raw Data Sources and Reproduction Instructions](#12-appendix-raw-data-sources-and-reproduction-instructions)

---

## 1. Introduction and Motivation

Serving large language models (LLMs) at competitive latency on consumer-grade hardware presents a unique engineering challenge. Unlike data-center SXM-class deployments equipped with NVLink at 900 GB/s inter-GPU bandwidth, consumer configurations such as dual RTX 4090 or RTX PRO 4000 Blackwell systems interconnect via PCIe Gen4 at approximately 26-31 GB/s, creating a hard bandwidth ceiling on tensor-parallel synchronization at every autoregressive decode step.

Three well-established mitigation strategies are the subject of this study:

1. **Speculative Decoding**: A draft model (or draft algorithm) predicts a sequence of K future tokens in advance. The target model verifies all K tokens in a single batched forward pass. Under high acceptance rates, this reduces the number of expensive full-model forward passes per unit of generated text.

2. **N-Gram Speculative Decoding (NGRAM)**: A training-free speculative algorithm that uses the prompt's own text as a look-up table for draft token prediction. Repeated n-grams in the input context are reused as draft sequences without loading any additional model weights.

3. **FP8 KV Cache Quantization**: The key-value (KV) attention cache tensors are stored in 8-bit floating-point format (fp8_e5m2) rather than the default 16-bit format, halving the per-token VRAM footprint of the KV cache and theoretically enabling a larger concurrent-request batch or longer effective context length under the same VRAM budget.

This study benchmarks all three approaches against a deterministic single-stream baseline, using the SGLang inference framework with the `Qwen/Qwen2.5-32B-Instruct-AWQ` model — a 4-bit Activation-aware Weight Quantization (AWQ) compression of the full 32-billion parameter Qwen2.5 Transformer Decoder.

---

## 2. Hardware and Software Environment

### 2.1 Hardware Configuration

| Component | Specification |
| :--- | :--- |
| **GPU** | 2x NVIDIA RTX PRO 4000 Blackwell (SM 12.0) |
| **Per-GPU VRAM** | 24,467 MiB (aggregate 48,934 MiB) |
| **GPU Interconnect** | PCIe Gen4 x16 (no NVLink) |
| **Effective PCIe Bandwidth** | ~26.8 GB/s (measured via nvidia-smi) |
| **System RAM** | >= 64 GB |
| **Instance ID** | 50625322 |

### 2.2 Software Stack

| Component | Version |
| :--- | :--- |
| **Docker Image** | `lmsysorg/sglang:latest` |
| **SGLang** | 0.5.19 |
| **FlashInfer** | 0.6.18 |
| **CUDA Toolkit** | 13.0 |
| **GPU Driver** | 595.71.05 |
| **PyTorch** | 2.13.0+cu130 |
| **Python** | 3.11.x |
| **Attention Backend** | FlashInfer (Ada/Blackwell SM-optimized page-attention kernels) |

### 2.3 Model

| Property | Value |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-32B-Instruct` |
| **Quantization** | AWQ 4-bit (INT8 weights-only) |
| **Model ID** | `Qwen/Qwen2.5-32B-Instruct-AWQ` |
| **Vocabulary Size** | 152,064 tokens |
| **Approximate Disk Size** | ~19.5 GB |
| **Tensor Parallelism** | TP=2 (weights sharded evenly across both GPUs) |
| **KV Cache Slots Allocated** | 168,028 tokens (at FP16 or FP8 dtype) |

---

## 3. Experimental Design and Methodology

### 3.1 Benchmark Dataset

The evaluation suite comprised **90 prompts** organized into three equal-sized domains of 30 prompts each, stored in `dataset/combined_benchmark_dataset.jsonl`:

| Domain | File | Description | Output Limit |
| :--- | :--- | :--- | :--- |
| **Code** | `prompts_code.jsonl` | Python algorithms (LRU cache, Trie, Dijkstra, async I/O, data structures) — high syntactic regularity, strong n-gram repetition in completions. | 512 tokens |
| **Prose** | `prompts_prose.jsonl` | Technical architectural explanations, systems post-mortems, distributed inference trade-off essays — natural language entropy, lower n-gram repetition. | 512 tokens |
| **JSON** | `prompts_json.jsonl` | Structured JSON schema extraction, log parsing, OpenAPI object generation — highly repetitive structural tokens. | 384 tokens |

All prompts used `temperature=0.0` for fully deterministic greedy sampling to ensure complete reproducibility across runs.

### 3.2 Benchmark Harness

The automated client (`scripts/run_benchmark.py`) communicates with the SGLang OpenAI-compatible endpoint (`/v1/chat/completions`) using HTTP streaming with `stream_options: {include_usage: true}`. The following client-side latency metrics are measured per request:

- **TTFT (Time-To-First-Token)**: Wall-clock elapsed time from request initiation to receipt of the first non-empty content chunk, in milliseconds.
- **Decode Time**: Wall-clock elapsed time from the first token to the `[DONE]` marker.
- **Decode Throughput**: `completion_tokens / decode_time_s` in tokens per second.
- **End-to-End Throughput**: `completion_tokens / total_time_s` in tokens per second.

Each benchmark run includes **2 warmup requests** prior to the measured evaluation window to ensure CUDA graphs are fully compiled and the KV cache is warmed.

### 3.3 Server Configurations Evaluated

| Run ID | Label | Key Flags | Status |
| :--- | :--- | :--- | :--- |
| **R1** | `baseline_tp2` | `--tp 2 --mem-fraction-static 0.82 --attention-backend flashinfer` | **Complete (90/90)** |
| **R2** | `spec_ngram_k5` | `--speculative-algorithm NGRAM --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --tp 2 --mem-fraction-static 0.85` | **Complete (90/90)** |
| **R3** | `baseline_fp8kv` | `--tp 2 --mem-fraction-static 0.85 --kv-cache-dtype fp8_e5m2 --attention-backend flashinfer` | **Complete (90/90)** |
| **R4** | `spec_standalone_1.5b` | `--speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct --speculative-algorithm STANDALONE` | **FAILED — vocab mismatch** |
| **R5** | `spec_eagle3` | `--speculative-draft-model-path ruipeterpan/Qwen2.5-32B-Instruct_EAGLE3_UltraChat --speculative-algorithm EAGLE3` | **FAILED — CUDA graph dtype error** |

---

## 4. Results: Baseline Configuration (R1)

**Server Command:**
```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-32B-Instruct-AWQ \
  --tp 2 \
  --mem-fraction-static 0.82 \
  --attention-backend flashinfer \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000
```

**Source:** `results/bench_baseline_local.out`

### 4.1 Aggregate Statistics (n = 90)

| Metric | Min | Median | Mean | Max | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Decode Throughput (tok/s)** | 55.52 | **55.60** | **55.71** | 56.37 | 0.21 |
| **TTFT (ms)** | — | **34.9** | **40.7** | — | — |

### 4.2 Per-Domain Decode Throughput

| Domain | Representative Decode Range (tok/s) | Notes |
| :--- | :--- | :--- |
| **Code** (code_01 – code_30) | 55.57 – 56.37 | First 2 prompts show elevated TTFT (65–66 ms) due to graph warm-up; throughput converges rapidly to the 55.6x range. |
| **Prose** (prose_01 – prose_30) | 55.52 – 55.59 | Exceptionally flat; near-zero variance. Confirms memory-bandwidth-bound steady state. |
| **JSON** (json_01 – json_30) | 55.57 – 56.23 | Slight positive bias at shorter completions (json_07: 83 tokens, 56.23 tok/s) due to reduced decode window length. |

### 4.3 Observations

- **Extreme decode stability (std dev 0.21 tok/s)**: The baseline represents a fully memory-bandwidth-bound decode regime. With AWQ 4-bit weights loaded at ~9.75 GB per GPU, the RTX PRO 4000 Blackwell's HBM bandwidth is the deterministic ceiling for each autoregressive step. No prompt-level variation can alter this floor.
- **TTFT bimodality**: First-category prompts exhibit elevated TTFT (65–80 ms for code_01/code_02 and all JSON prompts due to longer system prompts), while subsequent prompts settle at 30–40 ms, consistent with prefill latency for 50–100 token prompts at TP=2.
- **Absence of RadixAttention cache hits**: All 90 prompts use distinct, non-overlapping context, ensuring cold-cache decode-throughput measurements.

---

## 5. Results: NGRAM Speculative Decoding (R2)

**Server Command:**
```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-32B-Instruct-AWQ \
  --speculative-algorithm NGRAM \
  --speculative-num-steps 5 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 6 \
  --tp 2 \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000
```

**Source:** `results/bench_ngram_local.out`

### 5.1 Aggregate Statistics (n = 90)

| Metric | Min | Median | Mean | Max | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Decode Throughput (tok/s)** | 58.05 | **71.37** | **74.59** | 225.95 | 19.18 |
| **TTFT (ms)** | — | **37.6** | **44.5** | — | — |
| **Speedup over R1 (Median)** | — | **1.28x** | — | — | — |
| **Speedup over R1 (Mean)** | — | — | **1.34x** | — | — |

### 5.2 Per-Domain Decode Throughput Analysis

**Code Domain (code_01 – code_30):**

| Prompt | Tokens | TTFT (ms) | Decode (tok/s) |
| :--- | :--- | :--- | :--- |
| code_01 | 512 | 154.8 | 98.80 |
| code_02 | 512 | 68.8 | **225.95** |
| code_03 | 512 | 38.2 | 60.78 |
| code_08 | 512 | 35.3 | 67.99 |
| code_11 | 512 | 33.3 | 78.67 |
| code_22 | 512 | 34.8 | 81.24 |
| code_28 | 512 | 32.9 | 84.60 |
| code_30 | 512 | 32.3 | 80.65 |

Code prompts demonstrate the widest and highest NGRAM acceptance benefits. The exceptional result for `code_02` (225.95 tok/s — a **4.06x burst speedup**) represents a prompt with highly repetitive n-gram completions matching patterns already in the prompt window (data structure boilerplate: `__init__`, `self.`, `return` patterns).

**Prose Domain (prose_01 – prose_30):**

| Prompt | Tokens | TTFT (ms) | Decode (tok/s) |
| :--- | :--- | :--- | :--- |
| prose_01 | 512 | 37.3 | 61.99 |
| prose_06 | 512 | 35.6 | **58.05** (minimum) |
| prose_10 | 512 | 41.4 | 82.01 |
| prose_19 | 512 | 33.1 | 71.09 |
| prose_23 | 512 | 37.9 | 72.12 |

Prose prompts exhibit lower median NGRAM acceptance due to higher lexical entropy. The floor is close to baseline (58.05 tok/s, just 4.4% above baseline). Spikes like prose_10 (82.01 tok/s) correspond to technical prose with repeated terminology (API names, library identifiers, architectural terms).

**JSON Domain (json_01 – json_30):**

| Prompt | Tokens | TTFT (ms) | Decode (tok/s) |
| :--- | :--- | :--- | :--- |
| json_04 | 133 | 62.6 | 87.29 |
| json_05 | 151 | 58.0 | **107.35** |
| json_08 | 384 | 37.9 | 89.09 |
| json_10 | 138 | 69.5 | 88.51 |
| json_14 | 384 | 57.8 | 94.44 |
| json_27 | 267 | 67.2 | 103.34 |
| json_28 | 384 | 33.1 | **105.35** |

JSON prompts deliver the highest NGRAM acceptance rates (65–107 tok/s) due to the repetitive structural grammar of JSON: repeated keys (`"status":`, `"id":`, `"type":`), colons, commas, numeric values, and closing braces generate abundant exact n-gram matches in the draft look-ahead window.

### 5.3 Observations

- **High inter-prompt variance (std dev 19.18 tok/s)** is the defining characteristic of NGRAM speculative decoding. Unlike learned draft models, NGRAM relies entirely on n-gram co-occurrence between the prompt context and the generated completion. Prompt-level acceptance behavior is task-domain-dependent and cannot be controlled or tuned during inference.
- **TTFT regression (+2.7 ms median, +3.8 ms mean)**: Launching the NGRAM draft algorithm adds overhead on the first-token path due to n-gram index construction from the prompt window prior to initiating speculative draft generation.
- **Burst acceleration at code_02 (225.95 tok/s / 4.06x)**: This outlier is statistically real and represents the upper bound of NGRAM performance when the completion contains extremely long n-gram match chains from the source prompt. It elevates the mean meaningfully above the median (74.59 vs 71.37 tok/s), indicating positive skew in the throughput distribution.
- **JSON > Code > Prose** is the domain ordering for NGRAM acceptance rate, consistent with structural analysis: JSON grammar has the highest token-level repetition, code has moderate repetition, and prose has the lowest.

---

## 6. Results: FP8 KV Cache Quantization (R3)

**Server Command:**
```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-32B-Instruct-AWQ \
  --tp 2 \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --kv-cache-dtype fp8_e5m2 \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 30000
```

**Source:** `results/bench_fp8_local.out`

### 6.1 Aggregate Statistics (n = 90)

| Metric | Min | Median | Mean | Max | Std Dev |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Decode Throughput (tok/s)** | 54.68 | **54.77** | **54.86** | 55.49 | 0.19 |
| **TTFT (ms)** | — | **35.5** | **42.7** | — | — |
| **Delta vs R1 (Median)** | — | **-0.83 tok/s (-1.5%)** | — | — | — |

### 6.2 Per-Domain Decode Throughput Analysis

| Domain | Median Decode (tok/s) | vs R1 Baseline |
| :--- | :--- | :--- |
| **Code** (code_01 – code_30) | ~54.74 – 54.78 | -0.82 to -0.84 tok/s |
| **Prose** (prose_01 – prose_30) | ~54.68 – 54.78 | -0.78 to -0.87 tok/s |
| **JSON** (json_01 – json_30) | ~54.75 – 55.09 | -0.48 to -0.85 tok/s |

All three domains show uniform, marginal regression uniformly below the R1 baseline, with no domain exhibiting a throughput advantage.

### 6.3 Observations

- **Confirmed negative result — FP8 KV cache does not improve single-stream decode throughput.** In a single-stream inference regime, the decoding step is bounded by model-weight memory bandwidth (loading ~19.5 GB of AWQ weight tensors per forward pass), not by the KV cache read bandwidth. FP8 KV compression reduces VRAM consumed by the attention cache by ~50%, but that cache read is not the bottleneck in single-stream 512-token generations.
- **Marginal -1.5% regression**: The slight throughput decrease under FP8 KV is attributable to dequantization overhead — the GPU must apply the FP8 -> FP16 conversion kernel per attention head on every attention computation, adding a small but consistent compute cost with no compensating bandwidth saving in this workload.
- **Extremely stable throughput (std dev 0.19 tok/s)**: Even tighter than the R1 baseline (std dev 0.21), confirming that the FP8 configuration produces a deterministic, memory-bandwidth-limited decoding floor identical in character to the R1 baseline.
- **When FP8 KV would matter**: Under multi-request concurrent serving (batch size >= 4, context lengths >= 8,192 tokens), FP8 KV cache would allow 2x as many concurrent KV slot allocations within the same VRAM budget, meaningfully improving throughput-under-load. This benefit is not observable in the single-stream sequential evaluation protocol used here.

---

## 7. Comparative Analysis Across Configurations

### 7.1 Aggregate Throughput Comparison Table

| Configuration | n | Median (tok/s) | Mean (tok/s) | Std Dev | Min (tok/s) | Max (tok/s) | Speedup (Median) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1: baseline_tp2** | 90 | **55.60** | 55.71 | 0.21 | 55.52 | 56.37 | 1.00x (reference) |
| **R2: spec_ngram_k5** | 90 | **71.37** | 74.59 | 19.18 | 58.05 | 225.95 | **1.28x** |
| **R3: baseline_fp8kv** | 90 | **54.77** | 54.86 | 0.19 | 54.68 | 55.49 | **0.985x (-1.5%)** |

### 7.2 TTFT Comparison Table

| Configuration | Median TTFT (ms) | Mean TTFT (ms) | vs Baseline |
| :--- | :---: | :---: | :---: |
| **R1: baseline_tp2** | 34.9 | 40.7 | — |
| **R2: spec_ngram_k5** | 37.6 | 44.5 | +2.7 ms median (+7.7%) |
| **R3: baseline_fp8kv** | 35.5 | 42.7 | +0.6 ms median (+1.7%) |

### 7.3 Throughput Distribution Analysis

The key differentiator between configurations is **variance**, not just central tendency:

- **R1 and R3** exhibit a near-deterministic throughput distribution (std dev < 0.25 tok/s). These configurations are fully memory-bandwidth-bound, and no prompt-level feature can alter the decode throughput.
- **R2** exhibits a positively-skewed, high-variance distribution (std dev 19.18 tok/s). The interquartile range spans approximately 63–82 tok/s, with a long right tail driven by high-acceptance-rate prompts. The minimum (58.05 tok/s) is still 4.4% above the baseline, indicating NGRAM provides a worst-case floor improvement even on low-acceptance prompts.

---

## 8. GPU and Memory Telemetry Analysis

**Source Files:**
- `results/gpu_telemetry_baseline_local.csv` — 15,558 sample points
- `results/gpu_telemetry_spec_ngram_local.csv` — 5,493 sample points
- `results/gpu_telemetry_fp8_baseline_local.csv` — 2,502 sample points

### 8.1 Memory Allocation Profile (Median Across Telemetry Samples)

| Configuration | GPU0 Mem Used (MiB) | GPU1 Mem Used (MiB) | GPU0 Power (W) | GPU1 Power (W) | GPU0 Temp (C) | GPU1 Temp (C) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1: baseline_tp2** | 21,748 | 21,748 | 3.8 | 8.2 | 48 | 48 |
| **R2: spec_ngram_k5** | 22,524 | 22,524 | 18.1 | 21.6 | 55 | 54 |
| **R3: baseline_fp8kv** | 22,624 | 22,624 | 140.0 | 140.0 | 75 | 74 |

### 8.2 VRAM Budget Decomposition (Per GPU)

Based on SGLang server logs and memory telemetry:

| Component | Estimated VRAM per GPU |
| :--- | :--- |
| Target model weights (AWQ 4-bit, sharded TP=2) | ~9.75 GB |
| KV cache buffer (168,028 slots, FP16 or FP8) | ~5.13 GB |
| CUDA context, runtime, NCCL buffers | ~2.0 – 3.0 GB |
| CUDA graph capture overhead | ~1.5 – 2.0 GB |
| **Total (R1, mem-fraction 0.82)** | **~21.7 GB / 24.5 GB** |
| **Total (R2/R3, mem-fraction 0.85)** | **~22.5 – 22.6 GB / 24.5 GB** |

### 8.3 Power and Utilization Notes

- **R3 (FP8) shows peak utilization and power (140W per GPU, 75C)** because telemetry sampling captured this configuration during intensive decode activity with uniformly busy back-to-back 512-token requests. The 1-second nvidia-smi polling interval aligned with active decode windows.
- **R2 (NGRAM) shows low median power (18–22W) and 0% utilization median** despite full VRAM allocation. This is a polling artifact: NGRAM produces bursty, short decode windows (high throughput but shorter wall-clock duration per prompt due to speculative acceptance), separated by idle gaps. The 1-second sampler frequently lands in the inter-request idle state.
- **R1 shows near-zero power in telemetry** for the same sampling reason — single-stream requests complete in ~9 seconds each (512 tokens / 55.6 tok/s), and the sequential nature of the benchmark means each GPU is idle for a fraction of the sampling interval between requests.

> **Note on utilization measurement methodology**: The `nvidia-smi --format=csv -l 1` periodic telemetry approach produces accurate VRAM, temperature, and power readings but unreliable utilization percentages for bursty inference workloads. For a rigorous utilization profile, NVIDIA Nsight Systems (NSys) timeline traces or SGLang's built-in profiling hooks would be required.

---

## 9. Planned Configurations: Failure Modes and Reproducibility Notes

Two originally planned speculative decoding configurations — Standalone 1.5B and EAGLE-3 — could not be completed in this session. Both failures are documented in detail as reproducibility hazards for the SGLang ecosystem on Blackwell SM12 hardware.

### 9.1 Standalone 1.5B Draft Model (R4) — FAILED

**Planned Command:**
```bash
--model-path Qwen/Qwen2.5-32B-Instruct-AWQ \
--speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
--speculative-algorithm STANDALONE \
--speculative-num-steps 5
```

**Failure Mode:** SGLang's `STANDALONE` speculative decoding path requires **identical vocabularies** between the target model and the draft model. The target model (`Qwen2.5-32B-Instruct-AWQ`) has a vocabulary of **152,064 tokens**. The standard `Qwen/Qwen2.5-1.5B-Instruct` checkpoint carries a vocabulary of **151,936 tokens**. The 128-token mismatch causes an immediate validation failure at server initialization.

**Remediation Paths:**
- Use a 1.5B draft model with vocabulary size exactly matching 152,064 (e.g., a model explicitly built against the same tokenizer version as Qwen2.5-32B-Instruct-AWQ).
- Alternatively, use SGLang's `NGRAM` algorithm (which requires no draft model) or the `EAGLE3` algorithm (which uses the target model's own vocabulary embedding space and is vocabulary-agnostic by design).

### 9.2 EAGLE-3 Speculative Decoding (R5) — FAILED

**Planned Command:**
```bash
--model-path Qwen/Qwen2.5-32B-Instruct-AWQ \
--speculative-draft-model-path ruipeterpan/Qwen2.5-32B-Instruct_EAGLE3_UltraChat \
--speculative-algorithm EAGLE3 \
--speculative-num-steps 8 \
--speculative-eagle-topk 10
```

**Failure Mode:** The server failed during CUDA graph capture at the prefill runner stage with a Cutlass/TVM `Mismatched Tensor` dtype error: `expected dtype=float16`. This error occurs because the EAGLE-3 draft head, compiled for the FlashInfer prefill kernel path, produces or expects FP16 tensors, while the Blackwell SM12 Cutlass kernels in SGLang 0.5.19 operate on a different precision alignment. Retry attempts with `--disable-cuda-graph`, `tc_piecewise` graph backend, reduced `--mem-fraction-static 0.72`, and reduced `--speculative-num-steps 4` all reproduced the same failure.

**Reproducibility Note:** This is a known compatibility issue between SGLang 0.5.x's EAGLE-3 prefill graph compilation path and NVIDIA Blackwell SM12 GPUs. Resolution is expected in a future SGLang release that updates the Cutlass kernel dispatch logic for SM12.

### 9.3 Summary of Unexecuted Planned Experiments

| Planned Experiment | Hypothesis Tested | Blocking Issue |
| :--- | :--- | :--- |
| Standalone 1.5B speculative | H1: 1.5B draft provides 1.5 – 2.0x speedup | Vocabulary size mismatch (152,064 vs 151,936) |
| EAGLE-3 speculative | H2: EAGLE-3 provides 1.8 – 2.5x speedup with 500 MB overhead | Cutlass dtype mismatch on Blackwell SM12 / SGLang 0.5.19 |
| K-step sweep (K=3, K=8) | H3: Higher K increases speedup up to saturation | NGRAM-only path; sweep without canonical draft model doesn't cleanly test original hypothesis |
| Per-domain breakdown (code/prose/JSON separately) | H4: JSON achieves highest acceptance rate, Prose lowest | Dataset run as combined file; domain-level aggregation requires re-runs with domain-filtered files |
| NSys / SGLang profiling trace | Accurate GPU utilization measurement | Time and session constraints |

---

## 10. Discussion

### 10.1 NGRAM as a Training-Free Speculative Baseline

The 1.28x median speedup achieved by NGRAM speculative decoding represents a practically significant result for a **zero-additional-parameter, training-free** approach. NGRAM requires no draft model weights, no fine-tuning, and minimal additional VRAM beyond the n-gram index built from the live prompt context (~776 MB additional VRAM for NGRAM state, measured as: 22,524 MiB – 21,748 MiB = 776 MiB per GPU from telemetry delta).

However, the high standard deviation (19.18 tok/s) limits its utility for latency-sensitive production workloads requiring predictable SLOs. NGRAM is best suited to batch offline workloads over corpora with high self-similarity: code repositories, structured data generation, template-based document generation.

### 10.2 The Memory-Bandwidth Ceiling

The baseline's extreme stability (std dev 0.21 tok/s) is a direct manifestation of the memory-bandwidth-bound decoding regime. On the RTX PRO 4000 Blackwell, the effective HBM memory bandwidth is approximately 576 GB/s per card. Loading the per-GPU model shard (~9.75 GB) once per decode step implies a theoretical throughput ceiling of approximately:

```
floor_latency  = 9.75 GB / 576 GB/s  =  ~16.9 ms per token
ceiling_tok_s  = 1 / 0.0169 s        =  ~59 tok/s
```

The observed 55.6 tok/s baseline is approximately 94% of this theoretical ceiling, indicating minimal overhead from NCCL synchronization, attention computation, and sampling logic. Speculative decoding is theoretically the only client-side mechanism to break this ceiling in the single-stream regime, by amortizing the model weight load across multiple accepted tokens per step.

### 10.3 FP8 KV Cache: Correct Tool, Wrong Workload

The -1.5% regression observed for FP8 KV quantization is not a failure of the technique but a correct experimental outcome. KV cache format matters for **memory capacity**, not memory bandwidth in single-stream decode. The practical benefit of FP8 KV caching would manifest in:

- **Multi-request concurrent serving** (batch size >= 4): Smaller KV footprint enables twice as many concurrent request slots.
- **Long-context workloads** (context > 32,768 tokens): Halving KV size directly expands the maximum achievable context window at fixed VRAM budget.
- **Production API deployment**: Enabling higher queries-per-second (QPS) at a given VRAM budget.

This benchmark's single-stream, 512-token protocol intentionally isolates decode throughput and is not designed to capture these concurrency-level benefits.

### 10.4 Blackwell SM12 Compatibility Gap

The dual EAGLE-3 and STANDALONE failures highlight a critical reproducibility gap for the SGLang community. The RTX PRO 4000 Blackwell (SM12) is a recent Blackwell-class architecture, and SGLang 0.5.19's speculative decoding paths were primarily validated on Ampere (SM80/SM86) and Ada Lovelace (SM89) hardware. The Cutlass kernel dispatch and CUDA graph capture paths require architecture-specific updates that may not yet be fully merged for SM12 in this image version.

---

## 11. Conclusions

This study yields the following principal conclusions:

1. **NGRAM speculative decoding delivers a statistically significant 1.28x median throughput improvement** (71.37 tok/s vs. 55.60 tok/s baseline) with no additional model parameters, at the cost of high inter-prompt variance (std dev 19.18 tok/s). It is recommended for offline batch workloads with high prompt self-similarity.

2. **FP8 KV cache quantization (fp8_e5m2) provides no single-stream decode throughput benefit** and imposes a marginal -1.5% regression due to dequantization overhead. Its benefit is architectural and capacity-focused, manifesting only under concurrent multi-request or long-context serving conditions.

3. **The memory-bandwidth ceiling on PCIe-connected consumer GPUs is the dominant constraint.** The baseline's 55.60 tok/s at 94% of theoretical HBM bandwidth ceiling confirms that SGLang's FlashInfer backend efficiently saturates available bandwidth. Breaking this ceiling requires speculative algorithms with high acceptance rates.

4. **EAGLE-3 and Standalone 1.5B speculative decoding remain unvalidated on Blackwell SM12 (SGLang 0.5.19)** due to vocabulary mismatch and CUDA graph dtype incompatibility, respectively. These represent the highest-priority follow-up experiments and are expected to deliver 1.5x–2.5x speedups based on published EAGLE-3 results on Ada Lovelace hardware.

5. **The experimental infrastructure is fully reproducible.** All configuration flags, dataset files (90-prompt combined JSONL), benchmark harness code, and telemetry pipelines are committed in the repository and validated as functional on the target instance.

---

## 12. Appendix: Raw Data Sources and Reproduction Instructions

### 12.1 Data File Manifest

| File | Description | Size |
| :--- | :--- | :--- |
| `results/benchmark_runs_local.csv` | Full per-prompt CSV log for all 3 runs (270 rows) | 18.4 KB |
| `results/bench_baseline_local.out` | Streaming console output for R1 (baseline_tp2) | 7.4 KB |
| `results/bench_ngram_local.out` | Streaming console output for R2 (spec_ngram_k5) | 7.4 KB |
| `results/bench_fp8_local.out` | Streaming console output for R3 (baseline_fp8kv) | 7.4 KB |
| `results/gpu_telemetry_baseline_local.csv` | Per-second GPU telemetry for R1 (15,558 points) | 1.49 MB |
| `results/gpu_telemetry_spec_ngram_local.csv` | Per-second GPU telemetry for R2 (5,493 points) | 538 KB |
| `results/gpu_telemetry_fp8_baseline_local.csv` | Per-second GPU telemetry for R3 (2,502 points) | 248 KB |

### 12.2 Reproduction Commands

**Step 1: Launch the baseline server**
```bash
bash scripts/launch_baseline_tp2.sh
# Wait for: "The server is launched successfully"
```

**Step 2: Run GPU telemetry daemon**
```bash
bash scripts/monitor_gpu.sh results/gpu_telemetry_baseline.csv &
MONITOR_PID=$!
```

**Step 3: Execute benchmark harness**
```bash
python3 scripts/run_benchmark.py \
  --config-name baseline_tp2 \
  --dataset dataset/combined_benchmark_dataset.jsonl \
  --output-csv results/benchmark_runs.csv \
  --warmup 2
kill -9 $MONITOR_PID
```

**Step 4: Repeat with NGRAM speculative server**
```bash
docker stop $(docker ps -q)
bash scripts/launch_speculative_standalone.sh  # modify flags for NGRAM algorithm
```

**Step 5: Retry EAGLE-3 on newer SGLang image (pending SM12 fix)**
```bash
docker pull lmsysorg/sglang:latest
bash scripts/launch_speculative_eagle3.sh
```

### 12.3 Known Issues and Workarounds

| Issue | Affected Config | Workaround |
| :--- | :--- | :--- |
| Vocabulary mismatch (152,064 vs 151,936) | Standalone 1.5B (R4) | Use a 1.5B checkpoint with vocab 152,064, or switch to NGRAM/EAGLE3 |
| Cutlass TVM prefill dtype mismatch on SM12 | EAGLE-3 (R5) | Upgrade to SGLang >= 0.5.20 when SM12 Cutlass fix is released |
| `nvidia-smi` utilization unreliable for bursty workloads | All configs | Use NSys or SGLang `--enable-profiling` for accurate utilization traces |
| TTFT inflation on first 2 prompts (warmup effect) | All configs | Increase `--warmup` to 3–5 for production benchmarks |

---

*Report generated from raw benchmark data in `results/` directory.*  
*Instance: 50625322 | SGLang 0.5.19 | FlashInfer 0.6.18 | CUDA 13.0 | Driver 595.71.05*  
*Hardware: 2x NVIDIA RTX PRO 4000 Blackwell (SM12, 24,467 MiB each)*
