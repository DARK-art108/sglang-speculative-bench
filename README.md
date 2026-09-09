<div align="center">

# `sglang-speculative-bench`
### Rigorous Speculative Decoding Benchmarks for Qwen2.5-32B on Dual Consumer GPUs without NVLink

[![Engine: SGLang](https://img.shields.io/badge/Engine-SGLang_v0.4+-blue.svg)](https://github.com/sgl-project/sglang)
[![Model: Qwen2.5-32B-AWQ](https://img.shields.io/badge/Target_Model-Qwen2.5--32B--Instruct--AWQ-purple.svg)](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct-AWQ)
[![Hardware: 2x RTX 4090](https://img.shields.io/badge/Hardware-2x_RTX_4090_(48GB)-green.svg)](#hardware-memory--interconnect-budget)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-orange.svg)](LICENSE)

</div>

---

## Abstract

Autoregressive transformer decoding is strictly bounded by GPU memory bandwidth and inter-device communication latency. When serving medium-large parameter models (e.g., 32B) on consumer hardware under Tensor Parallelism ($\text{TP}=2$), the absence of high-bandwidth NVLink ($900\text{ GB/s}$) forces all-reduce tensor synchronizations over commodity PCIe buses ($\sim 11.7\text{ to }31.5\text{ GB/s}$). This creates an interconnect bottleneck where inter-GPU synchronization latency dominates decode time.

**`sglang-speculative-bench`** provides an end-to-end, reproducible research testbed evaluating how speculative decoding mitigates this interconnect bottleneck. By proposing $K$ draft tokens and verifying them in a **single batched target forward pass**, speculative decoding amortizes $K \times L$ all-reduce operations into a single verification step. This repository contains the complete benchmarking harness, standardized 90-prompt multi-domain dataset, automated Prometheus metric scrapers, and GPU telemetry daemons for SGLang.

---

## Key Research Questions & Hypotheses

* **$H_1$ (Interconnect Amortization):** Speculative decoding delivers a higher *relative* speedup on PCIe-constrained consumer hardware ($1.6\times\text{ to }2.8\times$) than on high-bandwidth NVLink systems because it bypasses per-token PCIe AllReduce latency.
* **$H_2$ (Domain Entropy Invariance):** Speculative acceptance rate ($\alpha$) correlates inversely with output sequence entropy:
  $$\alpha_{\text{JSON}} \; (75\%\text{--}90\%) \;>\; \alpha_{\text{Code}} \; (70\%\text{--}85\%) \;>\; \alpha_{\text{Prose}} \; (55\%\text{--}65\%)$$
* **$H_3$ (Draft Architecture Trade-offs):** A lightweight tree-speculation head (**EAGLE-3**, $\sim 500\text{ MB}$) achieves higher effective acceptance and throughput than a full autoregressive small language model (**Standalone Qwen2.5-1.5B**, $\sim 3.1\text{ GB}$) while consuming $84\%$ less draft VRAM.

---

## Experimental Architecture Matrix

```
                      +------------------------------------------+
                      |         User / Benchmark Client          |
                      |          (scripts/run_benchmark.py)       |
                      +------------------------------------------+
                                    |               ^
                   Streaming HTTP   |               | Prometheus Scrape
               /v1/chat/completions |               | /metrics (Accept Rate)
                                    v               |
                      +------------------------------------------+
                      |          SGLang Server Engine            |
                      |       (RadixAttention + FlashInfer)      |
                      +------------------------------------------+
                                   /              \
         [Standalone 1.5B or EAGLE3]              [Target Verification]
                     v                                      v
       +----------------------------+         +----------------------------+
       |   Draft Proposal (K=5-8)   |         | Qwen2.5-32B-Instruct-AWQ   |
       |  Low latency, minimal sync |         |  1 Batched AllReduce Sync  |
       +----------------------------+         +----------------------------+
                     \                              /
                      +----------------------------+
                      |      PCIe Interconnect     |
                      |  (PCIe Gen3/Gen4 x16 bus)  |
                      +----------------------------+
```

| Configuration | Target Model | Draft Architecture | Speculative Algorithm | Draft Steps ($K$) | Target $\text{TP}$ |
|---|---|---|---|---|---|
| **Control (Baseline)** | `Qwen/Qwen2.5-32B-Instruct-AWQ` | *None* | `None` | `N/A` | 2 |
| **Speculative A** | `Qwen/Qwen2.5-32B-Instruct-AWQ` | `Qwen/Qwen2.5-1.5B-Instruct` | `STANDALONE` | $5$ | 2 |
| **Speculative B** | `Qwen/Qwen2.5-32B-Instruct-AWQ` | `ruipeterpan/..._EAGLE3_UltraChat` | `EAGLE3` | $8$ (top-$k=10$) | 2 |

---

## Hardware & Memory Budget

### Static VRAM Allocation (Per GPU, Dual RTX 4090 24GB, $\text{TP}=2$)

```
Total VRAM per Card: 24,564 MiB
+------------------------------------+-------------------------+--------------------+
| Target Weights (TP=2): ~9,750 MiB  | Draft: ~1,550 MiB (1.5B)| Static Buffer: ~8GB| Dynamic Headroom  |
+------------------------------------+-------------------------+--------------------+--------------------+
0                                    10GB                      12GB                 20.1GB (0.82)        24.5GB
```

* **Target Weights:** $\sim 19.5\text{ GB}$ AWQ total $\rightarrow \sim 9.75\text{ GB}$ per GPU rank.
* **Draft Weights:** $\sim 3.1\text{ GB}$ (Standalone 1.5B) or $\sim 0.5\text{ GB}$ (EAGLE-3).
* **Static Allocation Cap (`--mem-fraction-static 0.82`):** Leaves $\sim 4.4\text{ GB}$ dynamic buffer per GPU to absorb NCCL ring buffers, activation spikes, and draft verification tensors without triggering CUDA OOM.
* **Interconnect:** PCIe Gen4 x16 ($\sim 31.5\text{ GB/s}$) or PCIe Gen3 x16 ($\sim 11.7\text{ GB/s}$) without NVLink.

---

## Standardized Benchmark Dataset (`dataset/`)

The evaluation suite comprises **90 curated, distinct prompts** stratified across three predictable entropy profiles to prevent RadixAttention prefix caching confounders during throughput analysis:

1. **`dataset/prompts_code.jsonl` (30 prompts):** Algorithmic routines, asynchronous pipelines, and data structures. Medium entropy; expected accept rate **$70\%\text{ to }85\%$**.
2. **`dataset/prompts_json.jsonl` (30 prompts):** Schema-constrained extraction, API specifications, and database records. Low entropy; expected accept rate **$75\%\text{ to }90\%$**.
3. **`dataset/prompts_prose.jsonl` (30 prompts):** Technical post-mortems, systems analysis, and architectural trade-offs. High entropy; expected accept rate **$55\%\text{--}65\%$**.
4. **`dataset/combined_benchmark_dataset.jsonl` (90 prompts):** Full multi-domain evaluation suite ($42,240$ total generated tokens).

---

## Repository Structure

```
sglang-speculative-bench/
├── dataset/
│   ├── combined_benchmark_dataset.jsonl   # Unified 90-prompt test suite
│   ├── generate_dataset.py               # Deterministic prompt generator
│   ├── prompts_code.jsonl                # 30 code generation prompts
│   ├── prompts_json.jsonl                # 30 structured JSON extraction prompts
│   └── prompts_prose.jsonl               # 30 natural language prose prompts
├── plans/
│   ├── GPU_COST_AND_INSTANCE_SELECTION_GUIDE.md # Cost matrix & Vast.ai CLI guide
│   ├── README-advanced-2xRTX4090-refined.md    # Production reference recipe
│   └── SGLANG_BENCHMARKING_PLAN.md             # Experimental plan & verification gates
├── results/
│   └── .gitkeep                          # Output directory for telemetry & CSV runs
├── scripts/
│   ├── launch_baseline_tp2.sh            # Control baseline launcher (TP=2)
│   ├── launch_speculative_standalone.sh  # Standalone 1.5B speculative launcher
│   ├── launch_speculative_eagle3.sh      # EAGLE-3 tree speculation launcher
│   ├── monitor_gpu.sh                    # 1Hz background GPU telemetry logger
│   └── run_benchmark.py                  # Client with Prometheus metric scraper
└── .gitignore                            # Environment and scratch artifact protection
```

---

## Quickstart & Replication

### 1. Launch SGLang Server
Select one configuration to evaluate:

```bash
# Option A: Baseline Non-Speculative (Control)
bash scripts/launch_baseline_tp2.sh

# Option B: Standalone Speculative (Qwen2.5-1.5B Draft, K=5)
bash scripts/launch_speculative_standalone.sh

# Option C: EAGLE-3 Speculative (Lightweight Head, K=8)
bash scripts/launch_speculative_eagle3.sh
```

Wait until server health checks pass:
```bash
until curl -s http://localhost:30000/health >/dev/null; do sleep 2; done
```

### 2. Execute Benchmark Harness
Start hardware telemetry logging and run the streaming evaluation client:

```bash
# Start background GPU power/VRAM logger
bash scripts/monitor_gpu.sh results/gpu_telemetry_spec_standalone.csv &
MONITOR_PID=$!

# Run evaluation client
python3 scripts/run_benchmark.py \
  --endpoint http://localhost:30000 \
  --dataset dataset/combined_benchmark_dataset.jsonl \
  --config-name spec_standalone_1.5b \
  --output-csv results/benchmark_runs.csv

# Terminate telemetry logger
kill -9 $MONITOR_PID
```

---

## Metrics & Observability

The client (`scripts/run_benchmark.py`) instruments both client-observed latency and engine-level Prometheus counters:

* **Time-To-First-Token ($\text{TTFT}$):** $\text{TTFT} = t_{\text{first token}} - t_{\text{start}}$ ($\text{ms}$).
* **Time-Per-Output-Token ($\text{TPOT}$):** $\text{TPOT} = (t_{\text{end}} - t_{\text{first token}}) / N_{\text{tokens}}$ ($\text{ms/tok}$).
* **Decode Throughput:** $\text{Throughput} = N_{\text{completion tokens}} / (t_{\text{end}} - t_{\text{first token}})$ ($\text{tok/s}$).
* **Speculative Acceptance Rate ($\alpha$):** Scraped directly from SGLang endpoint `/metrics`:
  $$\alpha = \frac{\Delta \text{sglang:spec\_accepted\_tokens}}{\Delta \text{sglang:spec\_drafted\_tokens}}$$
* **Average Accepted Length:** Mean tokens accepted per target verification pass (`sglang:spec_accept_length`).

---

## Reproducibility & Cost Optimization Guide

For rental commands, cost matrices across GPU providers (Vast.ai, RunPod), and cloud instance provisioning scripts, refer to:
👉 **[`plans/GPU_COST_AND_INSTANCE_SELECTION_GUIDE.md`](plans/GPU_COST_AND_INSTANCE_SELECTION_GUIDE.md)**

---

## Citation

If you use this benchmarking suite or recipe in your systems research, please cite:

```bibtex
@misc{sglang-speculative-bench-2026,
  author = {DARK-art108},
  title = {sglang-speculative-bench: Rigorous Speculative Decoding Benchmarks for Qwen2.5-32B on Dual Consumer GPUs},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/DARK-art108/sglang-speculative-bench}}
}
```
