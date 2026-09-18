# TurboLLM Autotune Benchmark Report

**Model:** `qwen3.8-27b gsq-rco|IQ3_S|12120016960` (Qwen3.8 27B, IQ3_S Quantization)
**Hardware:** 1x NVIDIA RTX 6000 Ada Generation (49,140 MB VRAM), 133 GB System RAM
**Date:** 2026-09-18
**Context Length:** 8,192 tokens

---

## Executive Summary

This report summarizes the autotuning runs for the IQ3_S quantized Qwen 27B model across two distinct KV Cache quantizations: **`q8_0`** and **`f16`**. The goal of the autotuning process was to determine the maximum number of GPU offloaded layers (`ngl`) that fit in VRAM while observing the resulting performance impact.

In both configurations, the autotuner successfully offloaded all **65 layers** to the GPU. 

**Key Findings:**
- Using `f16` (unquantized) KV cache yielded better overall performance (**26.3 tok/s** vs **23.9 tok/s**) with only a marginal increase in VRAM footprint (+240 MB) compared to `q8_0`.
- Prefill speeds and Time-To-First-Token (TTFT) were also notably superior when using the `f16` KV cache.
- The entire model fits comfortably within the 48GB VRAM of a single RTX 6000 Ada, utilizing only ~15.5 - 15.8 GB at full offload.

---

## Performance Comparison (Winner Configurations)

Both runs converged on full layer offloading (`ngl=65`). The table below compares the optimal configurations for both KV cache types.

| Metric | KV Cache: `f16` (Winner) | KV Cache: `q8_0` | Difference |
| :--- | :--- | :--- | :--- |
| **Decode Throughput** | **26.29 tok/s** | 23.86 tok/s | `f16` is +10.2% faster |
| **Prefill Throughput** | **1299.15 tok/s** | 1227.96 tok/s | `f16` is +5.8% faster |
| **Time-To-First-Token (TTFT)** | **5060.25 ms** | 5353.59 ms | `f16` is 293ms faster |
| **VRAM Usage (Final)** | 15,835 MB | **15,595 MB** | `q8_0` uses 240 MB less |
| **Max Offloaded Layers** | 65 / 65 | 65 / 65 | Parity |

---

## Detailed Autotuning Trajectories

The autotuner incrementally probes layer offload boundaries (`ngl`) to measure memory scaling before settling on the maximum stable value for the final benchmark.

### Run 1: `f16` KV Cache (Started 19:11:27Z)

*Unquantized FP16 KV Cache.*

| Probe Step | VRAM Absolute (MB) | Status |
| :--- | :--- | :--- |
| `ngl = 32` | 9,260 | ok |
| `ngl = 49` | 12,657 | ok |
| `ngl = 57` | 14,063 | ok |
| `ngl = 61` | 14,826 | ok |
| `ngl = 63` | 15,167 | ok |
| `ngl = 64` | 15,370 | ok |
| `ngl = 65` (Full) | 15,555 | ok |
| **Final Benchmark** | **15,845** | **26.3 tok/s** |

### Run 2: `q8_0` KV Cache (Started 19:37:57Z)

*8-bit Quantized KV Cache.*

| Probe Step | VRAM Absolute (MB) | Status |
| :--- | :--- | :--- |
| `ngl = 32` | 9,172 | ok |
| `ngl = 49` | 12,509 | ok |
| `ngl = 57` | 13,821 | ok |
| `ngl = 61` | 14,571 | ok |
| `ngl = 63` | 14,920 | ok |
| `ngl = 64` | 15,128 | ok |
| `ngl = 65` (Full) | 15,315 | ok |
| **Final Benchmark** | **15,605** | **23.9 tok/s** |

---

## Analysis & Recommendations

1. **Memory Ceiling Not Reached**: The RTX 6000 Ada has ~48 GB of VRAM. At `ngl=65` and 8k context, this IQ3_S quantization of Qwen 27B only consumes ~15.8 GB. This leaves massive headroom (~32 GB).
2. **KV Quantization Penalty**: The `q8_0` KV cache introduces a measurable performance regression (~10% decode, ~6% prefill) due to the overhead of quantizing and dequantizing the cache on the fly. 
3. **Recommendation**: Given the abundant VRAM headroom on the RTX 6000 Ada, **do not use KV cache quantization (`q8_0`) for this model/hardware pair**. Stick with `f16` KV caching to maximize inference throughput. The 240 MB saved by `q8_0` is immaterial when 32 GB of VRAM remains unused.
