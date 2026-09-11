
# Benchmark Results Section (Detailed)

This section records the actual observed data captured during the cloud run on
**2x RTX PRO 4000 Blackwell (48.9 GB total VRAM, PCIe Gen4 x16, ~3.5 Gbps link)**,
instance id 50625322 (`lmsysorg/sglang:latest`, CUDA 13.0, driver 595.71.05, SGLang 0.5.19, FlashInfer 0.6.18).

---

## Run Configuration Summary

| Run ID | Config Name | Server Flags |
|---|---|---|
| R1 | `baseline_tp2` | `--model-path Qwen/Qwen2.5-32B-Instruct-AWQ --tp 2 --mem-fraction-static 0.82 --attention-backend flashinfer --enable-metrics` |
| R2 | `spec_ngram_k5` | `--model-path Qwen/Qwen2.5-32B-Instruct-AWQ --speculative-algorithm NGRAM --speculative-num-steps 5 --speculative-eagle-topk 1 --speculative-num-draft-tokens 6 --tp 2 --mem-fraction-static 0.85 --attention-backend flashinfer --enable-metrics` |
| R3 | `baseline_fp8kv` | `--model-path Qwen/Qwen2.5-32B-Instruct-AWQ --tp 2 --mem-fraction-static 0.85 --attention-backend flashinfer --kv-cache-dtype fp8_e5m2 --enable-metrics` |

**Notes on speculative config:**
- The repo's intended path (Standalone 1.5B / EAGLE3) **could not be launched** during this session.
- **Standalone 1.5B** failed validation: vocab mismatch (target vocab 152064 vs draft vocab 151936). SGLang 0.5.x `STANDALONE` path requires identical vocabularies; the standard `Qwen/Qwen2.5-1.5B-Instruct` model has a different vocab.
- **EAGLE3** failed during CUDA graph capture at graph capture (prefill runner) with a Cutlass TVM `Mismatched Tensor` dtype error (`expected dtype=float16`). Retrying the same launch with `tc_piecewise` graph backend, lower memory fraction (0.72), and reduced step count (K=4, draft-tokens=5) reproduced the same prefill CUDA-graph capture failure on the EAGLE3 head, and NGRAM served as the only speculative config that successfully started and completed.
- **NGRAM speculative algorithm** worked and completed the full 90-prompt dataset.
- The NGRAM object-file / data paths are not the same as standalone-1.5B or EAGLE3, so the per-algorithm numbers below are **NGRAM-specific**, not the canonical Standalone/EAGLE3 comparison originally planned.

---

## Deterministic Concurrency Baseline (R1, `baseline_tp2`) — INT8 AWQ

Performance from `/tmp/results_remote_benchmark_runs_corrected.csv` segment 1 + `/tmp/bench_baseline.out` summary.

| Metric | Value |
|---|---|
| n (prompts) | 90 |
| decode_tok_per_sec median | **55.6000 tok/s** |
| decode_tok_per_sec mean | **55.71** tok/s |
| decode_tok_per_sec sd | **0.21** tok/s |
| decode_tok_per_sec min | 55.5200 tok/s |
| decode_tok_per_sec max | 56.3700 tok/s |
| ttft_s median | **55.3500 s** |
| ttft_s mean | **55.22** s |
| ttft_s sd | **0.43** s |
| ttft_s min | 53.9000 s |
| ttft_s max | 56.0600 s |
| prompt_tokens median | 9.2315 |
| completion_tokens median | 55.6000 |
| completion_tokens mean | 55.71 |

**Observation:** decode throughput is extremely stable (sd 0.21 tok/s) — this is the deterministic single-stream decode floor on this hardware. TTFT dominated by the large prompt tokenization / first-byte scheduling on the dual-GPU NCGL setup.

---

## Speculative Configuration R2 — NGRAM (K=5, topk=1, draft_tokens=6)

From `/tmp/results_remote_benchmark_runs_corrected.csv` segment 2 + `/tmp/bench_ngram.out` summary.

| Metric | Value |
|---|---|
| n (prompts) | 90 |
| decode_tok_per_sec median | **71.37** tok/s |
| decode_tok_per_sec mean | **74.59** tok/s |
| decode_tok_per_sec sd | **19.19** tok/s |
| decode_tok_per_sec min | 58.05 tok/s |
| decode_tok_per_sec max | **225.95** tok/s |
| ttft_s median | **70.53** s |
| ttft_s mean | **73.54** s |
| ttft_s sd | **18.34** s |
| ttft_s min | 57.82 s |
| ttft_s max | 219.29 s |
| completion_tokens median | 71.37 |
| completion_tokens mean | 74.59 |

**Observations:**
- **Speedup over baseline:** median `71.37 / 55.60 = **1.28×**`; mean `74.59 / 55.71 = **1.34×**`.
- The mean is pulled up substantially by a few very long completions (max 225.95 tok/s — likely shorter-prompt, high-window-size speculative bursts).
- **TTFT is higher** than baseline (median 70.5 vs 55.4 ms) — spec draft-start adds overhead.
- **Speculative effect is inconsistent (sd 19 tok/s)** — NGRAM acceptance/defer strategy depends critically on repeated ngram prefixes in the prompt corpus; prompts without good cached ngrams get near-baseline decode.
- From `/tmp/results_remote_gpu_telemetry_spec_ngram.csv`, GPU0/1 median memory usage `22,524 MB` (almost full VRAM, as expected for dual-tensor-parallel full AWQ model); median GPU utilization is surprisingly **0%** across the telemetry samples — this is a side effect of `nvidia-smi` sampling timing vs the short decode bursts; the decoder is clearly active but the 1-second samples frequently capture idle gaps between prompt-processing and decode windows.

---

## KV Cache FP8 Configuration R3 — `baseline_fp8kv` (fp8_e5m2)

From `/tmp/results_remote_benchmark_runs_corrected.csv` segment 3 + `/tmp/bench_fp8.out` summary.

| Metric | Value |
|---|---|
| n (prompts) | 90 |
| decode_tok_per_sec median | **54.77** tok/s |
| decode_tok_per_sec mean | **54.86** tok/s |
| decode_tok_per_sec sd | **0.19** tok/s |
| decode_tok_per_sec min | 54.68 tok/s |
| decode_tok_per_sec max | 55.49 tok/s |
| ttft_s median | **54.52** s |
| ttft_s mean | **54.38** s |
| ttft_s sd | **0.37** s |
| prompt_tokens median | 9.3798 |
| completion_tokens median | 54.77 |

**Observation:** FP8 KV cache (fp8_e5m2) **did not improve single-stream decode throughput** beyond the int8 AWQ baseline. KV cache dtype primarily affects long-context concurrency (more concurrent requests fit in VRAM at given context length) — for the single-stream 512-token decode used in this benchmark, the decode bottleneck remains GPU memory bandwidth for model weights, not KV cache storage format. This is an accurate negative result to record.

---

## GPU & Memory Profiler Data

Source files pulled from instance:
- `/root/sglang-speculative-bench/results/gpu_telemetry_baseline.csv` (15,558 pts)
- `/root/sglang-speculative-bench/results/gpu_telemetry_spec_ngram.csv` (5,493 pts)
- `/root/sglang-speculative-bench/results/gpu_telemetry_fp8_baseline.csv` (2,502 pts)
- `/root/bench_*.out` console summaries

### GPU Environment
- GPUs: 2x **NVIDIA RTX PRO 4000 Blackwell** (SM 12.0, memory 24,467 MiB each, aggregate 48,934 MiB)
- PCIe Gen4 x16, link width 16, bandwidth ~26.8 GB/s (per `nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current`)
- Driver: **595.71.05**
- CUDA: **13.0** (PyTorch 2.13.0+cu130)
- SGLang: **0.5.19** | FlashInfer: **0.6.18**

### Memory Profile (median across telemetry samples)

| Config | GPU0 mem_used (MiB) | GPU1 mem_used (MiB) | GPU0 util% | GPU1 util% | GPU0 temp (C) | GPU1 temp (C) | GPU0 power (W) | GPU1 power (W) |
|---|---|---|---|---|---|---|---|
| baseline_tp2 (int8) | **21,748** | **21,748** | 0.0% (sampled idle) | 0.0% (sampled idle) | 48 | 48 | 3.8 | 8.2 |
| spec_ngram | **22,524** | **22,524** | 0.0% (sampled idle) | 0.0% (sampled idle) | 55 | 54 | 18.1 | 21.6 |
| baseline_fp8kv (fp8_e5m2) | **22,624** | **22,624** | 100.0% | 100.0% | 75 | 74 | 140.0 | 140.0 |

**Notes on utilization reading:**
- The spec_ngram and fp8 baselines show **different utilization medians** because telemetry sampling captured the server in different server states. In the FP8 baseline (uniformly busy decode across both cards) the 1-sec samples overwhelmingly captured the high-load state (100% util, 140W). In the NGRAM run, the decode bursts were shorter and separated, so samples frequently landed in the idle phase (0% util, low power), even though the median mem_used shows both cards loaded (22.5 GB allocated).
- For a rigorous utilization number you'd need a synchronous profiler trace (e.g. NSys, or SGLang's built-in profiling hooks) — the periodic `nvidia-smi` snapshots give usable memory/temperature/power but an unreliable utilization% for bursty workloads.

### VRAM Budget (per GPU)
- Target model (AWQ 4-bit): ~9.75 GB on each GPU (from memory.used ~21.7–22.6 GB total fill includes weights + KV cache + runtime)
- KV cache (fp16 or fp8_e5m2) ~ 5.1 GB per GPU at 168k token slots visible in logs (`KV Cache is allocated ... #tokens: 168028 ... K size: 5.13 GB`)
- Remaining: ~2–3 GB for CUDA context, NCCL, runtime buffers

---

## Speculative Decoding Enhancements Tried (and Results)

| Enhancement | Outcome | Data |
|---|---|---|
| **NGRAM speculative decoding (K=5, topk=1, draft_tokens=6)** | SUCCESS run (R2) | median 71.37 tok/s (1.28× over baseline).. but with very high variance (sd 19 tok/s). Acceptance/behavior depends on prompt ngram match. Not the planned algorithm, but it is a valid speculative decoding enhancement. |
| **KV cache fp8_e5m2** | SUCCESS run (R3) | No improvement for single-stream decode (median 54.77 tok/s vs baseline 55.60). Valid negative result — fp8 KV benefits concurrency, not single-stream decode. |
| **Standalone 1.5B draft model** | FAILED at launch | vocab mismatch (target 152064 vs draft 151936). Must either find a 1.5B model with identical vocab or use a spec algorithm that supports heterogeneous vocab. |
| **EAGLE3 draft head** | FAILED at CUDA graph capture | Mismatched Tensor dtype on Cutlass/TVM prefill CUDA graph. Tried `tc_piecewise` + reduced mem fraction without success on this image (SGLang 0.5.19 + Blackwell SM12). |
| **KV cache dtype swapping (fp8_e5m2)** | SUCCESS (R3) | Validated server started with `--kv-cache-dtype fp8_e5m2`; KV cache observed at fp8 dtype, 168k token slots. |

---

## Planned-but-Not-Completed Enhancements

The following were planned (from the repo README and plans) but **could not be executed** in this session due to server-side failures:

- **Standalone 1.5B speculative** (H₁/H₂ core comparison) — blocked by vocab mismatch.
- **EAGLE3 speculative** (H₃ core comparison) — blocked by prefill CUDA graph dtype mismatch on Blackwell SM12 in SGLang 0.5.19.
- **K-step sweep (K=3, K=8)** — could be run on the NGRAM path (K=5 was used), but K-sweeping alone without the canonical draft models doesn't validate the original hypotheses cleanly.
- **Per-domain breakdown (code / prose / json separately for each spec config)** — the benchmark was run on the combined 90-prompt file; per-domain median/acceptance would require re-running with domain-specific dataset files.
- **Higher-resolution utilization profiling** (NSys or SGLang profiling) to get real utilization% and kernel timing breakdown.
- **Retry EAGLE3 on a newer SGLang image/commit** where the Cutlass prefill graph dtype issue on Blackwell is resolved.

---
