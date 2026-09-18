# TurboLLM artifacts (Vast.ai instance 51471731, 1x RTX 6000 Ada 48GB)

Session: 2026-09-18/19, image lmsysorg/sglang:latest + Node 22 + TurboLLM 1.13.5.
Instance destroyed after extraction; everything below was pulled first.

## Files
- `turbollm_full_log.txt` — SSH console capture: daemon log, /api/v1/status, versions, HF cache listing
- `turbollm_state.tgz` + extracted — TurboLLM state: config.json, SQLite db (+wal/shm), logs/, telemetry/, downloads/
- `SUMMARY.txt` — readable digest: manifest, daily usage, bench activity, engine log tail
- `engine_bench_extract.txt` — grep of perf lines from remote engine logs
- `logs/engine-61cedf30-*.log` — TurboLLM's alternate vLLM engine attempt (OOM, see below)
- `logs/engine-f9b0acc6-*.log` — active llama.cpp b9608 Vulkan engine (the one that served)

## What ran
- Engine: llama.cpp b9608 (Vulkan backend — TurboLLM auto-pick for this card)
- Model: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf` (13 GB GGUF, IQ3_S, MTP draft included) + BF16 mmproj
- Launch flags: `-c 8192 -ngl 65 --cache-type-k q8_0 --cache-type-v q8_0 --flash-attn on --spec-type draft-mtp --spec-draft-n-max 16`
- TurboLLM auto-benchmarks recorded for IQ3_S (f16 and q8_0 KV variants) in `telemetry/bench-rate-limit.json`
- Daily usage telemetry: 847 chat msgs, 15 code, 10 autotune, 5 link (2026-09-18)

## Failures captured (useful negatives)
- TurboLLM's vLLM alt-engine on BF16 Qwen3.8-27B: CUDA OOM at 47.3/47.36 GiB — 27B bf16 does NOT fit 48GB with default settings; only the IQ3_S GGUF served.
- The vulkan backend (not CUDA) was auto-provisioned for RTX 6000 Ada.
