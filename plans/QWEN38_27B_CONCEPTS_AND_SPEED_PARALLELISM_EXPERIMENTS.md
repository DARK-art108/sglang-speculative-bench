# Qwen3.8-27B on SGLang: Concepts, GPU Selection & Speed/Parallelism Experiments

Date: 2026-09-15. Sources: SGLang cookbook page `cookbook/autoregressive/Qwen/Qwen3.8-27B`
(selected cell `hw=h200 · variant=default · quant=fp8 · nodes=single · spec=none ·
tier=low-latency · ssmDtype=float32`), SGLang advanced-features docs
(`pd_disaggregation`, `llms.txt` index), and live Vast.ai inventory searched the same day.

> Audience: this file starts "baby-simple" (stories + numbers) and ends with
> copy-paste experiment commands. Estimates are marked `[ESTIMATE]` wherever they
> are derived by arithmetic rather than measured on the hardware.

---

## Part 1 — Concepts, baby-simple, with examples

### 1. Model weights vs KV cache (the two things eating VRAM)

The model is a **writer**.

- **Weights** = the writer's *brain* (everything learned in school). Fixed size,
  never changes. Qwen3.8-27B-FP8 ≈ **~27–28 GB**.
- **KV cache** = the writer's *scratchpad* (notes on THIS conversation).
  Grows with every token.

Example: load `Qwen/Qwen3.8-27B-FP8` → ~27 GB gone immediately. One user sends a
5,000-token prompt with `--kv-cache-dtype fp8_e4m3` (1 byte per element) →
KV scratchpad ≈ 5,000 × 32 KB ≈ **160 MB**. Two users → ~320 MB. Scratchpad is
torn up when the conversation ends.

### 2. The VRAM suitcase

A GPU is a **suitcase**. Packing order:

1. Suit (weights) — ~27 GB for this recipe.
2. Scratchpads (KV cache + GDN/Mamba state pool) — grows with traffic.
3. Shoes (CUDA graphs, NCCL, activations, draft heads) — ~3–5 GB.

Example: RTX 5090 = 32 GB suitcase. 27 GB weights + ~2 GB shoes = 29 GB, leaving
~3 GB for scratchpads. One request's GDN state pool alone is ~770 MB at float32
(S=5 slots × 154 MB/slot), so concurrency dies fast. That is why the cookbook
says: *"on this 32 GB card the GDN state pool, not KV, is what runs out first"*
— and why **FP8 has no verified 5090 cell** (only NVFP4 variants fit).

### 3. FP8 KV cache = shorthand notes

The writer normally writes scratchpad notes in full sentences (BF16 = 2 bytes
per number). FP8 = **shorthand, 1 byte per number**. Same notes, half the paper.

Verified on our own hardware (Qwen2.5-32B R3 run): FP8 KV gave **no
single-stream decode gain** (55.6 → 54.8 tok/s — noise). Shorthand saves
*paper*, not *pen speed*. FP8 KV is a **concurrency lever, not a speed lever**.

### 4. Prefill vs decode (reading vs writing)

- **Prefill** = the model *reads* your whole prompt at once. Parallel, fast per token.
- **Decode** = the model *writes* the answer **one token at a time**. Strictly
  serial — each word depends on all previous ones.

Our R1 telemetry: ~9k-token prompts dominated total time (TTFT region), then
~55 tokens decoded at ~55 tok/s (~1 s). Reading the question takes seconds;
writing 55 words takes ~1 second.

### 5. TTFT vs tok/s (two different clocks)

- **TTFT** (time to first token) = spinner time. Dominated by **prefill**.
- **tok/s** (decode throughput) = streaming speed after the first token.

Low-latency serving optimises both; high-throughput serving trades a little
TTFT for many more tokens/second in aggregate.

### 6. Speculative decoding = guess-and-check typist

The main writer is *slow but always right*. Hire a **fast intern who guesses**
the next few words. The writer verifies all guesses **in one parallel pass**
(checking 4 tokens costs ≈ producing 1). Right guesses = free tokens. Wrong
guess = discard from that point, writer emits one token herself.

**Acceptance rate** = % of guessed tokens accepted. ~70% acceptance on 4-token
drafts ≈ ~2.4× speedup.

Our R2 proof: NGRAM drafts on Qwen2.5-32B gave **55.6 → 71.4 tok/s median
(1.28×)**, but sd = 19 tok/s — code-like text guesses well (bursts to 226
tok/s), creative prose does not (58 tok/s). Intern quality is everything.

The interns, cheapest-first:

| Intern | How it guesses | SGLang flags |
|---|---|---|
| **MTP / EAGLE (in-checkpoint)** | tiny head attached to the big model; **already inside the Qwen3.8-27B checkpoint — no extra download** | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` |
| NGRAM | copies phrases seen before | what our R2 used; free but content-dependent |
| DSPARK | separate small draft model | `--speculative-algorithm DSPARK --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark` |
| DFlash2 | separate small draft model, 8 draft tokens | `--speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8` (H200 status: in-progress) |

### 7. PD disaggregation = separate reading room and writing desk

One café reads orders and writes replies at the same table → a huge incoming
order freezes everyone's replies (**prefill interruption**). Split the café:

- **Prefill GPU**: only reads prompts, builds the KV scratchpad.
- **Decode GPU**: only writes tokens from shipped scratchpads.
- **Router** (`sgl-router`): carries scratchpads prefill → decode.

```
User → [router :8000] → [prefill :30000] ──KV (≈1 GB per 32k-req)──→ [decode :30100] → User
```

Pays off at **high concurrency** (user #12's prefill no longer stalls user #7's
decode). At single-stream low-latency it only adds a transfer hop. KV must
travel over **NVLink/PCIe inside one box** — across the public internet
(~3 Gbps on Vast) 1 GB of KV = ~3 s of pure shipping. **Never split P/D across
rentals for latency work.**

### 8. Chunked prefill = reading in chapters

A 32k-token prompt arrives. Instead of freezing all decodes to read it in one
gulp, read it in **chunks** (`--chunked-prefill-size 32768` on H200;
`2048` on RTX PRO 6000 class) and interleave other users' decodes between
chunks. Nobody's stream stalls longer than one chunk.

### 9. Radix / prefix caching = don't re-read shared pages

Fifty users share the same 2,000-token system prompt. Reading it 50 times is
waste — **radix cache** keeps the scratchpad for shared prefixes; users 2–50
skip re-reading. TTFT on repeated prefixes: ~10× faster.

The Mamba twist: this hybrid model's GDN **state** is a fixed-size blob per
slot (~154 MB at fp32), not growable KV. `--mamba-radix-cache-strategy`
decides how many spare state blobs to keep: `extra_buffer` (S=5, low-latency),
`extra_buffer_lazy` (S=4, throughput), `no_buffer` (S=3).

### 10. HiCache = the overflow shelf

Scratchpads fill the desk (VRAM) → move *closed* conversations' notes to a
**shelf** (host RAM — ~258 GB on the H200 boxes found). Returning users reload
from the shelf: slower than desk, far faster than re-reading. Flags:
`--enable-hierarchical-cache --hicache-ratio 2`. A 100k-token conversation that
cannot fit VRAM survives on the shelf.

### 11. GDN state / SSM dtype = the writer's memory format

This model is **hybrid**: some layers use attention (per-token notes), some use
a **Mamba/GDN rolling memory** — one whiteboard per conversation, constantly
rewritten, constant size (cheap for long chats). Whiteboard in fine marker
(**float32**, 154 MB/slot) or thick marker (**bfloat16**, ~77 MB/slot).
Slightly sloppier, **2× more conversations fit**. You selected float32 = max
accuracy; bf16 is the throughput lever.

Memory math (from the cookbook's own calculator, fp32, no spec):

- `stateBytesPerSlot = 48 × (48×128×128×4 + 10240×3×2)` ≈ **154 MB**
- `kvBytesPerToken (fp8) = 16×4×256×2×1` = **32 KB** (64 KB at bf16)
- Low-latency slots S=5 → **~770 MB state + 164 MB KV per 5k-token request ≈ ~0.93 GB/req**
- H200 (141 GB, mem-fraction 0.85 → ~120 GB budget, ~85 GB after weights +
  overhead) ⇒ [ESTIMATE] **~80–90 concurrent 5k-token requests** before the
  pool, not the card, is the limit.

### 12. Concurrency and the two tiers

One fast bank teller (low-latency: `extra_buffer`, S=5) vs many registers
(high-throughput: `extra_buffer_lazy`, S=4). Our R1–R3 benchmark was
single-stream — tier choice and PD disagg had nothing to save. They start to
matter at `--max-concurrency ≥ 8–16`.

### Cheat sheet

| Concept | Baby version | Lever |
|---|---|---|
| Weights 27 GB | the brain | fixed cost |
| KV cache | scratchpad, grows per token | VRAM pressure |
| FP8 KV | shorthand notes | **concurrency** ↑ |
| GDN state bf16 | thick-marker whiteboard | **concurrency** ↑ |
| Radix cache | don't re-read shared pages | TTFT ↓ |
| HiCache | shelf for old notes | context length ↑ |
| Chunked prefill | read in chapters | fairness under load |
| Spec (MTP/EAGLE) | intern guesses, writer checks | **speed** ↑↑ (cheapest win) |
| PD disagg | reading room ≠ writing desk | **throughput** at high concurrency |
| Tier low-latency/high-throughput | 1 fast teller vs many registers | traffic-shape tuning |

---

## Part 2 — What the recipe demands (grounded in the cookbook)

Selected cell (`h200/default/fp8/single`, verified):

```bash
python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --host 0.0.0.0 --port 30000
```

Eligible hardware per the cookbook: **H200 (141 GB), RTX PRO 6000 (96 GB),
RTX 5090 (32 GB, NVFP4 quants only), DGX Spark (128 GB), GB300 (288 GB)** —
single GPU each. B200/H100 are **not** verified cells for this hybrid GDN
model.

---

## Part 3 — Vast.ai inventory (searched live 2026-09-15)

| GPU | VRAM | Cookbook status | Vast.ai offers | Best $/hr | Verdict |
|---|---|---|---|---|---|
| **H200** | 141 GB | ✅ verified (your exact cell) | **9** — 1x from **$3.29** (id `50416326`, rel 0.967); 1x $3.98 (`39605034/39605089`, rel 0.998–0.999) | **$3.29** | ✅ exact match |
| 2x H200 | 282 GB | ✅ + PD-capable, **NVLink present** (`bw_nvlink ≈ 478 GB/s` on all three 2x offers) | 3 — **$7.52** (`48539237`, rel 0.996), $7.95 (`40323390`, rel 0.999, 21 Gbps net), $10.00 | **$7.52** | ✅ for PD-disagg experiments |
| 4x H200 | 564 GB | ✅ scale-out | 2 — $14.32, $18.42 | $14.32 | ✅ only for DP/scale tests |
| RTX PRO 6000 | 96 GB | ✅ verified | **0** | — | ❌ out of stock |
| RTX 5090 | 32 GB | ⚠️ NVFP4-only (no FP8 cell fits) | **64**, from $0.35 | $0.35 | ❌ for FP8; ✅ only if you switch quant to NVFP4 |
| DGX Spark / GB300 / MI300X | 128/288/192 GB | ✅ | **0 each** | — | ❌ out of stock |
| B200 | 192 GB | ⚠️ **not a verified cell** for this model | 10, from $5.63 (rel 0.998) | $5.63 | ⚠️ works in principle, hybrid-GDN kernels unverified on SM100 |
| H100 / A100-80GB | 80 GB | ❌ not listed for this model | **0** | — | ❌ out of stock anyway |

**Best GPU, in one line: 1x H200** — it is the only in-stock card with a
verified FP8 cell for this model. Pick `39605034` ($3.98/hr, rel 0.999) for
reliability or `50416326` ($3.29/hr) to save ~₹60/hr. For anything involving
PD disaggregation, step up to **2x H200 `48539237` ($7.52/hr)** — real NVLink
between the pair, so KV ships at ~478 GB/s, not over the internet.

Cost sketch (≈ ₹88/USD): 1x H200 ≈ **₹290–350/hr**; 2x H200 ≈ **₹660/hr**;
RTX 5090 (NVFP4 route) ≈ **₹30/hr**. A 4-hour 1x-H200 session ≈ **₹1,200–1,400**.

---

## Part 4 — Hosting: from zero to serving (1x H200)

```bash
# 1. Rent (120 GB+ disk for weights + HF cache + results)
vastai create instance 39605034 \
  --image lmsysorg/sglang:latest \
  --disk 150 \
  --env '-e HF_TOKEN=<your-hf-token> -p 30000:30000' \
  --ssh --direct

# 2. On the box: env check + NVLink/CUDA sanity
nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.current,pcie.link.width.current \
  --format=csv
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python3 -c "import sglang; print(sglang.__version__)"

# 3. Serve the exact recipe cell
nohup python3 -m sglang.launch_server \
  --trust-remote-code --model-path Qwen/Qwen3.8-27B-FP8 \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
  --attention-backend flashinfer \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --mamba-ssm-dtype float32 \
  --mamba-radix-cache-strategy extra_buffer \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
  --enable-metrics --host 0.0.0.0 --port 30000 > server_h200_fp8.log 2>&1 &
until curl -sf http://localhost:30000/health >/dev/null; do sleep 10; done

# 4. Smoke test
curl http://localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.8-27B-FP8","messages":[{"role":"user","content":"Hello"}]}'
```

---

## Part 5 — Speed & parallelism experiment plan (the core of this file)

Goal: separate **speed** (tok/s at concurrency 1) from **parallelism** (how many
concurrent requests before TTFT/TPOT collapse). Run E1→E7 in order; each is
~15–40 min on 1x H200 except E6 (needs the 2x box).

Common harness (matches this repo's conventions):

```bash
# GPU telemetry in background for EVERY run
bash scripts/monitor_gpu.sh results/gpu_telemetry_<config>.csv &
# Load driver: cookbook's own bench template
python3 -m sglang.bench_serving \
  --backend sglang-oai --host localhost --port <PORT> \
  --model Qwen/Qwen3.8-27B-FP8 \
  --dataset-name random --random-input-len <ISL> --random-output-len <OSL> \
  --random-range-ratio 1 --num-prompts <N> --max-concurrency <C> \
  --request-rate inf --flush-cache --output-file results/bench_<config>.json
```

Record per run: total tok/s, median TTFT, median TPOT, max sustained C,
accept-length (spec runs, from `/metrics`: `sglang:spec_accept_length`),
GPU util/power/temp medians, $ cost.

| # | Experiment | Change vs E1 | What it proves | [ESTIMATE] expected on H200 |
|---|---|---|---|---|
| E1 | Baseline, C=1, ISL 5120/OSL 1024 | — | speed floor | ~110–140 tok/s |
| E2 | **Concurrency ramp C = 1,2,4,8,16,32,64** (N=C×4) | only `--max-concurrency` | the speed-vs-parallelism curve; find the knee where TPOT degrades | total tok/s climbs to ~8–15k; knee ≈ C 32–64 |
| E3 | KV dtype fp8_e4m3 vs bfloat16 at C=16/32 | `--kv-cache-dtype` | max-concurrency shift (~2× more reqs fit on fp8) | fp8 sustains ~2× the C of bf16 |
| E4 | SSM dtype float32 vs bfloat16 at C=16/32 | `--mamba-ssm-dtype` | state-pool headroom doubles (154→77 MB/slot) | bf16 ≈ +60–100% max C, tiny quality delta to verify on GSM8K |
| E5 | Spec MTP/EAGLE on vs off at C=1 and C=8 | add `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` | cheapest tok/s multiplier; accept-length from `/metrics` | +30–80% median tok/s at C=1 |
| E6 | **PD disagg intra-node** (2x H200 box) at C=16/32 | prefill GPU0 + decode GPU1 + router | whether splitting P/D beats collocated at load | wins only if E2 shows prefill-stall (TPOT p99) at high C |
| E7 | HiCache long-context (ISL 60k, C=2) | add `--enable-hierarchical-cache --hicache-ratio 2` | serving beyond VRAM without OOM | no OOM; TTFT within ~1.5× of E1-scaled |

E5 detail (worth it — free speed, in-checkpoint head):

```bash
python3 -m sglang.launch_server --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.85 --attention-backend flashinfer \
  --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
  --mamba-ssm-dtype float32 --mamba-radix-cache-strategy extra_buffer \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --enable-metrics --host 0.0.0.0 --port 30000
```

E6 detail (2x H200 `48539237`, NIXL backend = simplest single-node path, from
the SGLang PD docs):

```bash
pip install nixl  # transfer engine
# prefill on GPU 0
python3 -m sglang.launch_server --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 --attention-backend flashinfer \
  --disaggregation-mode prefill --disaggregation-transfer-backend nixl \
  --host 0.0.0.0 --port 30000 > server_prefill.log 2>&1 &
# decode on GPU 1
python3 -m sglang.launch_server --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.80 --attention-backend flashinfer \
  --disaggregation-mode decode --disaggregation-transfer-backend nixl \
  --base-gpu-id 1 --max-running-requests 128 \
  --host 0.0.0.0 --port 30100 > server_decode.log 2>&1 &
# router (bench against :8000)
python3 -m sglang_router.launch_router --pd-disaggregation \
  --prefill http://127.0.0.1:30000 --decode http://127.0.0.1:30100 \
  --host 0.0.0.0 --port 8000 > router.log 2>&1 &
```

> Mooncake variant: same shape with `--disaggregation-ib-device mlx5_0`
> instead of the nixl backend; add `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=INTRA_NODE_NVLINK`
> + `MC_INTRANODE_NVLINK=true` to force KV over NVLink. Only if NIXL
> misbehaves — NIXL/UCX is the shorter path on a Vast box.

Stop rule per run: if median TPOT at C is >2× the C=1 TPOT, that C is past the
knee — record it and move on. Do not burn hours past saturation.

---

## Part 6 — What we already proved (prior R1–R3, 2x RTX PRO 4000)

| Run | Config | Median tok/s | Lesson for this plan |
|---|---|---|---|
| R1 baseline_tp2 | Qwen2.5-32B AWQ, TP2 | 55.60 (sd 0.21) | single-stream floor; decode is bandwidth-bound |
| R2 spec NGRAM K=5 | + NGRAM drafts | 71.37 (1.28×, sd 19) | spec works but intern quality dominates variance |
| R3 fp8 KV | fp8_e5m2 KV | 54.77 (≈ baseline) | KV dtype = concurrency lever, not speed lever |

These three results are exactly why E2 (concurrency ramp) and E5 (in-checkpoint
MTP) are the highest-value experiments above: R1–R3 never tested C>1, and never
tested a neural draft head.

---

## Part 7 — Recommended session (≈ 4 h, ≈ ₹1,400 on 1x H200)

1. Boot 1x H200 (`39605034`), serve E1 cell, smoke test (30 min).
2. E2 ramp C=1→32 (90 min) + E3 fp8 vs bf16 at the knee (30 min).
3. E5 spec on/off at C=1 and C=8 (45 min).
4. E4 bf16 SSM at the knee (20 min) + GSM8K spot-check
   (`python3 -m sglang.test.run_eval --host http://localhost --port 30000
   --model Qwen/Qwen3.8-27B-FP8 --eval-name gsm8k --num-examples 200`).
5. `vastai stop instance <id>`; pull `results/bench_*.json` +
   `results/gpu_telemetry_*.csv`; destroy only when done analysing.
6. Book E6 (2x H200, ~₹660/hr) **only if** E2 shows TPOT-p99 pain at high C —
   otherwise PD is spend without a problem to solve.

## Links

- Recipe: https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
- PD disaggregation: https://docs.sglang.io/docs/advanced_features/pd_disaggregation.md
- HiCache: https://docs.sglang.io/docs/advanced_features/hicache.md
- Quantized KV: https://docs.sglang.io/docs/advanced_features/quantized_kv_cache.md
- Speculative decoding: https://docs.sglang.io/docs/advanced_features/speculative_decoding.md
- Bench serving: https://docs.sglang.io/docs/developer_guide/bench_serving.md
