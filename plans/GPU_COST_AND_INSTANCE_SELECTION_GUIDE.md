# SGLang Qwen2.5-32B Instance Selection & Cost Optimization Guide

A practical guide for choosing, renting, and configuring cloud GPU instances across daily exploration, standard benchmarking, and heavy throughput testing.

---

## 1. Activity Tier Matrix & Cost Comparison

| Activity Tier | Workload Description | Target Hardware | Hourly Cost (USD) | Hourly Cost (INR) | 2–3 Hr Session Cost |
|---|---|---|---|---|---|
| **Tier 1: Daily Exploration & Prototyping** | Prompt drafting, quick speculative checks, single-category tests, debugging | **2x RTX 3090** (48GB) or **1x RTX A6000** (48GB) | **$0.35 – $0.45/hr** | **₹29 – ₹38/hr** | **$0.70 – $1.35** (~₹60 – ₹115) |
| **Tier 2: Standard Benchmark Recipe** | Full 90-prompt suite, K-step sweeps ($K=3,5,8$), FP8 KV cache, official paper/lab numbers | **2x RTX 4090** (48GB) | **$0.70 – $0.95/hr** | **₹58 – ₹79/hr** | **$1.40 – $2.85** (~₹120 – ₹240) |
| **Tier 3: Heavy Serving & Stress Tests** | High-concurrency serving (`sglang.bench_serving`), long context ($\ge 32\text{k}$), multi-tenant simulation | **2x H100 SXM** (160GB) or **1x H200** (141GB) | **$2.80 – $4.50/hr** | **₹235 – ₹380/hr** | **$5.60 – $13.50** (~₹470 – ₹1,140) |

---

## 2. Tier Deep Dive & Setup

### Tier 1: Low-Cost Daily Exploration ($0.35 – $0.45/hr)

* **Goal:** Maximum bang for buck. Perfect for iterating on scripts, modifying prompts, and testing acceptance rates without burning budget.
* **Option A: 2x RTX 3090 (24GB x 2 = 48GB total)**
  * Native `--tp 2` matching repository scripts.
  * Ampere architecture (SM 8.6) with solid Triton attention backend performance.
  * **Search CLI:**
    ```bash
    vastai search offers 'reliability > 0.97 disk_space >= 80 inet_down > 500 num_gpus=2 gpu_name=RTX_3090' -o 'dph' --limit 5
    ```
  * **Launch CLI:**
    ```bash
    vastai create instance <OFFER_ID> \
      --image lmsysorg/sglang:latest \
      --disk 80 \
      --ssh --direct
    ```

* **Option B: 1x RTX A6000 (48GB VRAM)**
  * 48GB VRAM on a **single card**.
  * Zero PCIe communication overhead, no multi-GPU synchronization, simplest server orchestration (`--tp 1`).
  * **Search CLI:**
    ```bash
    vastai search offers 'reliability > 0.97 disk_space >= 80 inet_down > 500 num_gpus=1 gpu_name=RTX_A6000' -o 'dph' --limit 5
    ```

---

### Tier 2: Standard Production Recipe ($0.70 – $0.95/hr)

* **Goal:** Reproduce the exact benchmark targets defined in `plans/README-advanced-2xRTX4090-refined.md`.
* **Hardware:** **2x NVIDIA GeForce RTX 4090 (24GB x 2 = 48GB total)**
* **Key Advantages:**
  * Ada Lovelace (SM 8.9) with dedicated FP8 tensor cores.
  * Full FlashInfer kernel acceleration (`--attention-backend flashinfer`).
  * Highest consumer decode speeds (~30 tok/s baseline, ~55–90 tok/s speculative).
* **Search CLI:**
  ```bash
  vastai search offers 'reliability > 0.97 disk_space >= 80 inet_down > 500 num_gpus=2 gpu_name=RTX_4090' -o 'dph' --limit 5
  ```
* **Launch CLI:**
  ```bash
  vastai create instance <OFFER_ID> \
    --image lmsysorg/sglang:latest \
    --disk 80 \
    --ssh --direct
  ```

---

### Tier 3: Heavy Serving & Concurrency Stress ($2.80 – $4.50/hr)

* **Goal:** Concurrency testing ($BS \ge 16$), 32k+ token contexts, and multi-user simulation.
* **Option A: 2x H100 SXM (160GB VRAM, NVLink ~900 GB/s)**
  * Direct drop-in with `--tp 2`.
  * Inter-GPU AllReduce latency drops to $<2\mu\text{s}$ over NVLink.
  * Baseline throughput reaches 140–160 tok/s; speculative reaches 220–260 tok/s.
* **Option B: 1x H200 (141GB HBM3e VRAM, 4.8 TB/s Memory Bandwidth)**
  * Fits full unquantized BF16 32B model plus massive KV cache on a single GPU.
  * Requires changing `--tp 2` to `--tp 1` in launch scripts.

---

## 3. SGLang Parameter Matrix by Tier

| Engine Parameter | Tier 1 (2x 3090) | Tier 1 (1x A6000) | Tier 2 (2x 4090) | Tier 3 (2x H100) | Tier 3 (1x H200) |
|---|---|---|---|---|---|
| `--tp` | `2` | `1` | `2` | `2` | `1` |
| `--mem-fraction-static` | `0.80` | `0.85` | `0.82` | `0.85` | `0.75` |
| `--attention-backend` | `triton` | `triton` | `flashinfer` | `flashinfer` | `flashinfer` |
| `--kv-cache-dtype` | `auto` / `fp8_e5m2` | `auto` | `auto` / `fp8_e5m2` | `auto` | `auto` |
| `--speculative-algorithm` | `STANDALONE` | `STANDALONE` | `STANDALONE` | `STANDALONE` | `STANDALONE` |
| `--speculative-num-steps` | `5` | `5` | `5` | `5` | `5` |

---

## 4. Operational & Cost Optimization Rules

### Rule 1: The Fast-Network Filter (`inet_down > 500`)
* `Qwen2.5-32B-Instruct-AWQ` (~19.5 GB) + `1.5B` (~3.1 GB) = ~23 GB download.
* On a **100 Mbps link**: downloads take **~35 minutes** (you burn 50% of your paid 1-hour session waiting for downloads).
* On a **600–1000 Mbps link**: downloads take **~3 minutes**.
* **Always enforce `inet_down > 500`** in your Vast.ai search filters.

### Rule 2: Disk Size Allocation
* Always request **$\ge 80\text{ GB}$ disk** (100–120 GB is ideal).
* Docker image + base OS + model weights peak at ~65–75 GB during extraction.
* Never select 40–50 GB disks to save pennies; hitting `no space left on device` aborts the run and wastes the entire instance cost.

### Rule 3: Direct SGLang Template (No Docker-in-Docker)
* Set the template directly to `lmsysorg/sglang:latest` during rental.
* Running `docker run` inside a default Ubuntu instance frequently fails because nested Docker is disabled on host GPU daemons.
* Inside `lmsysorg/sglang:latest`, execute `python3 -m sglang.launch_server ...` directly.

### Rule 4: Stop vs. Destroy
* **`vastai stop instance <ID>`:** Stops GPU compute charges, but **continues billing storage disk** (~$0.01–$0.03/day). Useful if taking a 30-minute break between runs.
* **`vastai destroy instance <ID>`:** Deletes the container and permanently halts **all** billing. Run this immediately after SCP-ing your CSV results.

### Rule 5: Consumer Multi-GPU NCCL Watchout
* When renting dual RTX 3090s or dual RTX 4090s on dual-socket enterprise server boards, PCIe P2P may be blocked.
* If SGLang hangs during server boot at `NCCL initialization`, set:
  ```bash
  export NCCL_P2P_DISABLE=1
  ```
  before starting the server.

---

## 5. End-to-End Session Workflow (Copy-Paste)

### 1. Launch Instance (Tier 1 Example)
```bash
# 1. Search best live offer
vastai search offers 'reliability > 0.97 disk_space >= 80 inet_down > 500 num_gpus=2 gpu_name=RTX_3090' -o 'dph' --limit 3

# 2. Rent with SGLang template
vastai create instance <OFFER_ID> --image lmsysorg/sglang:latest --disk 80 --ssh --direct

# 3. Check connection string
vastai show instances
```

### 2. Run On-Instance
```bash
# Connect via SSH
ssh -p <PORT> root@<HOST>

# Clone benchmark suite
git clone <YOUR_REPO_URL>
cd benchmarking-sglang-qwen-recipe

# Start baseline server in background/tmux
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-32B-Instruct-AWQ \
  --tp 2 \
  --mem-fraction-static 0.85 \
  --enable-metrics \
  --host 0.0.0.0 --port 30000 > server.log 2>&1 &

# Wait for health check
until curl -s http://localhost:30000/health >/dev/null; do sleep 2; done

# Execute benchmark
python3 scripts/run_benchmark.py \
  --config-name baseline_tp2 \
  --dataset dataset/combined_benchmark_dataset.jsonl \
  --output-csv results/benchmark_runs.csv
```

### 3. Teardown & Local Export
```bash
# Copy results back to local machine
scp -P <PORT> root@<HOST>:/root/benchmarking-sglang-qwen-recipe/results/benchmark_runs.csv ./results/

# Destroy instance immediately
vastai destroy instance <INSTANCE_ID>
```
