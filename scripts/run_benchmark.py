#!/usr/bin/env python3
"""
Automated Benchmarking Harness for SGLang LLM Inference
Measures TTFT, TPOT, Decode Throughput (tok/s), and scrapes SGLang Prometheus metrics
for Speculative Decoding Acceptance Rate.
"""

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
import urllib.request
import urllib.error

def get_prometheus_metrics(endpoint_url):
    """Scrape /metrics endpoint and return parsed sglang gauges/counters."""
    metrics_url = f"{endpoint_url.rstrip('/')}/metrics"
    try:
        req = urllib.request.Request(metrics_url, headers={"User-Agent": "sglang-bench"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            content = resp.read().decode("utf-8")
        
        metrics = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                key, val_str = parts[0], parts[1]
                try:
                    metrics[key] = float(val_str)
                except ValueError:
                    pass
        return metrics
    except Exception:
        return {}

def send_chat_completion_stream(endpoint_url, model, prompt, max_tokens, temperature):
    """
    Send streaming chat completion request to /v1/chat/completions.
    Returns: (output_text, ttft, total_time, output_tokens, prompt_tokens)
    """
    url = f"{endpoint_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t_start = time.perf_counter()
    t_first_token = None
    output_chunks = []
    usage_info = {}
    
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if "usage" in chunk and chunk["usage"]:
                        usage_info = chunk["usage"]
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            output_chunks.append(content)
                except json.JSONDecodeError:
                    continue
                    
    t_end = time.perf_counter()
    total_time = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else total_time
    output_text = "".join(output_chunks)
    
    # Estimate tokens if server usage not provided in stream
    out_tokens = usage_info.get("completion_tokens")
    if out_tokens is None:
        # Fallback approximation: 1 token ~ 4 chars or whitespace count
        out_tokens = max(1, len(output_text.split()))
    prompt_tokens = usage_info.get("prompt_tokens", len(prompt.split()))
    
    return {
        "output_text": output_text,
        "ttft_s": ttft,
        "total_time_s": total_time,
        "decode_time_s": max(0.0001, total_time - ttft),
        "completion_tokens": out_tokens,
        "prompt_tokens": prompt_tokens,
        "decode_tok_per_sec": out_tokens / max(0.0001, total_time - ttft),
        "e2e_tok_per_sec": out_tokens / total_time
    }

def main():
    parser = argparse.ArgumentParser(description="SGLang LLM Inference Benchmark")
    parser.add_argument("--endpoint", default="http://localhost:30000", help="SGLang server URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ", help="Model name")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    parser.add_argument("--output-csv", default="results/benchmark_runs.csv", help="CSV output path")
    parser.add_argument("--config-name", default="speculative_standalone", help="Configuration label")
    parser.add_argument("--warmup", type=int, default=2, help="Number of warmup requests")
    parser.add_argument("--limit", type=int, default=0, help="Max requests to test (0 = all)")
    args = parser.parse_args()
    
    # Load dataset
    prompts = []
    with open(args.dataset, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
                
    if not prompts:
        print(f"Error: No prompts found in {args.dataset}")
        sys.exit(1)
        
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    
    # Check if CSV exists, if not write header
    csv_exists = os.path.exists(args.output_csv)
    csv_file = open(args.output_csv, "a", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow([
            "timestamp", "config_name", "model", "prompt_id", "category",
            "prompt_tokens", "completion_tokens", "ttft_s", "decode_time_s",
            "total_time_s", "decode_tok_per_sec", "e2e_tok_per_sec",
            "spec_accept_rate", "spec_accept_length"
        ])
        csv_file.flush()
        
    print(f"==================================================")
    print(f"Starting Benchmark: {args.config_name}")
    print(f"Endpoint: {args.endpoint} | Model: {args.model}")
    print(f"Dataset: {args.dataset} ({len(prompts)} items, limit={args.limit}, warmup={args.warmup})")
    print(f"==================================================")
    
    # Warmup
    if args.warmup > 0:
        print(f"Running {args.warmup} warmup requests...")
        for i in range(args.warmup):
            p = prompts[i % len(prompts)]
            send_chat_completion_stream(
                args.endpoint, args.model, p["prompt"],
                p.get("max_tokens", 256), p.get("temperature", 0.0)
            )
        print("Warmup complete.")
        
    # Benchmark runs
    results = []
    test_prompts = prompts[:args.limit]
    
    for idx, p in enumerate(test_prompts, start=1):
        prompt_id = p.get("id", f"p_{idx}")
        category = p.get("category", "default")
        max_tokens = p.get("max_tokens", 512)
        temp = p.get("temperature", 0.0)
        
        # Scrape metrics before/after to compute per-request speculative deltas
        m_before = get_prometheus_metrics(args.endpoint)
        res = send_chat_completion_stream(args.endpoint, args.model, p["prompt"], max_tokens, temp)
        m_after = get_prometheus_metrics(args.endpoint)

        # SGLang exposes spec metrics as cumulative counters; delta isolates this request.
        def _delta(key):
            b, a = m_before.get(key), m_after.get(key)
            if b is None or a is None:
                return None
            d = a - b
            return d if d > 0 else None

        accepted = _delta("sglang:spec_accepted_num_tokens")
        drafted = _delta("sglang:spec_num_draft_tokens")
        if accepted and drafted:
            spec_rate = accepted / drafted
            spec_len = accepted / max(1, _delta("sglang:spec_num_requests") or 1)
        else:
            spec_rate = None
            spec_len = None
        
        res["prompt_id"] = prompt_id
        res["category"] = category
        res["spec_accept_rate"] = spec_rate
        res["spec_accept_length"] = spec_len
        results.append(res)
        
        # Write to CSV
        csv_writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            res["prompt_tokens"], res["completion_tokens"],
            round(res["ttft_s"], 4), round(res["decode_time_s"], 4),
            round(res["total_time_s"], 4), round(res["decode_tok_per_sec"], 2),
            round(res["e2e_tok_per_sec"], 2),
            round(spec_rate, 4) if spec_rate is not None else "",
            round(spec_len, 2) if spec_len is not None else ""
        ])
        csv_file.flush()
        
        print(f"[{idx}/{len(test_prompts)}] {prompt_id} ({category}): "
              f"Tokens: {res['completion_tokens']} | "
              f"TTFT: {res['ttft_s']*1000:.1f}ms | "
              f"Decode: {res['decode_tok_per_sec']:.2f} tok/s"
              + (f" | Accept Rate: {spec_rate:.2%}" if spec_rate is not None else ""))
              
    csv_file.close()
    
    # Statistical Summary
    decode_speeds = [r["decode_tok_per_sec"] for r in results]
    ttfts = [r["ttft_s"] * 1000 for r in results]
    
    print("\n" + "="*50)
    print(f"SUMMARY FOR {args.config_name} (n={len(results)})")
    print("="*50)
    print(f"Decode Throughput (tok/s):")
    print(f"  Best (Max):   {max(decode_speeds):.2f} tok/s")
    print(f"  Median:       {statistics.median(decode_speeds):.2f} tok/s")
    print(f"  Mean:         {statistics.mean(decode_speeds):.2f} tok/s")
    print(f"  Min:          {min(decode_speeds):.2f} tok/s")
    if len(decode_speeds) > 1:
        print(f"  StdDev:       {statistics.stdev(decode_speeds):.2f} tok/s")
    print(f"Time-To-First-Token (TTFT):")
    print(f"  Median:       {statistics.median(ttfts):.1f} ms")
    print(f"  Mean:         {statistics.mean(ttfts):.1f} ms")
    
    rates = [r["spec_accept_rate"] for r in results if r["spec_accept_rate"] is not None]
    if rates:
        print(f"Speculative Accept Rate: {statistics.mean(rates):.2%}")
    print(f"Results saved to: {args.output_csv}")
    print("="*50)

if __name__ == "__main__":
    main()
