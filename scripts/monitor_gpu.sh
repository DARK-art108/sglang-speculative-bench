#!/usr/bin/env bash
# Real-time GPU telemetry logging daemon
set -euo pipefail

OUTPUT_FILE="${1:-results/gpu_telemetry.csv}"
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo "Logging GPU telemetry to $OUTPUT_FILE (Ctrl+C to stop)..."
nvidia-smi \
  --query-gpu=timestamp,index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
  --format=csv \
  -l 1 > "$OUTPUT_FILE"
