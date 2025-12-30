#!/usr/bin/env bash
set -euo pipefail

# Usage: ./run.sh [data_root] [model] [output_dir] [epoch] [lr] [bs] [max_length] [seed]
# Example: ./run.sh data bert-base-uncased experiments 3 2e-5 16 256 42

DATA_ROOT=${1:-data}
MODEL=${2:-bert-base-uncased}
OUTPUT_DIR=${3:-experiments}
EPOCH=${4:-4}
LR=${5:-2e-5}
BS=${6:-32}
MAX_LENGTH=${7:-512}
SEED=${8:-666}

python main.py \
  --data_root "$DATA_ROOT" \
  --model "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --epoch "$EPOCH" \
  --lr "$LR" \
  --bs "$BS" \
  --max_length "$MAX_LENGTH" \
  --seed "$SEED"
