#!/bin/bash

# The ~$10 tier of nanochat.
# Target: single-GPU pipeline (~2 hours on modern datacenter GPU) with chat-worthy checkpoints.

set -euo pipefail

export UV_HTTP_TIMEOUT=600   # seconds

PROFILE="h100"
EVE_ENABLED=false

while [ $# -gt 0 ]; do
  case "$1" in
    h100|rtx5090)
      PROFILE="$1"
      ;;
    eve)
      EVE_ENABLED=true
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: bash run10.sh [h100|rtx5090] [eve]" >&2
      exit 1
      ;;
  esac
  shift
done

SEQ_LEN=2048
DEVICE_BATCH=24
TOTAL_BATCH=49_152

if [ "$PROFILE" = "rtx5090" ]; then
  DEVICE_BATCH=12           # halve per-step tokens to fit 32GB
  TOTAL_BATCH=49_152        # two microsteps => same effective batch, keeps 20x tokens/param
fi

EVE_ARGS=()
if [ "$EVE_ENABLED" = true ]; then
  EVE_ARGS+=(--eve True)
fi

echo "Running run10 profile: $PROFILE"
echo "  device_batch_size = $DEVICE_BATCH"
echo "  total_batch_size  = $TOTAL_BATCH"
if [ "$EVE_ENABLED" = true ]; then
  echo "  eve dynamics     = enabled"
fi

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Environment setup via uv
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# Optional wandb logging
export WANDB_RUN="${WANDB_RUN:-dummy}"

# Fresh report
python -m nanochat.report reset

# Rust tokenizer toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Dataset + tokenizer (scaled for 3.7B tokens total)
python -m nanochat.dataset -n 16
python -m nanochat.dataset -n 120 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train --max_chars=200_000_000 --doc_cap=10_000 --vocab_size=65_536
python -m scripts.tok_eval

# Finish background shard download
wait "$DATASET_DOWNLOAD_PID"

# Base pretraining (depth-12, 1 GPU)
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
    --depth=12 \
    --device_batch_size="$DEVICE_BATCH" \
    --total_batch_size="$TOTAL_BATCH" \
    --num_iterations=75_500 \
    --eval_tokens=32_768 \
    --core_metric_every=-1 \
    --sample_every=-1 \
    "${EVE_ARGS[@]}" \
    --run="$WANDB_RUN"

torchrun --standalone --nproc_per_node=1 -m scripts.base_loss
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- --max-per-task=64

# Midtraining for conversational format/tooling
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
torchrun --standalone --nproc_per_node=1 -m scripts.mid_train -- \
    --device_batch_size=4 \
    --num_iterations=200 \
    "${EVE_ARGS[@]}" \
    --run="$WANDB_RUN"
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i mid -a ARC-Easy|ARC-Challenge -x 100

# Supervised finetune pass
torchrun --standalone --nproc_per_node=1 -m scripts.chat_sft -- \
    --device_batch_size=4 \
    --num_epochs=1 \
    --num_iterations=200 \
    "${EVE_ARGS[@]}" \
    --run="$WANDB_RUN"
torchrun --standalone --nproc_per_node=1 -m scripts.chat_eval -- -i sft -a ARC-Easy|ARC-Challenge -x 100

# Wrap up
python -m nanochat.report generate

echo "run10 complete. Report available in report.md. Try: python -m scripts.chat_web"
