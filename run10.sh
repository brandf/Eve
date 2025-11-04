#!/bin/bash

# The ~$10 tier of nanochat.
# Target: single-GPU pipeline (~2 hours on modern datacenter GPU) with chat-worthy checkpoints.

set -euo pipefail

export UV_HTTP_TIMEOUT=600   # seconds

PROFILE="h100"
EVE_ENABLED=false
OVERRIDE_ITERS=""
OVERRIDE_EVAL_TOKENS=""
OVERRIDE_EVE_BETA1=""
OVERRIDE_EVE_BETA2=""
OVERRIDE_EVE_ETA=""
OVERRIDE_DEVICE_BATCH=""
OVERRIDE_TOTAL_BATCH=""

while [ $# -gt 0 ]; do
  case "$1" in
    h100|rtx5090)
      PROFILE="$1"
      ;;
    eve)
      EVE_ENABLED=true
      ;;
    --iters=*)
      OVERRIDE_ITERS="${1#*=}"
      ;;
    --eval_tokens=*)
      OVERRIDE_EVAL_TOKENS="${1#*=}"
      ;;
    --eve_beta1=*)
      OVERRIDE_EVE_BETA1="${1#*=}"
      ;;
    --eve_beta2=*)
      OVERRIDE_EVE_BETA2="${1#*=}"
      ;;
    --eve_eta=*)
      OVERRIDE_EVE_ETA="${1#*=}"
      ;;
    --device_batch_size=*)
      OVERRIDE_DEVICE_BATCH="${1#*=}"
      ;;
    --total_batch_size=*)
      OVERRIDE_TOTAL_BATCH="${1#*=}"
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: bash run10.sh [h100|rtx5090] [eve] [--iters=N] [--eval_tokens=N] [--eve_beta1=X] [--eve_beta2=Y] [--eve_eta=Z] [--device_batch_size=N] [--total_batch_size=N]" >&2
      exit 1
      ;;
  esac
  shift
done

SEQ_LEN=2048
DEVICE_BATCH=48
TOTAL_BATCH=98_304
DEFAULT_ITERS=37_750

if [ "$PROFILE" = "rtx5090" ]; then
  DEVICE_BATCH=24           # tuned for 32GB RTX 5090
  TOTAL_BATCH=49_152        # single microstep at 24 x 2048 tokens
  DEFAULT_ITERS=75_500
fi

if [ -n "$OVERRIDE_DEVICE_BATCH" ]; then
  DEVICE_BATCH="$OVERRIDE_DEVICE_BATCH"
fi

if [ -n "$OVERRIDE_TOTAL_BATCH" ]; then
  TOTAL_BATCH="$OVERRIDE_TOTAL_BATCH"
fi

EVE_ARGS=()
if [ "$EVE_ENABLED" = true ]; then
  EVE_BETA1=${OVERRIDE_EVE_BETA1:-0.80}
  EVE_BETA2=${OVERRIDE_EVE_BETA2:-0.91}
  EVE_ETA=${OVERRIDE_EVE_ETA:-1.0}
  EVE_ARGS+=(--eve=True "--eve_beta1=$EVE_BETA1" "--eve_beta2=$EVE_BETA2" "--eve_eta=$EVE_ETA")
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

# Fresh report (skipped if NO_REPORT set)
if [ "${NO_REPORT:-0}" != "1" ]; then
  python -m nanochat.report reset
fi

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
    --num_iterations="${OVERRIDE_ITERS:-$DEFAULT_ITERS}" \
    --eval_tokens="${OVERRIDE_EVAL_TOKENS:-32_768}" \
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
if [ "${NO_REPORT:-0}" != "1" ]; then
  python -m nanochat.report generate
fi

echo "run10 complete. Report available in report.md. Try: python -m scripts.chat_web"
