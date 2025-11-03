#!/bin/bash
# Lightweight Eve hyperparameter tuner using run10.sh as the backend.

set -euo pipefail

PROFILE="h100"
ENABLE_EVE=true
STAGE1_TRIALS=3
STAGE1_ITERS=5000
STAGE2_TRIALS=4
STAGE2_ITERS=8000
EVAL_TOKENS=""
BASE_DIR="$(pwd)"
REPORT_ROOT="$BASE_DIR/autotune_runs"
EVE_DEFAULT_BETA1=0.90
EVE_DEFAULT_BETA2=0.9990
EVE_DEFAULT_ETA=1.0

usage() {
  cat <<EOF
Usage: bash scripts/autotune_eve.sh [options]

Options:
  --profile (h100|rtx5090)    GPU profile to use (default: h100)
  --stage1-trials N           Number of random trials in stage 1 (default: 3)
  --stage1-iters N            Iterations per stage 1 trial (default: 5000)
  --stage2-trials N           Number of refinement trials (default: 4)
  --stage2-iters N            Iterations per refinement trial (default: 8000)
  --eval-tokens N             Tokens for validation sampling (default: auto per profile)
  --baseline                  Run without Eve (baseline ResNet residual update)
  --help                      Show this message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2;;
    --stage1-trials) STAGE1_TRIALS="$2"; shift 2;;
    --stage1-iters) STAGE1_ITERS="$2"; shift 2;;
    --stage2-trials) STAGE2_TRIALS="$2"; shift 2;;
    --stage2-iters) STAGE2_ITERS="$2"; shift 2;;
    --eval-tokens) EVAL_TOKENS="$2"; shift 2;;
    --baseline) ENABLE_EVE=false; shift;;
    --help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

if $ENABLE_EVE; then
  MODE_DIR="$REPORT_ROOT/eve"
else
  MODE_DIR="$REPORT_ROOT/baseline"
fi
mkdir -p "$MODE_DIR"
SUMMARY_FILE="$MODE_DIR/summary.tsv"
printf "stage\ttrial\tbeta1\tbeta2\teta\tmin_bpb\titers\treport_path\n" > "$SUMMARY_FILE"

if [[ -z "$EVAL_TOKENS" ]]; then
  case "$PROFILE" in
    h100) EVAL_TOKENS=49152;;
    rtx5090) EVAL_TOKENS=24576;;
  esac
  echo "[autotune] Using default eval_tokens=$EVAL_TOKENS for profile $PROFILE"
fi

if $ENABLE_EVE; then
  echo "[autotune] Mode: Eve-enabled sweep"
else
  echo "[autotune] Mode: Baseline residual sweep (no Eve)"
fi

random_float() {
  local lo="$1"
  local hi="$2"
  python3 - "$lo" "$hi" <<'PY'
import random, sys
lo, hi = map(float, sys.argv[1:3])
print(f"{random.uniform(lo, hi):.6f}")
PY
}

extract_bpb() {
  local report="$1"
  grep -m1 "Validation bpb" "$report" | awk -F ":" '{print $NF+0}' || echo "inf"
}

run_trial() {
  local stage="$1"
  local trial="$2"
  local beta1="$3"
  local beta2="$4"
  local eta="$5"
  local iters="$6"
  local run_id="${stage}_${trial}_$(date +%H%M%S)"
  local log_dir="$MODE_DIR/$run_id"
  mkdir -p "$log_dir"
  echo "[autotune] Stage ${stage} trial ${trial}: beta1=$beta1 beta2=$beta2 eta=$eta iters=$iters"
  local args=("$PROFILE")
  local eve_flags=()
  if $ENABLE_EVE; then
    args+=(eve)
    eve_flags+=("--eve_beta1=$beta1" "--eve_beta2=$beta2" "--eve_eta=$eta")
  fi
  WANDB_RUN="autotune_$run_id" bash run10.sh "${args[@]}" \
    --iters="$iters" \
    --eval_tokens="$EVAL_TOKENS" \
    "${eve_flags[@]}" \
    > "$log_dir/run.log" 2>&1 || true
  local report="$BASE_DIR/report.md"
  if [[ -f "$report" ]]; then
    cp "$report" "$log_dir/"
    min_bpb=$(extract_bpb "$log_dir/report.md")
  else
    min_bpb="inf"
  fi
  printf "%s\t%s\t%.6f\t%.6f\t%.6f\t%s\t%s\t%s\n" \
    "$stage" "$trial" "$beta1" "$beta2" "$eta" "$min_bpb" "$iters" "$log_dir/report.md" \
    >> "$SUMMARY_FILE"
  echo "[autotune] Trial ${trial} -> min_bpb=${min_bpb}"
}

if $ENABLE_EVE; then
  echo "[autotune] Stage 1: random exploration with $STAGE1_TRIALS trials."
  for ((i=1;i<=STAGE1_TRIALS;i++)); do
    beta1=$(random_float 0.85 0.95)
    beta2=$(random_float 0.9985 0.9995)
    eta=$(random_float 0.8 1.2)
    run_trial "stage1" "$i" "$beta1" "$beta2" "$eta" "$STAGE1_ITERS"
  done
else
  echo "[autotune] Baseline mode: running single reference trial."
  run_trial "baseline" "1" "$EVE_DEFAULT_BETA1" "$EVE_DEFAULT_BETA2" "$EVE_DEFAULT_ETA" "$STAGE1_ITERS"
fi

if $ENABLE_EVE; then
  echo "[autotune] Stage 2: refining best candidates."
  best=$(sort -t$'\t' -k6,6 "$SUMMARY_FILE" | head -n $((STAGE2_TRIALS+1)))
  echo "$best" | tail -n +2 | while IFS=$'\t' read -r _ trial beta1 beta2 eta _ _ _; do
    run_trial "stage2" "${trial}a" "$beta1" "$beta2" "$eta" "$STAGE2_ITERS"
    beta1=$(python3 - "$beta1" <<'PY'
import random, sys
base = float(sys.argv[1])
print(f"{base + random.uniform(-0.01, 0.01):.6f}")
PY
)
    beta2=$(python3 - "$beta2" <<'PY'
import random, sys
base = float(sys.argv[1])
print(f"{base + random.uniform(-1e-4, 1e-4):.6f}")
PY
)
    eta=$(python3 - "$eta" <<'PY'
import random, sys
base = float(sys.argv[1])
print(f"{base + random.uniform(-0.05, 0.05):.6f}")
PY
)
    run_trial "stage2" "${trial}b" "$beta1" "$beta2" "$eta" "$STAGE2_ITERS"
  done
fi

echo "[autotune] Results (best first):"
awk 'BEGIN {printf "%-8s %-6s %-8s %-8s %-8s %-8s %-8s %s\n", "stage", "trial", "beta1", "beta2", "eta", "min_bpb", "iters", "report_path"}
{
  printf "%-8s %-6s %-8s %-8s %-8s %-8s %-8s %s\n",
    $1, $2, $3, $4, $5, $6, $7, $8;
}' < <(tail -n +2 "$SUMMARY_FILE" | sort -t$'\t' -k6,6)
