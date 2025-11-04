#!/bin/bash
# Lightweight Eve hyperparameter tuner that compares Eve sweeps against a baseline run.

set -euo pipefail

PROFILE="h100"
STAGE1_TRIALS=4
STAGE1_ITERS=500
STAGE2_TRIALS=4
STAGE2_ITERS=1000
BASELINE_ITERS=1000
EVAL_TOKENS=""

BASE_DIR="$(pwd)"
REPORT_ROOT="$BASE_DIR/autotune_runs"
EVE_DIR="$REPORT_ROOT/eve"
BASELINE_DIR="$REPORT_ROOT/baseline"
EVE_SUMMARY="$EVE_DIR/summary.tsv"
BASELINE_SUMMARY="$BASELINE_DIR/summary.tsv"

EVE_DEFAULT_BETA1=0.90
EVE_DEFAULT_BETA2=0.9990
EVE_DEFAULT_ETA=1.0

usage() {
  cat <<EOF
Usage: bash scripts/autotune_eve.sh [options]

Options:
  --profile (h100|rtx5090)    GPU profile to use (default: h100)
  --stage1-trials N           Number of random Eve trials in stage 1 (default: 3)
  --stage1-iters N            Iterations per stage 1 Eve trial (default: 5000)
  --stage2-trials N           Number of Eve refinement trials (default: 4)
  --stage2-iters N            Iterations per Eve refinement trial (default: 8000)
  --baseline-iters N          Iterations for the baseline (no Eve) run (default: 5000)
  --eval-tokens N             Tokens used during validation (default: auto per profile)
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
    --baseline-iters) BASELINE_ITERS="$2"; shift 2;;
    --eval-tokens) EVAL_TOKENS="$2"; shift 2;;
    --help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

mkdir -p "$EVE_DIR" "$BASELINE_DIR"
printf "stage\ttrial\tbeta1\tbeta2\teta\tmin_bpb\titers\treport_path\n" > "$EVE_SUMMARY"
printf "stage\ttrial\tbeta1\tbeta2\teta\tmin_bpb\titers\treport_path\n" > "$BASELINE_SUMMARY"

if [[ -z "$EVAL_TOKENS" ]]; then
  case "$PROFILE" in
    h100) EVAL_TOKENS=49152;;
    rtx5090) EVAL_TOKENS=24576;;
    *) echo "Unsupported profile: $PROFILE"; exit 1;;
  esac
  echo "[autotune] Using default eval_tokens=$EVAL_TOKENS for profile $PROFILE"
fi

echo "[autotune] Eve sweep configuration:"
echo "  profile          : $PROFILE"
echo "  stage1 trials    : $STAGE1_TRIALS (iters=$STAGE1_ITERS)"
echo "  stage2 trials    : $STAGE2_TRIALS (iters=$STAGE2_ITERS)"
echo "  baseline iters   : $BASELINE_ITERS"
echo "  eval_tokens      : $EVAL_TOKENS"

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
  local source_file="$1"
  if [[ ! -f "$source_file" ]]; then
    echo "inf"
    return
  fi
  local value
  value=$(grep -m1 -E "Validation bpb" "$source_file" | awk -F ":" '{print $NF+0}')
  if [[ -z "$value" ]]; then
    echo "inf"
  else
    printf "%s" "$value"
  fi
}

run_trial() {
  local mode="$1"   # eve | baseline
  local stage="$2"
  local trial="$3"
  local beta1="$4"
  local beta2="$5"
  local eta="$6"
  local iters="$7"

  local target_dir
  local summary_file
  if [[ "$mode" == "eve" ]]; then
    target_dir="$EVE_DIR"
    summary_file="$EVE_SUMMARY"
  else
    target_dir="$BASELINE_DIR"
    summary_file="$BASELINE_SUMMARY"
  fi

  local run_id="${stage}_${trial}_$(date +%H%M%S)"
  local log_dir="$target_dir/$run_id"
  mkdir -p "$log_dir"

  echo "[autotune][$mode] Stage ${stage} trial ${trial}: beta1=$beta1 beta2=$beta2 eta=$eta iters=$iters"

  local args=("$PROFILE")
  local eve_flags=()
  if [[ "$mode" == "eve" ]]; then
    args+=(eve)
    eve_flags+=("--eve_beta1=$beta1" "--eve_beta2=$beta2" "--eve_eta=$eta")
  fi

  WANDB_MODE=offline WANDB_RUN="autotune_${mode}_${run_id}" \
    bash run10.sh "${args[@]}" \
    --iters="$iters" \
    --eval_tokens="$EVAL_TOKENS" \
    "${eve_flags[@]}" \
    > "$log_dir/run.log" 2>&1 || true

  local report_path="$BASE_DIR/report.md"
  local min_bpb="inf"

  if [[ -f "$report_path" ]]; then
    cp "$report_path" "$log_dir/report.md"
    min_bpb=$(extract_bpb "$log_dir/report.md")
  fi

  if [[ "$min_bpb" == "inf" ]]; then
    min_bpb=$(extract_bpb "$log_dir/run.log")
  fi

  if [[ "$min_bpb" == "inf" ]]; then
    echo "[autotune][$mode] WARNING: no validation bpb found for ${stage}/${trial}."
    echo "[autotune][$mode] tail -n 40 $log_dir/run.log"
    tail -n 40 "$log_dir/run.log"
  fi

  printf "%s\t%s\t%.6f\t%.6f\t%.6f\t%s\t%s\t%s\n" \
    "$stage" "$trial" "$beta1" "$beta2" "$eta" "$min_bpb" "$iters" "$log_dir/report.md" \
    >> "$summary_file"

  echo "[autotune][$mode] Trial ${trial} -> min_bpb=${min_bpb}"
}

# Eve Stage 1 exploration
if (( STAGE1_TRIALS > 0 )); then
  echo "[autotune] Stage 1 (Eve): random exploration with $STAGE1_TRIALS trials."
  for ((i=1;i<=STAGE1_TRIALS;i++)); do
    beta1=$(random_float 0.85 0.95)
    beta2=$(random_float 0.9985 0.9995)
    eta=$(random_float 0.8 1.2)
    run_trial "eve" "stage1" "$i" "$beta1" "$beta2" "$eta" "$STAGE1_ITERS"
  done
fi

# Eve Stage 2 refinement
if (( STAGE2_TRIALS > 0 )); then
  echo "[autotune] Stage 2 (Eve): refining best candidates."
  best=$(tail -n +2 "$EVE_SUMMARY" | sort -t$'\t' -k6,6n | head -n "$STAGE2_TRIALS")
  if [[ -z "$best" ]]; then
    echo "[autotune] Stage 2 skipped (no stage1 results found)."
  else
    echo "$best" | while IFS=$'\t' read -r stage trial beta1 beta2 eta _ _ _; do
      run_trial "eve" "stage2" "${trial}a" "$beta1" "$beta2" "$eta" "$STAGE2_ITERS"
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
      run_trial "eve" "stage2" "${trial}b" "$beta1" "$beta2" "$eta" "$STAGE2_ITERS"
    done
  fi
fi

# Baseline reference run
echo "[autotune] Baseline: running single reference without Eve."
run_trial "baseline" "baseline" "1" "$EVE_DEFAULT_BETA1" "$EVE_DEFAULT_BETA2" "$EVE_DEFAULT_ETA" "$BASELINE_ITERS"

echo
echo "[autotune] Eve results (best first):"
awk 'BEGIN {printf "%-8s %-8s %-10s %-10s %-8s %-10s %-8s %s\n", "stage", "trial", "beta1", "beta2", "eta", "min_bpb", "iters", "report_path"}
{printf "%-8s %-8s %-10s %-10s %-8s %-10s %-8s %s\n", $1, $2, $3, $4, $5, $6, $7, $8}' \
  < <(tail -n +2 "$EVE_SUMMARY" | sort -t$'\t' -k6,6n)

echo
echo "[autotune] Baseline result:"
awk 'BEGIN {printf "%-8s %-8s %-10s %-10s %-8s %-10s %-8s %s\n", "stage", "trial", "beta1", "beta2", "eta", "min_bpb", "iters", "report_path"}
{printf "%-8s %-8s %-10s %-10s %-8s %-10s %-8s %s\n", $1, $2, $3, $4, $5, $6, $7, $8}' \
  < <(tail -n +2 "$BASELINE_SUMMARY")
