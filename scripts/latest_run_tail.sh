#!/bin/bash
set -euo pipefail

find_latest_log() {
  find . \( -path "./.venv" -o -path "./.git" \) -prune -o \
    -type f -name 'run.log' -print0 \
  | xargs -0 ls -t 2>/dev/null | head -n 1
}

LOG_FILE=$(find_latest_log)
if [[ -z "$LOG_FILE" ]]; then
  echo "No run.log files found." >&2
  exit 1
fi

echo "Monitoring $LOG_FILE"

# Print existing validation lines first
grep "Validation bpb" "$LOG_FILE" || true

# Then follow new ones as they arrive
tail -F "$LOG_FILE" | stdbuf -o0 grep --line-buffered "Validation bpb"
