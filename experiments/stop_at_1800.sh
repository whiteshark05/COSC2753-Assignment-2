#!/bin/bash
# Stops rung 1 at the 18:00 mark, provided it has passed epoch 50 by then.
# Records the matched-epoch comparison first. Exits harmlessly if the run ends itself.
R=/Users/tinnguyen/Downloads/Tinibig/COSC2753-Assignment-2
PROG=$R/experiments/progress_rung1-cifar-stem.log
OUT=$R/experiments/rung1_stopped_comparison.txt
MIN_EPOCH=50
DEADLINE=1800

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$R/experiments/stop_watcher.log"; }
log "watcher armed: stop at $DEADLINE once epoch >= $MIN_EPOCH"

for i in $(seq 1 400); do
  PID=$(ps aux | grep "[r]ung_ablation.py" | grep python | awk '{print $2}' | head -1)
  EP=$(grep -oE "^ep [0-9]+" "$PROG" 2>/dev/null | tail -1 | awk '{print $2}'); EP=${EP:-0}
  NOW=$(date +%H%M)

  if [ -z "$PID" ]; then
    log "training already ended at epoch $EP - nothing to stop"
    { echo "=== rung 1 ended on its own at epoch $EP ==="; "$R/experiments/compare_rungs.sh"; } > "$OUT" 2>&1
    exit 0
  fi

  if [ "$NOW" -ge "$DEADLINE" ] && [ "$EP" -ge "$MIN_EPOCH" ]; then
    log "18:00 mark reached at epoch $EP - recording, then stopping"
    { echo "=== rung 1 stopped at the 18:00 mark, epoch $EP ==="
      echo "This run predates the checkpointing patch: no model file and no full-eval"
      echo "metrics (strict / bootstrap CI / P@K). The probe curve below is the surviving"
      echo "evidence, and it is what the stem decision rests on."
      echo
      "$R/experiments/compare_rungs.sh"
    } > "$OUT" 2>&1
    kill "$PID" 2>/dev/null; sleep 5; kill -9 "$PID" 2>/dev/null
    log "stopped pid $PID at epoch $EP; comparison -> $OUT"
    exit 0
  fi
  sleep 30
done
log "watcher timed out"
