#!/bin/bash
# Waits until rung 1 passes epoch 55, records the matched-epoch comparison, then
# stops the training run. Exits early and harmlessly if the run finishes by itself.
R=/Users/tinnguyen/Downloads/Tinibig/COSC2753-Assignment-2
PROG=$R/experiments/progress_rung1-cifar-stem.log
OUT=$R/experiments/rung1_epoch55_comparison.txt
TARGET=55

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$R/experiments/stop_at_55.log"; }
log "watcher started; waiting for epoch >= $TARGET"

for i in $(seq 1 300); do            # up to 5h of polling
  PID=$(ps aux | grep "[r]ung_ablation.py" | grep python | awk '{print $2}' | head -1)
  EP=$(grep -oE "^ep [0-9]+" "$PROG" 2>/dev/null | tail -1 | awk '{print $2}')
  EP=${EP:-0}

  if [ -z "$PID" ]; then
    log "training process gone (finished or early-stopped) at epoch $EP - nothing to kill"
    { echo "=== rung 1 ended on its own at epoch $EP ==="; "$R/experiments/compare_rungs.sh"; } > "$OUT" 2>&1
    log "comparison written to $OUT"; exit 0
  fi

  if [ "$EP" -ge "$TARGET" ]; then
    log "epoch $EP >= $TARGET - recording comparison, then stopping"
    { echo "=== rung 1 stopped at epoch $EP (requested: after 55) ==="
      echo "NOTE: this run predates the checkpointing patch, so no model file or"
      echo "full-eval metrics (strict / bootstrap CI / P@K) exist - the probe curve below"
      echo "is the surviving evidence, and it is what the stem decision rests on."
      echo
      "$R/experiments/compare_rungs.sh"
    } > "$OUT" 2>&1
    kill "$PID" 2>/dev/null; sleep 5; kill -9 "$PID" 2>/dev/null
    log "stopped training pid $PID at epoch $EP"
    log "comparison saved to $OUT"
    exit 0
  fi
  sleep 60
done
log "watcher timed out"
