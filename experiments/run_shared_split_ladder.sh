#!/bin/bash
# Full 9-rung ladder, retrained on the team's SHARED split (data/processed).
#
# Every rung must be retrained, not re-scored: the old checkpoints were trained on
# data/processed_task4, whose training rows land in this split's query set, so scoring
# them here would leak. Ordered so the two numbers the report leans on -- rung 0 (the
# ladder baseline) and rung 5 (the selected model) -- land first.
#
# All rungs run the same 50-epoch budget. Previously rung3-colour ran 80 epochs because
# it predates the fixed budget; running it at 50 makes the ladder matched throughout.
cd "$(dirname "$0")"
P=../.venv/bin/python
L=log
mkdir -p "$L"

run () {                       # run <tag> <args...>
  tag="$1"; shift
  echo "[$(date '+%H:%M:%S')] START $tag" >> "$L/ladder_progress.log"
  $P rung_ablation.py "$@" --tag "$tag" > "$L/out_${tag}.log" 2>&1
  rc=$?
  map=$(../.venv/bin/python -c "import json;print(f\"{json.load(open('$L/result_${tag}.json'))['mAP10']:.4f}\")" 2>/dev/null || echo "n/a")
  echo "[$(date '+%H:%M:%S')] DONE  $tag rc=$rc mAP@10=$map" >> "$L/ladder_progress.log"
}

echo "=== shared-split ladder started $(date) ===" > "$L/ladder_progress.log"
run rung0-50
run rung5-arcface-50    --arcface
run rung3b-w0.10-50     --head colour --colour-loss ce --colour-weight 0.10
run rung6-arc-colour-50 --head colour --colour-loss ce --colour-weight 0.03 --arcface
run rung2-gem-50        --pool gem
run rung3-colour        --head colour
run rung3b-ce-50        --head colour --colour-loss ce
run rung3b-w0.03-50     --head colour --colour-loss ce --colour-weight 0.03
run rung4-aux-50        --aux
echo "=== ladder finished $(date) ===" >> "$L/ladder_progress.log"
