#!/bin/bash
# 5-epoch directional screen over the colour weight -- same method Task 1 used
# (EPOCHS_SCREEN = 5). Pass criterion: sem_loss descending BELOW the 0.20 margin
# while col_acc climbs, i.e. both blocks alive.
cd "$(dirname "$0")"
while pgrep -f "rung_ablation.py --pool gem --tag rung2-gem-50" >/dev/null; do sleep 20; done
for W in 0.30 0.10 0.03; do
  ../.venv/bin/python rung_ablation.py --head colour --colour-loss ce --colour-weight $W \
      --epochs 5 --tag screen-w$W > out_screen-w$W.log 2>&1
done
