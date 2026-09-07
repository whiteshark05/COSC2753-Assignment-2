#!/bin/bash
# Full 50-epoch runs at the two weights that survived the screen. 0.03 is the healthy
# one; 0.10 brackets it from above (borderline at epoch 5 -- may or may not recover).
cd "$(dirname "$0")"
../.venv/bin/python rung_ablation.py --head colour --colour-loss ce --colour-weight 0.03 \
    --tag rung3b-w0.03-50 > out_rung3b-w0.03-50.log 2>&1
../.venv/bin/python rung_ablation.py --head colour --colour-loss ce --colour-weight 0.10 \
    --tag rung3b-w0.10-50 > out_rung3b-w0.10-50.log 2>&1
