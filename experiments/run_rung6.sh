#!/bin/bash
# Rung 6: the two winners combined -- ArcFace on the semantic block (rung 5, +0.038 mAP@10)
# plus colour CE at w=0.03 on the colour block (rung 3b, +0.258 strict).
cd "$(dirname "$0")"
while pgrep -f "rung_ablation.py --aux --tag rung4-aux-50" >/dev/null; do sleep 20; done
../.venv/bin/python rung_ablation.py --head colour --colour-loss ce --colour-weight 0.03 \
    --arcface --tag rung6-arc-colour-50 > out_rung6-50.log 2>&1
