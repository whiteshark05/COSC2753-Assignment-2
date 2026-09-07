#!/bin/bash
# 50-epoch ladder: re-baseline rung 0, then the two rungs, all at an identical budget.
cd "$(dirname "$0")"
P=../.venv/bin/python
$P rung_ablation.py                                        --tag rung0-50    > out_rung0-50.log    2>&1
$P rung_ablation.py --head colour --colour-loss ce         --tag rung3b-ce-50 > out_rung3b-ce-50.log 2>&1
$P rung_ablation.py --pool gem                             --tag rung2-gem-50 > out_rung2-gem-50.log 2>&1
