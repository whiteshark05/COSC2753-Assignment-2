#!/bin/bash
# Rung 5 first (ArcFace looked strongest in the 2-epoch smoke), then rung 4.
cd "$(dirname "$0")"
../.venv/bin/python rung_ablation.py --arcface --tag rung5-arcface-50 > out_rung5-50.log 2>&1
../.venv/bin/python rung_ablation.py --aux     --tag rung4-aux-50     > out_rung4-50.log 2>&1
