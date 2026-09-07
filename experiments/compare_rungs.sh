#!/bin/bash
# Matched-epoch comparison of rung 1 against rung 0's recorded curve.
cd /Users/tinnguyen/Downloads/Tinibig/COSC2753-Assignment-2
/Users/tinnguyen/Downloads/Tinibig/COSC2753-Assignment-2/.venv/bin/python - <<'PY'
import json, re, pathlib
r0 = {int(k): v for k, v in json.loads(pathlib.Path("experiments/rung0_probe_curve.json").read_text()).items()}
p = pathlib.Path("experiments/progress_rung1-cifar-stem.log")
r1, tmin = {}, 0.0
for line in p.read_text().splitlines():
    m = re.match(r"ep (\d+)/80 .*probe_mAP@10 ([\d.]+).*elapsed ([\d.]+)min", line)
    if m: r1[int(m.group(1))] = float(m.group(2)); tmin = float(m.group(3))
if not r1:
    print("no epochs yet"); raise SystemExit
last = max(r1)
print(f"rung 1 at epoch {last}/80  ({tmin:.0f} min elapsed, {tmin/last:.2f} min/epoch)")
print(f"{'epoch':>6} {'rung0 (default)':>16} {'rung1 (cifar)':>15} {'delta':>8}")
for e in sorted(r1):
    if e % 5 == 0 or e == last:
        a, b = r0.get(e), r1[e]
        astr = f"{a:.4f}" if a is not None else "-"
        dstr = f"{b - a:+.4f}" if a is not None else ""
        print(f"{e:>6} {astr:>16} {b:>15.4f} {dstr:>8}")
done_ = [e for e in sorted(r1) if e in r0]
if done_:
    wins = sum(1 for e in done_ if r1[e] > r0[e])
    msg = f"\nrung1 ahead in {wins}/{len(done_)} matched epochs"
    if last in r0:
        msg += f"; delta at epoch {last}: {r1[last] - r0[last]:+.4f}"
    print(msg)
    print(f"rung0 reference: epoch 76 probe {r0[max(r0)]:.4f}, full-eval mAP@10 0.7961")
PY
