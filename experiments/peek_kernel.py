"""Report the live stage of a running notebook kernel, read-only.

nbconvert buffers every cell's output until the run ends, so there is no on-disk
progress signal at all. This subscribes to the kernel's iopub channel and reports the
most recent progress line it hears. It never calls execute(), so it cannot disturb the
run. Pass a listen budget in seconds (default 45).
"""
import re, subprocess, sys, time

def connection_file():
    ps = subprocess.run(["ps", "-A", "-o", "command="], capture_output=True, text=True).stdout
    for ln in ps.splitlines():
        if "ipykernel_launcher" in ln:
            m = re.search(r"(/\S+\.json)", ln)
            if m:
                return m.group(1)
    return None

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 45.0
cf = connection_file()
if not cf:
    print("NO KERNEL: no ipykernel_launcher process running")
    raise SystemExit(2)

from jupyter_client import BlockingKernelClient
kc = BlockingKernelClient(connection_file=cf)
kc.load_connection_file()
kc.start_channels()
last, cell, t0 = None, None, time.time()
while time.time() - t0 < budget:
    try:
        msg = kc.get_iopub_msg(timeout=3)
    except Exception:
        continue
    mt, c = msg["msg_type"], msg.get("content", {})
    if mt == "stream":
        for ln in (c.get("text") or "").replace("\r", "\n").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith(("|", "#")):
                last = ln
    elif mt == "execute_input":
        src = (c.get("code") or "").strip().splitlines()
        if src:
            cell = src[0][:90]
kc.stop_channels()
if cell:
    print(f"CELL {cell}")
print(f"LAST {last}" if last else "SILENT (no output in this window)")
