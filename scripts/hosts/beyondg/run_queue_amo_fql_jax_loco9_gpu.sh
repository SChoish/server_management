#!/usr/bin/env bash
# Resume the live AMO-fql JAX loco9 T-init-5 alpha_lr GPU queues.
# Does nothing if GPU trainers/launchers are already alive.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export EIGEN_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
AMO="/home/ext_csv/AMO-fql"
PY="/home/ext_csv/miniconda3/envs/amo-jax/bin/python"
STORE="/raid/ext_csv/AMO_store"
if pgrep -f "$AMO/train.py" | grep -v grep >/dev/null 2>&1; then
  if ! pgrep -af "$AMO/train.py" | grep -E -- '--device=cpu|--device cpu' >/dev/null 2>&1 \
     || pgrep -af "$AMO/train.py" | grep -E -- '--device=cuda|--device cuda' >/dev/null 2>&1; then
    echo "GPU train.py already running; not starting extra GPU queue"
    exit 0
  fi
fi
if pgrep -f "launch_fql_amo_jax_loco9_tinit5_alrgrid.py" >/dev/null 2>&1; then
  if pgrep -af "launch_fql_amo_jax_loco9_tinit5_alrgrid.py" | grep -v -- '--device=cpu' >/dev/null 2>&1; then
    echo "GPU launcher already running; not starting extra GPU queue"
    exit 0
  fi
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi unavailable here; run via launch_amo_gpu_in_docker.sh" >&2
  exit 1
fi
"$PY" - <<'PY'
from pathlib import Path
import json
import numpy as np

codes = {-1, -2, -15, 129, 130, 143}
root = Path("/raid/ext_csv/AMO_store")
for failed in root.glob("fql_amo_jax_loco9_tinit5_alr*_seeds0to3/jobs/*/FAILED.json"):
    try:
        rec = json.loads(failed.read_text())
        rc = rec.get("return_code")
    except Exception:
        continue
    if rc in codes:
        failed.rename(failed.with_name(f"FAILED.interrupt.{failed.stat().st_mtime_ns}.json"))

for run in root.glob("fql_amo_jax_loco9_tinit5_alr*_seeds0to3/runs/*"):
    ckpt = run / "checkpoint.npz"
    metrics = run / "metrics.jsonl"
    if not ckpt.is_file() or not metrics.is_file():
        continue
    try:
        meta = json.loads(str(np.load(ckpt, allow_pickle=False)["__metadata__"]))
        step = int(meta["steps"])
    except Exception as exc:
        print(f"skip trim {run.name}: {exc}")
        continue
    rows = []
    changed = False
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row_step = int(row.get("step") or 0)
        except Exception:
            rows.append(line)
            continue
        if row_step > step:
            changed = True
            continue
        rows.append(line)
    if changed:
        metrics.write_text("\n".join(rows) + ("\n" if rows else ""))
        print(f"trimmed metrics to step<={step} for {run.name}")
PY
start_one() {
  local config="$1"
  local tag="$2"
  local out="$STORE/fql_amo_jax_loco9_tinit5_alr${tag}_seeds0to3"
  local pid=""
  if [[ -f "$out/launcher_process.json" ]]; then
    pid="$("$PY" -c 'import json,os,sys; from pathlib import Path
p=json.load(open(sys.argv[1])).get("pid",-1)
os.kill(int(p),0)
cmd=Path(f"/proc/{int(p)}/cmdline").read_bytes()
raise SystemExit(0 if b"launch_fql_amo_jax_loco9" in cmd else 1)' "$out/launcher_process.json" 2>/dev/null && echo yes || true)"
  fi
  if [[ -n "$pid" ]]; then
    echo "launcher already alive for $tag"
    return 0
  fi
  "$PY" "$AMO/scripts/launch_fql_amo_jax_loco9_tinit5_alrgrid.py" \
    --gpus 0,1 --max-used-mib 80000 --max-parallel 2 \
    --out "$out" --config "$config" --device cuda:0 --retry-failed --detach
}
start_one "$AMO/configs/fql_amo_tinit5_alr3e-4.yaml" "3e-4"
start_one "$AMO/configs/fql_amo_tinit5_alr1e-3.yaml" "1e-3"
start_one "$AMO/configs/fql_amo_tinit5_alr2e-3.yaml" "2e-3"
echo "GPU queues requested"
