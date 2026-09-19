#!/usr/bin/env bash
# CPU failover for the live AMO-fql JAX antmaze6 T-init-5 queues.
# Refuses to start if GPU trainers still own the same results cells.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_THREADS=1
export MKL_NUM_THREADS=1 EIGEN_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
AMO="/home/ext_csv/AMO-fql"
PY="/home/ext_csv/miniconda3/envs/amo-jax/bin/python"
STORE="/raid/ext_csv/AMO_store"
if pgrep -af "$AMO/train.py" | grep -E -- '--device=cuda|--device cuda' >/dev/null 2>&1; then
  echo "GPU train.py still running; refusing CPU overlap on the same cells" >&2
  exit 1
fi
if pgrep -af "launch_fql_amo_jax_antmaze6_tinit5_alrgrid.py" | grep -v -- '--device=cpu' | grep -v grep >/dev/null 2>&1; then
  echo "GPU launcher still running; refusing CPU overlap" >&2
  exit 1
fi
"$PY" - <<'PY'
from pathlib import Path
import json
codes = {-1, -2, -15, 129, 130, 143}
root = Path("/raid/ext_csv/AMO_store")
for failed in root.glob("fql_amo_jax_antmaze6_tinit5_alr*_seeds0to3/jobs/*/FAILED.json"):
    try:
        rec = json.loads(failed.read_text())
        rc = rec.get("return_code")
    except Exception:
        continue
    if rc in codes:
        failed.rename(failed.with_name(f"FAILED.interrupt.{failed.stat().st_mtime_ns}.json"))
PY
start_one() {
  local config="$1"
  local tag="$2"
  local out="$STORE/fql_amo_jax_antmaze6_tinit5_alr${tag}_seeds0to3"
  if [[ -f "$out/launcher_process.json" ]]; then
    if "$PY" -c 'import json,os,sys; from pathlib import Path
p=json.load(open(sys.argv[1])).get("pid",-1)
os.kill(int(p),0)
cmd=Path(f"/proc/{int(p)}/cmdline").read_bytes()
raise SystemExit(0 if b"launch_fql_amo_jax_antmaze6" in cmd else 1)' "$out/launcher_process.json" 2>/dev/null; then
      echo "CPU/GPU launcher pid still alive for $tag; not starting extra"
      return 0
    fi
  fi
  "$PY" "$AMO/scripts/launch_fql_amo_jax_antmaze6_tinit5_alrgrid.py" \
    --gpus 0,1 --max-used-mib 80000 --max-parallel 2 \
    --out "$out" --config "$config" --device cpu --retry-failed --detach
}
start_one "$AMO/configs/fql_amo_tinit5_alr3e-4.yaml" "3e-4"
start_one "$AMO/configs/fql_amo_tinit5_alr1e-3.yaml" "1e-3"
start_one "$AMO/configs/fql_amo_tinit5_alr2e-3.yaml" "2e-3"
echo "CPU queues requested"
