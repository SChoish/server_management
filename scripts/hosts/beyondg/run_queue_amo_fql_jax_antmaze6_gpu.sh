#!/usr/bin/env bash
# Start FQL+AMO JAX antmaze6 T-init-5 alpha_lr GPU queues (priority over loco9).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export EIGEN_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
AMO="/home/ext_csv/AMO-fql"
PY="/home/ext_csv/miniconda3/envs/amo-jax/bin/python"
STORE="/raid/ext_csv/AMO_store"

"$PY" - <<'PY'
from pathlib import Path
import sys

def real_cmd(cmd: bytes) -> bool:
    # Ignore Cursor/tool wrapper shells that embed the needle in their argv.
    if b"snap=$(command cat" in cmd or b"bash -O extglob" in cmd:
        return False
    return True

for pid_dir in Path("/proc").iterdir():
    if not pid_dir.name.isdigit():
        continue
    try:
        cmd = (pid_dir / "cmdline").read_bytes()
    except OSError:
        continue
    if not real_cmd(cmd):
        continue
    if b"launch_fql_amo_jax_loco9_tinit5_alrgrid.py" in cmd and b"--device=cpu" not in cmd and b"--device cpu" not in cmd:
        print("loco9 GPU launcher still running; free GPUs before antmaze priority", file=sys.stderr)
        raise SystemExit(1)
    if b"AMO-fql/train.py" in cmd and b"fql_amo_jax_loco9_tinit5" in cmd and b"--device=cpu" not in cmd:
        print("loco9 GPU train.py still running; free GPUs before antmaze priority", file=sys.stderr)
        raise SystemExit(1)
PY

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
      echo "launcher pid still alive for $tag; not starting extra"
      return 0
    fi
  fi
  "$PY" "$AMO/scripts/launch_fql_amo_jax_antmaze6_tinit5_alrgrid.py" \
    --gpus 0,1 --max-used-mib 80000 --max-parallel 2 \
    --out "$out" --config "$config" --device cuda:0 --retry-failed --detach
}
start_one "$AMO/configs/fql_amo_tinit5_alr3e-4.yaml" "3e-4"
start_one "$AMO/configs/fql_amo_tinit5_alr1e-3.yaml" "1e-3"
start_one "$AMO/configs/fql_amo_tinit5_alr2e-3.yaml" "2e-3"
echo "antmaze6 GPU queues requested"
