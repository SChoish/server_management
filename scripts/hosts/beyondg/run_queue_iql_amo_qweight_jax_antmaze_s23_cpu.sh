#!/usr/bin/env bash
# CPU failover for IQL-AMO qweight AntMaze seeds 2–3. Same store, resume from checkpoint.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export EIGEN_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
PY="/home/ext_csv/miniconda3/envs/amo-jax/bin/python"
LAUNCH="/home/ext_csv/AMO-main/scripts/launch_iql_amo_qweight_jax_beta125_rho_loco_antmaze.py"
OUT="/raid/ext_csv/AMO_store/iql_amo_qweight_jax_beta125_rho_loco_antmaze_seeds0to3"

if "$PY" - <<'PY'
from pathlib import Path
import sys

for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        raw = (proc / "cmdline").read_bytes()
    except OSError:
        continue
    if b"snap=$(command cat" in raw or b"bash -O extglob" in raw:
        continue
    cmd = raw.replace(b"\0", b" ")
    if b"iql_amo_qweight" not in cmd and b"launch_iql_amo_qweight" not in cmd:
        continue
    if b"--device=cpu" in cmd or b"--device cpu" in cmd:
        continue
    if b"--device=cuda" in cmd or b"--device cuda" in cmd:
        print(f"GPU qweight still running pid={proc.name}", file=sys.stderr)
        raise SystemExit(0)
raise SystemExit(1)
PY
then
  echo "GPU qweight still running; refusing CPU overlap" >&2
  exit 1
fi

if "$PY" - <<'PY'
from pathlib import Path
needle = b"launch_iql_amo_qweight_jax_beta125_rho_loco_antmaze.py"
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        raw = (proc / "cmdline").read_bytes()
    except OSError:
        continue
    if b"snap=$(command cat" in raw or b"bash -O extglob" in raw:
        continue
    cmd = raw.replace(b"\0", b" ")
    if needle in cmd and b"--seeds 2,3" in cmd and b"--device cpu" in cmd:
        print(f"cpu launcher already live pid={proc.name}")
        raise SystemExit(0)
raise SystemExit(1)
PY
then
  exit 0
fi

nohup "$PY" "$LAUNCH" \
  --detach \
  --gpus 0,1 \
  --max-used-mib 80000 \
  --max-parallel 6 \
  --device cpu \
  --seeds 2,3 \
  --env-substr antmaze \
  --out "$OUT" \
  >>"$OUT/launcher.log" 2>&1 &
echo "antmaze seeds 2-3 CPU queue requested"
