#!/usr/bin/env bash
# CPU failover for IQL qbc_deterministic_w2 seed0 T-sweep then seeds 1-3. Same save dir; resume ckpts.
# Starts the canvas/git 5m loop (canvas + origin/main STATUS push) on GPU-loss host failover.
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export EIGEN_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu IQL_QBC_DET_DEVICE=cpu
PY=/home/ext_csv/miniconda3/envs/offrl/bin/python
QUEUE=/home/ext_csv/MPI_sweep/scripts/hosts/beyondg/run_queue_iql_qbc_det_w2_seed0.sh
LOOP=/home/ext_csv/MPI_sweep/scripts/refresh_iql_qbc_det_w2_5m.sh
LOGDIR=/raid/ext_csv/MPI_store/iql_qbc_deterministic_w2_hscale_seed0/logs
mkdir -p "$LOGDIR" /home/ext_csv/logs

start_canvas_git_loop() {
  nohup bash "$LOOP" >>/home/ext_csv/logs/iql_qbc_det_w2_seed0_canvas_5m.nohup 2>&1 &
  echo "canvas/git 5m loop requested pid=$!"
}

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
    if b"run_queue_iql_qbc_det_w2_seed0_gpu.sh" in raw:
        continue
    env = b""
    env_path = proc / "environ"
    if env_path.exists():
        try:
            env = env_path.read_bytes()
        except OSError:
            env = b""
    gpuish = (
        b"launch_mpi_sweep.py" in raw or b"train_iql_mpi.py" in raw
        or b"run_queue_iql_qbc_det_w2_seed0.sh" in raw
    ) and b"qbc_deterministic_w2" in raw or b"run_queue_iql_qbc_det_w2_seed0.sh" in raw
    if not gpuish:
        continue
    if b"IQL_QBC_DET_DEVICE=cpu" in env or b"JAX_PLATFORMS=cpu" in env or b"--cpu-jobs" in raw:
        continue
    print(f"GPU qbc_det still running pid={proc.name}", file=sys.stderr)
    raise SystemExit(0)
raise SystemExit(1)
PY
then
  echo "GPU qbc_det still running; refusing CPU overlap" >&2
  exit 1
fi

start_canvas_git_loop

if "$PY" - <<'PY'
from pathlib import Path
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit():
        continue
    try:
        raw = (proc / "cmdline").read_bytes()
    except OSError:
        continue
    if b"snap=$(command cat" in raw:
        continue
    env = b""
    env_path = proc / "environ"
    if env_path.exists():
        try:
            env = env_path.read_bytes()
        except OSError:
            env = b""
    if b"run_queue_iql_qbc_det_w2_seed0.sh" in raw and (
        b"IQL_QBC_DET_DEVICE=cpu" in env or b"JAX_PLATFORMS=cpu" in env
    ):
        print(f"cpu queue already live pid={proc.name}")
        raise SystemExit(0)
    if b"launch_mpi_sweep.py" in raw and b"qbc_deterministic_w2" in raw and b"--cpu-jobs" in raw:
        print(f"cpu launcher already live pid={proc.name}")
        raise SystemExit(0)
raise SystemExit(1)
PY
then
  exit 0
fi

nohup bash "$QUEUE" >>"$LOGDIR/queue.log" 2>&1 &
echo "iql qbc_deterministic_w2 seed0 CPU queue requested pid=$!"
