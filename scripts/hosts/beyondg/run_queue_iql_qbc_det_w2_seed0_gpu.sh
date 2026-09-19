#!/usr/bin/env bash
# GPU watcher entry for IQL qbc_deterministic_w2 seed0 T-sweep then seeds 1-3.
# Always (re)starts the canvas/git 5m loop (canvas + origin/main STATUS push).
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export EIGEN_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/home/ext_csv/miniconda3/envs/offrl/bin/python
QUEUE=/home/ext_csv/MPI_sweep/scripts/hosts/beyondg/run_queue_iql_qbc_det_w2_seed0.sh
LOOP=/home/ext_csv/MPI_sweep/scripts/refresh_iql_qbc_det_w2_5m.sh
LOGDIR=/raid/ext_csv/MPI_store/iql_qbc_deterministic_w2_hscale_seed0/logs
mkdir -p "$LOGDIR" /home/ext_csv/logs

start_canvas_git_loop() {
  nohup bash "$LOOP" >>/home/ext_csv/logs/iql_qbc_det_w2_seed0_canvas_5m.nohup 2>&1 &
  echo "canvas/git 5m loop requested pid=$!"
}

start_canvas_git_loop

if "$PY" - <<'PY'
from pathlib import Path
needles = (
    b"run_queue_iql_qbc_det_w2_seed0.sh",
    b"qbc_deterministic_w2",
)
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
    if b"IQL_QBC_DET_DEVICE=cpu" in env or b"JAX_PLATFORMS=cpu" in env:
        continue
    if any(n in raw for n in needles) and (
        b"launch_mpi_sweep.py" in raw or b"train_iql_mpi.py" in raw
        or b"run_queue_iql_qbc_det_w2_seed0.sh" in raw
    ):
        print(f"gpu queue already live pid={proc.name}")
        raise SystemExit(0)
raise SystemExit(1)
PY
then
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi unavailable; GPU queue not started" >&2
  exit 1
fi

unset CUDA_VISIBLE_DEVICES JAX_PLATFORMS JAX_PLATFORM_NAME
export IQL_QBC_DET_DEVICE=cuda
nohup bash "$QUEUE" >>"$LOGDIR/queue.log" 2>&1 &
echo "iql qbc_deterministic_w2 seed0 GPU queue requested pid=$!"
