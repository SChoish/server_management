#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export EIGEN_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
nvidia-smi >/dev/null
exec "${AMO_PYTHON:-/home/ext_csv/miniconda3/envs/amo-jax/bin/python}" \
  "$(dirname "$0")/run_fql_queue.py" gpu
