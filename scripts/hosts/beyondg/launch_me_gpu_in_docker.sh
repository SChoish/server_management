#!/usr/bin/env bash
# SSH into the allocated Beyond-G docker (host:published-port) and resume ME GPU.
# Soft-stops the local CPU ME queue first so the same cells are not dual-run.
set -uo pipefail

HOST="${BEYONDG_HOST:-127.0.0.1}"
PORT="${BEYONDG_SSH_PORT:-}"
USER="${BEYONDG_SSH_USER:-ext_csh}"
REMOTE_ROOT="${BEYONDG_REMOTE_ROOT:-/home/ext_csh/MPI_sweep}"
REMOTE_CMD="${BEYONDG_REMOTE_CMD:-bash ${REMOTE_ROOT}/logs/mpi_tau40_ext/run_queue_imp_me_k14_s0123_gpu.sh}"
STATE=/home/ext_csh/MPI_sweep/logs/beyondg
CPU_PIDFILE=/home/ext_csh/MPI_sweep/sweep_results/diagnostics/gpu_idle_failover/cpu_queue.pid
KEY="${BEYONDG_SSH_KEY:-/home/ext_csh/.ssh/gpubox_host}"

if [[ -z "$PORT" ]]; then
  echo "set BEYONDG_SSH_PORT (docker published SSH port)" >&2
  exit 2
fi

mkdir -p "$STATE"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -p "$PORT")
if [[ -n "$KEY" ]]; then
  ssh_opts+=(-i "$KEY" -o IdentitiesOnly=yes)
fi

echo "[beyondg] probe ${USER}@${HOST}:${PORT}"
if ! ssh "${ssh_opts[@]}" "${USER}@${HOST}" "test -x ${REMOTE_ROOT}/logs/mpi_tau40_ext/run_queue_imp_me_k14_s0123_gpu.sh"; then
  echo "[beyondg] ssh or remote script missing" >&2
  exit 3
fi

if [[ -f "$CPU_PIDFILE" ]]; then
  cpu_pid=$(cat "$CPU_PIDFILE" || true)
  if [[ -n "${cpu_pid:-}" && -d "/proc/${cpu_pid}" ]]; then
    echo "[beyondg] SIGTERM local CPU queue pid=${cpu_pid}"
    kill -TERM "$cpu_pid" || true
    for _ in $(seq 1 120); do
      pgrep -u ext_csh -f 'train_td3bc.py' >/dev/null || break
      sleep 2
    done
  fi
fi

echo "[beyondg] start remote ME GPU queue"
ssh "${ssh_opts[@]}" "${USER}@${HOST}" \
  "pgrep -f 'python -u .*train_td3bc.py' >/dev/null \
   || pgrep -f 'python -u .*launch_mpi_sweep.py' >/dev/null \
   || pgrep -f 'bash .*/run_queue_imp_me_k14_s0123_gpu.sh' >/dev/null \
   && echo already && exit 0; \
   setsid nohup ${REMOTE_CMD} >> ${REMOTE_ROOT}/logs/mpi_tau40_ext/queue_imp_me_k14_docker.log 2>&1 < /dev/null & echo \$!"
echo "$PORT" >"$STATE/last_port"
date -Is >"$STATE/last_launch"
