#!/usr/bin/env bash
# SSH into the allocated Beyond-G docker and resume AMO GPU.
# Soft-stops the local CPU queue first so the same cells are not dual-run.
set -uo pipefail

HOST="${BEYONDG_HOST:-${BEYONDG_DOCKER_HOST:-166.104.28.73}}"
PORT="${BEYONDG_SSH_PORT:-${BEYONDG_DOCKER_PORT:-}}"
USER="${BEYONDG_SSH_USER:-ext_csv}"
REMOTE_ROOT="${BEYONDG_REMOTE_ROOT:-/home/ext_csv/MPI_sweep}"
REMOTE_CMD="${BEYONDG_REMOTE_CMD:-bash ${REMOTE_ROOT}/scripts/hosts/beyondg/run_queue_amo_fql_jax_loco9_gpu.sh}"
STATE=/home/ext_csv/MPI_sweep/logs/beyondg
CPU_PIDFILE="$STATE/cpu_queue.pid"
KEY="${BEYONDG_SSH_KEY:-/home/ext_csv/.ssh/gpubox_host}"
KNOWN="${BEYONDG_KNOWN_HOSTS:-/home/ext_csv/.ssh/gpubox_known_hosts}"

if [[ -z "$PORT" && -f "$STATE/docker.port" ]]; then
  PORT="$(tr -d '[:space:]' <"$STATE/docker.port")"
fi
if [[ -z "$PORT" ]]; then
  echo "set BEYONDG_SSH_PORT (docker published SSH port)" >&2
  exit 2
fi

mkdir -p "$STATE"
ssh_opts=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new
          -o "UserKnownHostsFile=$KNOWN" -p "$PORT")
if [[ -n "$KEY" ]]; then
  ssh_opts+=(-i "$KEY" -o IdentitiesOnly=yes)
fi

echo "[beyondg] probe ${USER}@${HOST}:${PORT}"
if ! ssh "${ssh_opts[@]}" "${USER}@${HOST}" "echo gpubox-ok" | grep -q gpubox-ok; then
  ssh-keygen -f "$KNOWN" -R "[$HOST]:$PORT" >/dev/null 2>&1 || true
  if ! ssh "${ssh_opts[@]}" "${USER}@${HOST}" "echo gpubox-ok" | grep -q gpubox-ok; then
    echo "[beyondg] ssh missing" >&2
    exit 3
  fi
fi

if [[ -f "$CPU_PIDFILE" ]]; then
  cpu_pid=$(cat "$CPU_PIDFILE" || true)
  if [[ -n "${cpu_pid:-}" && -d "/proc/${cpu_pid}" ]]; then
    echo "[beyondg] SIGTERM local CPU queue pid=${cpu_pid}"
    kill -TERM "$cpu_pid" || true
    for _ in $(seq 1 120); do
      pgrep -u ext_csv -f 'AMO-fql/train.py' | grep -E -- '--device=cpu|--device cpu' >/dev/null || break
      sleep 2
    done
  fi
fi

echo "[beyondg] start remote AMO GPU queue"
ssh "${ssh_opts[@]}" "${USER}@${HOST}" "setsid nohup ${REMOTE_CMD} >> ${STATE}/queue_amo_fql_docker.log 2>&1 < /dev/null & echo \$!"
echo "$PORT" >"$STATE/last_port"
date -Is >"$STATE/last_launch"
