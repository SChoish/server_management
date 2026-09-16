#!/usr/bin/env bash
# Keep the Beyond-G lease watcher alive. Portal retry is inside the watcher.
set -uo pipefail
ROOT=/home/ext_csv/MPI_sweep
PY="${PYTHON:-python3}"
WATCH="$ROOT/scripts/hosts/beyondg/watch_beyondg_lease.py"
PORT_WATCH="$ROOT/scripts/hosts/beyondg/watch_docker_port.py"
LOG="$ROOT/logs/beyondg/lease.watch.log"
PIDFILE="$ROOT/logs/beyondg/lease.watch.pid"
GUARD="$ROOT/logs/beyondg/ensure.pid"

mkdir -p "$ROOT/logs/beyondg"
echo $$ >"$GUARD"

alive() {
  local pidfile="$1" needle="$2" pid cmd
  pid=$(cat "$pidfile" 2>/dev/null || true)
  if [[ -n "${pid:-}" && -r "/proc/${pid}/cmdline" ]]; then
    cmd=$(tr '\0' ' ' <"/proc/${pid}/cmdline")
    [[ "$cmd" == *"$needle"* ]] && return 0
  fi
  pgrep -f "$needle --watch" >/dev/null
}

start_watch() {
  nohup "$PY" -u "$WATCH" --watch --apply --interval 60 >>"$LOG" 2>&1 &
  echo $! >"$ROOT/logs/beyondg/lease.watch.nohup.pid"
}

start_port() {
  nohup "$PY" -u "$PORT_WATCH" --watch --interval 60 >>"$ROOT/logs/beyondg/docker_port.stdout" 2>&1 &
  echo $! >"$ROOT/logs/beyondg/docker_port.pid"
}

while true; do
  if ! alive "$PIDFILE" watch_beyondg_lease.py; then
    echo "[ensure] restart watcher $(date -Is)" >>"$LOG"
    start_watch
  fi
  if ! alive "$ROOT/logs/beyondg/docker_port.pid" watch_docker_port.py; then
    echo "[ensure] restart docker port watcher $(date -Is)" >>"$LOG"
    start_port
  fi
  sleep 60
done
