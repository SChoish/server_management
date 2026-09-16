#!/usr/bin/env bash
# Keep the Beyond-G lease watcher alive. Portal retry is inside the watcher.
set -uo pipefail
ROOT=/home/ext_csh/MPI_sweep
PY=/home/ext_csh/miniconda3/envs/capo_jax/bin/python
WATCH="$ROOT/scripts/hosts/beyondg/watch_beyondg_lease.py"
LOG="$ROOT/logs/beyondg/lease.watch.log"
PIDFILE="$ROOT/logs/beyondg/lease.watch.pid"
GUARD="$ROOT/logs/beyondg/ensure.pid"

mkdir -p "$ROOT/logs/beyondg"
echo $$ >"$GUARD"

alive() {
  local pid cmd
  pid=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${pid:-}" && -r "/proc/${pid}/cmdline" ]]; then
    cmd=$(tr '\0' ' ' <"/proc/${pid}/cmdline")
    [[ "$cmd" == *watch_beyondg_lease.py* ]] && return 0
  fi
  pgrep -f 'miniconda3/.*/python -u .*/watch_beyondg_lease.py --watch' >/dev/null
}

start_watch() {
  nohup "$PY" -u "$WATCH" --watch --apply --interval 60 >>"$LOG" 2>&1 &
  echo $! >"$ROOT/logs/beyondg/lease.watch.nohup.pid"
}

while true; do
  if ! alive; then
    echo "[ensure] restart watcher $(date -Is)" >>"$LOG"
    start_watch
  fi
  sleep 60
done
