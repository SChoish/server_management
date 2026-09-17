#!/usr/bin/env bash
# One supervisor and one controller per profile, including across checkouts.
set -uo pipefail
umask 077
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
WATCH="$HERE/watch_beyondg_lease.py"
STATE="$("$PY" "$WATCH" --print-state-dir "$@")" || exit 1
mkdir -p "$STATE"
exec 9>"$STATE/ensure.lock"
flock -n 9 || exit 0
echo $$ >"$STATE/ensure.pid"
child=""
cleanup() {
  trap - TERM INT
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" 2>/dev/null || true
  fi
  rm -f "$STATE/ensure.pid"
  exit 0
}
trap cleanup TERM INT
while true; do
  "$PY" -u "$WATCH" --watch --apply "$@" 9>&- &
  child=$!
  wait "$child" || true
  child=""
  sleep 5
done
