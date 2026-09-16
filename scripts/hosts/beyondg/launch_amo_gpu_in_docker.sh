#!/usr/bin/env bash
# Start the live AMO-fql GPU queues inside the published GPU container.
# No-op if those queues/trainers are already running.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
LOG_DIR="$REPO/logs/beyondg"
QUEUE="$REPO/scripts/hosts/beyondg/run_queue_amo_fql_jax_loco9_gpu.sh"
HOST="${BEYONDG_DOCKER_HOST:-166.104.28.73}"
USER="${BEYONDG_SSH_USER:-ext_csv}"
KEY="${BEYONDG_SSH_KEY:-$HOME/.ssh/gpubox_host}"
KNOWN="$HOME/.ssh/gpubox_known_hosts"
PORT="${BEYONDG_DOCKER_PORT:-}"
if [[ -z "$PORT" && -f "$LOG_DIR/docker.port" ]]; then
  PORT="$(tr -d '[:space:]' <"$LOG_DIR/docker.port")"
fi
if [[ -z "$PORT" ]]; then
  echo "docker ssh port unknown (set BEYONDG_DOCKER_PORT or logs/beyondg/docker.port)" >&2
  exit 1
fi
ssh_cmd() {
  ssh -p "$PORT" \
    -o BatchMode=yes \
    -o ConnectTimeout=8 \
    -o "UserKnownHostsFile=$KNOWN" \
    -o StrictHostKeyChecking=accept-new \
    -o IdentitiesOnly=yes \
    ${KEY:+-i "$KEY"} \
    "$USER@$HOST" \
    "$@"
}
if ! ssh_cmd 'echo gpubox-ok' | grep -q gpubox-ok; then
  ssh-keygen -f "$KNOWN" -R "[$HOST]:$PORT" >/dev/null 2>&1 || true
  ssh_cmd 'echo gpubox-ok' | grep -q gpubox-ok
fi
ssh_cmd "bash $QUEUE"
