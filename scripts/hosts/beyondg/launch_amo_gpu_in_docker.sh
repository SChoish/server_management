#!/usr/bin/env bash
# Compatibility entrypoint: all handoffs go through the common controller.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" -u "$HERE/watch_beyondg_lease.py" --apply "$@"
