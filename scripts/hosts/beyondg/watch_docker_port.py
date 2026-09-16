#!/usr/bin/env python3
"""When Beyond-G queue publishes a docker SSH port, launch GPU inside it.

Write the published port to logs/beyondg/docker.port (one integer).
After a successful launch, record last_port so the same mapping does not
fire twice. The port file is kept so the lease watcher can still SSH.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/ext_csv/MPI_sweep")
STATE = ROOT / "logs" / "beyondg"
PORT_FILE = STATE / "docker.port"
LAUNCH = ROOT / "scripts" / "hosts" / "beyondg" / "launch_me_gpu_in_docker.sh"


def read_port(path: Path) -> int | None:
    if not path.is_file():
        return None
    raw = path.read_text(encoding="utf-8").strip().splitlines()
    if not raw:
        return None
    try:
        port = int(raw[0].strip())
    except ValueError:
        return None
    if not 1 <= port <= 65535:
        return None
    return port


def already_launched(port: int) -> bool:
    last = STATE / "last_port"
    return last.is_file() and last.read_text(encoding="utf-8").strip() == str(port)


def tick(*, apply: bool) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    port = read_port(PORT_FILE)
    if port is None:
        print("[beyondg] waiting for", PORT_FILE, flush=True)
        return 0
    if already_launched(port):
        print(f"[beyondg] port {port} already launched", flush=True)
        return 0
    print(f"[beyondg] saw docker port {port}", flush=True)
    if not apply:
        return 0
    env = os.environ.copy()
    env["BEYONDG_SSH_PORT"] = str(port)
    env.setdefault("BEYONDG_HOST", "166.104.28.73")
    env.setdefault("BEYONDG_SSH_USER", "ext_csv")
    rc = subprocess.call(["bash", str(LAUNCH)], env=env)
    if rc == 0:
        (STATE / "last_port").write_text(f"{port}\n", encoding="utf-8")
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.watch:
        while True:
            tick(apply=bool(args.apply))
            time.sleep(max(5, args.interval))
    return tick(apply=bool(args.apply))


if __name__ == "__main__":
    raise SystemExit(main())
