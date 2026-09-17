#!/usr/bin/env python3
"""Replace legacy watchers with the common controller without stopping jobs."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from queue_worker import identity, read_process, script_argument
from watch_beyondg_lease import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('ext_csh', 'ext_csv'))
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    cfg = load_config(args.profile, args.config)
    names = {'ensure_beyondg_watchers.sh', 'watch_beyondg_lease.py', 'watch_docker_port.py'}
    stopped = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        proc = read_process(int(entry.name))
        if not proc:
            continue
        script = script_argument(proc['argv'])
        if script and Path(script).name in names and 'beyondg' in script:
            stopped.append(proc)
            try:
                os.kill(proc['signal_pid'], signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 45
    while stopped and time.monotonic() < deadline:
        stopped = [p for p in stopped if (current := read_process(p['pid']))
                   and identity(p) == identity(current)]
        if stopped:
            time.sleep(0.2)
    if stopped:
        raise SystemExit('Previous watcher is still shutting down; retry after it exits.')
    directory = Path(cfg['worker']['state_dir'])
    directory.mkdir(parents=True, exist_ok=True)
    command = ['bash', str(Path(__file__).with_name('ensure_beyondg_watchers.sh'))]
    if args.profile:
        command += ['--profile', args.profile]
    if args.config:
        command += ['--config', str(args.config.resolve())]
    with (directory / 'lease.watch.log').open('ab') as handle:
        proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                                stderr=subprocess.STDOUT, start_new_session=True,
                                env={**os.environ, 'PYTHON': sys.executable})
    print(f'Common watcher supervisor started: pid={proc.pid}; log={directory / "lease.watch.log"}')


if __name__ == '__main__':
    main()
