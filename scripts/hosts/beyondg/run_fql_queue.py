#!/usr/bin/env python3
"""Resume the existing three FQL sweeps; the launcher owns checkpoints/jobs."""
import argparse
import json
import os
import subprocess
from pathlib import Path

from queue_worker import option, read_process, script_argument


def launcher_alive(launcher, output):
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        proc = read_process(int(entry.name))
        if not proc:
            continue
        script = script_argument(proc['argv'])
        if script and Path(script).name == launcher.name:
            if option(proc['argv'], '--out') == str(output):
                return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('cpu', 'gpu'))
    args = parser.parse_args()
    amo = Path(os.environ.get('AMO_ROOT', '/home/ext_csv/AMO-fql'))
    store = Path(os.environ.get('AMO_STORE', '/raid/ext_csv/AMO_store'))
    python = os.environ.get('AMO_PYTHON', '/home/ext_csv/miniconda3/envs/amo-jax/bin/python')
    launcher = amo / 'scripts/launch_fql_amo_jax_loco9_tinit5_alrgrid.py'
    if not launcher.is_file():
        raise SystemExit(f'Missing experiment launcher: {launcher}')
    for tag in ('3e-4', '1e-3', '2e-3'):
        output = store / f'fql_amo_jax_loco9_tinit5_alr{tag}_seeds0to3'
        if launcher_alive(launcher, output):
            continue
        # Preserve interrupted jobs for the launcher's retry/resume path. Do not
        # remove checkpoints or rewrite metrics from another active job.
        for failed in output.glob('jobs/*/FAILED.json'):
            try:
                record = json.loads(failed.read_text())
                if record.get('return_code') in {-1, -2, -15, 129, 130, 143}:
                    failed.rename(failed.with_name(f'FAILED.interrupt.{failed.stat().st_mtime_ns}.json'))
            except (OSError, ValueError):
                continue
        subprocess.run([
            python, str(launcher), '--gpus', '0,1', '--max-used-mib', '80000',
            '--max-parallel', '2', '--out', str(output),
            '--config', str(amo / f'configs/fql_amo_tinit5_alr{tag}.yaml'),
            '--device', 'cpu' if args.mode == 'cpu' else 'cuda:0',
            '--retry-failed', '--detach',
        ], check=True, cwd=amo)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
