#!/usr/bin/env python3
"""Register the desired CPU/GPU queue pair for the running host watcher."""
import argparse
import fcntl
import json
from pathlib import Path

from queue_worker import atomic_json
from watch_beyondg_lease import config_path, load_config


def register(path, cpu, gpu, cwd, processes):
    path = Path(path).resolve()
    cwd = Path(cwd).resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError("cwd must be a directory")

    def script(value):
        value = Path(value)
        value = (value if value.is_absolute() else cwd / value).resolve(strict=True)
        if not value.is_file():
            raise ValueError(f"Not a script file: {value}")
        return str(value)

    cpu, gpu = script(cpu), script(gpu)
    if cpu == gpu:
        raise ValueError("CPU and GPU queue scripts must be different")
    patterns = list(dict.fromkeys(script(value) for value in processes))
    if not patterns:
        raise ValueError("Register the trainer and any detached launcher with --process")
    # Serialize registrations and atomically replace the JSON so the watcher
    # never reads a partially written queue pair. No process is started here.
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        load_config(path=path)
        cfg = json.loads(path.read_text())
        worker = cfg["worker"]
        worker["process_patterns"] = patterns
        for mode, queue in (("cpu", cpu), ("gpu", gpu)):
            job = worker[mode]
            job.update(command=["bash", queue], cwd=str(cwd), queue_patterns=[queue])
            # Environment overrides belong to the previous experiment. Put
            # environment setup in each queue script for the new experiment.
            job.pop("env", None)
        atomic_json(path, cfg)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ext_csh", "ext_csv"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--cpu-queue", required=True, type=Path)
    parser.add_argument("--gpu-queue", required=True, type=Path)
    parser.add_argument("--cwd", required=True, type=Path)
    parser.add_argument("--process", required=True, nargs="+",
                        help="Trainer and launcher paths, absolute or relative to cwd")
    args = parser.parse_args(argv)
    path = config_path(args.profile, args.config)
    try:
        cfg = register(path, args.cpu_queue, args.gpu_queue, args.cwd, args.process)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Registration failed: {error}\n")
    print(f"Registered queue pair in {path}")
    for mode in ("cpu", "gpu"):
        print(f"{mode}: {cfg['worker'][mode]['command']}")
    print("The running host watcher will reload on its next poll (default 30s).")
    print("A different active queue is preserved; replacement waits until it exits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
