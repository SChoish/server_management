#!/usr/bin/env python3
"""Small stdlib-only process supervisor, also sent over SSH on stdin."""
from __future__ import annotations

import fcntl
import fnmatch
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_CHILDREN = []


def reap_children():
    _CHILDREN[:] = [child for child in _CHILDREN if child.poll() is None]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def option(argv, name):
    for i, arg in enumerate(argv):
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
        if arg == name and i + 1 < len(argv):
            return argv[i + 1]
    return None


def script_argument(argv):
    """Inspect executable/script arguments, never shell/Python command text."""
    if not argv:
        return None
    name = Path(argv[0]).name
    if name.startswith(("python", "pypy")) or name in {"bash", "sh", "dash", "zsh"}:
        for arg in argv[1:]:
            if arg in {"-c", "-m", "-", "--command"}:
                return None
            if not arg.startswith("-"):
                return arg
        return None
    return argv[0]


def classify(argv, env, cfg):
    script = script_argument(argv)
    if script is None:
        return None
    for mode in ("cpu", "gpu"):
        if any(fnmatch.fnmatch(script, p) for p in cfg[mode]["queue_patterns"]):
            return mode
    if not any(fnmatch.fnmatch(script, p) for p in cfg["process_patterns"]):
        return None
    device = option(argv, "--device") or ""
    cpu_jobs = option(argv, "--cpu-jobs")
    try:
        cpu_jobs = float(cpu_jobs or 0) > 0
    except ValueError:
        cpu_jobs = False
    if (device.startswith("cpu") or cpu_jobs or env.get("JAX_PLATFORMS") == "cpu"
            or env.get("JAX_PLATFORM_NAME") == "cpu"
            or env.get("CUDA_VISIBLE_DEVICES") == ""):
        return "cpu"
    return "gpu"


def read_process(pid):
    root = Path("/proc") / str(pid)
    try:
        if root.stat().st_uid != os.getuid():
            return None
        status = (root / "status").read_text().splitlines()
        ids = next((x.split()[1:] for x in status if x.startswith("NSpid:")), [str(pid)])
        own_ids = next((x.split()[1:] for x in Path("/proc/self/status").read_text().splitlines()
                        if x.startswith("NSpid:")), [str(os.getpid())])
        depth = len(own_ids)
        if len(ids) < depth:
            return None
        # /proc may be host-mounted in a PID namespace. Never signal a sibling
        # namespace using a coincidentally equal PID. On the host (depth == 1),
        # its /proc PIDs also address children inside Docker correctly.
        if depth > 1 and os.readlink(root / "ns/pid") != namespace():
            return None
        signal_pid = int(ids[depth - 1])
        raw = (root / "stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] in {"Z", "X"}:
            return None
        argv = (root / "cmdline").read_bytes().decode(errors="replace").split("\0")
        argv = [x for x in argv if x]
        if not argv:
            return None
        try:
            environ = (root / "environ").read_bytes().decode(errors="replace")
            env = dict(x.split("=", 1) for x in environ.split("\0") if "=" in x)
        except (PermissionError, FileNotFoundError):
            env = {}
        return {"pid": int(pid), "signal_pid": signal_pid,
                "ppid": int(fields[1]), "start": fields[19],
                "argv": argv, "env": env}
    except (OSError, ValueError, IndexError):
        return None


def identity(proc):
    return f"{proc['pid']}:{proc['start']}"


def namespace():
    return os.readlink("/proc/self/ns/pid")


def own_proc_pid():
    return int(Path("/proc/self").resolve().name)


def launched_process(pid):
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            proc = read_process(int(entry.name))
            if proc and proc["signal_pid"] == pid and proc["ppid"] == own_proc_pid():
                return proc
    return None


def snapshot(cfg, record):
    procs = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            proc = read_process(int(entry.name))
            if proc:
                procs[proc["pid"]] = proc
    # Never select this worker or the SSH/controller/shell that invoked it.
    excluded = set()
    pid = own_proc_pid()
    while pid in procs and pid not in excluded:
        excluded.add(pid)
        pid = procs[pid]["ppid"]
    # Long-lived result-sync helpers are not experiment workers. Exclude their
    # descendants as well, even if an earlier snapshot tracked those PIDs.
    auxiliary = {pid for pid, proc in procs.items()
                 if (script := script_argument(proc["argv"])) is not None
                 and any(fnmatch.fnmatch(script, p) for p in cfg.get("ignore_patterns", []))}
    while True:
        children = {pid for pid, proc in procs.items() if proc["ppid"] in auxiliary}
        if children <= auxiliary:
            break
        auxiliary.update(children)
    excluded.update(auxiliary)
    groups = {"cpu": set(), "gpu": set()}
    for pid, proc in procs.items():
        if pid in excluded:
            continue
        mode = classify(proc["argv"], proc["env"], cfg)
        if mode:
            groups[mode].add(pid)
    if record.get("namespace") == namespace():
        for mode in groups:
            known = set(record.get(mode, {}).get("tracked", []))
            groups[mode].update(pid for pid, p in procs.items()
                                if pid not in excluded and identity(p) in known)
    # Track children before stopping a shell/launcher so orphaned trainers remain
    # visible even after the parent exits or the watcher is restarted.
    for mode, pids in groups.items():
        while True:
            children = {pid for pid, p in procs.items()
                        if p["ppid"] in pids and pid not in excluded}
            if children <= pids:
                break
            pids.update(children)
    return {mode: [procs[p] for p in sorted(pids)] for mode, pids in groups.items()}


def gpu_available():
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return proc.returncode == 0 and bool(proc.stdout.strip())
    except (OSError, subprocess.TimeoutExpired):
        return False


def supervise(cfg, action, mode, *, require_gpu=False):
    reap_children()
    directory = Path(cfg["state_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    # Per-PID-namespace records prevent host/container PID-number collisions.
    suffix = namespace().replace(":", "_").replace("[", "").replace("]", "")
    record_path = directory / f"workers.{suffix}.json"
    with (directory / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        groups = snapshot(cfg, record)
        for key in groups:
            record.setdefault(key, {})
            record[key]["tracked"] = [identity(p) for p in groups[key]]
        record["namespace"] = namespace()
        available = gpu_available() if require_gpu else None
        result = {"ok": True, "gpu_available": available}
        if action == "stop":
            sent = set(record[mode].get("signalled", []))
            for proc in groups[mode]:
                token = identity(proc)
                current = read_process(proc["pid"])
                if token in sent or not current or identity(current) != token:
                    continue
                try:
                    os.kill(proc["signal_pid"], signal.SIGTERM)
                    sent.add(token)
                except ProcessLookupError:
                    pass
            record[mode]["signalled"] = sorted(sent)
            groups = snapshot(cfg, record)
            result["stopped"] = not groups[mode]
        elif action == "start":
            other = "gpu" if mode == "cpu" else "cpu"
            if groups[other]:
                result.update(ok=False, error=f"{other}_still_running")
            elif groups[mode]:
                result["already_running"] = True
            elif require_gpu and not available:
                result.update(ok=False, error="gpu_not_available")
            elif time.time() - record[mode].get("last_start", 0) < cfg.get("start_retry_s", 30):
                result.update(ok=False, error="start_cooldown")
            else:
                record[mode]["signalled"] = []
                record[mode]["last_start"] = time.time()
                job = cfg[mode]
                env = os.environ.copy()
                env.update(job.get("env", {}))
                # Children keep an explicit device environment for recognition.
                if mode == "cpu":
                    env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu")
                else:
                    if env.get("CUDA_VISIBLE_DEVICES") == "":
                        env.pop("CUDA_VISIBLE_DEVICES")
                    if env.get("JAX_PLATFORMS") == "cpu":
                        env.pop("JAX_PLATFORMS")
                    if env.get("JAX_PLATFORM_NAME") == "cpu":
                        env.pop("JAX_PLATFORM_NAME")
                logfile = Path(job["log"])
                logfile.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with logfile.open("ab") as handle:
                        child = subprocess.Popen(
                            job["command"], cwd=job["cwd"], env=env,
                            stdin=subprocess.DEVNULL, stdout=handle,
                            stderr=subprocess.STDOUT, start_new_session=True,
                        )
                    _CHILDREN.append(child)
                    child_proc = launched_process(child.pid)
                    if child_proc:
                        record[mode]["tracked"].append(identity(child_proc))
                    # Detect immediate command/path errors; do not claim a queued
                    # shell spawn proves that training successfully resumed.
                    time.sleep(0.1)
                    result["launch_returncode"] = child.poll()
                    result["start_requested"] = child.returncode in (None, 0)
                    if child.returncode not in (None, 0):
                        result.update(ok=False, error="queue_launch_failed")
                    groups = snapshot(cfg, record)
                except OSError as error:
                    result.update(ok=False, error=f"queue_launch:{error.__class__.__name__}")
        elif action != "status":
            raise ValueError(f"Unknown worker action: {action}")
        for key in groups:
            record[key]["tracked"] = [identity(p) for p in groups[key]]
        atomic_json(record_path, record)
        result["cpu"] = [identity(p) for p in groups["cpu"]]
        result["gpu"] = [identity(p) for p in groups["gpu"]]
        return result


def main():
    try:
        cfg = json.loads(sys.argv[1])
        result = supervise(cfg, sys.argv[2], sys.argv[3],
                           require_gpu=len(sys.argv) > 4 and sys.argv[4] == "gpu")
    except Exception as error:
        result = {"ok": False, "error": f"worker:{error.__class__.__name__}:{error}"}
    print(json.dumps(result), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
