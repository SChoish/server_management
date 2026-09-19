#!/usr/bin/env python3
"""One controller: portal queue -> GPU -> renew -> CPU failover -> GPU."""
from __future__ import annotations

import argparse
import fcntl
import getpass
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from http.cookiejar import CookieJar
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import queue_worker

REPO = Path(__file__).resolve().parents[3]
QUEUED = {"queued", "waiting", "pending"}
RELEASED = {"stopped", "exited", "expired", "preempted", "revoked", "terminated"}
KNOWN_STATES = {"running"} | QUEUED | RELEASED


class PortalError(RuntimeError):
    pass


class AuthError(PortalError):
    pass


def read_env(path):
    result = {}
    if Path(path).is_file():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                name, value = line.split("=", 1)
                result[name.strip()] = value.strip().strip("\"'")
    return result


def load_config(profile=None, path=None):
    profile = profile or os.environ.get("BEYONDG_PROFILE") or getpass.getuser()
    path = Path(path) if path else REPO / "config" / f"{profile}.json"
    cfg = json.loads(path.read_text())
    # Expand only the repository placeholder; never interpret shell expressions.
    cfg = json.loads(json.dumps(cfg).replace("${REPO}", str(REPO)))
    for key in ("sid", "portal_urls", "ssh", "worker", "env_file"):
        if not cfg.get(key):
            raise ValueError(f"Missing config field: {key}")
    if not cfg["sid"].startswith("dgx-"):
        raise ValueError("Configure an explicit portal server ID")
    for mode in ("cpu", "gpu"):
        job = cfg["worker"][mode]
        if not isinstance(job["command"], list) or not job["command"]:
            raise ValueError(f"{mode}.command must be a nonempty argv list")
    return cfg


def parse_status(payload, sid):
    """Missing/malformed target is unknown, never evidence of GPU loss."""
    if not isinstance(payload, dict) or not isinstance(payload.get(sid), dict):
        raise PortalError(f"target_missing:{sid}")
    row = payload[sid]
    state = str(row.get("state", "")).lower()
    if state not in KNOWN_STATES:
        raise PortalError(f"unknown_state:{state}")
    remaining = row.get("lease_hours_left")
    cooldown = row.get("cooldown_min_left")
    try:
        remaining = float(remaining) if remaining is not None else None
        cooldown = float(cooldown) if cooldown is not None else 0.0
        port = int(row["ssh_port"]) if row.get("ssh_port") else None
    except (ValueError, TypeError) as error:
        raise PortalError("invalid_status_fields") from error
    if port is not None and not 1 <= port <= 65535:
        raise PortalError("invalid_ssh_port")
    if remaining is not None and not math.isfinite(remaining):
        raise PortalError("invalid_remaining_time")
    if not math.isfinite(cooldown) or cooldown < 0:
        cooldown = 0.0
    return {"sid": sid, "state": state, "remaining_h": remaining,
            "host": row.get("host"), "ssh_port": port,
            "cooldown_min_left": cooldown}


class Portal:
    def __init__(self, cfg):
        self.cfg = cfg
        self.base = None
        self.opener = None
        self.login_at = 0.0

    def credentials(self):
        env = {**read_env(self.cfg["env_file"]), **os.environ}
        user = env.get("BEYONDG_USER") or env.get("BEYONDG_USERNAME")
        password = env.get("BEYONDG_PASS") or env.get("BEYONDG_PASSWORD")
        if not user or not password:
            raise AuthError("NEED_PORTAL_CREDS")
        return user, password

    def request(self, method, path, data=None, *, expect_json=True):
        data = urllib.parse.urlencode(data).encode() if data is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        try:
            with self.opener.open(req, timeout=self.cfg.get("http_timeout_s", 15)) as resp:
                status, body, url = resp.status, resp.read(), resp.geturl()
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise AuthError(f"http_{error.code}") from error
            raise PortalError(f"http_{error.code}") from error
        except (OSError, urllib.error.URLError) as error:
            raise PortalError(f"network:{error.__class__.__name__}") from error
        if status != 200:
            raise PortalError(f"http_{status}")
        if not expect_json:
            return None
        if "/auth/login" in url:
            raise AuthError("login_redirect")
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError) as error:
            raise AuthError("non_json_status_or_expired_login") from error
        if not isinstance(payload, dict):
            raise PortalError("non_object_response")
        return payload

    def login(self, base):
        user, password = self.credentials()
        self.base = base.rstrip("/")
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar()))
        self.request("POST", "/auth/login", {"username": user, "password": password},
                     expect_json=False)
        self.login_at = time.time()

    def status(self):
        if self.opener and time.time() - self.login_at < 25 * 60:
            try:
                return parse_status(self.request("GET", "/container/status"), self.cfg["sid"])
            except PortalError:
                self.opener = None
        env = {**read_env(self.cfg["env_file"]), **os.environ}
        bases = [env.get("BEYONDG_PORTAL"), self.base, *self.cfg["portal_urls"]]
        errors = []
        for base in dict.fromkeys(x for x in bases if x):
            try:
                self.login(base)
                return parse_status(self.request("GET", "/container/status"), self.cfg["sid"])
            except PortalError as error:
                errors.append(str(error))
                self.opener = None
        raise PortalError(";".join(errors) or "portal_unavailable")

    def start(self):
        if not self.opener:
            self.status()
        sid = urllib.parse.quote(self.cfg["sid"], safe="")
        payload = self.request("POST", f"/container/{sid}/start", {})
        if payload.get("ok") is not True:
            raise PortalError("start_rejected")
        return payload


class Workers:
    def __init__(self, cfg):
        self.cfg = cfg
        self.worker = cfg["worker"]
        self.source = Path(queue_worker.__file__).read_text()

    def local(self, action="status", mode="cpu"):
        return queue_worker.supervise(self.worker, action, mode)

    def remote(self, box, action="status", mode="gpu"):
        ssh = self.cfg["ssh"]
        host = box.get("host") or ssh["host"]
        port = box.get("ssh_port")
        if not port:
            return {"ok": False, "error": "portal_has_no_port"}
        # Reallocations can recreate sshd on the same published port. Keep
        # host-key continuity within each portal-confirmed allocation.
        known = Path(ssh["known_hosts"] + "." + str(box["allocation_token"]))
        known.parent.mkdir(parents=True, exist_ok=True)
        # The helper arrives on stdin, so the Docker image need not contain this
        # checkout. Only the configured experiment/queue paths must be mounted.
        remote = shlex.join([ssh.get("python", "python3"), "-",
                             json.dumps(self.worker), action, mode, "gpu"])
        argv = ["ssh", "-i", ssh["key"], "-p", str(port),
                "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
                "-o", "ConnectTimeout=8", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={known}",
                "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
                f"{ssh['user']}@{host}", remote]
        try:
            proc = subprocess.run(argv, input=self.source, text=True,
                                  capture_output=True, timeout=30)
            if proc.returncode not in (0, 1):
                return {"ok": False, "error": "ssh_failed", "detail": proc.stderr[-400:]}
            result = json.loads(proc.stdout)
            if not isinstance(result, dict):
                raise ValueError("invalid worker response")
            return result
        except (OSError, subprocess.TimeoutExpired, ValueError) as error:
            return {"ok": False, "error": f"remote:{error.__class__.__name__}"}


class Controller:
    def __init__(self, cfg, portal, workers, state=None, *, clock=time.time):
        self.cfg, self.portal, self.workers = cfg, portal, workers
        self.state = dict(state or {})
        # Honor evidence saved by either legacy watcher during migration.
        self.state.setdefault("gpu_seen", bool(self.state.get("gpu_held")))
        self.clock = clock
        self.events = []

    def event(self, text):
        self.events.append(text)

    def observe(self, box):
        if box["state"] == "running":
            if self.state.get("portal_state") != "running" or not self.state.get("allocation_token"):
                self.state["allocation_token"] = str(time.time_ns())
            self.state["gpu_seen"] = True
        elif self.state.get("gpu_seen"):
            self.state["loss_confirmed"] = True
        self.state["portal_state"] = box["state"]
        self.state["remaining_h"] = box["remaining_h"]
        self.state["ssh_port"] = box["ssh_port"]
        self.state["cooldown_min_left"] = box.get("cooldown_min_left") or 0.0
        box["allocation_token"] = self.state.get("allocation_token", "unknown")

    def set_phase(self, phase):
        self.state.update(phase=phase, ts=self.clock(), events=list(self.events))
        return dict(self.state)

    def ready_for(self, key, interval):
        previous = self.state.get(key)
        return previous is None or self.clock() - previous >= interval

    def tick(self, *, apply=False):
        self.events = []
        try:
            box = self.portal.status()
        except PortalError as error:
            self.event(f"portal_unknown:{error}")
            return self.set_phase("wait_portal")
        prev_state = self.state.get("portal_state")
        prev_cooldown = float(self.state.get("cooldown_min_left") or 0)
        self.observe(box)
        if not apply:
            return self.set_phase("observe_" + box["state"])

        # Renew in place: a failed HTTP request must not kill a healthy job.
        retry_s = self.cfg.get("request_retry_s", 60)
        due = (box["state"] == "running" and box["remaining_h"] is not None
               and box["remaining_h"] <= self.cfg.get("reload_remaining_h", 6.5))
        renew = (due and self.ready_for("last_reload_success", 30 * 60)
                 and self.ready_for("last_request", retry_s))
        cooldown = float(box.get("cooldown_min_left") or 0)
        # GPU loss: POST /start on the same tick, even if a reload just ran.
        # If the portal is in cooldown, do not spam; the instant
        # cooldown_min_left hits 0, POST again without waiting request_retry_s.
        just_lost = prev_state == "running" and box["state"] in RELEASED
        cooldown_cleared = cooldown <= 0 and prev_cooldown > 0
        acquire = (box["state"] in RELEASED
                   and (just_lost or cooldown_cleared
                        or self.ready_for("last_request", retry_s)))
        if renew or acquire:
            self.state["last_request"] = self.clock()
            accepted = False
            before = box
            try:
                self.portal.start()
                accepted = True
                self.event("reload_requested" if renew else "gpu_requested")
            except PortalError as error:
                self.event(f"request_failed:{error}")
            # A timeout can hide a successful request: re-read before deciding
            # whether to start CPU or GPU, even when POST raised an exception.
            try:
                box = self.portal.status()
            except PortalError as error:
                self.event(f"post_request_status_unknown:{error}")
                return self.set_phase("wait_portal")
            self.observe(box)
            if renew and accepted:
                if (box["state"] in QUEUED or
                        (box["state"] == "running" and box["remaining_h"] is not None
                         and box["remaining_h"] > before["remaining_h"] + 0.05)):
                    self.state["last_reload_success"] = self.clock()
                    self.event("reload_confirmed")
                else:
                    self.event("reload_not_yet_confirmed")

        local = self.workers.local()
        if not local.get("ok"):
            self.event("local_probe_failed:" + str(local.get("error")))
            return self.set_phase("wait_local")
        if box["state"] != "running":
            # Initial queue waiting is different from an allocation being taken.
            # Once loss is confirmed, keep CPU active throughout GPU queueing.
            if not self.state.get("loss_confirmed"):
                return self.set_phase("wait_gpu")
            if local.get("gpu"):
                stopped = self.workers.local("stop", "gpu")
                if not stopped.get("ok") or not stopped.get("stopped"):
                    return self.set_phase("stopping_lost_gpu")
            if local.get("cpu"):
                return self.set_phase("cpu_waiting_gpu")
            started = self.workers.local("start", "cpu")
            self.event("cpu_start_requested" if started.get("ok") else
                       "cpu_start_failed:" + str(started.get("error")))
            return self.set_phase("cpu_waiting_gpu" if started.get("cpu") else "cpu_starting")

        remote = self.workers.remote(box)
        if not remote.get("ok") or not remote.get("gpu_available"):
            self.event("gpu_not_ready:" + str(remote.get("error", "no_gpu")))
            # Do not stop CPU just because the portal says Running; wait for an
            # authenticated Docker connection and a visible GPU.
            return self.set_phase("wait_docker")
        if local.get("cpu"):
            # If a legacy/manual GPU job overlaps CPU, stop GPU first. Never
            # start a new writer while CPU is still saving its checkpoint.
            if remote.get("gpu"):
                stopped = self.workers.remote(box, "stop", "gpu")
                if not stopped.get("ok") or not stopped.get("stopped"):
                    return self.set_phase("stopping_overlapping_gpu")
            stopped = self.workers.local("stop", "cpu")
            self.event("cpu_stop_requested")
            if not stopped.get("ok") or not stopped.get("stopped"):
                return self.set_phase("stopping_cpu")
        # A second observation prevents delayed CPU shutdown being mistaken for
        # completion and catches detached launchers/children.
        local = self.workers.local()
        if not local.get("ok") or local.get("cpu"):
            return self.set_phase("stopping_cpu")
        if remote.get("cpu"):
            stopped = self.workers.remote(box, "stop", "cpu")
            if not stopped.get("ok") or not stopped.get("stopped"):
                return self.set_phase("stopping_cpu")
        result = self.workers.remote(box, "start", "gpu")
        if not result.get("ok"):
            self.event("gpu_start_failed:" + str(result.get("error")))
            return self.set_phase("gpu_starting")
        if result.get("gpu"):
            self.state["loss_confirmed"] = False
            self.event("gpu_running")
            return self.set_phase("gpu_running")
        self.event("gpu_start_requested")
        return self.set_phase("gpu_starting")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("ext_csh", "ext_csv"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--print-state-dir", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.profile, args.config)
    directory = Path(cfg["worker"]["state_dir"])
    if args.print_state_dir:
        print(directory)
        return 0
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / "controller.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Controller already running for this profile", flush=True)
        return 0
    state_path = directory / "lease.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    controller = Controller(cfg, Portal(cfg), Workers(cfg), state)
    running = True
    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    pid_path = directory / "lease.watch.pid"
    if args.watch:
        pid_path.write_text(str(os.getpid()) + "\n")
    try:
        while running:
            try:
                result = controller.tick(apply=args.apply)
            except Exception as error:
                controller.event(f"tick_error:{error.__class__.__name__}:{error}")
                result = controller.set_phase("error")
            # Observations cannot replace authoritative state of an apply run.
            if args.apply:
                queue_worker.atomic_json(state_path, result)
            line = json.dumps(result, ensure_ascii=False)
            with (directory / "watcher.log").open("a") as handle:
                handle.write(line + "\n")
            print(line, flush=True)
            if not args.watch:
                break
            deadline = time.monotonic() + max(1, args.interval)
            while running and time.monotonic() < deadline:
                time.sleep(min(1, max(0, deadline - time.monotonic())))
    finally:
        if args.watch and pid_path.exists() and pid_path.read_text().strip() == str(os.getpid()):
            pid_path.unlink()
        lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
