#!/usr/bin/env python3
"""Portal reload when reachable; GPU-usage failover always.

Port of hosts/beyondg/watch_beyondg_lease.py for this host (ext_csv / DGX #2).
Live experiment here is AMO-fql JAX loco9 T-init-5, not Imp ME.

1. Portal (best-effort). Laptop VPN can drop. Prefer :5000 locally, else DGX1.
   While the portal answers, reload before remaining hits 6h (trigger 6.5h).
2. GPU usage (always). Host nvidia-smi is denied; read util via docker SSH,
   plus our train PIDs. If the box/SSH is gone and we are not in the portal
   queue, the GPU was taken → CPU failover. If GPU comes back, stop CPU and
   resume GPU. No dual-run.
3. Portal queue: keep polling until Running and SSH is up. Then resume GPU
   if the experiment is down. Do not CPU-failover while queued.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from http.cookiejar import CookieJar
from pathlib import Path

ROOT = Path("/home/ext_csv/MPI_sweep")
STATE_DIR = ROOT / "logs" / "beyondg"
STATE_PATH = STATE_DIR / "lease.json"
EVENT_LOG = STATE_DIR / "watcher.log"
PORT_FILE = STATE_DIR / "docker.port"
ENV_FILE = STATE_DIR / "portal.env"
KEY_DEFAULT = Path("/home/ext_csv/.ssh/gpubox_host")
KNOWN_HOSTS = Path("/home/ext_csv/.ssh/gpubox_known_hosts")
CPU_QUEUE = ROOT / "scripts/hosts/beyondg/run_queue_amo_fql_jax_loco9_cpu.sh"
CPU_LOG = STATE_DIR / "queue_amo_fql_cpu_failover.log"
CPU_PIDFILE = STATE_DIR / "cpu_queue.pid"
LAUNCH_IN_DOCKER = ROOT / "scripts/hosts/beyondg/launch_amo_gpu_in_docker.sh"
GPU_QUEUE = ROOT / "scripts/hosts/beyondg/run_queue_amo_fql_jax_loco9_gpu.sh"
GPU_LOG = STATE_DIR / "queue_amo_fql_docker.log"
KST = timezone(timedelta(hours=9))
PORTALS = ("http://127.0.0.1:5000", "http://166.104.28.71:5000")
RELOGIN_S = 25 * 60
TAKEN_CPU_S = 5 * 60
RELOAD_REMAINING_H = 6.5
RELOAD_COOLDOWN_S = 30 * 60
UTIL_HELD = 10.0
WATCHER_REV = "20260917-reload-resume-2"
TRAIN_NEEDLE = "AMO-fql/train.py"
LAUNCHER_NEEDLE = "launch_fql_amo_jax_loco9_tinit5_alrgrid.py"
QUEUE_NEEDLE = "run_queue_amo_fql_jax_loco9_gpu.sh"
DEFAULT_HOST = "166.104.28.73"
DEFAULT_USER = "ext_csv"
DEFAULT_PORT = 23021
_SESSION = None
_SESSION_AT = 0.0
_PORTAL = PORTALS[0]
REMOTE_LIST_EXP = (
    "pgrep -af '[A]MO-fql/train.py' || true; "
    "pgrep -af '[l]aunch_fql_amo_jax_loco9_tinit5_alrgrid.py' || true; "
    "pgrep -af '[r]un_queue_amo_fql_jax_loco9_gpu.sh' || true"
)


def now() -> datetime:
    return datetime.now(KST)


def load_state() -> dict:
    if STATE_PATH.is_file():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(payload: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATE_PATH)


def parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text)


def remaining_hours(lease_start: datetime, lease_hours: float) -> float:
    elapsed = (now() - lease_start).total_seconds() / 3600.0
    return lease_hours - elapsed


def log_event(msg: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{now().isoformat(timespec='seconds')} {msg}\n"
    with EVENT_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line)
    print(line, end="", flush=True)


def decide(
    *,
    gpu_held: bool,
    portal_ok: bool,
    portal_queued: bool,
    remaining_h: float,
    taken_s: float,
    reload_at: float = RELOAD_REMAINING_H,
    taken_cpu_s: float = TAKEN_CPU_S,
    seconds_since_reload: float | None = None,
    reload_cooldown_s: float = RELOAD_COOLDOWN_S,
) -> str:
    """Reload only if portal is up, and before remaining reaches 6h.

    Queue on the portal: wait for allocation. Do not CPU-failover while queued.
    CPU is usage-only: GPU gone and not queued → continue on CPU.
    A recent successful reload must not loop just because lease.json is stale.
    """
    if gpu_held:
        recently = (
            seconds_since_reload is not None
            and 0 <= seconds_since_reload < reload_cooldown_s
        )
        if recently:
            return "watch"
        if portal_ok and remaining_h <= reload_at:
            return "reload"
        return "watch"
    if portal_queued:
        return "wait_queue"
    if taken_s >= taken_cpu_s:
        return "cpu"
    return "wait_gpu"


def is_real_exp_cmd(cmd: str) -> bool:
    if not cmd:
        return False
    if any(s in cmd for s in ("watch_beyondg_lease", "pgrep", "pkill")):
        return False
    if "--device=cpu" in cmd or "--device cpu" in cmd:
        return False
    return any(n in cmd for n in (TRAIN_NEEDLE, LAUNCHER_NEEDLE, QUEUE_NEEDLE))


def exp_really_alive(cmds: list[str], usage: list[dict]) -> bool:
    """Train.py counts. Launchers count only if the GPU is actually busy."""
    util_held = any(float(row.get("util") or 0) >= UTIL_HELD for row in usage)
    train = any(TRAIN_NEEDLE in cmd and is_real_exp_cmd(cmd) for cmd in cmds)
    if train:
        return True
    launcher = any(LAUNCHER_NEEDLE in cmd and is_real_exp_cmd(cmd) for cmd in cmds)
    return bool(launcher and util_held)


def should_resume_gpu(
    *,
    ssh_up: bool,
    portal_running: bool,
    portal_queued: bool,
    exp_alive: bool,
) -> bool:
    return bool(ssh_up and portal_running and not portal_queued and not exp_alive)


def read_env_file(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def portal_creds() -> tuple[str, str] | None:
    merged = {**read_env_file(ENV_FILE), **os.environ}
    user = merged.get("BEYONDG_USER") or merged.get("BEYONDG_USERNAME")
    password = merged.get("BEYONDG_PASS") or merged.get("BEYONDG_PASSWORD")
    if user and password:
        return user, password
    return None


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


def iter_pids(needle: str) -> list[int]:
    found = []
    proc = Path("/proc")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        cmd = _cmdline(int(entry.name))
        if needle in cmd and "watch_beyondg_lease" not in cmd:
            found.append(int(entry.name))
    return found


def gpu_train_pids() -> list[int]:
    out = []
    for pid in iter_pids(TRAIN_NEEDLE):
        cmd = _cmdline(pid)
        if "--device=cpu" in cmd or "--device cpu" in cmd:
            continue
        out.append(pid)
    return out


def cpu_train_pids() -> list[int]:
    out = []
    for pid in iter_pids(TRAIN_NEEDLE):
        cmd = _cmdline(pid)
        if "--device=cpu" in cmd or "--device cpu" in cmd:
            out.append(pid)
    return out


def gpu_launcher_pids() -> list[int]:
    out = []
    for pid in iter_pids(LAUNCHER_NEEDLE):
        cmd = _cmdline(pid)
        if "--device=cpu" in cmd or "--device cpu" in cmd:
            continue
        out.append(pid)
    return out


def ssh_cmd(state: dict, remote: str, timeout: int = 20) -> subprocess.CompletedProcess:
    port = int(state.get("ssh_port") or DEFAULT_PORT)
    if PORT_FILE.is_file():
        raw = PORT_FILE.read_text(encoding="utf-8").strip().splitlines()
        if raw:
            try:
                port = int(raw[0])
            except ValueError:
                pass
    key = state.get("ssh_key") or str(KEY_DEFAULT)
    host = state.get("host") or DEFAULT_HOST
    user = state.get("ssh_user") or DEFAULT_USER
    cmd = [
        "ssh",
        "-i",
        key,
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={timeout}",
        "-o",
        f"UserKnownHostsFile={KNOWN_HOSTS}",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-p",
        str(port),
        f"{user}@{host}",
        remote,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
    except (subprocess.TimeoutExpired, OSError) as error:
        return subprocess.CompletedProcess(cmd, 1, "", str(error))
    err = (proc.stderr or "") + (proc.stdout or "")
    if proc.returncode != 0 and (
        "REMOTE HOST IDENTIFICATION HAS CHANGED" in err
        or "Host key verification failed" in err
    ):
        subprocess.run(
            ["ssh-keygen", "-f", str(KNOWN_HOSTS), "-R", f"[{host}]:{port}"],
            capture_output=True,
            text=True,
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)
    return proc


def ssh_up(state: dict) -> bool:
    proc = ssh_cmd(state, "echo ok")
    return proc.returncode == 0 and "ok" in (proc.stdout or "")


def box_gpu_usage(state: dict) -> list[dict]:
    proc = ssh_cmd(
        state,
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
        "--format=csv,noheader,nounits",
    )
    if proc.returncode != 0:
        return []
    rows = []
    for line in (proc.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "util": float(parts[1]),
                    "memory_used_mib": float(parts[2]),
                }
            )
        except ValueError:
            continue
    return rows


def docker_exp_cmds(state: dict) -> list[str]:
    proc = ssh_cmd(state, REMOTE_LIST_EXP)
    if proc.returncode != 0:
        return []
    return [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]


def gpu_queue_alive(state: dict) -> bool:
    return any(is_real_exp_cmd(cmd) for cmd in docker_exp_cmds(state))


def host_gpu_busy() -> bool:
    return bool(gpu_train_pids() or gpu_launcher_pids())


def is_cpu_queue_pid(pid: int) -> bool:
    cmd = _cmdline(pid)
    if not cmd or "watch_beyondg_lease" in cmd:
        return False
    return "run_queue_amo_fql_jax_loco9_cpu.sh" in cmd or (
        LAUNCHER_NEEDLE in cmd and ("--device=cpu" in cmd or "--device cpu" in cmd)
    )


def cpu_queue_pid() -> int | None:
    if CPU_PIDFILE.is_file():
        try:
            pid = int(CPU_PIDFILE.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = 0
        if is_cpu_queue_pid(pid):
            return pid
        CPU_PIDFILE.unlink(missing_ok=True)
    for pid in iter_pids("run_queue_amo_fql_jax_loco9_cpu.sh"):
        if is_cpu_queue_pid(pid):
            return pid
    if cpu_train_pids():
        return cpu_train_pids()[0]
    return None


def _sigterm_wait(pids: list[int], timeout_s: float = 180.0) -> None:
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            continue
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not any(Path(f"/proc/{pid}").exists() for pid in pids):
            return
        time.sleep(2)


def soft_stop_cpu() -> None:
    pid = cpu_queue_pid()
    pids = []
    if pid:
        pids.append(pid)
    pids.extend(iter_pids("run_queue_amo_fql_jax_loco9_cpu.sh"))
    pids.extend(cpu_train_pids())
    _sigterm_wait(sorted(set(pids)))


def start_cpu() -> int | None:
    live = cpu_queue_pid()
    if live:
        return live
    if host_gpu_busy():
        log_event("skip CPU failover; GPU experiment still live")
        return None
    CPU_LOG.parent.mkdir(parents=True, exist_ok=True)
    handle = CPU_LOG.open("a", encoding="utf-8")
    handle.write(f"\n[{now().isoformat(timespec='seconds')}] CPU failover start\n")
    handle.flush()
    proc = subprocess.Popen(
        ["bash", str(CPU_QUEUE)],
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    CPU_PIDFILE.write_text(f"{proc.pid}\n", encoding="utf-8")
    return int(proc.pid)


def soft_stop_gpu(state: dict) -> None:
    _sigterm_wait(gpu_launcher_pids() + gpu_train_pids(), timeout_s=180.0)
    ssh_cmd(
        state,
        "pkill -TERM -f run_queue_amo_fql_jax_loco9_gpu.sh || true; "
        "pkill -TERM -f launch_fql_amo_jax_loco9_tinit5_alrgrid.py || true; "
        "pkill -TERM -f 'AMO-fql/train.py' || true",
    )
    for _ in range(90):
        if not gpu_queue_alive(state):
            return
        time.sleep(2)


def start_gpu(state: dict) -> bool:
    if not ssh_up(state):
        log_event("SSH not up; refusing to start GPU queue")
        return False
    cmds = docker_exp_cmds(state)
    usage = box_gpu_usage(state)
    if exp_really_alive(cmds, usage):
        return False
    env = os.environ.copy()
    env["BEYONDG_DOCKER_PORT"] = str(state.get("ssh_port") or "")
    env["BEYONDG_DOCKER_HOST"] = str(state.get("host") or DEFAULT_HOST)
    GPU_LOG.parent.mkdir(parents=True, exist_ok=True)
    handle = GPU_LOG.open("a", encoding="utf-8")
    subprocess.Popen(
        ["bash", str(LAUNCH_IN_DOCKER)],
        cwd=str(ROOT),
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return True


def parse_jobs_page(html: str) -> dict:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    low = text.lower()
    running = bool(re.search(r"\brunning\b", low))
    waiting = (not running) and bool(
        re.search(r"\b(waiting|queued|pending|in queue)\b", low)
    )
    port = None
    ssh_m = re.search(r"(\d+\.\d+\.\d+\.\d+):(\d{4,5})", text)
    if ssh_m:
        port = int(ssh_m.group(2))
    rem = None
    rem_m = re.search(r"auto-stops in\s+([0-9.]+)\s*h", low)
    if rem_m:
        rem = float(rem_m.group(1))
    return {
        "running": running,
        "waiting": waiting and not running,
        "ssh_port": port,
        "portal_remaining_h": rem,
        "logged_in": "log in" not in low or "username" not in low,
    }


def parse_container_status(payload: object) -> dict:
    if not isinstance(payload, dict) or not payload:
        return {
            "portal": "missing",
            "remaining_h": None,
            "state": None,
            "host": None,
            "ssh_port": None,
            "sid": None,
        }
    sid = None
    row = None
    for key, value in payload.items():
        if isinstance(value, dict) and value.get("state"):
            sid, row = key, value
            if value.get("state") in {"running", "queued"}:
                break
    if row is None:
        return {
            "portal": "ok",
            "remaining_h": None,
            "state": "stopped",
            "host": None,
            "ssh_port": None,
            "sid": next(iter(payload), None),
        }
    state = str(row.get("state") or "")
    portal = "queued" if state == "queued" else "ok"
    remaining = row.get("lease_hours_left")
    try:
        remaining_h = float(remaining) if remaining is not None else None
    except (TypeError, ValueError):
        remaining_h = None
    port = row.get("ssh_port")
    try:
        ssh_port = int(port) if port is not None else None
    except (TypeError, ValueError):
        ssh_port = None
    return {
        "portal": portal,
        "remaining_h": remaining_h,
        "state": state,
        "host": row.get("host") or DEFAULT_HOST,
        "ssh_port": ssh_port,
        "sid": sid,
        "raw": row,
    }


class _UrllibSession:
    def __init__(self) -> None:
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(CookieJar())
        )
        self.base = _PORTAL

    def post(self, url, data=None, timeout=20, allow_redirects=True):
        payload = None
        if isinstance(data, dict):
            payload = urllib.parse.urlencode(data).encode()
        elif isinstance(data, bytes):
            payload = data
        req = urllib.request.Request(url, data=payload, method="POST")
        return self._open(req, timeout)

    def get(self, url, timeout=20):
        return self._open(urllib.request.Request(url), timeout)

    def _open(self, req, timeout):
        try:
            resp = self.opener.open(req, timeout=timeout)
            body = resp.read()
            return _Resp(resp.getcode(), body, dict(resp.headers))
        except urllib.error.HTTPError as error:
            body = error.read() if error.fp else b""
            return _Resp(error.code, body, dict(error.headers or {}))


class _Resp:
    def __init__(self, status_code, body: bytes, headers: dict) -> None:
        self.status_code = status_code
        self.content = body
        self.text = body.decode("utf-8", "replace")
        self.headers = {k.lower(): v for k, v in headers.items()}

    def json(self):
        return json.loads(self.text)


def _make_session():
    try:
        import requests

        return requests.Session()
    except ImportError:
        return _UrllibSession()


def portal_session(*, force: bool = False):
    global _SESSION, _SESSION_AT, _PORTAL
    creds = portal_creds()
    if creds is None:
        return None, {"ok": False, "error": "NEED_PORTAL_CREDS"}
    age = time.time() - _SESSION_AT
    if _SESSION is not None and not force and age < RELOGIN_S:
        return _SESSION, {"ok": True, "reused": True, "age_s": round(age)}
    user, password = creds
    explicit = os.environ.get("BEYONDG_PORTAL", "").strip().rstrip("/")
    bases = []
    if explicit:
        bases.append(explicit)
    bases.extend(PORTALS)
    last_error = "portal_down"
    for base in bases:
        session = _make_session()
        if isinstance(session, _UrllibSession):
            session.base = base
        try:
            session.post(
                f"{base}/auth/login",
                data={"username": user, "password": password},
                timeout=20,
                allow_redirects=True,
            )
            jobs = session.get(f"{base}/", timeout=20)
        except Exception as error:
            last_error = f"portal_down:{error.__class__.__name__}"
            continue
        parsed = parse_jobs_page(jobs.text)
        if jobs.status_code != 200 or not parsed["logged_in"]:
            last_error = "login_failed"
            continue
        _PORTAL = base
        _SESSION = session
        _SESSION_AT = time.time()
        return session, {"ok": True, "reused": False, "portal": base, **parsed}
    _SESSION = None
    return None, {"ok": False, "error": last_error}


def portal_jobs() -> dict:
    session, info = portal_session()
    if session is None:
        return info
    try:
        status = session.get(f"{_PORTAL}/container/status", timeout=20)
        payload = status.json()
        parsed = parse_container_status(payload)
        running = parsed.get("state") == "running"
        waiting = parsed.get("state") in {"queued", "waiting", "pending"}
        return {
            "ok": True,
            "running": running,
            "waiting": waiting and not running,
            "ssh_port": parsed.get("ssh_port"),
            "portal_remaining_h": parsed.get("remaining_h"),
            "sid": parsed.get("sid"),
            "host": parsed.get("host"),
            "logged_in": True,
        }
    except Exception:
        try:
            jobs = session.get(f"{_PORTAL}/", timeout=20)
        except Exception as error:
            session2, info2 = portal_session(force=True)
            if session2 is None:
                return info2
            return {"ok": False, "error": f"portal_down:{error.__class__.__name__}"}
        parsed = parse_jobs_page(jobs.text)
        if not parsed["logged_in"]:
            session2, info2 = portal_session(force=True)
            if session2 is None:
                return info2
            try:
                jobs = session2.get(f"{_PORTAL}/", timeout=20)
                parsed = parse_jobs_page(jobs.text)
            except Exception as error:
                return {"ok": False, "error": f"portal_down:{error.__class__.__name__}"}
        return {"ok": True, **parsed, "html_len": len(jobs.text)}


def portal_reload(state: dict) -> dict:
    session, info = portal_session(force=True)
    if session is None:
        return info
    sid = "dgx-h200-2"
    url = ""
    method = "POST"
    resp = None
    try:
        status = session.get(f"{_PORTAL}/container/status", timeout=20).json()
        if isinstance(status, dict) and status:
            parsed = parse_container_status(status)
            sid = parsed.get("sid") or next(iter(status))
        url = f"{_PORTAL}/container/{sid}/start"
        resp = session.post(url, timeout=60, data={})
        body = (
            resp.json()
            if str(resp.headers.get("content-type", "")).startswith("application/json")
            else {}
        )
    except Exception as error:
        return {"ok": False, "error": f"portal_down:{error.__class__.__name__}"}
    if not (isinstance(body, dict) and body.get("ok")):
        return {
            "ok": False,
            "error": body.get("msg") if isinstance(body, dict) else "reload_failed",
            "status": getattr(resp, "status_code", None),
            "sid": sid,
        }
    parsed = {}
    try:
        after = session.get(f"{_PORTAL}/container/status", timeout=20).json().get(sid, {})
        parsed["running"] = after.get("state") == "running"
        parsed["waiting"] = after.get("state") in {"queued", "waiting", "pending"}
        parsed["ssh_port"] = after.get("ssh_port")
        parsed["portal_remaining_h"] = after.get("lease_hours_left")
        if after.get("host"):
            state["host"] = after["host"]
    except Exception:
        after = {}
    if parsed.get("ssh_port"):
        state["ssh_port"] = parsed["ssh_port"]
        PORT_FILE.write_text(f"{state['ssh_port']}\n", encoding="utf-8")
    return {
        "ok": getattr(resp, "status_code", 200) < 400,
        "status": getattr(resp, "status_code", None),
        "url": url or f"{_PORTAL}/container/{sid}/start",
        "method": method,
        "port": state.get("ssh_port"),
        "waiting": parsed.get("waiting"),
        "running": parsed.get("running"),
        "lease_hours_left": parsed.get("portal_remaining_h"),
    }


def default_state() -> dict:
    port = DEFAULT_PORT
    if PORT_FILE.is_file():
        try:
            port = int(PORT_FILE.read_text(encoding="utf-8").strip().splitlines()[0])
        except (ValueError, IndexError):
            pass
    return {
        "lease_start": now().isoformat(timespec="seconds"),
        "lease_hours": 10.0,
        "reload_remaining_hours": RELOAD_REMAINING_H,
        "host": DEFAULT_HOST,
        "ssh_port": port,
        "ssh_user": DEFAULT_USER,
        "ssh_key": str(KEY_DEFAULT),
        "label": "DGX H200 #2",
        "last_reload_at": None,
        "idle_since": None,
        "actions": [],
    }


def _persist_reload_clock(state: dict, reload_info: dict) -> None:
    state["last_reload_at"] = now().isoformat(timespec="seconds")
    remaining = reload_info.get("lease_hours_left")
    if remaining is not None:
        try:
            rem = float(remaining)
            elapsed_h = float(state.get("lease_hours") or 10.0) - rem
            state["lease_start"] = (now() - timedelta(hours=elapsed_h)).isoformat(
                timespec="seconds"
            )
        except (TypeError, ValueError):
            pass
    save_state({**load_state(), **state})


def do_reload(state: dict, gpu_held: bool) -> list[str]:
    """SIGTERM, portal start, persist the new clock, then resume GPU.

    Exceptions after SIGTERM must not skip resume or lease.json writes.
    """
    actions = []
    try:
        if gpu_held:
            soft_stop_gpu(state)
            actions.append("sigterm_gpu")
        reload_info = portal_reload(state)
        actions.append(f"reload:{reload_info.get('error') or reload_info.get('ok')}")
        if not reload_info.get("ok"):
            log_event(f"EVENT RELOAD_FAIL {reload_info.get('error')}")
            return actions
        _persist_reload_clock(state, reload_info)
        log_event("EVENT RELOAD_OK waiting for GPU allocation")
        for _ in range(36):
            if ssh_up(state):
                break
            time.sleep(5)
        if ssh_up(state):
            actions.append("reload_allocated")
            if start_gpu(state):
                actions.append("resume_gpu")
                log_event("EVENT EXP_DOWN resume GPU queue after reload")
        else:
            actions.append("reload_waiting")
            log_event("EVENT PORTAL_QUEUE after reload; poll until GPU is held")
        return actions
    except Exception as error:
        log_event(f"EVENT RELOAD_ERROR {error.__class__.__name__}: {error}")
        actions.append(f"reload_error:{error.__class__.__name__}")
        try:
            if ssh_up(state) and start_gpu(state):
                actions.append("resume_gpu")
                log_event("EVENT EXP_DOWN resume GPU queue after reload error")
        except Exception as resume_error:
            log_event(
                f"EVENT RESUME_ERROR {resume_error.__class__.__name__}: {resume_error}"
            )
        return actions


def tick(*, apply: bool) -> dict:
    state = {**default_state(), **load_state()}
    start = parse_iso(state["lease_start"])
    rem = remaining_hours(start, float(state["lease_hours"]))
    portal = portal_jobs()
    if portal.get("host"):
        state["host"] = portal["host"]
    if portal.get("ssh_port"):
        state["ssh_port"] = portal["ssh_port"]
        PORT_FILE.write_text(f"{state['ssh_port']}\n", encoding="utf-8")
    if portal.get("portal_remaining_h") is not None and portal.get("running"):
        rem = float(portal["portal_remaining_h"])
        elapsed_h = float(state["lease_hours"]) - rem
        state["lease_start"] = (now() - timedelta(hours=elapsed_h)).isoformat(
            timespec="seconds"
        )
    up = ssh_up(state)
    cmds = docker_exp_cmds(state) if up else []
    queue = any(is_real_exp_cmd(cmd) for cmd in cmds)
    usage = box_gpu_usage(state) if up else []
    util_held = any(row["util"] >= UTIL_HELD for row in usage)
    exp_alive = exp_really_alive(cmds, usage)
    portal_ok = bool(portal.get("ok"))
    if not portal_ok:
        portal_session(force=True)
        portal = portal_jobs()
        portal_ok = bool(portal.get("ok"))
        if portal.get("ssh_port"):
            state["ssh_port"] = portal["ssh_port"]
            PORT_FILE.write_text(f"{state['ssh_port']}\n", encoding="utf-8")
            up = ssh_up(state)
            cmds = docker_exp_cmds(state) if up else []
            queue = any(is_real_exp_cmd(cmd) for cmd in cmds)
            usage = box_gpu_usage(state) if up else []
            util_held = any(row["util"] >= UTIL_HELD for row in usage)
            exp_alive = exp_really_alive(cmds, usage)
    portal_queued = bool(
        portal_ok and portal.get("waiting") and not portal.get("running")
    )
    gpu_held = bool(up and not portal_queued)
    just_allocated = gpu_held and not bool(state.get("gpu_held"))
    seconds_since_reload = None
    if state.get("last_reload_at"):
        try:
            seconds_since_reload = (
                now() - parse_iso(str(state["last_reload_at"]))
            ).total_seconds()
        except ValueError:
            seconds_since_reload = None
    if gpu_held:
        idle_since = None
        taken_s = 0.0
    elif portal_queued:
        idle_since = None
        taken_s = 0.0
    else:
        idle_since = float(state["idle_since"] or time.time())
        taken_s = time.time() - idle_since
    action = decide(
        gpu_held=gpu_held,
        portal_ok=portal_ok,
        portal_queued=portal_queued,
        remaining_h=rem,
        taken_s=taken_s,
        reload_at=float(state["reload_remaining_hours"]),
        seconds_since_reload=seconds_since_reload,
    )
    actions = [action]
    if portal_ok and not bool(state.get("portal_ok")):
        actions.append("PORTAL_BACK")
        log_event("EVENT PORTAL_BACK")
    if portal_queued and not bool(state.get("portal_queued")):
        log_event("EVENT PORTAL_QUEUE waiting until GPU is allocated")
    if just_allocated:
        log_event(
            f"EVENT GPU_ALLOCATED port={state.get('ssh_port')} "
            f"exp_up={int(exp_alive)} util={usage}"
        )
    if apply:
        try:
            if gpu_held and cpu_queue_pid():
                soft_stop_cpu()
                actions.append("stop_cpu")
                log_event("EVENT STOP_CPU gpu held again")
            if action == "reload":
                actions.extend(do_reload(state, gpu_held))
                rem = remaining_hours(
                    parse_iso(state["lease_start"]), float(state["lease_hours"])
                )
                if state.get("last_reload_at"):
                    try:
                        seconds_since_reload = (
                            now() - parse_iso(str(state["last_reload_at"]))
                        ).total_seconds()
                    except ValueError:
                        pass
            elif action == "cpu":
                pid = start_cpu()
                actions.append(f"cpu_pid={pid}")
                log_event(f"EVENT CPU_FAILOVER pid={pid} taken_s={taken_s:.0f}")
            elif should_resume_gpu(
                ssh_up=up,
                portal_running=bool(portal.get("running")),
                portal_queued=portal_queued,
                exp_alive=exp_alive,
            ):
                if start_gpu(state):
                    actions.append("resume_gpu")
                    log_event("EVENT EXP_DOWN resume GPU queue")
        except Exception as error:
            log_event(f"EVENT TICK_APPLY_ERROR {error.__class__.__name__}: {error}")
            actions.append(f"tick_error:{error.__class__.__name__}")
            try:
                if should_resume_gpu(
                    ssh_up=ssh_up(state),
                    portal_running=bool(portal.get("running")),
                    portal_queued=portal_queued,
                    exp_alive=False,
                ) and start_gpu(state):
                    actions.append("resume_gpu")
                    log_event("EVENT EXP_DOWN resume GPU queue after tick error")
            except Exception as resume_error:
                log_event(
                    f"EVENT RESUME_ERROR {resume_error.__class__.__name__}: {resume_error}"
                )
    payload = {
        **state,
        "ts": now().isoformat(timespec="seconds"),
        "remaining_hours": round(rem, 3),
        "due": action == "reload",
        "ssh_up": up,
        "gpu_queue_alive": queue,
        "exp_alive": exp_alive,
        "gpu_held": gpu_held,
        "gpu_usage": usage,
        "idle_since": idle_since,
        "taken_s": round(taken_s, 1),
        "portal_ok": portal_ok,
        "portal_queued": portal_queued,
        "portal_error": portal.get("error"),
        "portal_creds": portal_creds() is not None,
        "cpu_pid": cpu_queue_pid(),
        "apply": apply,
        "actions": actions,
        "watcher_rev": WATCHER_REV,
    }
    save_state(payload)
    utils = "/".join(str(int(row["util"])) for row in usage) or "-"
    log_event(
        f"tick action={action} portal={int(portal_ok)} queued={int(portal_queued)} "
        f"gpu_held={int(gpu_held)} rem={rem:.2f} ssh={int(up)} "
        f"exp={int(exp_alive)} util={utils} cpu={cpu_queue_pid() or '-'}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    src = Path(__file__)
    mtime = datetime.fromtimestamp(src.stat().st_mtime, KST).isoformat(
        timespec="seconds"
    )
    log_event(
        f"EVENT WATCHER_START pid={os.getpid()} rev={WATCHER_REV} mtime={mtime}"
    )
    if args.watch:
        (STATE_DIR / "lease.watch.pid").write_text(f"{os.getpid()}\n")
        while True:
            try:
                tick(apply=bool(args.apply))
            except Exception as error:
                log_event(f"EVENT TICK_ERROR {error.__class__.__name__}: {error}")
            time.sleep(max(10, args.interval))
    tick(apply=bool(args.apply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
