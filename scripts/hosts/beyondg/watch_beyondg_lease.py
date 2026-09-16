#!/usr/bin/env python3
"""Portal reload when reachable; GPU-usage failover always.

1. Portal (best-effort). Laptop VPN can drop; this host talks to :5000 locally.
   While the portal answers, reload before remaining hits 6h (trigger 6.5h)
   so the allocation is not stolen.
2. GPU usage (always). Host nvidia-smi is denied; read util via docker SSH,
   plus our train PIDs. If the box/SSH is gone and we are not in the portal
   queue, the GPU was taken → CPU ME. If GPU comes back, stop CPU and resume
   GPU. No dual-run.
3. Portal queue: keep polling until the job is Running and SSH is up. Then
   check whether the ME GPU experiment is down and resume it.

Human log: logs/beyondg/watcher.log. JSON snapshot: logs/beyondg/lease.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path("/home/ext_csh/MPI_sweep")
STATE_DIR = ROOT / "logs" / "beyondg"
STATE_PATH = STATE_DIR / "lease.json"
EVENT_LOG = STATE_DIR / "watcher.log"
PORT_FILE = STATE_DIR / "docker.port"
ENV_FILE = STATE_DIR / "portal.env"
KEY_DEFAULT = Path("/home/ext_csh/.ssh/gpubox_host")
CPU_QUEUE = ROOT / "logs/mpi_tau40_ext/run_queue_imp_me_k14_s0123_cpu.sh"
CPU_LOG = ROOT / "logs/mpi_tau40_ext/queue_imp_me_k14_cpu_failover.log"
CPU_PIDFILE = ROOT / "sweep_results/diagnostics/gpu_idle_failover/cpu_queue.pid"
KST = timezone(timedelta(hours=9))
PORTAL = "http://127.0.0.1:5000"
RELOGIN_S = 25 * 60
TAKEN_CPU_S = 5 * 60
RELOAD_REMAINING_H = 6.5
RELOAD_COOLDOWN_S = 30 * 60
UTIL_HELD = 10.0
GPU_ALIVE_SH = (
    "pgrep -f 'python -u .*train_td3bc.py' >/dev/null "
    "|| pgrep -f 'python -u .*launch_mpi_sweep.py' >/dev/null "
    "|| pgrep -f 'bash .*/run_queue_imp_me_k14_s0123_gpu.sh' >/dev/null"
)
GPU_QUEUE = (
    "/raid/ext_csh/artifacts/MPI_sweep/logs/mpi_tau40_ext/"
    "run_queue_imp_me_k14_s0123_gpu.sh"
)
GPU_QUEUE_FALLBACK = str(
    ROOT / "logs/mpi_tau40_ext/run_queue_imp_me_k14_s0123_gpu.sh"
)
GPU_LOG = (
    "/raid/ext_csh/artifacts/MPI_sweep/logs/mpi_tau40_ext/"
    "queue_imp_me_k14_docker.log"
)
_SESSION = None
_SESSION_AT = 0.0


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


def sync_lease_from_remaining(state: dict, rem: float, when: datetime | None = None) -> str:
    when = when or now()
    elapsed_h = float(state["lease_hours"]) - float(rem)
    return (when - timedelta(hours=elapsed_h)).isoformat(timespec="seconds")


def parse_container_box(boxes) -> dict:
    if not isinstance(boxes, dict) or not boxes:
        return {}
    sid = next(iter(boxes))
    box = boxes[sid] if isinstance(boxes[sid], dict) else {}
    rem = box.get("lease_hours_left")
    return {
        "sid": sid,
        "ssh_port": box.get("ssh_port"),
        "portal_remaining_h": float(rem) if rem is not None else None,
        "running": box.get("state") == "running",
        "waiting": box.get("state") in {"queued", "waiting", "pending"},
        "queue_length": box.get("queue_length"),
        "state": box.get("state"),
        "raw": box,
    }


def portal_reload_result(
    sid: str,
    status: int,
    parsed: dict,
    port,
) -> dict:
    return {
        "ok": status < 400,
        "status": status,
        "url": f"{PORTAL}/container/{sid}/start",
        "method": "POST",
        "port": port,
        "waiting": parsed.get("waiting"),
        "running": parsed.get("running"),
        "lease_hours_left": parsed.get("portal_remaining_h"),
    }


def source_mtime() -> float:
    try:
        return Path(__file__).resolve().stat().st_mtime
    except OSError:
        return 0.0


_SOURCE_MTIME = source_mtime()


def maybe_reexec(argv: list[str]) -> None:
    if source_mtime() == _SOURCE_MTIME:
        return
    log_event("EVENT REEXEC watcher source changed")
    os.execv(
        sys.executable,
        [sys.executable, "-u", str(Path(__file__).resolve()), *argv],
    )


def reload_cooldown_ok(last_reload_at, when: datetime | None = None) -> bool:
    if not last_reload_at:
        return True
    when = when or now()
    try:
        gap = (when - parse_iso(str(last_reload_at))).total_seconds()
    except ValueError:
        return True
    return gap >= RELOAD_COOLDOWN_S


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
    rem_from_portal: bool = True,
    reload_cooldown_ok: bool = True,
) -> str:
    """Reload only if portal is up, and before remaining reaches 6h.

    Queue on the portal: wait for allocation. Do not CPU-failover while queued.
    CPU is usage-only: GPU gone and not queued → continue ME on CPU.
    Never reload from a stale local clock (rem_from_portal=False).
    """
    if gpu_held:
        if (
            portal_ok
            and rem_from_portal
            and reload_cooldown_ok
            and remaining_h <= reload_at
        ):
            return "reload"
        return "watch"
    if portal_queued:
        return "wait_queue"
    if taken_s >= taken_cpu_s:
        return "cpu"
    return "wait_gpu"


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


def ssh_cmd(state: dict, remote: str, timeout: int = 20) -> subprocess.CompletedProcess:
    port = int(state.get("ssh_port") or 23023)
    if PORT_FILE.is_file():
        raw = PORT_FILE.read_text(encoding="utf-8").strip().splitlines()
        if raw:
            try:
                port = int(raw[0])
            except ValueError:
                pass
    key = state.get("ssh_key") or str(KEY_DEFAULT)
    host = state.get("host") or "127.0.0.1"
    user = state.get("ssh_user") or "ext_csh"
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
        "StrictHostKeyChecking=accept-new",
        "-p",
        str(port),
        f"{user}@{host}",
        remote,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5)


def ssh_up(state: dict) -> bool:
    try:
        proc = ssh_cmd(state, "echo ok")
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and "ok" in (proc.stdout or "")


def box_gpu_usage(state: dict) -> list[dict]:
    try:
        proc = ssh_cmd(
            state,
            "nvidia-smi --query-gpu=index,utilization.gpu,memory.used "
            "--format=csv,noheader,nounits",
        )
    except (subprocess.TimeoutExpired, OSError):
        return []
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


def gpu_queue_alive(state: dict) -> bool:
    try:
        proc = ssh_cmd(state, GPU_ALIVE_SH)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0


def host_gpu_busy() -> bool:
    proc = subprocess.run(
        ["pgrep", "-af", "launch_mpi_sweep.py"], capture_output=True, text=True
    )
    return any(
        "launch_mpi_sweep.py" in line and "--cpu-jobs" not in line
        for line in proc.stdout.splitlines()
    )


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    except OSError:
        return ""


def is_cpu_queue_pid(pid: int) -> bool:
    cmd = _cmdline(pid)
    if not cmd or "watch_beyondg_lease" in cmd:
        return False
    return "run_queue_imp_me_k14_s0123_cpu.sh" in cmd or (
        "launch_mpi_sweep.py" in cmd and "--cpu-jobs" in cmd
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
    proc = subprocess.run(
        ["pgrep", "-af", "run_queue_imp_me_k14_s0123_cpu.sh"],
        capture_output=True,
        text=True,
    )
    for line in proc.stdout.splitlines():
        pid = line.strip().split(None, 1)[0]
        if pid.isdigit() and is_cpu_queue_pid(int(pid)):
            return int(pid)
    return None


def soft_stop_cpu() -> None:
    pid = cpu_queue_pid()
    if pid:
        subprocess.run(["kill", "-TERM", str(pid)], check=False)
    subprocess.run(
        ["pkill", "-TERM", "-f", "run_queue_imp_me_k14_s0123_cpu.sh"],
        check=False,
    )
    for _ in range(60):
        if cpu_queue_pid() is None:
            return
        time.sleep(2)


def start_cpu() -> int | None:
    live = cpu_queue_pid()
    if live:
        return live
    CPU_LOG.parent.mkdir(parents=True, exist_ok=True)
    CPU_PIDFILE.parent.mkdir(parents=True, exist_ok=True)
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
    ssh_cmd(
        state,
        "pkill -TERM -f run_queue_imp_me_k14_s0123_gpu.sh || true; "
        "pkill -TERM -f 'launch_mpi_sweep.py' || true; "
        "pkill -TERM -f 'train_td3bc.py' || true",
    )
    for _ in range(90):
        proc = ssh_cmd(state, "pgrep -f train_td3bc.py >/dev/null")
        if proc.returncode != 0:
            return
        time.sleep(2)


def start_gpu_remote(queue: str, log: str) -> str:
    return (
        f"{GPU_ALIVE_SH} && exit 0; "
        f"setsid nohup bash {queue} >> {log} 2>&1 < /dev/null & echo $!"
    )


def start_gpu(state: dict) -> None:
    queue = GPU_QUEUE if Path(GPU_QUEUE).is_file() else GPU_QUEUE_FALLBACK
    ssh_cmd(state, start_gpu_remote(queue, GPU_LOG))


def find_reload_targets(html: str, base: str) -> list[tuple[str, str, dict[str, str]]]:
    found: list[tuple[str, str, dict[str, str]]] = []
    for match in re.finditer(
        r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>(.*?)</a>',
        html,
        re.I | re.S,
    ):
        href, text = match.group(1), re.sub(r"<[^>]+>", "", match.group(2))
        blob = f"{href} {text}".lower()
        if "reload" in blob or "extend" in blob or "renew" in blob:
            found.append(("GET", urljoin(base, href), {}))
    for match in re.finditer(r"<form([^>]*)>(.*?)</form>", html, re.I | re.S):
        attrs, body = match.group(1), match.group(2)
        action_m = re.search(r'action=["\']([^"\']+)["\']', attrs, re.I)
        method_m = re.search(r'method=["\']([^"\']+)["\']', attrs, re.I)
        action = urljoin(base, action_m.group(1) if action_m else "")
        method = (method_m.group(1) if method_m else "POST").upper()
        blob = f"{attrs} {body}".lower()
        if "reload" in blob or "extend" in blob or "renew" in blob:
            fields = {
                name: value
                for name, value in re.findall(
                    r'<input[^>]+name=["\']([^"\']+)["\'][^>]*value=["\']([^"\']*)["\']',
                    body,
                    re.I,
                )
            }
            found.append((method, action or base, fields))
    return found


def parse_jobs_page(html: str) -> dict:
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    low = text.lower()
    running = bool(re.search(r"\brunning\b", low))
    waiting = (not running) and bool(
        re.search(r"\b(waiting|queued|pending|in queue)\b", low)
    )
    port = None
    port_m = re.search(r":(\d{4,5})\b", text)
    if port_m:
        port = int(port_m.group(1))
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


def portal_session(*, force: bool = False):
    global _SESSION, _SESSION_AT
    creds = portal_creds()
    if creds is None:
        return None, {"ok": False, "error": "NEED_PORTAL_CREDS"}
    try:
        import requests
    except ImportError:
        return None, {"ok": False, "error": "requests_missing"}
    age = time.time() - _SESSION_AT
    if _SESSION is not None and not force and age < RELOGIN_S:
        return _SESSION, {"ok": True, "reused": True, "age_s": round(age)}
    user, password = creds
    session = requests.Session()
    try:
        session.post(
            f"{PORTAL}/auth/login",
            data={"username": user, "password": password},
            timeout=20,
            allow_redirects=True,
        )
        jobs = session.get(f"{PORTAL}/jobs/", timeout=20)
    except Exception as error:  # portal drop — keep looping
        _SESSION = None
        return None, {"ok": False, "error": f"portal_down:{error.__class__.__name__}"}
    parsed = parse_jobs_page(jobs.text)
    if jobs.status_code != 200 or not parsed["logged_in"]:
        _SESSION = None
        return None, {"ok": False, "error": "login_failed", "status": jobs.status_code}
    _SESSION = session
    _SESSION_AT = time.time()
    return session, {"ok": True, "reused": False, **parsed}


def portal_jobs() -> dict:
    session, info = portal_session()
    if session is None:
        return info
    try:
        jobs = session.get(f"{PORTAL}/jobs/", timeout=20)
    except Exception as error:
        session2, info2 = portal_session(force=True)
        if session2 is None:
            return {**info, "ok": False, "error": f"portal_down:{error.__class__.__name__}"}
        try:
            jobs = session2.get(f"{PORTAL}/jobs/", timeout=20)
            session = session2
        except Exception as error2:
            return {"ok": False, "error": f"portal_down:{error2.__class__.__name__}"}
    parsed = parse_jobs_page(jobs.text)
    if not parsed["logged_in"]:
        session2, info2 = portal_session(force=True)
        if session2 is None:
            return info2
        try:
            jobs = session2.get(f"{PORTAL}/jobs/", timeout=20)
            parsed = parse_jobs_page(jobs.text)
        except Exception as error:
            return {"ok": False, "error": f"portal_down:{error.__class__.__name__}"}
    return {"ok": True, **parsed, "html_len": len(jobs.text)}


def portal_reload(state: dict) -> dict:
    session, info = portal_session(force=True)
    if session is None:
        return info
    try:
        jobs = session.get(f"{PORTAL}/jobs/", timeout=20)
    except Exception as error:
        return {"ok": False, "error": f"portal_down:{error.__class__.__name__}"}
    sid = "dgx-h200-1"
    try:
        status = session.get(f"{PORTAL}/container/status", timeout=20).json()
        if isinstance(status, dict) and status:
            sid = next(iter(status))
        resp = session.post(f"{PORTAL}/container/{sid}/start", timeout=60)
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    except Exception as error:
        return {"ok": False, "error": f"portal_down:{error.__class__.__name__}"}
    if not (isinstance(body, dict) and body.get("ok")):
        return {
            "ok": False,
            "error": body.get("msg") if isinstance(body, dict) else "reload_failed",
            "status": resp.status_code,
            "sid": sid,
        }
    parsed = parse_jobs_page("")
    try:
        after = session.get(f"{PORTAL}/container/status", timeout=20).json().get(sid, {})
        parsed["running"] = after.get("state") == "running"
        parsed["waiting"] = after.get("state") in {"queued", "waiting", "pending"}
        parsed["ssh_port"] = after.get("ssh_port")
        parsed["portal_remaining_h"] = after.get("lease_hours_left")
    except Exception:
        after = {}
    if parsed.get("ssh_port"):
        state["ssh_port"] = parsed["ssh_port"]
        PORT_FILE.write_text(f"{state['ssh_port']}\n", encoding="utf-8")
    return portal_reload_result(
        sid, resp.status_code, parsed, state.get("ssh_port")
    )


def default_state() -> dict:
    return {
        "lease_start": "2026-09-17T01:58:47+09:00",
        "lease_hours": 10.0,
        "reload_remaining_hours": RELOAD_REMAINING_H,
        "host": "127.0.0.1",
        "ssh_port": 23023,
        "ssh_user": "ext_csh",
        "ssh_key": str(KEY_DEFAULT),
        "label": "DGX H200 #1",
        "last_reload_at": None,
        "idle_since": None,
        "actions": [],
    }


def resume_gpu_if_down(state: dict, actions: list[str], reason: str) -> None:
    if not ssh_up(state) or gpu_queue_alive(state):
        return
    start_gpu(state)
    actions.append("resume_gpu")
    log_event(f"EVENT EXP_DOWN resume ME GPU queue ({reason})")


def do_reload(state: dict, _gpu_held: bool = False) -> list[str]:
    """Extend the lease in place. Do not SIGTERM trains first.

    Yesterday's hole: SIGTERM then crash in portal_reload left GPU empty.
    POST /start keeps the same SSH port when the box is already running.
    """
    actions: list[str] = []
    try:
        reload_info = portal_reload(state)
        actions.append(f"reload:{reload_info.get('error') or reload_info.get('ok')}")
        if not reload_info.get("ok"):
            log_event(f"EVENT RELOAD_FAIL {reload_info.get('error')}")
            resume_gpu_if_down(state, actions, "reload_fail")
            return actions
        if reload_info.get("lease_hours_left") is not None:
            rem = float(reload_info["lease_hours_left"])
            state["lease_start"] = sync_lease_from_remaining(state, rem)
            state["remaining_hours"] = rem
        state["last_reload_at"] = now().isoformat(timespec="seconds")
        log_event("EVENT RELOAD_OK waiting for GPU allocation")
        for _ in range(36):
            if ssh_up(state):
                break
            time.sleep(5)
        if ssh_up(state):
            actions.append("reload_allocated")
            resume_gpu_if_down(state, actions, "after_reload")
        else:
            actions.append("reload_waiting")
            log_event("EVENT PORTAL_QUEUE after reload; poll until GPU is held")
        return actions
    except Exception as error:
        log_event(f"EVENT RELOAD_ERROR {error.__class__.__name__}: {error}")
        actions.append(f"reload_error:{error.__class__.__name__}")
        resume_gpu_if_down(state, actions, "reload_error")
        return actions


def recover_after_tick_error(*, apply: bool) -> None:
    if not apply:
        return
    state = {**default_state(), **load_state()}
    try:
        resume_gpu_if_down(state, [], "tick_error")
    except Exception as error:
        log_event(f"EVENT RECOVER_FAIL {error.__class__.__name__}: {error}")


def tick(*, apply: bool) -> dict:
    state = {**default_state(), **load_state()}
    start = parse_iso(state["lease_start"])
    rem = remaining_hours(start, float(state["lease_hours"]))
    rem_from_portal = False
    portal = portal_jobs()
    box = {}
    try:
        session, _ = portal_session()
        if session is not None:
            boxes = session.get(f"{PORTAL}/container/status", timeout=20).json()
            box = parse_container_box(boxes)
            if box.get("ssh_port"):
                portal["ssh_port"] = box["ssh_port"]
            if box.get("portal_remaining_h") is not None:
                portal["portal_remaining_h"] = box["portal_remaining_h"]
            if box:
                portal["running"] = box["running"]
                portal["waiting"] = box["waiting"]
                portal["ok"] = True
    except Exception:
        box = {}
    if portal.get("ssh_port"):
        state["ssh_port"] = portal["ssh_port"]
        PORT_FILE.write_text(f"{state['ssh_port']}\n", encoding="utf-8")
    if portal.get("portal_remaining_h") is not None and portal.get("running"):
        rem = float(portal["portal_remaining_h"])
        state["lease_start"] = sync_lease_from_remaining(state, rem)
        rem_from_portal = True
    prev_qlen = (state.get("after") or {}).get("queue_length")
    if box.get("queue_length") and box.get("queue_length") != prev_qlen:
        log_event(f"EVENT PORTAL_QUEUE_LEN {box['queue_length']}")
    up = ssh_up(state)
    queue = gpu_queue_alive(state) if up else False
    usage = box_gpu_usage(state) if up else []
    util_held = any(row["util"] >= UTIL_HELD for row in usage)
    portal_ok = bool(portal.get("ok"))
    if not portal_ok:
        portal_session(force=True)
        portal = portal_jobs()
        portal_ok = bool(portal.get("ok"))
        if portal.get("ssh_port"):
            state["ssh_port"] = portal["ssh_port"]
            PORT_FILE.write_text(f"{state['ssh_port']}\n", encoding="utf-8")
            up = ssh_up(state)
            queue = gpu_queue_alive(state) if up else False
            usage = box_gpu_usage(state) if up else []
            util_held = any(row["util"] >= UTIL_HELD for row in usage)
    portal_queued = bool(portal_ok and portal.get("waiting") and not portal.get("running"))
    gpu_held = bool(up and (queue or host_gpu_busy() or util_held) and not portal_queued)
    just_allocated = gpu_held and not bool(state.get("gpu_held"))
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
        rem_from_portal=rem_from_portal,
        reload_cooldown_ok=reload_cooldown_ok(state.get("last_reload_at")),
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
            f"exp_up={int(queue)} util={usage}"
        )
    if apply:
        if gpu_held and cpu_queue_pid():
            soft_stop_cpu()
            actions.append("stop_cpu")
            log_event("EVENT STOP_CPU gpu held again")
        if action == "reload":
            actions.extend(do_reload(state, gpu_held))
            rem = float(state.get("remaining_hours") or rem)
            up = ssh_up(state)
            queue = gpu_queue_alive(state) if up else False
        elif action == "cpu":
            pid = start_cpu()
            actions.append(f"cpu_pid={pid}")
            log_event(f"EVENT CPU_FAILOVER pid={pid} taken_s={taken_s:.0f}")
        if action != "cpu" and gpu_held and up and not queue:
            resume_gpu_if_down(state, actions, "tick")
    payload = {
        **state,
        "ts": now().isoformat(timespec="seconds"),
        "remaining_hours": round(rem, 3),
        "due": action == "reload",
        "ssh_up": up,
        "gpu_queue_alive": queue,
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
        "rem_from_portal": rem_from_portal,
        "after": box.get("raw") or state.get("after"),
    }
    save_state(payload)
    utils = "/".join(str(int(row["util"])) for row in usage) or "-"
    log_event(
        f"tick action={action} portal={int(portal_ok)} queued={int(portal_queued)} "
        f"gpu_held={int(gpu_held)} rem={rem:.2f} ssh={int(up)} "
        f"exp={int(queue)} util={utils} cpu={cpu_queue_pid() or '-'}"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    argv = []
    if args.watch:
        argv.append("--watch")
    if args.apply:
        argv.append("--apply")
    argv.extend(["--interval", str(args.interval)])
    if args.watch:
        (STATE_DIR / "lease.watch.pid").write_text(f"{os.getpid()}\n")
        while True:
            maybe_reexec(argv)
            try:
                tick(apply=bool(args.apply))
            except Exception as error:
                log_event(f"EVENT TICK_ERROR {error.__class__.__name__}: {error}")
                recover_after_tick_error(apply=bool(args.apply))
            time.sleep(max(10, args.interval))
    tick(apply=bool(args.apply))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
