from datetime import datetime, timedelta, timezone

from scripts.hosts.beyondg.watch_beyondg_lease import (
    GPU_ALIVE_SH,
    PORTAL,
    decide,
    parse_container_box,
    portal_reload_result,
    remaining_hours,
    reload_cooldown_ok,
    start_gpu_remote,
    sync_lease_from_remaining,
)


def test_remaining_hits_6_5_after_3_5h(monkeypatch):
    kst = timezone(timedelta(hours=9))
    start = datetime(2026, 9, 17, 1, 58, 47, tzinfo=kst)
    now = start + timedelta(hours=3, minutes=30)
    monkeypatch.setattr(
        "scripts.hosts.beyondg.watch_beyondg_lease.now",
        lambda: now,
    )
    rem = remaining_hours(start, 10.0)
    assert abs(rem - 6.5) < 1e-9


def test_reload_only_if_portal_up_and_before_6h():
    assert (
        decide(
            gpu_held=True,
            portal_ok=True,
            portal_queued=False,
            remaining_h=6.5,
            taken_s=0,
        )
        == "reload"
    )
    assert (
        decide(
            gpu_held=True,
            portal_ok=False,
            portal_queued=False,
            remaining_h=6.5,
            taken_s=0,
        )
        == "watch"
    )


def test_queue_waits_and_does_not_cpu():
    assert (
        decide(
            gpu_held=False,
            portal_ok=True,
            portal_queued=True,
            remaining_h=6.0,
            taken_s=3600,
        )
        == "wait_queue"
    )


def test_cpu_only_when_taken_and_not_queued():
    assert (
        decide(
            gpu_held=False,
            portal_ok=False,
            portal_queued=False,
            remaining_h=8.0,
            taken_s=300,
        )
        == "cpu"
    )
    assert (
        decide(
            gpu_held=False,
            portal_ok=True,
            portal_queued=False,
            remaining_h=8.0,
            taken_s=10,
        )
        == "wait_gpu"
    )


def test_no_reload_without_portal_remaining():
    assert (
        decide(
            gpu_held=True,
            portal_ok=True,
            portal_queued=False,
            remaining_h=6.5,
            taken_s=0,
            rem_from_portal=False,
        )
        == "watch"
    )


def test_no_reload_during_cooldown():
    assert (
        decide(
            gpu_held=True,
            portal_ok=True,
            portal_queued=False,
            remaining_h=6.4,
            taken_s=0,
            rem_from_portal=True,
            reload_cooldown_ok=False,
        )
        == "watch"
    )


def test_reload_cooldown_gap():
    kst = timezone(timedelta(hours=9))
    start = datetime(2026, 9, 17, 6, 0, 7, tzinfo=kst)
    assert not reload_cooldown_ok(start.isoformat(), start + timedelta(minutes=10))
    assert reload_cooldown_ok(start.isoformat(), start + timedelta(minutes=31))


def test_portal_reload_result_has_literal_url():
    out = portal_reload_result(
        "dgx-h200-1",
        200,
        {"waiting": False, "running": True, "portal_remaining_h": 10.0},
        23023,
    )
    assert out["url"] == f"{PORTAL}/container/dgx-h200-1/start"
    assert out["method"] == "POST"
    assert out["ok"] is True
    assert out["lease_hours_left"] == 10.0
    assert out["port"] == 23023


def test_sync_lease_from_remaining():
    kst = timezone(timedelta(hours=9))
    when = datetime(2026, 9, 17, 8, 35, 47, tzinfo=kst)
    start = sync_lease_from_remaining({"lease_hours": 10.0}, 9.9, when)
    assert start == "2026-09-17T08:29:47+09:00"


def test_parse_container_box_remaining():
    box = parse_container_box(
        {
            "dgx-h200-1": {
                "state": "running",
                "ssh_port": 23023,
                "lease_hours_left": 9.8,
                "queue_length": 1,
            }
        }
    )
    assert box["running"] is True
    assert box["portal_remaining_h"] == 9.8
    assert box["ssh_port"] == 23023
    assert box["queue_length"] == 1


def test_start_gpu_skips_if_alive():
    remote = start_gpu_remote("/tmp/q.sh", "/tmp/q.log")
    assert remote.startswith(f"{GPU_ALIVE_SH} && exit 0;")
