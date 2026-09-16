import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "hosts" / "beyondg"))

from watch_beyondg_lease import (
    decide,
    exp_really_alive,
    is_real_exp_cmd,
    parse_container_status,
    should_resume_gpu,
)


def test_reload_only_when_portal_ok_and_remaining_le_6_5():
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
            portal_ok=True,
            portal_queued=False,
            remaining_h=6.51,
            taken_s=0,
        )
        == "watch"
    )
    assert (
        decide(
            gpu_held=True,
            portal_ok=False,
            portal_queued=False,
            remaining_h=5.0,
            taken_s=0,
        )
        == "watch"
    )


def test_portal_queued_waits_even_if_taken_is_long():
    assert (
        decide(
            gpu_held=False,
            portal_ok=True,
            portal_queued=True,
            remaining_h=9.0,
            taken_s=10_000,
        )
        == "wait_queue"
    )


def test_cpu_failover_only_when_not_queued_and_taken_ge_300():
    assert (
        decide(
            gpu_held=False,
            portal_ok=False,
            portal_queued=False,
            remaining_h=9.0,
            taken_s=299,
        )
        == "wait_gpu"
    )
    assert (
        decide(
            gpu_held=False,
            portal_ok=False,
            portal_queued=False,
            remaining_h=9.0,
            taken_s=300,
        )
        == "cpu"
    )
    assert (
        decide(
            gpu_held=False,
            portal_ok=True,
            portal_queued=True,
            remaining_h=9.0,
            taken_s=300,
        )
        == "wait_queue"
    )
    assert (
        decide(
            gpu_held=True,
            portal_ok=False,
            portal_queued=False,
            remaining_h=9.0,
            taken_s=300,
        )
        == "watch"
    )


def test_parse_container_status_running_and_queued():
    running = parse_container_status(
        {
            "dgx-h200-2": {
                "state": "running",
                "host": "166.104.28.73",
                "ssh_port": 23021,
                "lease_hours_left": 8.8,
            }
        }
    )
    assert running["portal"] == "ok"
    assert running["ssh_port"] == 23021
    assert running["remaining_h"] == 8.8
    queued = parse_container_status(
        {
            "dgx-h200-2": {
                "state": "queued",
                "queue_position": 1,
                "queue_length": 2,
            }
        }
    )
    assert queued["portal"] == "queued"


def test_recent_reload_does_not_loop_on_stale_remaining():
    assert (
        decide(
            gpu_held=True,
            portal_ok=True,
            portal_queued=False,
            remaining_h=6.5,
            taken_s=0,
            seconds_since_reload=60,
        )
        == "watch"
    )
    assert (
        decide(
            gpu_held=True,
            portal_ok=True,
            portal_queued=False,
            remaining_h=6.5,
            taken_s=0,
            seconds_since_reload=31 * 60,
        )
        == "reload"
    )


def test_pgrep_probe_is_not_a_live_experiment():
    assert not is_real_exp_cmd("pgrep -f AMO-fql/train.py")
    assert not is_real_exp_cmd(
        "python3 /home/ext_csv/MPI_sweep/scripts/hosts/beyondg/watch_beyondg_lease.py --watch"
    )
    assert is_real_exp_cmd("python /home/ext_csv/AMO-fql/train.py --device cuda:0")
    assert not is_real_exp_cmd("python /home/ext_csv/AMO-fql/train.py --device=cpu")


def test_launcher_without_gpu_util_is_not_alive():
    cmds = ["python launch_fql_amo_jax_loco9_tinit5_alrgrid.py --gpus 0,1"]
    assert not exp_really_alive(cmds, [{"util": 0.0}, {"util": 0.0}])
    assert exp_really_alive(cmds, [{"util": 66.0}, {"util": 15.0}])
    assert exp_really_alive(
        ["python /home/ext_csv/AMO-fql/train.py --device cuda:0"],
        [{"util": 0.0}, {"util": 0.0}],
    )


def test_resume_gpu_after_reload_when_ssh_up_and_exp_down():
    assert should_resume_gpu(
        ssh_up=True,
        portal_running=True,
        portal_queued=False,
        exp_alive=False,
    )
    assert not should_resume_gpu(
        ssh_up=True,
        portal_running=True,
        portal_queued=False,
        exp_alive=True,
    )
    assert not should_resume_gpu(
        ssh_up=True,
        portal_running=False,
        portal_queued=True,
        exp_alive=False,
    )
