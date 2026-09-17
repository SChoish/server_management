"""Transition regressions plus real Linux process/checkpoint handoff tests."""
import copy
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parents[1] / 'scripts/hosts/beyondg'
sys.path.insert(0, str(HERE))
import queue_worker as worker
from watch_beyondg_lease import (
    AuthError, Controller, Portal, PortalError, Workers, load_config, parse_status,
)


def box(state='running', remaining=9.0):
    return {'sid':'dgx-h200-2', 'state':state, 'remaining_h':remaining,
            'host':'127.0.0.1', 'ssh_port':23021 if state == 'running' else None}


class FakePortal:
    def __init__(self, current=None):
        self.current = current or box()
        self.posts = 0
        self.error = None
        self.post_error = None
        self.after_post = None

    def status(self):
        if self.error:
            raise PortalError(self.error)
        return copy.deepcopy(self.current)

    def start(self):
        self.posts += 1
        if self.after_post:
            self.current = self.after_post
        if self.post_error:
            raise PortalError(self.post_error)
        return {'ok':True}


class FakeWorkers:
    def __init__(self):
        self.cpu = []
        self.gpu = []
        self.local_gpu = []
        self.remote_cpu = []
        self.ssh_ok = True
        self.gpu_available = True
        self.stop_pending = False
        self.calls = []

    def local(self, action='status', mode='cpu'):
        self.calls.append(('local', action, mode))
        if action == 'start':
            if self.local_gpu:
                return {'ok':False,'error':'gpu_still_running'}
            self.cpu = ['cpu:1']
        if action == 'stop':
            if mode == 'cpu' and not self.stop_pending:
                self.cpu = []
            if mode == 'gpu':
                self.local_gpu = []
        return {'ok':True,'cpu':list(self.cpu),'gpu':list(self.local_gpu),
                'stopped':not (self.cpu if mode == 'cpu' else self.local_gpu)}

    def remote(self, current, action='status', mode='gpu'):
        self.calls.append(('remote', action, mode))
        if not self.ssh_ok:
            return {'ok':False, 'error':'ssh_failed'}
        if action == 'stop':
            if mode == 'cpu':
                self.remote_cpu = []
            else:
                self.gpu = []
        if action == 'start':
            if self.cpu or self.remote_cpu:
                raise AssertionError('GPU started before CPU checkpoint writer exited')
            self.gpu = ['gpu:1']
        return {'ok':True,'cpu':list(self.remote_cpu),'gpu':list(self.gpu),
                'gpu_available':self.gpu_available,
                'stopped':not (self.remote_cpu if mode == 'cpu' else self.gpu)}


class Transitions(unittest.TestCase):
    def setUp(self):
        self.portal = FakePortal()
        self.workers = FakeWorkers()
        self.now = 10_000
        self.control = Controller({}, self.portal, self.workers, clock=lambda:self.now)

    def tick(self):
        return self.control.tick(apply=True)

    def test_initial_queue_waits_without_cpu(self):
        self.portal.current = box('queued', None)
        self.assertEqual(self.tick()['phase'], 'wait_gpu')
        self.assertEqual(self.portal.posts, 0)
        self.assertFalse(self.workers.cpu)

    def test_empty_new_gpu_starts_queue_without_utilization(self):
        result = self.tick()
        self.assertEqual(result['phase'], 'gpu_running')
        self.assertTrue(self.workers.gpu)

    def test_renew_at_6_5_in_place_and_preserve_jobs(self):
        self.workers.gpu = ['existing:1']
        self.portal.current = box(remaining=6.5)
        self.portal.after_post = box(remaining=10.0)
        result = self.tick()
        self.assertEqual(self.portal.posts, 1)
        self.assertIn('reload_confirmed', result['events'])
        self.assertFalse(any(call[1] == 'stop' for call in self.workers.calls))

    def test_dont_renew_at_6_51_or_from_missing_remaining(self):
        for remaining in (6.51, None):
            self.portal.current = box(remaining=remaining)
            self.tick()
        self.assertEqual(self.portal.posts, 0)

    def test_failed_reload_still_resumes_and_backs_off(self):
        self.portal.current = box(remaining=6.4)
        self.portal.post_error = 'http_503'
        result = self.tick()
        self.assertEqual(result['phase'], 'gpu_running')
        self.assertFalse(any(call[1] == 'stop' for call in self.workers.calls))
        self.now += 30
        self.tick()
        self.assertEqual(self.portal.posts, 1)
        self.now += 31
        self.tick()
        self.assertEqual(self.portal.posts, 2)

    def test_lost_gpu_runs_cpu_while_requeued(self):
        self.tick()
        self.workers.gpu = []
        self.portal.current = box('stopped', None)
        self.portal.after_post = box('queued', None)
        result = self.tick()
        self.assertTrue(result['loss_confirmed'])
        self.assertEqual(result['phase'], 'cpu_waiting_gpu')
        self.assertTrue(self.workers.cpu)
        for _ in range(3):
            self.tick()
        self.assertEqual(self.portal.posts, 1, 'Do not resubmit an existing queue entry')
        self.assertTrue(self.workers.cpu)

    def test_cpu_shutdown_must_finish_before_gpu_start(self):
        self.control.state.update(gpu_seen=True, loss_confirmed=True)
        self.workers.cpu = ['cpu:1']
        self.workers.stop_pending = True
        self.assertEqual(self.tick()['phase'], 'stopping_cpu')
        self.assertFalse(self.workers.gpu)
        self.workers.stop_pending = False
        self.assertEqual(self.tick()['phase'], 'gpu_running')
        stop_index = self.workers.calls.index(('local','stop','cpu'))
        start_index = self.workers.calls.index(('remote','start','gpu'))
        self.assertLess(stop_index, start_index)

    def test_ssh_failure_alone_does_not_start_cpu(self):
        self.control.state['gpu_seen'] = True
        self.workers.ssh_ok = False
        self.assertEqual(self.tick()['phase'], 'wait_docker')
        self.assertFalse(self.workers.cpu)
        self.assertNotIn('loss_confirmed', self.control.state)

    def test_portal_failure_alone_does_not_start_cpu(self):
        self.control.state['gpu_seen'] = True
        self.portal.error = 'network_down'
        self.assertEqual(self.tick()['phase'], 'wait_portal')
        self.assertEqual(self.workers.calls, [])

    def test_cpu_keeps_running_until_docker_gpu_is_ready(self):
        self.workers.cpu = ['cpu:1']
        self.workers.gpu_available = False
        self.assertEqual(self.tick()['phase'], 'wait_docker')
        self.assertTrue(self.workers.cpu)
        self.assertNotIn(('local','stop','cpu'), self.workers.calls)

    def test_initial_stopped_requests_gpu_without_cpu(self):
        self.portal.current = box('stopped', None)
        self.portal.after_post = box('queued', None)
        self.assertEqual(self.tick()['phase'], 'wait_gpu')
        self.assertEqual(self.portal.posts, 1)
        self.assertFalse(self.workers.cpu)

    def test_reload_that_requeues_transitions_to_cpu(self):
        self.portal.current = box(remaining=6.5)
        self.portal.after_post = box('queued', None)
        self.assertEqual(self.tick()['phase'], 'cpu_waiting_gpu')
        self.assertTrue(self.workers.cpu)

    def test_lost_local_gpu_must_stop_before_cpu(self):
        self.control.state['gpu_seen'] = True
        self.portal.current = box('queued', None)
        self.workers.local_gpu = ['gpu:old']
        self.tick()
        self.assertLess(self.workers.calls.index(('local','stop','gpu')),
                        self.workers.calls.index(('local','start','cpu')))

    def test_legacy_state_migrates_loss_evidence(self):
        self.control = Controller({}, self.portal, self.workers, {'gpu_held':True})
        self.portal.current = box('queued', None)
        self.assertEqual(self.tick()['phase'], 'cpu_waiting_gpu')

    def test_post_timeout_rechecks_authoritative_state(self):
        self.portal.current = box(remaining=6.4)
        self.portal.after_post = box('queued', None)
        self.portal.post_error = 'timeout_after_acceptance'
        self.assertEqual(self.tick()['phase'], 'cpu_waiting_gpu')

    def test_observe_mode_has_no_process_or_post_mutations(self):
        self.portal.current = box(remaining=6.4)
        self.control.tick(apply=False)
        self.assertEqual(self.workers.calls, [])
        self.assertEqual(self.portal.posts, 0)

    def test_allocation_key_scope_changes_only_on_reallocation(self):
        self.tick()
        old = self.control.state['allocation_token']
        self.tick()
        self.assertEqual(old, self.control.state['allocation_token'])
        self.portal.current = box('queued', None)
        self.tick()
        self.portal.current = box()
        self.tick()
        self.assertNotEqual(old, self.control.state['allocation_token'])


class PortalTests(unittest.TestCase):
    def test_selects_exact_sid_independent_of_dict_order(self):
        rows = {'dgx-h200-1':{'state':'running','ssh_port':23023},
                'dgx-h200-2':{'state':'queued'}}
        for payload in (rows, dict(reversed(list(rows.items())))):
            self.assertEqual(parse_status(payload, 'dgx-h200-2')['state'], 'queued')

    def test_missing_target_or_unknown_state_is_not_loss(self):
        for payload in ({'error':'unauthorized'}, {}, {'dgx-h200-2':{'state':'unknown'}},
                        {'dgx-h200-2':{'state':'running','lease_hours_left':'nan'}}):
            with self.assertRaises(PortalError):
                parse_status(payload, 'dgx-h200-2')

    def test_http401_invalidates_login(self):
        portal = Portal({'env_file':'/nonexistent'})
        portal.base = 'http://example.invalid'
        portal.opener = Mock()
        portal.opener.open.side_effect = urllib.error.HTTPError(
            portal.base, 401, 'Unauthorized', {}, None)
        with self.assertRaises(AuthError):
            portal.request('GET','/container/status')

    def test_expired_session_reauthenticates(self):
        cfg = {'env_file':'/nonexistent', 'portal_urls':['http://example.invalid'], 'sid':'dgx-h200-2'}
        portal = Portal(cfg)
        portal.base, portal.opener, portal.login_at = cfg['portal_urls'][0], Mock(), time.time()
        with patch.object(portal, 'request', side_effect=[AuthError('expired'), {'dgx-h200-2':{'state':'queued'}}]), \
                patch.object(portal, 'login') as login:
            self.assertEqual(portal.status()['state'], 'queued')
            login.assert_called_once()

    def test_start_posts_only_configured_sid(self):
        portal = Portal({'sid':'dgx-h200-2'})
        portal.opener = Mock()
        with patch.object(portal,'request',return_value={'ok':True}) as request:
            portal.start()
            request.assert_called_once_with('POST','/container/dgx-h200-2/start',{})

    def test_checked_in_profiles_load_and_pin_distinct_servers(self):
        a, b = load_config('ext_csh'), load_config('ext_csv')
        self.assertEqual(a['sid'], 'dgx-h200-1')
        self.assertEqual(b['sid'], 'dgx-h200-2')
        self.assertNotIn('${REPO}', json.dumps(b))


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.script = self.root / 'trainer.py'
        self.cfg = {'state_dir':str(self.root / 'state'), 'start_retry_s':0,
                    'process_patterns':[str(self.script)]}
        for mode in ('cpu','gpu'):
            self.cfg[mode] = {'queue_patterns':[str(self.root / (mode+'.sh'))],
                              'command':[sys.executable,str(self.script)],
                              'cwd':str(self.root),'log':str(self.root/(mode+'.log'))}

    def test_probe_command_text_is_never_a_training_process(self):
        for argv in (['bash','-c',f'pgrep -f {self.script}'],
                     ['ssh','host',f'python {self.script}'],
                     ['python3','-c',f'print("{self.script}")'],
                     ['python3','-',json.dumps(self.cfg)]):
            self.assertIsNone(worker.classify(argv,{},self.cfg))

    def test_cpu_flag_forms_and_environment(self):
        for args, env in ((['--device','cpu'],{}), (['--device=cpu'],{}),
                          (['--cpu-jobs','2'],{}), ([],{'JAX_PLATFORMS':'cpu'}),
                          ([],{'CUDA_VISIBLE_DEVICES':''})):
            self.assertEqual(worker.classify(['python','-u',str(self.script),*args],env,self.cfg),'cpu')
        self.assertEqual(worker.classify(['python',str(self.script),'--cpu-jobs','0'],{},self.cfg),'gpu')

    def test_checkpoint_writer_finishes_before_gpu_and_no_duplicate_start(self):
        self.script.write_text(
            'import signal,time\nfrom pathlib import Path\n'
            'root=Path(__file__).parent\n'
            'def stop(*args):\n'
            '    with (root/"signals").open("a") as f: f.write("TERM\\n")\n'
            '    (root/"saving").touch()\n'
            '    while not (root/"release").exists(): time.sleep(.02)\n'
            '    (root/"checkpoint").write_text("saved")\n'
            '    raise SystemExit(0)\n'
            'signal.signal(signal.SIGTERM,stop)\n'
            'while True: time.sleep(.02)\n')
        owned = []
        try:
            first = worker.supervise(self.cfg,'start','cpu')
            self.assertTrue(first['cpu'])
            owned += first['cpu']
            repeated = worker.supervise(self.cfg,'start','cpu')
            self.assertTrue(repeated['already_running'])
            self.assertEqual(first['cpu'], repeated['cpu'])
            stopping = worker.supervise(self.cfg,'stop','cpu')
            self.assertFalse(stopping['stopped'])
            again = worker.supervise(self.cfg,'stop','cpu')
            self.assertFalse(again['stopped'])
            blocked = worker.supervise(self.cfg,'start','gpu')
            self.assertEqual(blocked['error'],'cpu_still_running')
            (self.root/'release').touch()
            deadline=time.monotonic()+4
            while worker.supervise(self.cfg,'status','cpu')['cpu'] and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertEqual((self.root/'checkpoint').read_text(),'saved')
            self.assertEqual((self.root/'signals').read_text(),'TERM\n')
            gpu = worker.supervise(self.cfg,'start','gpu')
            owned += gpu['gpu']
            self.assertTrue(gpu['gpu'])
            self.assertFalse(gpu['cpu'])
        finally:
            (self.root/'release').touch()
            for token in owned:
                pid=int(token.split(':')[0])
                current=worker.read_process(pid)
                if current and worker.identity(current)==token:
                    os.kill(current['signal_pid'],signal.SIGTERM)
            deadline=time.monotonic()+2
            while time.monotonic()<deadline:
                live=[t for t in owned if worker.read_process(int(t.split(':')[0]))]
                if not live: break
                time.sleep(.02)
            worker.reap_children()

    def test_failed_queue_does_not_report_running(self):
        self.cfg['cpu']['command']=['bash',str(self.root/'missing.sh')]
        result=worker.supervise(self.cfg,'start','cpu')
        self.assertFalse(result['ok'])
        self.assertEqual(result['error'],'queue_launch_failed')
        self.assertFalse(result['cpu'])

    def test_remote_stdin_protocol_and_proc_mapping(self):
        proc = subprocess.run(
            [sys.executable, '-', json.dumps(self.cfg), 'status', 'gpu'],
            input=(HERE / 'queue_worker.py').read_text(),
            capture_output=True, text=True, check=True,
        )
        result = json.loads(proc.stdout)
        self.assertTrue(result['ok'])
        self.assertEqual(result['cpu'], [])
        self.assertEqual(result['gpu'], [])
        current = worker.read_process(worker.own_proc_pid())
        self.assertEqual(current['signal_pid'], os.getpid())


if __name__ == '__main__':
    unittest.main()
