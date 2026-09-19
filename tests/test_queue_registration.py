"""Queue replacement must not hide old writers or restart live training."""
import copy
import contextlib
import io
import json
import signal
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/hosts/beyondg'))
import queue_worker as worker
import watch_beyondg_lease as lease
from register_queue import register


class ReloadTests(unittest.TestCase):
    def setUp(self):
        self.cfg = lease.load_config('ext_csh')
        self.portal = Mock()
        self.portal.status.return_value = {
            'state': 'running', 'ssh_port': 23023, 'remaining_h': 8.0}
        self.workers = Mock()
        self.workers.local.return_value = {'ok': True, 'cpu': [], 'gpu': []}
        self.workers.remote.return_value = {'ok': True, 'cpu': [], 'gpu': []}
        self.control = lease.Controller(self.cfg, self.portal, self.workers,
                                        {'gpu_seen': True, 'allocation_token': '123'})
        self.next = copy.deepcopy(self.cfg)
        self.next['worker']['gpu']['command'] = ['bash', '/next/gpu.sh']
        self.next['worker']['cpu']['command'] = ['bash', '/next/cpu.sh']

    def test_identical_config_needs_no_probe(self):
        self.assertIsNone(lease.refresh_controller(self.control, copy.deepcopy(self.cfg)))
        self.assertFalse(self.workers.mock_calls)

    def test_live_local_cpu_defers_without_signalling(self):
        self.workers.local.return_value['cpu'] = ['123:456']
        self.assertEqual(lease.refresh_controller(self.control, self.next),
                         'config_reload_deferred:active_queue')
        self.assertIs(self.control.cfg, self.cfg)
        self.assertEqual(self.workers.mock_calls, [unittest.mock.call.local()])

    def test_live_remote_gpu_defers_without_signalling(self):
        self.workers.remote.return_value['gpu'] = ['456:789']
        self.assertEqual(lease.refresh_controller(self.control, self.next),
                         'config_reload_deferred:active_queue')
        self.assertIs(self.control.workers, self.workers)
        self.assertEqual(self.workers.remote.call_args.args[0]['allocation_token'], '123')
        self.assertEqual(len(self.workers.remote.call_args.args), 1)

    def test_detached_old_writer_blocks_new_queue(self):
        self.workers.remote.return_value['cpu'] = ['999:888']
        self.assertEqual(lease.refresh_controller(self.control, self.next),
                         'config_reload_deferred:active_queue')
        self.assertIs(self.control.cfg, self.cfg)

    def test_finished_queue_adopts_latest_config_and_preserves_lease(self):
        with patch.object(lease, 'Workers') as factory, patch.object(lease, 'Portal'):
            self.assertEqual(lease.refresh_controller(self.control, self.next), 'config_reloaded')
        factory.assert_called_once_with(self.next)
        self.assertIs(self.control.cfg, self.next)
        self.assertEqual(self.control.state['allocation_token'], '123')
        self.assertTrue(self.control.state['gpu_seen'])
        self.assertEqual(self.control.set_phase('gpu_starting')['queue_commands']['gpu'],
                         ['bash', '/next/gpu.sh'])

    def test_unknown_remote_or_portal_keeps_old_target(self):
        self.workers.remote.return_value = {'ok': False, 'error': 'ssh_failed'}
        self.assertEqual(lease.refresh_controller(self.control, self.next),
                         'config_reload_deferred:remote_unknown')
        self.portal.status.side_effect = lease.PortalError('network')
        self.assertEqual(lease.refresh_controller(self.control, self.next),
                         'config_reload_deferred:portal_unknown')
        self.assertIs(self.control.cfg, self.cfg)

    def test_pattern_update_for_same_queue_applies_without_stop(self):
        updated = copy.deepcopy(self.cfg)
        updated['worker']['process_patterns'].append('/project/another_launcher.py')
        with patch.object(lease, 'Workers'), patch.object(lease, 'Portal'):
            self.assertEqual(lease.refresh_controller(self.control, updated), 'config_reloaded')
        self.assertFalse(self.workers.mock_calls)

    def test_server_and_lock_directory_changes_require_restart(self):
        for changed in ('sid', 'ssh', 'state_dir'):
            updated = copy.deepcopy(self.cfg)
            if changed == 'sid':
                updated['sid'] = 'dgx-h200-2'
            elif changed == 'ssh':
                updated['ssh']['host'] = 'another-host'
            else:
                updated['worker']['state_dir'] = '/other/state'
            self.assertEqual(lease.refresh_controller(self.control, updated),
                             'config_reload_requires_restart:sid_ssh_or_state_dir')
        self.assertFalse(self.workers.mock_calls)

    def test_legacy_health_removed_but_recovery_evidence_survives(self):
        control = lease.Controller(self.cfg, self.portal, self.workers, {
            'gpu_usage': [{'util': 99}], 'gpu_queue_alive': True, 'gpu_held': True,
            'loss_confirmed': True, 'last_reload_success': 123})
        self.assertNotIn('gpu_usage', control.state)
        self.assertNotIn('gpu_queue_alive', control.state)
        self.assertTrue(control.state['gpu_seen'])
        self.assertTrue(control.state['loss_confirmed'])
        self.assertEqual(control.state['last_reload_success'], 123)


class RegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'profile.json'
        self.cfg = lease.load_config('ext_csh')
        self.cfg['worker']['gpu']['env'] = {'OLD_PROJECT': 'old'}
        self.path.write_text(json.dumps(self.cfg))
        for name in ('cpu.sh', 'gpu.sh', 'trainer.py', 'launcher.py'):
            (self.root / name).write_text('# test script\n')

    def test_registration_updates_both_modes_and_preserves_host_settings(self):
        register(self.path, 'cpu.sh', 'gpu.sh', self.root, ['trainer.py', 'launcher.py'])
        cfg = lease.load_config(path=self.path)
        self.assertEqual(cfg['ssh'], self.cfg['ssh'])
        self.assertEqual(cfg['sid'], self.cfg['sid'])
        self.assertEqual(cfg['worker']['state_dir'], self.cfg['worker']['state_dir'])
        for mode in ('cpu', 'gpu'):
            script = str(self.root / (mode + '.sh'))
            self.assertEqual(cfg['worker'][mode]['command'], ['bash', script])
            self.assertEqual(cfg['worker'][mode]['queue_patterns'], [script])
        self.assertNotIn('env', cfg['worker']['gpu'])
        argv = ['python3', str(self.root / 'launcher.py'), '--cpu-jobs', '4']
        self.assertEqual(worker.classify(argv, {}, cfg['worker']), 'cpu')

    def test_missing_or_identical_scripts_leave_config_unchanged(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileNotFoundError):
            register(self.path, 'missing.sh', 'gpu.sh', self.root, ['trainer.py'])
        self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises(ValueError):
            register(self.path, 'cpu.sh', 'cpu.sh', self.root, ['trainer.py'])
        self.assertEqual(self.path.read_bytes(), before)

    def test_bad_config_is_rejected_without_changing_controller(self):
        control = lease.Controller(self.cfg, Mock(), Mock())
        self.path.write_text('{')
        with self.assertRaises(ValueError):
            lease.refresh_controller(control, lease.load_config(path=self.path))
        self.assertIs(control.cfg, self.cfg)
        malformed = copy.deepcopy(self.cfg)
        malformed['worker']['process_patterns'] = []
        self.path.write_text(json.dumps(malformed))
        with self.assertRaises(ValueError):
            lease.load_config(path=self.path)

    def test_result_sync_and_children_do_not_hold_queue_open(self):
        cfg = self.cfg['worker']
        sync = cfg['ignore_patterns'][0]
        procs = {
            101: {'pid': 101, 'ppid': 1, 'start': '10', 'argv': ['bash', sync], 'env': {}},
            102: {'pid': 102, 'ppid': 101, 'start': '11', 'argv': ['git', 'push'], 'env': {}},
            103: {'pid': 103, 'ppid': 1, 'start': '12',
                  'argv': ['python', '/home/ext_csh/MPI_sweep/launch_mpi_sweep.py',
                           '--cpu-jobs', '4'], 'env': {}},
        }
        record = {'namespace': 'test', 'gpu': {'tracked': ['101:10', '102:11']}}
        with patch.object(worker.Path, 'iterdir', return_value=iter(
                [Path('/proc') / str(pid) for pid in procs])), \
                patch.object(worker, 'read_process', side_effect=lambda pid: procs[pid]), \
                patch.object(worker, 'own_proc_pid', return_value=999), \
                patch.object(worker, 'namespace', return_value='test'):
            result = worker.snapshot(cfg, record)
        self.assertEqual(result['gpu'], [])
        self.assertEqual([p['pid'] for p in result['cpu']], [103])

    def test_running_loop_recovers_from_partial_write_then_adopts_registered_queue(self):
        self.cfg['worker']['state_dir'] = str(self.root / 'state')
        self.path.write_text(json.dumps(self.cfg))
        portal, workers = Mock(), Mock()
        portal.status.side_effect = lambda: {
            'state': 'running', 'ssh_port': 23023, 'remaining_h': 8.0}
        workers.local.return_value = {'ok': True, 'cpu': [], 'gpu': []}
        workers.remote.side_effect = lambda box, action='status', mode='gpu': {
            'ok': True, 'cpu': [], 'gpu': ['123:456'] if action == 'start' else [],
            'gpu_available': True}
        handlers, elapsed = {}, [0.0]
        cycles = [0]

        def pause(seconds):
            elapsed[0] += seconds
            cycles[0] += 1
            if cycles[0] == 1:
                self.path.write_text('{')
            elif cycles[0] == 2:
                self.path.write_text(json.dumps(self.cfg))
                register(self.path, 'cpu.sh', 'gpu.sh', self.root,
                         ['trainer.py', 'launcher.py'])
            else:
                handlers[signal.SIGTERM]()

        with patch.object(lease, 'Portal', return_value=portal), \
                patch.object(lease, 'Workers', return_value=workers), \
                patch.object(lease.signal, 'signal', side_effect=lambda sig, fn: handlers.update({sig: fn})), \
                patch.object(lease.time, 'monotonic', side_effect=lambda: elapsed[0]), \
                patch.object(lease.time, 'sleep', side_effect=pause), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(lease.main(['--config', str(self.path), '--watch', '--apply',
                                         '--interval', '1']), 0)
        lines = [json.loads(line) for line in
                 (self.root / 'state/watcher.log').read_text().splitlines()]
        self.assertEqual(len(lines), 3)
        self.assertTrue(any(x.startswith('config_reload_failed:') for x in lines[1]['events']))
        self.assertEqual(lines[1]['queue_commands'], lines[0]['queue_commands'])
        self.assertIn('config_reloaded', lines[2]['events'])
        self.assertEqual(lines[2]['queue_commands']['gpu'], ['bash', str(self.root / 'gpu.sh')])
        self.assertEqual(lines[2]['phase'], 'gpu_running')
        self.assertFalse((self.root / 'state/lease.watch.pid').exists())


if __name__ == '__main__':
    unittest.main()
