"""The ext_csv result publisher must never keep a failed training queue alive."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/hosts/beyondg'))
import queue_worker as worker
import watch_beyondg_lease as lease


class ExtCsvRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.config = lease.load_config('ext_csv')
        self.cfg = self.config['worker']
        self.sync = '/home/ext_csv/MPI_sweep/scripts/refresh_iql_qbc_det_w2_5m.sh'

    def sample(self, procs, record):
        with patch.object(worker.Path, 'iterdir', return_value=iter(
                Path('/proc') / str(pid) for pid in procs)), \
                patch.object(worker, 'read_process', side_effect=lambda pid: procs[pid]), \
                patch.object(worker, 'own_proc_pid', return_value=999), \
                patch.object(worker, 'namespace', return_value='test'):
            return worker.snapshot(self.cfg, record)

    def processes(self, mode):
        env = {'IQL_QBC_DET_DEVICE':mode}
        return {
            100: {'pid':100, 'ppid':1, 'start':'9',
                  'argv':self.cfg[mode]['command'], 'env':env},
            101: {'pid':101, 'ppid':100, 'start':'10',
                  'argv':['bash', self.sync], 'env':env},
            102: {'pid':102, 'ppid':101, 'start':'11',
                  'argv':['sleep', '300'], 'env':env},
            103: {'pid':103, 'ppid':100, 'start':'12',
                  'argv':['python', '/home/ext_csv/MPI_sweep/train_iql_mpi.py'], 'env':env},
        }

    def test_live_training_excludes_sync_and_children_in_both_modes(self):
        for mode in ('cpu', 'gpu'):
            with self.subTest(mode=mode):
                result = self.sample(self.processes(mode), {})
                self.assertEqual([p['pid'] for p in result[mode]], [100, 103])
                self.assertEqual(result['gpu' if mode == 'cpu' else 'cpu'], [])

    def test_legacy_tracked_sync_cannot_keep_failed_queue_alive(self):
        for mode in ('cpu', 'gpu'):
            with self.subTest(mode=mode):
                procs = self.processes(mode)
                # This is exactly the record an older worker can have persisted
                # while the wrapper, publisher, and trainer were all alive.
                record = {'namespace':'test', mode:{'tracked':[
                    worker.identity(p) for p in procs.values()]}}
                del procs[100], procs[103]
                procs[101]['ppid'] = 1
                result = self.sample(procs, record)
                self.assertEqual(result, {'cpu':[], 'gpu':[]})

    def test_detached_checkpoint_writer_still_blocks_new_gpu(self):
        procs = self.processes('cpu')
        record = {'namespace':'test', 'cpu':{'tracked':[
            worker.identity(p) for p in procs.values()]}}
        del procs[100]
        procs[101]['ppid'] = procs[103]['ppid'] = 1
        # Even a detached child with a different argv must remain tracked until
        # it finishes saving; only the publisher is explicitly excluded.
        procs[103]['argv'] = ['python', '/temporary/checkpoint_writer.py']
        result = self.sample(procs, record)
        self.assertEqual([p['pid'] for p in result['cpu']], [103])
        self.assertEqual(result['gpu'], [])

    def test_manual_shared_queue_is_recognized_without_persisted_record(self):
        argv = ['bash', '/home/ext_csv/MPI_sweep/scripts/hosts/beyondg/run_queue_iql_qbc_det_w2_seed0.sh']
        for mode, env in (('cpu', {'IQL_QBC_DET_DEVICE':'cpu'}),
                          ('gpu', {'IQL_QBC_DET_DEVICE':'cuda'})):
            with self.subTest(mode=mode):
                self.assertEqual(worker.classify(argv, env, self.cfg), mode)

    def test_both_configured_wrappers_are_shipped(self):
        for mode in ('cpu', 'gpu'):
            command = self.cfg[mode]['command']
            self.assertEqual(command[0], 'bash')
            self.assertTrue(Path(command[1]).is_file(), command)
            self.assertEqual(worker.classify(command, {}, self.cfg), mode)


if __name__ == '__main__':
    unittest.main()
