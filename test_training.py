import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
import numpy as np
import long_act_profile as profile
from offline_metrics import sampling_plan
from long_act_validation import update_selection

class TrainingTests(unittest.TestCase):
    def test_plan_contains_legacy_and_fifty_unique_per_episode(self):
        lengths={ep:470+ep for ep in range(24,31)}
        full=profile.sampling_plan_350(lengths);old=sampling_plan(lengths)
        self.assertEqual(len(full),350)
        self.assertEqual([{k:r[k] for k in old[0]} for r in full if r['legacy']],old)
        for ep in lengths:self.assertEqual(len({r['frame'] for r in full if r['episode']==ep}),50)

    def test_training_order_crosses_epoch_without_truncation(self):
        batches=profile.batch_order(10,12,2,44);flat=sum(batches,[])
        self.assertEqual(len(flat),24);self.assertEqual(sorted(flat[:10]),list(range(10)))
        self.assertEqual(sorted(flat[10:20]),list(range(10)));self.assertNotEqual(flat[:10],flat[10:20])
        self.assertEqual(batches,profile.batch_order(10,12,2,44))

    def test_control_exact_updates_and_no_extra_fetch(self):
        updates=[];saves=[];valid=[]
        def loader():
            yield from range(6)
            raise AssertionError('extra data fetch')
        result=profile.run_training_loop(loader(),lambda raw,n:updates.append((raw,n)),lambda final:saves.append(final),
            lambda d:valid.append(d) is None,lambda:0,100,6,2)
        self.assertEqual(result,(6,False));self.assertEqual(len(updates),6)
        self.assertEqual(saves,[False,False,True]);self.assertEqual(len(valid),3)

    def test_cutoff_after_read_prevents_optimizer_update(self):
        now=[0]
        def loader():now[0]=10;yield 1
        def forbidden(*a):raise AssertionError('operation after cutoff')
        self.assertEqual(profile.run_training_loop(loader(),forbidden,forbidden,forbidden,lambda:now[0],10,6,2),(0,True))
        self.assertFalse(profile.may_start(9,10,1))

    def test_failed_save_stops_before_validation(self):
        def fail(*a):raise OSError('disk failure')
        def forbidden(*a):raise AssertionError('validation after failed save')
        with self.assertRaises(OSError):profile.run_training_loop([1],lambda *a:None,fail,forbidden,lambda:0,10,1,1)

    def test_failed_validation_stops_updates(self):
        seen=[]
        with self.assertRaises(ValueError):
            profile.run_training_loop([1,2],lambda r,n:seen.append(n),lambda f:None,
                lambda d:(_ for _ in ()).throw(ValueError('validation failed')),lambda:0,10,2,1)
        self.assertEqual(seen,[1])

    def test_hard_operation_timeout(self):
        code="import time;from long_act_profile import bounded_operation\nwith bounded_operation(.05,'fixture'):time.sleep(3)"
        p=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).parent,capture_output=True,timeout=5)
        self.assertEqual(p.returncode,124);self.assertIn(b'TIMEOUT',p.stdout)

    def test_rng_restored_even_on_error(self):
        class Torch:
            def __init__(self):self.value=1;self.cuda=self
            def get_rng_state(self):return self.value
            def set_rng_state(self,s):self.value=s
            def get_rng_state_all(self):return [self.value]
            def set_rng_state_all(self,s):self.value=s[0]
        torch=Torch();random.seed(44);np.random.seed(44)
        before=(random.getstate(),np.random.get_state())
        with self.assertRaises(ValueError):
            with profile.preserve_rng(torch,np):random.random();np.random.rand();torch.value=9;raise ValueError()
        self.assertEqual(before[0],random.getstate());self.assertTrue(np.array_equal(before[1][1],np.random.get_state()[1]));self.assertEqual(torch.value,1)

    def test_best_indices_can_diverge_without_claiming_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'selection.json'
            a=dict(total_steps=1000,normalized_l1=.4,first_above_001rad=20,full_max_excess_rad=.3)
            b=dict(total_steps=2000,normalized_l1=.3,first_above_001rad=25,full_max_excess_rad=.2)
            update_selection(p,a,'a');r=update_selection(p,b,'b')
            self.assertEqual(r['best_error']['total_steps'],2000);self.assertEqual(r['best_limits']['total_steps'],1000)
            self.assertFalse(r['best_error']['admissible'])
