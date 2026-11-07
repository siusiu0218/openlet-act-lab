import unittest
from openlet_act.long_act import resume_order, validate_config
from openlet_act.long_act_profile import batch_order

class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.config = dict(dataset_root='data', checkpoint='model', output='run',
            updates=12, batch_size=2, save_every=3, budget_seconds=30, seed=44,
            device='cpu', live_enabled=False, input_sha256={'input': 'a' * 64})
        self.state = dict(cursor=5, contract=dict(length=7, seed=44, batch_size=2,
            updates=12, device='cpu', input_sha256={'input': 'a' * 64}))

    def test_cursor_restores_exact_remaining_batches(self):
        self.assertEqual(resume_order(self.state, 7, self.config), batch_order(7, 12, 2, 44)[5:])

    def test_changed_inputs_rejected(self):
        for field, value in [('seed', 45), ('updates', 13), ('device', 'cuda'),
                             ('input_sha256', {'input': 'b' * 64})]:
            with self.assertRaises(ValueError):
                resume_order(self.state, 7, dict(self.config, **{field: value}))
        with self.assertRaises(ValueError):
            resume_order(self.state, 8, self.config)

    def test_invalid_cursor_and_config(self):
        for cursor in (-1, 13, True):
            with self.assertRaises(ValueError):
                resume_order(dict(self.state, cursor=cursor), 7, self.config)
        validate_config(self.config)
        for field, value in [('updates', 0), ('batch_size', True), ('live_enabled', True)]:
            with self.assertRaises(ValueError):
                validate_config(dict(self.config, **{field: value}))
