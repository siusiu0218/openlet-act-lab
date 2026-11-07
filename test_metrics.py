import unittest
import numpy as np
from example import fixture
from offline_metrics import metrics, target_window, sampling_plan
from long_offline_metrics import metrics as long_metrics


class MetricsTests(unittest.TestCase):
    def test_padding_excluded_and_hand_units(self):
        arrays, records, limits, mean = fixture()
        arrays['prediction'][0, :5, 7] = 1.
        arrays['prediction'][0, 5:, 7] = 100.
        result = metrics(arrays, records, limits, mean)
        self.assertEqual(result['valid_target_vectors'], 5)
        self.assertAlmostEqual(result['model_error']['valid_chunk']['mae'][7], 1.57)
        self.assertEqual(result['prediction_valid']['position']['violating_vectors'], 0)
        self.assertEqual(result['prediction_full']['position']['violating_vectors'], 95)

    def test_nan_and_identity_mask_rejected(self):
        arrays, records, limits, mean = fixture()
        arrays['prediction'][0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            metrics(arrays, records, limits, mean)
        arrays['prediction'][0, 0, 0] = 0
        arrays['is_pad'][0, 0] = True
        with self.assertRaises(ValueError):
            metrics(arrays, records, limits, mean)

    def test_sampling_and_end_padding(self):
        plan = sampling_plan({ep: 5 for ep in range(24, 31)})
        self.assertEqual(len(plan), 35)
        actions = np.arange(105, dtype=np.float32).reshape(5, 21)
        target, pad = target_window(plan[4], actions)
        np.testing.assert_array_equal(target, np.repeat(actions[-1:], 100, axis=0))
        self.assertEqual(int((~pad).sum()), 1)

    def test_long_metrics_matches_short(self):
        self.assertEqual(metrics(*fixture()), long_metrics(*fixture(), max_inputs=350))


if __name__ == '__main__':
    unittest.main()
