"""Synthetic offline example. Limits are illustrative only."""
import json
import numpy as np
from openlet_act.offline_metrics import metrics


def fixture():
    arrays = {
        'state': np.zeros((1, 18), dtype=np.float32),
        'target': np.zeros((1, 100, 21), dtype=np.float32),
        'normalized': np.zeros((1, 100, 21), dtype=np.float32),
        'prediction': np.zeros((1, 100, 21), dtype=np.float32),
        'is_pad': np.arange(100)[None, :] >= 5,
    }
    records = [dict(episode=24, frame=0, length=5)]
    limits = dict(predict_steps=100, execute_steps=1, direct_to_wbc=False,
                  future_tolerance_s=0, max_actions=1, live_enabled=False,
                  position_min=[-1.] * 18, position_max=[1.] * 18,
                  first_delta=[.5] * 18, step_delta=[.5] * 18,
                  tracking_delta=[.5] * 18, base_abs_max=[1.] * 3)
    for name in ('max_age_s', 'max_receive_age_s', 'max_skew_s',
                 'inference_timeout_s', 'control_budget_s', 'start_timeout_s',
                 'watchdog_period_s', 'stop_window_s', 'period_s'):
        limits[name] = .1
    return arrays, records, limits, np.zeros(21)


if __name__ == '__main__':
    print(json.dumps(metrics(*fixture()), indent=2))
