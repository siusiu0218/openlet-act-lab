"""CPU scalar-regression demonstration of checkpoint and resume control."""
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from checkpoint_io import atomic_json, save_checkpoint
from long_act_profile import batch_order, run_training_loop


def stage(root, initial, batches):
    state = dict(initial)
    def update(indices, count):
        x = np.asarray(indices, dtype=float) + 1
        gradient = float(np.mean(2 * x * (state['weight'] * x - 2 * x)))
        state['weight'] -= .001 * gradient
        state['cursor'] += 1
    def snapshot(final):
        destination = root / ('step-%03d' % state['cursor'])
        return save_checkpoint(destination, root / 'latest.json', state['cursor'],
            [('state', lambda p: atomic_json(p / 'state.json', state))], lambda *a, **kw: None)
    run_training_loop(batches, update, snapshot, lambda p: True,
                      time.monotonic, time.monotonic() + 30, len(batches), len(batches))
    return state


def main():
    batches = batch_order(7, 12, 2, 44)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        initial = dict(weight=0., cursor=0)
        expected = stage(root / 'continuous', initial, batches)
        stage(root / 'interrupted', initial, batches[:5])
        saved = json.loads((root / 'interrupted/step-005/state.json').read_text())
        actual = stage(root / 'resumed', saved, batches[saved['cursor']:])
        if actual != expected:
            raise AssertionError('resumed and uninterrupted results differ')
        print(json.dumps(dict(status='PASS', demonstration='scalar regression control flow',
                              resumed_equals_continuous=True, final=actual)))

if __name__ == '__main__':
    main()
