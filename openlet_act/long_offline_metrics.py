"""Deterministic ACT evaluation contracts and metrics; NumPy only, no model/ROS."""
import hashlib

import numpy as np

from openlet_act.inference_contract import validate_limits

EPISODES = tuple(range(24, 31))
HORIZON = 100
SCALE = np.array([1.] * 7 + [1.57] * 11 + [1.] * 3)
UNITS = ['rad'] * 18 + ['m/s', 'm/s', 'rad/s']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array(value, shape, name, dtype=None):
    a = value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)
    require(a.shape == shape and np.isfinite(a).all(), 'shape/nonfinite: ' + name)
    if dtype is not None:
        require(a.dtype == dtype, 'dtype: ' + name)
    return a


def sampling_plan(lengths):
    require(set(lengths) == set(EPISODES), 'held-out episodes must be exactly 24..30')
    require(all(type(n) is int and n >= 5 for n in lengths.values()), 'episode length/duplicate samples')
    plan, offset = [], 0
    for ep in EPISODES:
        n = lengths[ep]
        for quarter in range(5):
            frame = quarter * (n - 1) // 4
            plan.append(dict(episode=ep, frame=frame, length=n, dataset_index=offset + frame))
        offset += n
    return plan


def target_window(plan, episode_actions):
    a = array(episode_actions, (plan['length'], 21), 'episode actions', np.dtype('float32'))
    indices = plan['frame'] + np.arange(HORIZON)
    pad = indices >= plan['length']
    return a[np.minimum(indices, plan['length'] - 1)], pad


def validate_raw(raw, plan, expected_state, expected_target):
    """Check actual loader output against independently read numeric Parquet rows."""
    identity = {k: float(raw[k].item()) for k in ('episode_index', 'frame_index', 'timestamp')}
    require(identity['episode_index'] == plan['episode'] and identity['frame_index'] == plan['frame'], 'sample identity')
    require(abs(identity['timestamp'] - plan['frame'] / 10) <= 1e-4, 'sample timestamp')
    state = array(raw['observation.state'], (18,), 'state', np.dtype('float32'))
    target = array(raw['action'], (100, 21), 'target', np.dtype('float32'))
    pad = array(raw['action_is_pad'], (100,), 'mask', np.dtype('bool'))
    require(np.array_equal(state, expected_state), 'state/Parquet mismatch')
    require(np.array_equal(target, expected_target), 'target alignment/Parquet mismatch')
    expected_pad = plan['frame'] + np.arange(100) >= plan['length']
    require(np.array_equal(pad, expected_pad), 'padding alignment')
    hashes = {'observation.state': hashlib.sha256(state.tobytes()).hexdigest()}
    for key in ('observation.images.head_cam_h', 'observation.images.wrist_cam_r'):
        image = array(raw[key], (3, 480, 640), key, np.dtype('float32'))
        require(image.min() >= 0 and image.max() <= 1, 'image range')
        hashes[key] = hashlib.sha256(image.tobytes()).hexdigest()
    return dict(identity=identity, input_sha256=hashes), state.copy(), target.copy(), pad.copy()


def validate_evidence(a, records, max_inputs=35):
    n = len(records)
    require(n > 0 and n <= max_inputs, 'sample count')
    for key, tail in [('state', (18,)), ('target', (100, 21)),
                      ('normalized', (100, 21)), ('prediction', (100, 21))]:
        array(a[key], (n, *tail), key, np.dtype('float32'))
    array(a['is_pad'], (n, 100), 'mask', np.dtype('bool'))
    require(len({(r['episode'], r['frame']) for r in records}) == n, 'duplicate sample')
    for i, r in enumerate(records):
        require(r['episode'] in EPISODES and type(r['frame']) is int and type(r['length']) is int
                and r['length'] >= 5 and 0 <= r['frame'] < r['length'], 'sample bounds')
        require(np.array_equal(a['is_pad'][i], r['frame'] + np.arange(100) >= r['length']), 'mask disagrees with identity')


def _ranges(values, lo, hi):
    v = values.reshape(-1, values.shape[-1]).astype(np.float64)
    delta = np.maximum(np.maximum(lo - v, v - hi), 0)
    return dict(vectors=len(v), violating_vectors=int((delta > 0).any(axis=1).sum()),
                per_axis_count=(delta > 0).sum(axis=0).tolist(),
                max_excess=(delta.max(axis=0) if len(v) else np.zeros(len(lo))).tolist())


def _errors(values):
    v = values.reshape(-1, values.shape[-1]).astype(np.float64)
    require(len(v) > 0, 'empty error denominator')
    return dict(vectors=len(v), mae=np.abs(v).mean(0).tolist(), rmse=np.sqrt((v*v).mean(0)).tolist(),
                p95=np.percentile(np.abs(v), 95, axis=0).tolist(), maximum=np.abs(v).max(0).tolist())


def _delta(values, limits):
    v = values.reshape(-1, 18).astype(np.float64)
    return dict(vectors=len(v), exceeding_vectors=int((v > limits).any(1).sum()),
                per_axis_count=(v > limits).sum(0).tolist(),
                maximum=(v.max(0) if len(v) else np.zeros(18)).tolist())


def metrics(a, records, limits, action_mean, max_inputs=35):
    """All numeric metrics are physical units; no scalar mix of rad and m/s."""
    validate_limits(limits)
    validate_evidence(a, records, max_inputs)
    mean = array(action_mean, (21,), 'training action mean').astype(np.float64)
    pred, target = a['prediction'].astype(np.float64)*SCALE, a['target'].astype(np.float64)*SCALE
    state = a['state'].astype(np.float64)*SCALE[:18]
    valid = ~a['is_pad']
    lo, hi = np.asarray(limits['position_min'])*SCALE[:18], np.asarray(limits['position_max'])*SCALE[:18]
    base = np.asarray(limits['base_abs_max'])
    def ranges(v):
        return dict(position=_ranges(v[..., :18], lo, hi), base=_ranges(v[..., 18:], -base, base))
    result = dict(units=UNITS, input_count=len(records), valid_target_vectors=int(valid.sum()),
                  prediction_first=ranges(pred[:, 0]), prediction_full=ranges(pred),
                  prediction_valid=ranges(pred[valid]), target_first=ranges(target[:, 0]),
                  target_valid=ranges(target[valid]), state=_ranges(state, lo, hi))
    bad_pos = ((pred[..., :18] < lo) | (pred[..., :18] > hi)).any(axis=2)
    bad_base = (np.abs(pred[..., 18:]) > base).any(axis=2)
    result['inputs_with_any_prediction_violation'] = int((bad_pos | bad_base).any(axis=1).sum())
    hold = np.concatenate([state, np.zeros((len(state), 3))], axis=1)[:, None, :]
    for name, estimate in [('model', pred), ('hold_state_base_zero', hold), ('train_mean', mean*SCALE)]:
        error = estimate - target
        result[name + '_error'] = dict(first=_errors(error[:, 0]), valid_chunk=_errors(error[valid]))
    for name, value in [('prediction', pred), ('target', target)]:
        first = np.abs(value[:, 0, :18] - state)
        result[name + '_first_jump'] = {k: _delta(first, np.asarray(limits[k])*SCALE[:18])
                                       for k in ('first_delta', 'tracking_delta')}
        step = np.abs(np.diff(value[..., :18], axis=1))
        bound = np.asarray(limits['step_delta'])*SCALE[:18]
        result[name + '_chunk_adjacent'] = dict(full=_delta(step, bound),
            valid=_delta(step[valid[:, 1:] & valid[:, :-1]], bound))
    return result


def report(a, records, limits, action_mean):
    result = metrics(a, records, limits, action_mean)
    result['by_episode'] = {}
    for ep in sorted({r['episode'] for r in records}):
        indices = [i for i, r in enumerate(records) if r['episode'] == ep]
        result['by_episode'][str(ep)] = metrics({k: v[indices] for k, v in a.items()},
                                               [records[i] for i in indices], limits, action_mean)
    return result
