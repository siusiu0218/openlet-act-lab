"""CPU-only ACT/ROS boundary. No ROS, Torch, pickle, or implicit clipping."""
import json
import math
import re
import struct
import time

import numpy as np

SOURCES = ('head_cam_h', 'wrist_cam_r', 'joint_q', 'gripper')
SHAPES = {'head_cam_h': (3, 480, 640), 'wrist_cam_r': (3, 480, 640),
          'state': (18,)}
MAX_HEADER = 8192
MAX_BODY = (2 * 3 * 480 * 640 + 18) * 4


class ContractError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise ContractError(reason)


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), 'invalid ' + name)
    return float(value)


def vector(value, size, name):
    a = np.asarray(value)
    require(a.dtype.kind in 'fiu', 'non-numeric ' + name)
    a = a.astype(np.float64)
    require(a.shape == (size,) and np.isfinite(a).all(), 'invalid ' + name)
    return a.copy()


def identity(run, seq):
    require(isinstance(run, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run), 'invalid run')
    require(type(seq) is int and 0 <= seq < 2**53, 'invalid sequence')


def timing(meta, now_ros, now_mono, limits):
    """Header stamps share the ROS clock; receive/deadline ages use monotonic time."""
    finite(now_ros, 'ROS now'); finite(now_mono, 'monotonic now')
    validate_metadata(meta)
    for field in ('stamps', 'received'):
        require(set(meta[field]) == set(SOURCES), 'missing source timestamps')
        for value in meta[field].values():
            finite(value, field)
    stamps = list(meta['stamps'].values())
    require(max(stamps) - min(stamps) <= limits['max_skew_s'], 'source skew')
    for stamp in stamps:
        require(-limits['future_tolerance_s'] <= now_ros - stamp <= limits['max_age_s'], 'header age')
    for received in meta['received'].values():
        require(0 <= now_mono - received <= limits['max_receive_age_s'], 'receive age')
    finite(meta['sim_time'], 'sim time')
    require(meta['frame_alignment'] is False, 'unexpected alignment claim')


def validate_metadata(meta):
    identity(meta['run'], meta['seq'])
    for field in ('stamps', 'received'):
        require(isinstance(meta.get(field), dict) and set(meta[field]) == set(SOURCES), 'source timestamp fields')
        for value in meta[field].values():
            finite(value, field)
    finite(meta.get('sim_time'), 'sim time')
    require(meta.get('frame_alignment') is False, 'unexpected alignment claim')


def validate_limits(p):
    """No numerical defaults: a draft profile cannot become an action policy."""
    require(p.get('predict_steps') == 100 and p.get('execute_steps') == 1, 'unsupported action cadence')
    require(p.get('direct_to_wbc') is False, 'direct WBC control prohibited')
    for name in ('max_age_s', 'max_receive_age_s', 'max_skew_s', 'inference_timeout_s',
                 'control_budget_s', 'start_timeout_s', 'watchdog_period_s', 'stop_window_s', 'period_s'):
        require(finite(p.get(name), name) > 0, 'nonpositive ' + name)
    require(finite(p.get('future_tolerance_s'), 'future tolerance') >= 0, 'negative future tolerance')
    require(p['period_s'] >= .1, 'control rate exceeds 10 Hz wall time')
    require(p['watchdog_period_s'] <= min(p['max_receive_age_s'], p['inference_timeout_s'], p['stop_window_s']),
            'watchdog period exceeds monitored bound')
    require(type(p.get('max_actions')) is int and p['max_actions'] > 0, 'missing action budget')
    lo = vector(p.get('position_min'), 18, 'position min')
    hi = vector(p.get('position_max'), 18, 'position max')
    require((lo < hi).all(), 'unordered position limits')
    require((lo[:7] >= -3.14).all() and (hi[:7] <= 3.14).all(), 'arm software range')
    require((lo[7:] * 1.57 >= -5).all() and (hi[7:] * 1.57 <= 5).all(), 'hand publisher clipping range')
    for name in ('first_delta', 'step_delta', 'tracking_delta'):
        require((vector(p.get(name), 18, name) > 0).all(), 'nonpositive ' + name)
    require((vector(p.get('base_abs_max'), 3, 'base limits') >= 0).all(), 'negative base limit')
    return p


def check_action(value, measured, previous, limits):
    a = vector(value, 21, 'postprocessed action')
    q = vector(measured, 18, 'measured state')
    require((a[:18] >= limits['position_min']).all() and (a[:18] <= limits['position_max']).all(), 'position range')
    require((np.abs(a[18:]) <= limits['base_abs_max']).all(), 'base range')
    require((np.abs(a[:18] - q) <= limits['tracking_delta']).all(), 'measured tracking jump')
    reference = q if previous is None else vector(previous, 21, 'previous action')[:18]
    bound = limits['first_delta'] if previous is None else limits['step_delta']
    require((np.abs(a[:18] - reference) <= bound).all(), 'first/step jump')
    return a


def encode_observation(meta, arrays):
    validate_metadata(meta)
    blocks = []
    require(set(arrays) == set(SHAPES), 'observation keys')
    for key, shape in SHAPES.items():
        a = np.asarray(arrays[key])
        require(a.shape == shape and a.dtype == np.float32 and np.isfinite(a).all(), 'tensor ' + key)
        if key != 'state':
            require(a.min() >= 0 and a.max() <= 1, 'image range')
        blocks.append(a.astype('<f4', copy=False).tobytes(order='C'))
    header = dict(meta, kind='observation', version=1, dtype='<f4', shapes={k: list(v) for k, v in SHAPES.items()})
    return pack(header, b''.join(blocks))


def pack(header, body=b''):
    encoded = json.dumps(header, allow_nan=False, separators=(',', ':')).encode('utf-8')
    require(0 < len(encoded) <= MAX_HEADER and len(body) <= MAX_BODY, 'message length')
    return struct.pack('!II', len(encoded), len(body)) + encoded + body


def recv_packet(sock, deadline, clock=time.monotonic):
    """One absolute deadline covers all reads, including a trickling peer."""
    def read(size):
        result = bytearray()
        while len(result) < size:
            left = deadline - clock()
            require(left > 0, 'IPC deadline')
            sock.settimeout(left)
            part = sock.recv(size - len(result))
            require(bool(part), 'IPC EOF')
            result.extend(part)
        return bytes(result)
    h, b = struct.unpack('!II', read(8))
    require(0 < h <= MAX_HEADER and b <= MAX_BODY, 'message length')
    def unique(pairs):
        out = {}
        for k, v in pairs:
            require(k not in out, 'duplicate JSON key')
            out[k] = v
        return out
    header = json.loads(read(h).decode('utf-8'), object_pairs_hook=unique,
                        parse_constant=lambda s: (_ for _ in ()).throw(ContractError('nonfinite JSON')))
    require(isinstance(header, dict), 'header object required')
    return header, read(b)


def decode_observation(header, body):
    identity(header['run'], header['seq'])
    require(header.get('kind') == 'observation' and header.get('version') == 1
            and header.get('dtype') == '<f4' and header.get('shapes') == {k: list(v) for k, v in SHAPES.items()}
            and len(body) == MAX_BODY, 'observation wire contract')
    arrays, offset = {}, 0
    for key, shape in SHAPES.items():
        size = int(np.prod(shape)) * 4
        arrays[key] = np.frombuffer(body[offset:offset + size], dtype='<f4').reshape(shape).copy()
        offset += size
    encode_observation(header, arrays)  # Validate numeric/range boundaries too.
    return arrays


def first_action(chunk):
    """Input is the complete postprocessed block, never normalized predictions."""
    a = np.asarray(chunk)
    require(a.shape == (1, 100, 21) and a.dtype == np.float32 and np.isfinite(a).all(), 'action chunk')
    return a[0, 0].tolist()


class SocketPolicyClient:
    """Connected local socket; single caller and one outstanding request only."""
    def __init__(self, sock, clock=time.monotonic):
        self.sock, self.clock = sock, clock

    def exchange(self, meta, arrays, deadline):
        packet = encode_observation(meta, arrays)
        remaining = deadline - self.clock()
        require(remaining > 0, 'IPC deadline')
        self.sock.settimeout(remaining)
        self.sock.sendall(packet)
        header, body = recv_packet(self.sock, deadline, self.clock)
        require(not body and header.get('kind') == 'action' and header.get('version') == 1
                and header.get('space') == 'dataset_postprocessed', 'action wire contract')
        identity(header['run'], header['seq'])
        require(header['run'] == meta['run'] and header['seq'] == meta['seq'], 'response identity')
        return vector(header['action'], 21, 'action')
