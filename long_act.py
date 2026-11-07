"""Local ACT continuation with checkpointed optimizer, RNG and batch cursor."""
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
from checkpoint_io import atomic_json, digest, save_checkpoint
from long_act_profile import batch_order, restore_rng, run_training_loop


def validate_config(config):
    for key in ('dataset_root', 'checkpoint', 'output'):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError('missing path: ' + key)
    for key in ('updates', 'batch_size', 'save_every', 'budget_seconds'):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError('expected positive integer: ' + key)
    if type(config.get('seed')) is not int:
        raise ValueError('seed must be an integer')
    if config.get('device') not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if config.get('live_enabled') is not False:
        raise ValueError('offline mode required')
    if not config.get('input_sha256'):
        raise ValueError('input_sha256 must pin checkpoint and dataset files')
    return config


def verify_inputs(config):
    for name, expected in config['input_sha256'].items():
        if digest(name) != expected:
            raise ValueError('input SHA mismatch: ' + name)
    # Every local data and model file must be included in the explicit manifest.
    pinned = {Path(name).resolve() for name in config['input_sha256']}
    for folder in (config['dataset_root'], config['checkpoint']):
        root = Path(folder)
        if not root.is_dir():
            raise ValueError('missing input directory: ' + str(root))
        actual = {p.resolve() for p in root.rglob('*') if p.is_file()}
        if not actual or not actual.issubset(pinned):
            raise ValueError('input manifest is incomplete: ' + str(root))


def resume_order(state, length, config):
    contract = dict(length=length, seed=config['seed'], batch_size=config['batch_size'],
                    updates=config['updates'], device=config['device'], input_sha256=config['input_sha256'])
    if state['contract'] != contract:
        raise ValueError('resume configuration or input identity changed')
    cursor = state['cursor']
    if type(cursor) is not int or not 0 <= cursor <= config['updates']:
        raise ValueError('invalid resume cursor')
    return batch_order(length, config['updates'], config['batch_size'], config['seed'])[cursor:]


def verify_checkpoint(folder):
    folder = Path(folder)
    receipt = json.loads((folder / 'checkpoint-receipt.json').read_text())
    if receipt.get('complete') is not True:
        raise ValueError('incomplete checkpoint')
    required = {'optimizer.pt', 'training-state.pt', 'checkpoint/config.json',
                'checkpoint/model.safetensors', 'checkpoint/policy_preprocessor.json',
                'checkpoint/policy_postprocessor.json'}
    if not required.issubset(receipt['files']):
        raise ValueError('missing checkpoint components')
    for name, expected in receipt['files'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path) != expected:
            raise ValueError('invalid checkpoint file: ' + name)
    return receipt


def run(config, resume=None):
    validate_config(config)
    verify_inputs(config)
    import torch
    from torch.utils.data import DataLoader
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.processor import (PolicyProcessorPipeline, batch_to_transition,
        transition_to_batch, policy_action_to_transition, transition_to_policy_action)
    from act_runtime import load_act_config

    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    source = Path(config['checkpoint'])
    state = None
    if resume:
        receipt = verify_checkpoint(resume)
        source = Path(resume) / 'checkpoint'
        # Only load checkpoints created by this runner in trusted local storage.
        state = torch.load(Path(resume) / 'training-state.pt', map_location='cpu', weights_only=False)
        if state['cursor'] != receipt['total_steps']:
            raise ValueError('checkpoint step mismatch')
    policy = ACTPolicy.from_pretrained(source, config=load_act_config(source, device),
                                      local_files_only=True, strict=True).to(device)
    pre = PolicyProcessorPipeline.from_pretrained(source, config_filename='policy_preprocessor.json',
        local_files_only=True, to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline.from_pretrained(source, config_filename='policy_postprocessor.json',
        local_files_only=True, to_transition=policy_action_to_transition, to_output=transition_to_policy_action)
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=1e-5, weight_decay=1e-4)
    if resume:
        optimizer.load_state_dict(torch.load(Path(resume) / 'optimizer.pt', map_location=device, weights_only=True))
    dataset = LeRobotDataset(repo_id='OpenLET/task1-local', root=Path(config['dataset_root']),
        episodes=list(range(24)), video_backend='pyav', download_videos=False,
        tolerance_s=1e-4, delta_timestamps={'action': [i / 10 for i in range(100)]})
    if len(dataset) <= 0 or dataset.fps != 10:
        raise ValueError('expected nonempty 10Hz dataset')
    contract = dict(length=len(dataset), seed=config['seed'], batch_size=config['batch_size'],
                    updates=config['updates'], device=config['device'], input_sha256=config['input_sha256'])
    batches = resume_order(state, len(dataset), config) if state else batch_order(
        len(dataset), config['updates'], config['batch_size'], config['seed'])
    cursor = state['cursor'] if state else 0
    if state and state['torch_version'] != torch.__version__:
        raise ValueError('resume requires the original PyTorch version')
    if cursor == config['updates']:
        raise ValueError('checkpoint already completed the requested stage')
    loader = DataLoader(dataset, batch_sampler=batches, num_workers=0,
                        generator=torch.Generator().manual_seed(config['seed']))
    output = Path(config['output'])
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'config.json', config)
    if state:
        restore_rng(state, torch, np)
    started = time.monotonic()
    policy.train()

    def trace(event, **values):
        print(json.dumps(dict(event=event, **values)), flush=True)

    def snapshot(final=False):
        folder = output / ('step-%06d' % cursor)
        def write_state(destination):
            torch.save(dict(contract=contract, cursor=cursor, torch_version=torch.__version__, python=random.getstate(),
                numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all()), destination / 'training-state.pt')
        writers = [('model', lambda p: policy.save_pretrained(p / 'checkpoint')),
                   ('pre', lambda p: pre.save_pretrained(p / 'checkpoint')),
                   ('post', lambda p: post.save_pretrained(p / 'checkpoint')),
                   ('optimizer', lambda p: torch.save(optimizer.state_dict(), p / 'optimizer.pt')),
                   ('state', write_state)]
        return save_checkpoint(folder, output / 'latest.json', cursor, writers, trace)

    with (output / 'steps.jsonl').open('x') as log:
        def update(raw, count):
            nonlocal cursor
            batch = pre(raw)
            if tuple(batch['action'].shape) != (config['batch_size'], 100, 21):
                raise ValueError('expected batch x 100 x 21 actions')
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy(batch)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10., error_if_nonfinite=True)
            optimizer.step()
            if device == 'cuda':
                torch.cuda.synchronize()
            cursor += 1
            log.write(json.dumps(dict(step=cursor, loss=float(loss), grad_norm=float(norm))) + '\n')
            log.flush()
        completed, stopped = run_training_loop(loader, update, snapshot,
            lambda folder: bool(verify_checkpoint(folder)), time.monotonic,
            started + config['budget_seconds'], len(batches), config['save_every'])
    final = output / ('step-%06d' % cursor)
    if not final.exists():
        snapshot(True)
    atomic_json(output / 'summary.json', dict(status='partial_budget' if stopped else 'complete',
        cursor=cursor, updates_this_run=completed, elapsed_seconds=time.monotonic()-started,
        torch_version=torch.__version__, device=device, ros_started=False))
    return 2 if stopped else 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--resume')
    parser.add_argument('--check-config', action='store_true')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if args.check_config:
        validate_config(config)
        print('Configuration structure valid; input hashes and runtime checked at execution.')
    else:
        raise SystemExit(run(config, args.resume))
