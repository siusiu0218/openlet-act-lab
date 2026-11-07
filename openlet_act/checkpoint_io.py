"""Publish a checkpoint only after every writer and digest has completed."""
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(4*1024**2):h.update(data)
    return h.hexdigest()


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('x',encoding='utf-8') as f:
        json.dump(value,f,indent=2);f.write('\n');f.flush()
    temporary.replace(path)


def save_checkpoint(destination,latest,total_steps,writers,trace,*,existing_root=False):
    destination,latest=Path(destination),Path(latest)
    if destination.exists() and not existing_root:raise FileExistsError(destination)
    staging=(destination/'.checkpoint.incomplete' if existing_root else destination.with_name('.'+destination.name+'.incomplete'))
    staging.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    for label,writer in writers:
        before=time.monotonic();trace('checkpoint_write_begin',part=label,total_steps=total_steps)
        writer(staging)
        trace('checkpoint_write_end',part=label,seconds=time.monotonic()-before,total_steps=total_steps)
    files={}
    # Only checkpoint-owned files; never scan the environment or dataset.
    paths=sorted(p for p in staging.glob('*') if p.is_file())
    checkpoint=staging/'checkpoint'
    if checkpoint.exists():paths+=sorted(p for p in checkpoint.iterdir() if p.is_file())
    if not paths:raise ValueError('empty checkpoint')
    for path in paths:
        name=path.relative_to(staging).as_posix();before=time.monotonic()
        trace('checkpoint_hash_begin',part=name,total_steps=total_steps)
        files[name]=digest(path)
        trace('checkpoint_hash_end',part=name,bytes=path.stat().st_size,seconds=time.monotonic()-before,total_steps=total_steps)
    atomic_json(staging/'checkpoint-receipt.json',dict(total_steps=total_steps,complete=True,files=files))
    trace('checkpoint_publish_begin',total_steps=total_steps)
    if existing_root:
        # Final run root already owns logs. Never replace an existing checkpoint component.
        children=list(staging.iterdir());receipt=staging/'checkpoint-receipt.json'
        for path in children:
            if (destination/path.name).exists():raise FileExistsError(destination/path.name)
        for path in children:
            if path!=receipt:path.rename(destination/path.name)
        receipt.rename(destination/receipt.name);staging.rmdir()
    else:staging.rename(destination)
    atomic_json(latest,dict(total_steps=total_steps,path=str(destination),receipt_sha256=digest(destination/'checkpoint-receipt.json')))
    trace('checkpoint_publish_end',total_steps=total_steps,seconds=time.monotonic()-started)
    return destination
