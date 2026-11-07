"""In-process fixed validation; immutable arrays and receipts, no optimizer or ROS calls."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from checkpoint_io import digest
from offline_metrics import SCALE, target_window, validate_raw
from long_offline_metrics import metrics
from long_act_profile import sampling_plan_350,preserve_rng,bounded_operation,selection_key
from checkpoint_io import atomic_json


def evaluate(policy,pre,post,dataset,numeric,config,destination,step,torch):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    lengths={ep:len(v['state']) for ep,v in numeric.items()}
    plan=sampling_plan_350(lengths)
    data={k:[] for k in ('state','target','is_pad','normalized','prediction')};records=[];norm_errors=[]
    began=time.monotonic();was_training=policy.training
    with preserve_rng(torch,np),bounded_operation(600,'validation_'+str(step)):
        try:
            policy.eval()
            for i,item in enumerate(plan):
                start=time.monotonic()
                with bounded_operation(60,'sample_read_'+str(i)):
                    raw=dataset[item['dataset_index']]
                read_seconds=time.monotonic()-start
                if read_seconds>1:
                    print(json.dumps(dict(stage='validation_slow_read',step=step,index=i,seconds=read_seconds)),flush=True)
                with bounded_operation(30,'sample_compute_'+str(i)):
                    rows=numeric[item['episode']];target,_=target_window(item,rows['action'])
                    meta,state,target,pad=validate_raw(raw,item,rows['state'][item['frame']],target)
                    batch=pre(torch.utils.data.default_collate([raw]));policy.reset()
                    with torch.inference_mode():
                        normalized=policy.predict_action_chunk(batch)
                        physical=post(normalized.clone())
                    torch.cuda.synchronize() if next(policy.parameters()).is_cuda else None
                    norm=normalized.cpu().numpy()[0];pred=physical.cpu().numpy()[0]
                    expected=batch['action'].detach().cpu().numpy()[0]
                    assert norm.shape==pred.shape==expected.shape==(100,21)
                    assert np.isfinite(norm).all() and np.isfinite(pred).all() and np.isfinite(expected).all()
                    norm_errors.append(np.abs(norm.astype(float)-expected.astype(float))[~pad])
                    for k,v in zip(data,(state,target,pad,norm,pred)):data[k].append(v)
                    records.append(dict(item,**meta,seconds=time.monotonic()-start,read_seconds=read_seconds))
            arrays={k:np.stack(v) for k,v in data.items()}
            result=metrics(arrays,records,config['limits'],config['action_mean'],max_inputs=350)
            legacy=[i for i,p in enumerate(plan) if p['legacy']]
            assert len(legacy)==35
            result['legacy35']=metrics({k:v[legacy] for k,v in arrays.items()},[records[i] for i in legacy],config['limits'],config['action_mean'])
            result['normalized_l1']=float(np.concatenate(norm_errors).mean())
            pos=arrays['prediction'].astype(float)[...,:18]*SCALE[:18]
            lo=np.array(config['limits']['position_min'])*SCALE[:18];hi=np.array(config['limits']['position_max'])*SCALE[:18]
            excess=np.maximum(0,np.maximum(lo-pos,pos-hi))
            result.update(total_steps=step,first_above_001rad=int((excess[:,0]>.01).any(1).sum()),
                full_max_excess_rad=float(excess.max()),seconds=time.monotonic()-began,
                rng_preserved=True,optimizer_updates=0,ros_started=False)
            np.savez_compressed(destination/'arrays.npz',**arrays)
            np.savez_compressed(destination/'normalized-errors.npz',errors=np.concatenate(norm_errors))
            atomic_json(destination/'records.json',records);atomic_json(destination/'metrics.json',result)
            files={n:dict(sha256=digest(destination/n),bytes=(destination/n).stat().st_size)
                for n in ('arrays.npz','normalized-errors.npz','records.json','metrics.json')}
            assert all(v['bytes']<=10*1024**2 for v in files.values())
            atomic_json(destination/'receipt.json',dict(status='VALIDATION_COMPLETE',total_steps=step,files=files))
            return result
        finally:policy.train(was_training);policy.reset()


def update_selection(path,result,checkpoint):
    path=Path(path)
    selected=json.loads(path.read_text()) if path.exists() else {}
    for kind in ('best_error','best_limits'):
        key=selection_key(result,kind)
        if kind not in selected or key<tuple(selected[kind]['key']):
            selected[kind]=dict(total_steps=result['total_steps'],key=list(key),checkpoint=str(checkpoint),
                admissible=False,meaning='relative offline selection; not robot authorization')
    atomic_json(path,selected)
    return selected
