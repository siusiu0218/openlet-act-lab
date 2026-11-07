"""Portable training helpers extracted from the validated ACT experiment."""

import contextlib
import math
import os
import random
import threading

def sampling_plan_350(lengths):
    if sorted(lengths)!=list(range(24,31)):raise ValueError('episodes')
    plan=[];offset=0
    for ep,n in sorted(lengths.items()):
        if n<50:raise ValueError('episode too short')
        original={q*(n-1)//4 for q in range(5)}
        frames=set(original)
        # Fill from a denser deterministic grid; old five inputs are mandatory.
        candidates=[q*(n-1)//49 for q in range(50)]+list(range(n))
        for frame in candidates:
            if len(frames)==50:break
            frames.add(frame)
        for frame in sorted(frames):plan.append(dict(episode=ep,frame=frame,length=n,dataset_index=offset+frame,legacy=frame in original))
        offset+=n
    return plan

def batch_order(length,updates,batch_size,seed):
    # Each epoch is a new permutation; persisted flat order is the resume cursor basis.
    rng=random.Random(seed);order=[]
    while len(order)<updates*batch_size:
        epoch=list(range(length));rng.shuffle(epoch);order.extend(epoch)
    return [order[i:i+batch_size] for i in range(0,updates*batch_size,batch_size)]

def may_start(now,stop_at,required_seconds=0):
    return now+required_seconds<stop_at

@contextlib.contextmanager
def bounded_operation(seconds,label):
    # A Python exception cannot reliably interrupt a blocked CUDA/C call; terminate this
    # direct training child on expiry. Host/step cleanup remains independently supervised.
    def expired():
        print('LONG_ACT_OPERATION_TIMEOUT '+label,flush=True)
        os._exit(124)
    timer=threading.Timer(seconds,expired);timer.daemon=True;timer.start()
    try:yield
    finally:timer.cancel();timer.join()

@contextlib.contextmanager
def preserve_rng(torch,np):
    state=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    try:yield
    finally:
        random.setstate(state[0]);np.random.set_state(state[1]);torch.set_rng_state(state[2]);torch.cuda.set_rng_state_all(state[3])

def selection_key(result,kind):
    if kind=='best_error':return (result['normalized_l1'],)
    return (result['first_above_001rad'],result['full_max_excess_rad'],result['normalized_l1'])

def run_training_loop(loader,update,snapshot,validate,clock,stop_at,updates,save_every):
    iterator=iter(loader);completed=0
    for count in range(1,updates+1):
        if not may_start(clock(),stop_at):return completed,True
        with bounded_operation(60,'training_update_'+str(count)):
            raw=next(iterator)
            if not may_start(clock(),stop_at):return completed,True
            update(raw,count);completed=count
        if count%save_every==0 or count==updates:
            with bounded_operation(600,'checkpoint'):destination=snapshot(count==updates)
            if not validate(destination):return completed,True
    return completed,False

def restore_rng(state,torch,np):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
    assert random.getstate()==state['python']
    actual=np.random.get_state();saved=state['numpy']
    assert actual[0]==saved[0] and np.array_equal(actual[1],saved[1]) and actual[2:]==saved[2:]
    assert torch.equal(torch.get_rng_state(),state['torch'])
    assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state['cuda']))
