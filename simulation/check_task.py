import json,os,sys
from pathlib import Path
def assess(root,scope):
 def read(n):
  p=root/n;return p.read_text(errors='replace') if p.exists() else ''
 errors=[];metrics={};log=read('policy.log')
 if read('process-exit-code.txt').strip()!='0': errors.append('policy process did not exit zero')
 if 'Traceback' in log or 'Task 1 V2 failed:' in log:errors.append('policy exception')
 rows=[json.loads(s) for s in read('task1_grasp_diagnostics.jsonl').splitlines() if s.strip()]
 expected_seed=int(os.environ.get('OPENLET_TEST_SEED','28'))
 if {r.get('seed') for r in rows}!={expected_seed}:errors.append('missing or mismatched seed')
 names=['cylinder_1'] if scope=='first' else ['cylinder_1','cylinder_2','cylinder_3']
 for name in names:
  events={r['event']:r for r in rows if r.get('cylinder')==name}
  if not all(k in events for k in ('pre_close','post_close','post_lift','post_release')):
   errors.append(name+': missing phase');continue
  lift=events['post_lift']['object_position_world'][2]-events['pre_close']['object_position_world'][2]
  placed=events['post_release'].get('fully_in_target') is True
  metrics[name]=dict(lift_m=lift,official_center_in_bin=placed,release=events['post_release']['object_position_world'])
  if lift<0.08:errors.append(name+': actual lift below 8 cm')
  if not placed:errors.append(name+': official bin check failed')
 if scope=='first':
  if not (root/'first-complete.txt').exists():errors.append('first diagnostic incomplete')
 else:
  if read('task_result.txt').strip()!='success':errors.append('full task result not success')
  for marker in ('V2 source bin handoff triggered at lever ','V2 source bin conveyor motion complete and settled'):
   if marker not in log:errors.append('missing milestone: '+marker)
 return dict(passed=not errors,errors=errors,metrics=metrics,scope=scope,seed=expected_seed,placement_definition='new official center-in-bin check; not proof of whole-object containment or platform score')
if __name__=='__main__':
 root=Path(sys.argv[1]);result=assess(root,sys.argv[2]);(root/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));sys.exit(not result['passed'])
