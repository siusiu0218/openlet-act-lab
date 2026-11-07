import json
import os
from pathlib import Path
import re

out = Path(os.environ.get('OPENLET_OUTPUT', '/run-data'))
log_path = out / 'model-entry.log'
oversized = log_path.exists() and log_path.stat().st_size >= 64 * 1024 * 1024
log = ''
if log_path.exists():
    with log_path.open('rb') as stream:
        log = stream.read(64 * 1024 * 1024).decode(errors='replace')
records = [dict(name=m[0], process=int(m[1]), tee=int(m[2])) for m in
           re.findall(r'\[NODE_EXIT\] name=(\S+) process=(\d+) tee=(\d+)', log)]
errors = ['model-entry.log exceeded 64 MiB acceptance limit'] if oversized else []
if (out / 'resource-limit.txt').exists():
    errors.append('resource limit exceeded')
if not (out / 'egl-selection-passed.txt').exists():
    errors.append('Release EGL PCI preflight missing')
for name in ('mobile_manipulator_sqp_mpc_node', 'nodelet'):
    matches = [r for r in records if r['name'] == name]
    if not matches or any(r['process'] != 0 or r['tee'] != 0 for r in matches):
        errors.append('missing or nonzero node status: ' + name)
for marker in ('terminate called', 'double free', 'corruption (!prev)', 'Broken promise',
               'SIGKILL', 'SIGTERM', 'Segmentation fault', '_glfwPlatformGetTls',
               'OpenGL error', 'camera_render_error', 'context_destroyed=false',
               'LCM instance not initialized'):
    if marker in log:
        errors.append('fatal/forced marker: ' + marker)
for marker in ('Task 1 model simulator ready at seed', '[MPC_SHUTDOWN] solver callbacks drained',
               'Physics thread exited.', '[MUJOCO_SHUTDOWN] graphics resources released',
               '[DRAKE_SHUTDOWN] released duplicate Drake and config ownership',
               '[MODEL_EXIT] roslaunch_code=',
               '[CONTROLLER_SHUTDOWN] owned controllers released'):
    if marker not in log:
        errors.append('missing milestone: ' + marker)
if log.count('Task 1 model simulator ready at seed '+os.environ.get('OPENLET_TEST_SEED','28')+';') != 1 or log.count('Task 1 model simulator ready at seed') != 1:
    errors.append('expected exactly one requested-seed ready')
if 'continuing after failed task' in log:
    errors.append('internal failed-task retry is not accepted')
for role in ('viewer', 'three_cameras'):
    if '[OPENLET_EGL] role=' + role + ' vendor=NVIDIA Corporation renderer=NVIDIA' not in log:
        errors.append('missing NVIDIA context: ' + role)
if log.count('[OPENLET_EGL] context_destroyed=true') != 2:
    errors.append('expected two clean EGL context destructions')
for key in ('OPENLET_INITIAL_RC', 'OPENLET_FORCED'):
    if os.environ[key] != '0':
        errors.append(key + '=' + os.environ[key])
if os.environ['OPENLET_ENTRY_RC'] not in ('0', '130'):
    errors.append('unexpected entry status: ' + os.environ['OPENLET_ENTRY_RC'])
for relative in ('observations/summary.json', 'runtime-maps.json'):
    path = out / relative
    if not path.exists() or not json.loads(path.read_text()).get('passed'):
        errors.append('acceptance missing or failed: ' + relative)
result = dict(passed=not errors, errors=errors, node_exits=records,
              requested_stop_signal='SIGINT', entry_exit_code=int(os.environ['OPENLET_ENTRY_RC']),
              entry_forced=os.environ['OPENLET_FORCED'] != '0',
              scope='isolated Release official task1 requested-seed short startup/images/state/clean exit; task acceptance is checked separately')
(out / 'exit-check.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
raise SystemExit(0 if not errors else 1)
