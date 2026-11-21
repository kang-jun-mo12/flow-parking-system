"""Training rows of the BEV avoidance Flow from collected lot episodes (drive.py + system_server.py):
one row per teacher decision (flow_decisions.npz, every 0.2 s while the Flow drives), with what the BEV Flow reads
and what it has to give.

Per row
  pose [3] world x, y, yaw deg; goal [3] x, y, yaw rad; static: the truth static occupancy of bev_truth, bit-packed;
  walkers [3, 5] x, y, vx, vy (ego frame, m and m/s), valid; vector [VEC] (see vector()); labels: actions [4, 2] steer
  and speed fraction, gears [4] (0 forward, 1 stop, 2 reverse), intent (INTENTS), trajectory [15, 3] the car's own
  pose 0.2 .. 3 s later in the ego frame of now (x, y m, yaw rad) with its mask (what the teacher's driving led to);
  change: the intent differs from the row before in the same intervention.
The border of the lot and the target slot are drawn from pose and goal when a batch is made (bev_flow.py).
  static_perceived, walkers_perceived, perceived: the same two as the perception gave them (bev_perceived.py: the
  refiner's static mask and its people), where the episode has them - bev_refined.npz of a run with images (the
  evidence frame at or just before the decision), or what the student itself read in a run with the perception in the
  loop (flow_decisions.npz student_static, student_walkers).

Episodes that parked, rows the teacher marked valid - whoever drove: in a DAgger run (system_server.py
--student) the BEV Flow drives and the teacher's decision at its state is the label; the vector is then the one the
student had, and the trajectory aid is left out (the driving was not the teacher's).
usage (Python 3.11, from scripts/): python structure_flow/bev_flow_data.py OUT.npz ROOT [ROOT ...]
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bev_truth as T                                   # noqa: E402

INTENTS = ('stage', 'wait', 'forward_left', 'forward_right', 'back_left', 'back_right', 'back_away', 'forward_away')
MIRROR = (0, 1, 3, 2, 5, 4, 6, 7)          # the intent of the mirrored scene
WAIT = ('cross_ahead_wait', 'cross_side_wait', 'along_let_pass', 'along_behind_wait', 'behind_let_pass', 'prop_wait',
        'stop', 'prop_stop_to_turn', 'along_wait_aside')
STAGE = ('stage_plan', 'staged', 'stage_blocked')
WALKERS = 3
HORIZON = 15                # trajectory steps of 0.2 s
HISTORY = 5                 # ego poses 0.2 .. 1 s back
AISLE_FROM_SLOT = 5.45      # safety.AISLE_FROM_SLOT
STILL_KEPT, STILL_EVERY = 5, 5      # rows of a standing student in a standing scene: the first 5, then every 5th
NO_STATIC = np.zeros(T.ROWS*T.COLS//8, np.uint8)
VEC = 4+4+4+18+3*HISTORY+len(INTENTS)+1


def intent_of(source, gear):
    """Teacher reason (the rule teacher's name of its manoeuvre) and the first gear of its chunk -> index into INTENTS."""
    if source in STAGE:
        return 0
    if source in WAIT or gear == 1:
        return 1
    if source.endswith('_left') or source.endswith('_right'):
        side = 0 if source.endswith('_left') else 1
        return (4 if gear == 2 else 2)+side
    return 6 if gear == 2 else 7


def to_ego(pose, x, y):
    c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
    dx, dy = x-pose[0], y-pose[1]
    return c*dx+s*dy, -s*dx+c*dy


def stage_features(pose, goal):
    """The car in the aisle's frame at the goal slot, as safety.aisle_frame: s along the aisle from abreast of
    the slot (the aisle direction closer to the car's heading), lateral offset, heading error."""
    gx, gy = to_ego(pose, goal[0], goal[1])
    gyaw = goal[2]-math.radians(pose[2])
    centre = np.array([gx, gy])+AISLE_FROM_SLOT*np.array([math.cos(gyaw), math.sin(gyaw)])
    a = np.array([-math.sin(gyaw), math.cos(gyaw)])
    if a[0] < 0:
        a = -a
    n = np.array([-a[1], a[0]])
    return float(a @ -centre), float(n @ -centre), math.atan2(a[1], a[0])


def relative(pose, other):
    """other pose in the ego frame of pose: x, y (m), yaw (rad)."""
    x, y = to_ego(pose, other[0], other[1])
    return x, y, (math.radians(other[2]-pose[2])+math.pi) % (2*math.pi)-math.pi


def vector(speed, steer, reverse, pose, goal, condition, history, previous_intent, elapsed):
    """[VEC]: speed (signed, / 3 m/s), the steer and the direction applied before this decision (the frame recorded
    at the decision holds the control the decision gave: read from it the first steer and the gear of the answer
    would leak in), the parking planner's steer; the goal slot (x / 20, y / 20,
    cos, sin of its yaw); the aisle (s / 10, lateral / 3, sin, cos of the heading error); the tail of the previous
    plan (18, the worker's condition); the car's poses 0.2 .. 1 s ago; the previous intent (one hot); decisions since
    the Flow took over / 50."""
    gx, gy = to_ego(pose, goal[0], goal[1])
    gyaw = goal[2]-math.radians(pose[2])
    s, lateral, heading = stage_features(pose, goal)
    out = [np.clip(speed/3., -1.5, 1.5), steer, -1. if reverse else 1., float(condition[4]),
           np.clip(gx/20., -1., 1.), np.clip(gy/20., -1., 1.), math.cos(gyaw), math.sin(gyaw),
           np.clip(s/10., -2., 2.), np.clip(lateral/3., -1.5, 1.5), math.sin(heading), math.cos(heading)]
    out += [float(v) for v in condition[10:28]]
    for other in history:
        out += list(relative(pose, other))
    one_hot = [0.]*len(INTENTS)
    if previous_intent is not None:
        one_hot[previous_intent] = 1.
    return np.asarray(out+one_hot+[min(elapsed/50., 2.)], np.float32)


def episode_rows(folder):
    if not os.path.exists(os.path.join(folder, 'flow_decisions.npz')):
        return None
    scene = T.Scene(folder)
    z = np.load(os.path.join(folder, 'flow_decisions.npz'), allow_pickle=True)
    outcome = scene.meta['result']['outcome']
    if outcome != 'parked' and ('student_vector' not in z.files or bool(np.all(z['expert']))):
        return None             # the teacher drove and failed: not an example (also with a student riding along)
    # the student drove: the teacher's decisions at its states count whatever the end - but not the last second
    # before a collision, when no decision could help any more
    usable = len(z['t'])-(5 if outcome == 'collision' else 0)
    rows = dict(pose=[], goal=[], static=[], walkers=[], vector=[], actions=[], gears=[], intent=[], trajectory=[],
                trajectory_mask=[], change=[], frame=[], t=[], static_perceived=[], walkers_perceived=[], perceived=[])
    previous_segment, previous_intent, elapsed, still = None, None, 0, 0
    student = 'student_vector' in z.files
    teacher_drove = bool(np.all(z['expert']))
    refined = None
    if os.path.exists(os.path.join(folder, 'bev_refined.npz')):
        with np.load(os.path.join(folder, 'bev_refined.npz')) as saved:
            refined = dict(frames=saved['frames'], static=saved['static'], walkers=saved['walkers'])
    in_loop = (z['student_static'], z['student_walkers']) if 'student_static' in z.files else None
    for i in range(len(z['t'])):
        index = int(round(float(z['t'][i])*10))
        segment = int(z['segment'][i])
        if segment != previous_segment:
            previous_segment, previous_intent, elapsed = segment, None, 0
        intent = intent_of(str(z['target_source'][i]), int(z['target_gears'][i][0]))
        if index >= len(scene) or not bool(z['valid'][i]) or i >= usable:
            previous_intent, elapsed = intent, elapsed+1
            continue
        pose = scene.pose(index)
        frame = scene.frames[index]
        walkers = np.zeros((WALKERS, 5), np.float32)
        c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
        for k, (x, y, vx, vy) in enumerate(scene.walkers(index)[:WALKERS]):
            ex, ey = to_ego(pose, x, y)
            walkers[k] = (ex, ey, c*vx+s*vy, -s*vx+c*vy, 1.)
        # the student stands and nothing moves (a stuck run: decision after decision with the same input and the same
        # answer): after the first second every fifth of them is kept
        still = still+1 if student and abs(frame['speed']) < .05 and not np.any(np.abs(walkers[:, 2:4]) > .05) else 0
        if still > STILL_KEPT and still % STILL_EVERY:
            previous_intent, elapsed = intent, elapsed+1
            continue
        history = [scene.pose(max(0, index-2*k)) for k in range(1, HISTORY+1)]
        trajectory, mask = np.zeros((HORIZON, 3), np.float32), np.zeros(HORIZON, np.float32)
        for k in range(HORIZON):
            later = index+2*(k+1)
            if later < len(scene) and teacher_drove:        # what the teacher's driving led to, not the student's
                trajectory[k], mask[k] = relative(pose, scene.pose(later)), 1.
        rows['pose'].append(pose)
        rows['goal'].append(scene.goal)
        rows['static'].append(np.packbits(scene.render(index)[0] > .5))
        rows['walkers'].append(walkers)
        if refined is not None:         # made again from the saved evidence: the refiner in use now
            k = max(0, int(np.searchsorted(refined['frames'], index, 'right'))-1)
            rows['static_perceived'].append(refined['static'][k])
            rows['walkers_perceived'].append(refined['walkers'][k])
        elif in_loop is not None:       # what the refiner of that run gave the student
            rows['static_perceived'].append(in_loop[0][i])
            rows['walkers_perceived'].append(in_loop[1][i])
        else:
            rows['static_perceived'].append(NO_STATIC)
            rows['walkers_perceived'].append(np.zeros((WALKERS, 5), np.float32))
        rows['perceived'].append(in_loop is not None or refined is not None)
        before = scene.frames[max(0, index-1)]        # the control applied up to this decision
        if student:             # a run the BEV Flow took part in: the vector it had (its own previous plan and intent)
            rows['vector'].append(z['student_vector'][i])
        else:
            rows['vector'].append(vector(frame['speed'], before['steer'], before['reverse'], pose, scene.goal,
                                         z['condition'][i], history, previous_intent, elapsed))
        rows['actions'].append(z['target_actions'][i])
        rows['gears'].append(z['target_gears'][i])
        rows['intent'].append(intent)
        rows['trajectory'].append(trajectory)
        rows['trajectory_mask'].append(mask)
        rows['change'].append(previous_intent is not None and intent != previous_intent)
        rows['frame'].append(index)
        rows['t'].append(float(z['t'][i]))
        previous_intent, elapsed = intent, elapsed+1
    if not rows['pose']:
        return None
    meta = scene.meta
    return rows, dict(name=meta['name'], n=meta['n'], slot=meta['slot'], flip=meta.get('flip', 0), case=meta.get('case'))


def build(roots, out):
    folders = []
    for root in roots:
        folders += sorted(str(p) for p in Path(root).iterdir() if (p/'meta.json').exists() and (p/'frames.json').exists())
    data, episodes = {}, []
    for folder in folders:
        result = episode_rows(folder)
        if result is None:
            continue
        rows, info = result
        info.update(folder=folder, first=len(data.get('intent', [])), count=len(rows['intent']))
        episodes.append(info)
        for key, value in rows.items():
            data.setdefault(key, []).extend(value)
    arrays = dict(pose=np.asarray(data['pose'], np.float32), goal=np.asarray(data['goal'], np.float32),
                  static=np.stack(data['static']), walkers=np.stack(data['walkers']), vector=np.stack(data['vector']),
                  actions=np.asarray(data['actions'], np.float32), gears=np.asarray(data['gears'], np.int64),
                  intent=np.asarray(data['intent'], np.int64), trajectory=np.stack(data['trajectory']),
                  trajectory_mask=np.stack(data['trajectory_mask']), change=np.asarray(data['change'], bool),
                  frame=np.asarray(data['frame'], np.int32), t=np.asarray(data['t'], np.float32),
                  static_perceived=np.stack(data['static_perceived']),
                  walkers_perceived=np.stack(data['walkers_perceived']).astype(np.float32),
                  perceived=np.asarray(data['perceived'], bool),
                  episode=np.concatenate([np.full(e['count'], k, np.int32) for k, e in enumerate(episodes)]))
    np.savez_compressed(out, episodes=json.dumps(episodes), intents=np.asarray(INTENTS), **arrays)
    return arrays, episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('roots', nargs='+')
    args = parser.parse_args()
    arrays, episodes = build(args.roots, args.out)
    counts = np.bincount(arrays['intent'], minlength=len(INTENTS))
    print('episodes', len(episodes), 'rows', len(arrays['intent']), 'intent changes', int(arrays['change'].sum()))
    print({name: int(c) for name, c in zip(INTENTS, counts)})
    print('gears of the first step', np.bincount(arrays['gears'][:, 0], minlength=3).tolist(),
          'rows with a walker', int((arrays['walkers'][:, :, 4].sum(1) > 0).sum()),
          'rows with a perceived BEV', int(arrays['perceived'].sum()))


if __name__ == '__main__':
    main()
