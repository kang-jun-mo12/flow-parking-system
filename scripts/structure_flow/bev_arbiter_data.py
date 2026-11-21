"""Rows for the arbiter: a learned model in place of the rule that says which of the two drives -
the parking model or the avoidance Flow.

The rule (supervisor.FlowDriver.update -> relevance.RelevanceGate) forecasts the car 3 s along the
parking model's plan and every walker at constant velocity, the props standing: a box overlap within 0.25 m is a
conflict, and the Flow takes over. It reads the simulator's walkers. The arbiter is to say the same from the perceived
BEV and the plan, so its label is that rule's own verdict of every decision step - `raw_conflict`, apart for walkers
and props; the 0.6 s the rule waits before it lets go, the staging and the quiet time after a hand-back are time, not
perception, and stay rules (the supervisor).

replay(folder): the rule run again over a recorded episode (frames.json, teacher.npz, meta.json) - the plan the tracker
had at each decision step, where on it the car was, the forecast the gate made, its verdict. Where an episode
carries the verdict the rule gave at run time (frames.json `gate`), `check` compares the two.

usage (Python 3.11, from scripts/):
  python structure_flow/bev_arbiter_data.py check ROOT [ROOT ...]         replayed verdicts against the recorded ones
  python structure_flow/bev_arbiter_data.py build OUT.npz ROOT [ROOT ...]  rows of the episodes with a perceived BEV
"""
import argparse
import glob
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import supervisor as DF                                 # noqa: E402  (the gate, its configuration, the car's sizes)

DECISION = 2                # frames.json rows (10 Hz) per decision step (0.2 s)
SAMPLES = 31                # the gate's forecast: 0 .. 3 s every 0.1 s
VALIDATION_SLOTS = (19, 26, 34, 45)         # as bev_flow_data.py and bev_refiner.py


class Plan:
    """benchmark.Tracker as far as the gate reads it: the path of one plan in the world and where on it the car is."""

    def __init__(self, rear, rows):
        rows = np.asarray(rows, float)
        c, s = math.cos(rear[2]), math.sin(rear[2])
        self.path = np.stack((rear[0]+c*rows[:, 0]-s*rows[:, 1], rear[1]+s*rows[:, 0]+c*rows[:, 1]), 1)
        self.speed, self.phase = np.abs(rows[:, 3]), rows[:, 4].astype(int)
        self.gears = np.sign(rows[:, 3])
        for phase in set(self.phase.tolist()):
            part = self.phase == phase
            self.gears[part] = np.sign(self.gears[part].sum()) or 1
        self.yaw = rear[2]+rows[:, 2]
        self.index = 0

    def follow(self, rear, speed):
        """One tracker step (Tracker.control): the nearest point of the next 60 within the phase; the next phase once
        the car stands at the end of this one."""
        if self.index >= len(self.path):
            return
        last = int(np.nonzero(self.phase == self.phase[self.index])[0][-1])
        window = np.arange(self.index, min(self.index+60, last+1))
        self.index = int(window[np.argmin(np.linalg.norm(self.path[window]-rear, axis=1))])
        if self.index >= last and abs(speed) < .05 and last+1 < len(self.path):
            self.index = last+1

    def states(self, pose):
        """FlowDriver.path_states: the plan left, [x, y, yaw, direction, 0, km/h] in the actor frame."""
        if len(self.path) < 2:
            return []
        yaw0 = math.radians(pose[2])
        rear = np.array([pose[0]-DF.SPECS['rear_axle_to_center']*math.cos(yaw0),
                         pose[1]-DF.SPECS['rear_axle_to_center']*math.sin(yaw0)])
        start = max(0, min(self.index, len(self.path)-1)-2)
        near = start+int(np.argmin(np.hypot(self.path[start:, 0]-rear[0], self.path[start:, 1]-rear[1])))
        c, s = math.cos(yaw0), math.sin(yaw0)
        dx, dy = self.path[near:, 0]-pose[0], self.path[near:, 1]-pose[1]
        return np.stack((c*dx+s*dy, -s*dx+c*dy, (self.yaw[near:]-yaw0+math.pi) % (2*math.pi)-math.pi,
                         np.where(self.gears[near:] >= 0, 1., -1.), np.zeros(len(dx)), 3.6*self.speed[near:]), 1).tolist()


def to_local(pose, x, y, vx=0., vy=0.):
    yaw = math.radians(pose[2])
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x-pose[0], y-pose[1]
    return c*dx+s*dy, -s*dx+c*dy, c*vx+s*vy, -s*vx+c*vy


def replay(folder):
    """The rule over a recorded episode: one dict per decision step - t, frame (index into frames.json), pose, speed,
    forecast [SAMPLES, 3] (x, y, yaw of the car's centre in the actor frame of now), raw / walker / prop (the label),
    active (with the rule's 0.6 s), conflict_time, gap, controller, and `recorded` (the verdict kept at run time)."""
    with open(os.path.join(folder, 'meta.json'), encoding='utf-8') as stream:
        meta = json.load(stream)
    with open(os.path.join(folder, 'frames.json'), encoding='utf-8') as stream:
        frames = json.load(stream)
    teacher = np.load(os.path.join(folder, 'teacher.npz'), allow_pickle=True)
    plans = [None if bool(teacher['hold'][p]) else (teacher['rear'][p], teacher['rows'][p]) for p in range(len(teacher['t']))]
    props = [(p['x'], p['y'], math.radians(p['yaw']), p['length'], p['width']) for p in (meta.get('props') or [])]
    sizes = [(max(.6, w.get('length', .6)), max(.6, w.get('width', .6))) for w in (meta.get('walkers') or [])]
    gate = DF.relevance.RelevanceGate(dict(DF.GATE_CONFIG), DF.SPECS)
    out, plan, plan_index = [], None, -1
    for i, f in enumerate(frames):
        pose = (f['x'], f['y'], f['yaw'])
        yaw0 = math.radians(pose[2])
        rear = np.array([pose[0]-DF.SPECS['rear_axle_to_center']*math.cos(yaw0),
                         pose[1]-DF.SPECS['rear_axle_to_center']*math.sin(yaw0)])
        if i % DECISION == 0:
            tracks = []
            for k, (x, y, vx, vy) in enumerate(f['walkers']):
                lx, ly, lvx, lvy = to_local(pose, x, y, vx, vy)
                length, width = sizes[k] if k < len(sizes) else (.6, .6)
                tracks.append(dict(object_id='walker-{}'.format(k), class_name='pedestrian', x=lx, y=ly, vx=lvx, vy=lvy,
                                   yaw=math.atan2(lvy, lvx) if math.hypot(lvx, lvy) > .1 else 0., length=length,
                                   width=width, age_s=0., hits=10, ever_moving=True))
            for k, (x, y, yaw, length, width) in enumerate(props):
                lx, ly, _, _ = to_local(pose, x, y)
                tracks.append(dict(object_id='prop-{}'.format(k), class_name='vehicle', x=lx, y=ly, vx=0., vy=0.,
                                   yaw=yaw-yaw0, length=length, width=width, age_s=0., hits=10, ever_moving=True))
            states = plan.states(pose) if plan is not None else []
            signed = 3.6*f['speed'] if not f['reverse'] else -3.6*abs(f['speed'])
            report = gate.update(float(f['t'])+1e-9*(i+1), tracks, states, 0, signed)
            forecast = DF.relevance.forecast_path(states, 0, DF.GATE_CONFIG, DF.SPECS, signed)
            ids = [str(r['object_id']) for r in report['tracks']]
            out.append(dict(t=float(f['t']), frame=i, pose=pose, speed=float(f['speed']), reverse=bool(f['reverse']),
                            forecast=np.asarray(forecast, np.float32)[:, 1:], planned=plan is not None,
                            raw=bool(report['raw_conflict']), walker=any(not s.startswith('prop') for s in ids),
                            prop=any(s.startswith('prop') for s in ids), active=bool(report['active']),
                            conflict_time=float(report['conflict_time_s']), gap=float(report['minimum_gap_m']),
                            controller=f['controller'], recorded=f.get('gate') if f.get('gate') and
                            abs(f['gate']['t']-f['t']) < 1e-3 else None))
        # after the decision of this tick: the plan asked for now (frames.json `plan`: the number of the latest)
        if f['plan'] != plan_index:
            plan_index = f['plan']
            plan = None if not 0 <= plan_index < len(plans) or plans[plan_index] is None else Plan(*plans[plan_index])
        if plan is not None and f['controller'] != 'flow':
            plan.follow(rear, f['speed'])
    return meta, out


def folders_of(roots):
    out = []
    for root in roots:
        out += sorted(d for d in glob.glob(os.path.join(root, 'ep*')) if all(
            os.path.exists(os.path.join(d, n)) for n in ('meta.json', 'frames.json', 'teacher.npz')))
    return out


def check(roots):
    total = dict(steps=0, active=0, raw_same=0, walker_same=0, active_same=0, active_recorded=0, onsets=0, onset_within=0)
    worst = []
    for folder in folders_of(roots):
        try:
            _, rows = replay(folder)
        except Exception as error:                      # noqa: BLE001
            print(os.path.basename(folder), 'failed:', repr(error)[:120])
            continue
        rows = [r for r in rows if r['recorded'] is not None]
        if not rows:
            continue
        mine = np.array([r['active'] for r in rows])
        theirs = np.array([r['recorded']['active'] for r in rows])
        walker = np.array([r['walker'] for r in rows]) == np.array([r['recorded']['walker'] for r in rows])
        raw = np.array([r['raw'] for r in rows]) == np.array([bool(r['recorded']['conflicts']) for r in rows])
        total['steps'] += len(rows)
        total['active_same'] += int((mine == theirs).sum())
        total['walker_same'] += int(walker.sum())
        total['raw_same'] += int(raw.sum())
        total['active_recorded'] += int(theirs.sum())
        for k in np.nonzero(theirs[1:] & ~theirs[:-1])[0]+1:        # the rule's onsets: mine within 0.4 s
            total['onsets'] += 1
            total['onset_within'] += bool(mine[max(0, k-2):k+3].any() and not mine[max(0, k-3)])
        worst.append((float((mine != theirs).mean()), os.path.basename(folder), len(rows)))
    n = max(1, total['steps'])
    print(json.dumps(dict(steps=total['steps'], active_agreement=round(total['active_same']/n, 4),
                          raw_agreement=round(total['raw_same']/n, 4), walker_agreement=round(total['walker_same']/n, 4),
                          recorded_active_share=round(total['active_recorded']/n, 3), onsets=total['onsets'],
                          onsets_within_0_4_s=total['onset_within'])))
    print('least alike:', [(name, round(d, 3), k) for d, name, k in sorted(worst, reverse=True)[:6]])


def build(out, roots):
    """One row per decision step that has a perceived BEV (bev_refined.npz of bev_perceived.py): what the arbiter
    reads (pose, goal, the perceived static mask and people, the gate's forecast of the car along the plan, speed)
    and the rule's verdict of that step."""
    keys = ('episode', 't', 'pose', 'goal', 'static', 'walkers', 'forecast', 'planned', 'speed', 'reverse', 'walker',
            'prop', 'active', 'conflict_time', 'gap', 'flow')
    data = {k: [] for k in keys}
    episodes = []
    for folder in folders_of(roots):
        refined = os.path.join(folder, 'bev_refined.npz')
        if not os.path.exists(refined):
            continue
        try:
            meta, rows = replay(folder)
        except Exception as error:                      # noqa: BLE001
            print(os.path.basename(folder), 'failed:', repr(error)[:120])
            continue
        if meta.get('result', {}).get('outcome') in (None, 'skipped') or not rows:
            continue
        with np.load(refined) as saved:
            frames, static, walkers = saved['frames'], saved['static'], saved['walkers']
        kept = 0
        for r in rows:
            k = int(np.searchsorted(frames, r['frame'], 'right'))-1
            if k < 0 or r['frame']-int(frames[k]) > 1:
                continue                    # no perception at this step (after the parking was done)
            data['episode'].append(len(episodes))
            data['t'].append(r['t'])
            data['pose'].append(r['pose'])
            data['goal'].append(meta['goal'])
            data['static'].append(static[k])
            data['walkers'].append(walkers[k])
            data['forecast'].append(r['forecast'])
            data['planned'].append(r['planned'])
            data['speed'].append(-abs(r['speed']) if r['reverse'] else r['speed'])
            data['reverse'].append(r['reverse'])
            data['walker'].append(r['walker'])
            data['prop'].append(r['prop'])
            data['active'].append(r['active'])
            data['conflict_time'].append(r['conflict_time'])
            data['gap'].append(r['gap'])
            data['flow'].append(r['controller'] == 'flow')
            kept += 1
        if kept:
            episodes.append(dict(folder=folder.replace(chr(92), '/'), slot=int(meta['slot']), case=meta.get('case'),
                                 outcome=meta['result']['outcome'], rows=kept))
    arrays = dict(episode=np.asarray(data['episode'], np.int32), t=np.asarray(data['t'], np.float32),
                  pose=np.asarray(data['pose'], np.float32), goal=np.asarray(data['goal'], np.float32),
                  static=np.stack(data['static']), walkers=np.stack(data['walkers']).astype(np.float32),
                  forecast=np.stack(data['forecast']).astype(np.float32), planned=np.asarray(data['planned'], bool),
                  speed=np.asarray(data['speed'], np.float32), reverse=np.asarray(data['reverse'], bool),
                  walker=np.asarray(data['walker'], bool), prop=np.asarray(data['prop'], bool),
                  active=np.asarray(data['active'], bool), conflict_time=np.asarray(data['conflict_time'], np.float32),
                  gap=np.asarray(data['gap'], np.float32), flow=np.asarray(data['flow'], bool))
    np.savez_compressed(out, episodes=json.dumps(episodes), **arrays)
    validation = np.array([e['slot'] in VALIDATION_SLOTS for e in episodes])[arrays['episode']]
    print('episodes', len(episodes), 'rows', len(arrays['t']), 'validation rows', int(validation.sum()))
    print('share of the steps: walker conflict %.3f, prop conflict %.3f, rule active %.3f, the Flow drives %.3f, no plan %.3f' % (
        arrays['walker'].mean(), arrays['prop'].mean(), arrays['active'].mean(), arrays['flow'].mean(),
        1.-arrays['planned'].mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('check', 'build'))
    parser.add_argument('paths', nargs='+')
    args = parser.parse_args()
    if args.command == 'check':
        check(args.paths)
    else:
        build(args.paths[0], args.paths[1:])


if __name__ == '__main__':
    main()
