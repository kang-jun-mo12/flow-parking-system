"""The perceived BEV in the avoidance Flow's own terms: the refiner (bev_refiner.py) run on the evidence of the frozen
perception (bev_evidence.py), its output read as what bev_flow.bev_truth() takes in place of the simulator's objects -
the static occupancy mask and the people as a list (the blobs of the dynamic layer, their velocity from the velocity
layers). The Flow's BEV is drawn from these exactly as it is drawn from the truth, so one Flow reads either.

The people are kept over time (People): what the refiner shows where no camera looks, beside the car, is carried on
from what was seen before, not started there.

Perception: the refiner with the memory of its last evidence frames, one frame at a time (run time). Offline a whole
episode: FOLDER/bev_evidence.npz -> FOLDER/bev_refined.npz with frames [N] (recorded frame of each evidence frame),
static [N, ROWS*COLS/8] bit-packed, walkers [N, WALKERS, 5] (ego x, y m, vx, vy m/s, valid; nearest first).

usage (Python 3.11, from scripts/):
  python structure_flow/bev_perceived.py CHECKPOINT ROOT [ROOT ...] [--redo]
  python structure_flow/bev_perceived.py CHECKPOINT ROOT [ROOT ...] --compare       (against the truth, nothing saved)
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bev_truth as T                                   # noqa: E402
import bev_refiner as R                                 # noqa: E402

WALKERS = 3                 # bev_flow_data.WALKERS
MIN_CELLS = 6               # smaller blobs of the dynamic layer are dropped (the disc of a person is 28 cells)
# the dynamic layer is trained with its positives weighted 40 x (bev_refiner.loss_of): logit 0 is a true probability
# of some 2 %. A person is a blob of cells above MASK_AT whose strongest cell is above PEAK_AT
MASK_AT, PEAK_AT = 0., 0.
LOT = T.LOT_BOUND


def border_of(pose):
    """[B, ROWS, COLS] float: the cells outside the lot (the map prior; bev_flow.bev_truth draws the same)."""
    x, y = R.base_grid(pose.device)
    yaw = torch.deg2rad(pose[:, 2])
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    px = pose[:, 0, None, None]+c*x-s*y
    py = pose[:, 1, None, None]+s*x+c*y
    return (~((px > LOT[0]) & (px < LOT[1]) & (py > LOT[2]) & (py < LOT[3]))).float()


def frames_batch(model, evidence, velocity, poses, ks, device):
    """The refiner's batch for the evidence frames ks of an episode (arrays over all its frames): what
    bev_refiner.Episode.sample gives, without the truth."""
    def tensor(a):
        return torch.from_numpy(np.ascontiguousarray(a)).to(device).float()
    ks = np.asarray(ks)
    past = np.stack([np.maximum(0, ks-p) for p in R.PAST], 1)
    batch = dict(evidence=tensor(evidence[ks]), velocity=tensor(velocity[ks].astype(np.float32)), past=tensor(evidence[past]),
                 pose=tensor(poses[ks]), past_pose=tensor(poses[past]))
    if model.memory:
        back = np.stack([np.maximum(0, ks-h) for h in R.HISTORY], 1)
        batch.update(history=tensor(evidence[back]), history_pose=tensor(poses[back]))
    return batch


@torch.no_grad()
def refine(model, batch):
    """batch: evidence [B, C, R, C], velocity [B, 2, R, C], past [B, len(PAST), C, R, C], pose [B, 3], past_pose
    [B, len(PAST), 3] (floats on the device, the evidence in its uint8 values). Returns numpy static [B, R, C] bool,
    the dynamic logits [B, R, C], velocity [B, 2, R, C] m/s."""
    batch = dict(batch, border=border_of(batch['pose']))
    with torch.autocast('cuda', dtype=torch.float16, enabled=batch['pose'].device.type != 'cpu'):
        out = model(R.make_input(batch), R.make_history(batch) if model.memory else None).float()
    return (out[:, 0] > 0).cpu().numpy(), out[:, 1].cpu().numpy(), (out[:, 2:]*R.SPEED_SCALE).cpu().numpy()


def blobs(logit, flow, mask_at=None):
    """logit [R, C] of the dynamic layer, flow [2, R, C] m/s -> [(ego x, y, vx, vy, the strongest logit)] of its
    blobs above mask_at."""
    import cv2
    dynamic = logit > (MASK_AT if mask_at is None else mask_at)
    if not dynamic.any():
        return []
    count, labels, stats, centres = cv2.connectedComponentsWithStats(dynamic.astype(np.uint8), connectivity=8)
    found = []
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < MIN_CELLS:
            continue
        left, top, width, height = stats[i, :4]
        box = (slice(top, top+height), slice(left, left+width))
        peak = float(logit[box][labels[box] == i].max())
        x, y = R.cell_to_ego(centres[i][1], centres[i][0])
        # the velocity layers are trained on the person's disc and 2 cells around it only: read them at the middle
        r, c = int(round(centres[i][1])), int(round(centres[i][0]))
        window = (slice(max(0, r-2), r+3), slice(max(0, c-2), c+3))
        cells = labels[window] == i
        if not cells.any():
            window, cells = (slice(None), slice(None)), labels == i
        found.append((x, y, float(flow[0][window][cells].mean()), float(flow[1][window][cells].mean()), peak))
    return found


def people(logit, flow, mask_at=None, peak_at=None):
    """The blobs whose strongest cell is above peak_at as [WALKERS, 5] ego x, y, vx, vy, valid, the nearest first
    (frame by frame, without People's memory)."""
    peak_at = PEAK_AT if peak_at is None else peak_at
    out = np.zeros((WALKERS, 5), np.float32)
    found = sorted((math.hypot(b[0], b[1]),)+b[:4] for b in blobs(logit, flow, mask_at) if b[4] >= peak_at)
    for k, row in enumerate(found[:WALKERS]):
        out[k] = row[1:]+(1.,)
    return out


class People:
    """The people of the refined BEV over time, kept in the world frame. Beside the car, within a metre of its
    flanks, no camera of the rig looks: a walker passing there is out of sight for longer than the refiner's 0.8 s of
    evidence, and with its positives weighted 40 x the dynamic layer tends to fill that strip with phantoms from its
    prior. So: a blob starts a person only when it is strong (START), a known person is carried on by
    any blob near where it should be (hysteresis) and, unseen, coasts at its velocity - COAST_BLIND steps where the
    cameras do not look, COAST elsewhere."""
    START, CONFIRM = 3., 2          # logit of the strongest cell that starts a track; hits before it is given out
    SURE = 4.                       # a blob this strong is given out at once
    GATE = 1.                       # m between a blob and the predicted position
    COAST, COAST_BLIND = 2, 10      # decision steps (0.2 s) a track lives unseen
    BLIND = (3.4, 2.7)              # |x|, |y| of the strip beside the car in its frame, m
    NO_START = (3., 2.6)            # no person starts in there: nobody is first seen where no camera looks
    STEP = .2

    def __init__(self, **changes):
        for key, value in changes.items():
            setattr(self, key, value)
        self.reset()

    def reset(self):
        self.tracks = []

    def step(self, found, pose):
        """found: blobs() of this frame (ego frame), pose (x, y, yaw deg) -> [WALKERS, 5] ego x, y, vx, vy, valid."""
        c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
        seen = [(pose[0]+c*x-s*y, pose[1]+s*x+c*y, c*vx-s*vy, s*vx+c*vy, peak) for x, y, vx, vy, peak in found]
        for track in self.tracks:
            track['x'] += track['vx']*self.STEP
            track['y'] += track['vy']*self.STEP
        pairs = sorted((math.hypot(t['x']-b[0], t['y']-b[1]), i, j) for i, t in enumerate(self.tracks) for j, b in enumerate(seen))
        taken, matched = set(), set()
        for d, i, j in pairs:
            if d > self.GATE or i in matched or j in taken:
                continue
            matched.add(i)
            taken.add(j)
            track, blob = self.tracks[i], seen[j]
            track.update(x=blob[0], y=blob[1], vx=.5*track['vx']+.5*blob[2], vy=.5*track['vy']+.5*blob[3],
                         hits=track['hits']+1, unseen=0, sure=track['sure'] or blob[4] >= self.SURE)
        kept = []
        for i, track in enumerate(self.tracks):
            ex, ey = c*(track['x']-pose[0])+s*(track['y']-pose[1]), -s*(track['x']-pose[0])+c*(track['y']-pose[1])
            track['ego'] = (ex, ey)
            if i not in matched:
                track['unseen'] += 1
            blind = abs(ex) < self.BLIND[0] and abs(ey) < self.BLIND[1]
            given = track['hits'] >= self.CONFIRM or track['sure']
            if abs(ex) > T.X_HALF or abs(ey) > T.Y_HALF or track['unseen'] > (self.COAST_BLIND if blind and given else self.COAST):
                continue
            kept.append(track)
        for j, blob in enumerate(seen):
            if j not in taken and blob[4] >= self.START:
                ex, ey = c*(blob[0]-pose[0])+s*(blob[1]-pose[1]), -s*(blob[0]-pose[0])+c*(blob[1]-pose[1])
                if abs(ex) < self.NO_START[0] and abs(ey) < self.NO_START[1]:
                    continue
                kept.append(dict(x=blob[0], y=blob[1], vx=blob[2], vy=blob[3], hits=1, unseen=0, sure=blob[4] >= self.SURE,
                                 ego=(ex, ey)))
        self.tracks = kept
        out = np.zeros((WALKERS, 5), np.float32)
        given = [(math.hypot(*t['ego']), t) for t in kept if t['hits'] >= self.CONFIRM or t['sure']]
        for k, (_, t) in enumerate(sorted(given, key=lambda g: g[0])[:WALKERS]):
            out[k] = (t['ego'][0], t['ego'][1], c*t['vx']+s*t['vy'], -s*t['vx']+c*t['vy'], 1.)
        return out


class Perception:
    """The refiner one evidence frame (0.2 s) at a time: it keeps the frames its input looks back to."""

    def __init__(self, checkpoint=None, device='cuda', model=None):
        self.model, self.device = model if model is not None else R.load(str(checkpoint), device), device
        self.reset()

    def reset(self):
        self.memory = []            # (evidence, pose) of the evidence frames looked back to, the newest last
        self.people = People()

    def step(self, evidence, velocity, pose):
        """evidence uint8 [C, ROWS, COLS], velocity [2, ROWS, COLS], pose (x, y, yaw deg) -> static [ROWS, COLS] bool,
        walkers [WALKERS, 5]."""
        pose = np.asarray(pose, np.float32)
        # as bev_refiner.Episode.sample: before there are that many frames, the first one stands in
        past = [self.memory[-p] if len(self.memory) >= p else (self.memory[0] if self.memory else (evidence, pose))
                for p in R.PAST]

        def tensor(a):
            return torch.from_numpy(np.ascontiguousarray(a)).to(device=self.device, dtype=torch.float32)[None]
        batch = dict(evidence=tensor(evidence), velocity=tensor(np.asarray(velocity, np.float32)),
                     past=tensor(np.stack([e for e, _ in past])), pose=tensor(pose),
                     past_pose=tensor(np.stack([p for _, p in past])))
        if self.model.memory:
            back = [self.memory[-h] if len(self.memory) >= h else (self.memory[0] if self.memory else (evidence, pose))
                    for h in R.HISTORY]
            batch.update(history=tensor(np.stack([e for e, _ in back])), history_pose=tensor(np.stack([p for _, p in back])))
        static, dynamic, flow = refine(self.model, batch)
        self.memory = (self.memory+[(evidence, pose)])[-max(R.PAST+(R.HISTORY if self.model.memory else ())):]
        return static[0], self.people.step(blobs(dynamic[0], flow[0]), pose)


def episode(model, folder, device, batch_size=16):
    """Every evidence frame of the folder -> frames [N], static [N, ROWS, COLS] bool, walkers [N, WALKERS, 5]."""
    data = np.load(os.path.join(folder, 'bev_evidence.npz'))
    with open(os.path.join(folder, 'frames.json'), encoding='utf-8') as stream:
        recorded = json.load(stream)
    frames, evidence, velocity = data['frames'], data['evidence'], data['velocity']
    poses = np.array([(recorded[i]['x'], recorded[i]['y'], recorded[i]['yaw']) for i in frames], np.float32)

    statics, walkers, tracker = [], [], People()
    for start in range(0, len(frames), batch_size):
        ks = np.arange(start, min(start+batch_size, len(frames)))
        static, dynamic, flow = refine(model, frames_batch(model, evidence, velocity, poses, ks, device))
        statics.append(static)
        walkers += [tracker.step(blobs(dynamic[j], flow[j]), poses[k]) for j, k in enumerate(ks)]
    return frames, np.concatenate(statics), np.stack(walkers)


def compare(model, folders, device):
    """The people of the refined BEV against the simulator's, as the Flow gets them: per truth walker on the grid
    (standing ones too) the nearest perceived one within 1 m."""
    rows = dict(moving=[0, 0], standing=[0, 0])
    error, speed, extra, frames_n, inter, union = [], [], 0, 0, 0, 0
    x, y = T.cell_centres()
    area = (np.hypot(x, y) < 12.5) & ~((np.abs(x) <= 2.45) & (np.abs(y) <= 1.05))
    for folder in folders:
        scene = T.Scene(folder)
        frames, static, walkers = episode(model, folder, device)
        for k, index in enumerate(frames):
            pose = scene.pose(index)
            truth = scene.render(index)[0] > .5
            inter += int((static[k] & truth & area).sum())
            union += int(((static[k] | truth) & area).sum())
            c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
            seen = [w for w in walkers[k] if w[4] > .5]
            used = set()
            frames_n += 1
            for wx, wy, vx, vy in scene.walkers(index):
                ex, ey = c*(wx-pose[0])+s*(wy-pose[1]), -s*(wx-pose[0])+c*(wy-pose[1])
                if abs(ex) > T.X_HALF-.5 or abs(ey) > T.Y_HALF-.5:
                    continue
                kind = rows['moving' if math.hypot(vx, vy) >= .3 else 'standing']
                kind[0] += 1
                best, best_d = None, 1.
                for j, w in enumerate(seen):
                    d = math.hypot(w[0]-ex, w[1]-ey)
                    if d < best_d and j not in used:
                        best, best_d = j, d
                if best is None:
                    continue
                used.add(best)
                kind[1] += 1
                error.append(best_d)
                speed.append(math.hypot(seen[best][2]-(c*vx+s*vy), seen[best][3]-(-s*vx+c*vy)))
            extra += len(seen)-len(used)
    return dict(episodes=len(folders), frames=frames_n, static_iou=round(inter/max(1, union), 4),
                moving=dict(truth=rows['moving'][0], recall=round(rows['moving'][1]/max(1, rows['moving'][0]), 4)),
                standing=dict(truth=rows['standing'][0], recall=round(rows['standing'][1]/max(1, rows['standing'][0]), 4)),
                position_error=round(float(np.mean(error)), 3) if error else None,
                velocity_error_median=round(float(np.median(speed)), 3) if speed else None,
                velocity_error_mean=round(float(np.mean(speed)), 3) if speed else None,
                extra_people_per_frame=round(extra/max(1, frames_n), 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint')
    parser.add_argument('roots', nargs='+')
    parser.add_argument('--redo', action='store_true')
    parser.add_argument('--compare', action='store_true', help='print the comparison with the truth, per split; saves nothing')
    parser.add_argument('--limit', type=int, default=40, help='episodes of the training split in --compare')
    args = parser.parse_args()
    device = 'cuda'
    model = R.load(args.checkpoint, device)
    folders = R.folders_of(args.roots)
    if args.compare:
        train, validation = R.split(folders)
        step = max(1, len(train)//args.limit)
        print(json.dumps(dict(validation=compare(model, validation, device), train=compare(model, train[::step][:args.limit], device)),
                         indent=1))
        return
    began = time.perf_counter()
    done = 0
    for folder in folders:
        out = os.path.join(folder, 'bev_refined.npz')
        if os.path.exists(out) and not args.redo:
            continue
        frames, static, walkers = episode(model, folder, device)
        np.savez_compressed(out, frames=frames, static=np.packbits(static.reshape(len(static), -1), axis=1), walkers=walkers,
                            checkpoint=str(args.checkpoint))
        done += 1
        if done % 25 == 0:
            print(done, 'episodes', round(time.perf_counter()-began), 's', flush=True)
    print('refined', done, 'episodes', round(time.perf_counter()-began), 's', flush=True)


if __name__ == '__main__':
    main()
