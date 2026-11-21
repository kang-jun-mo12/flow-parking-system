"""BEV refiner: the evidence BEV of the frozen perception (bev_evidence.py: DDP front end, YOLO-World, metric depth)
-> static occupancy, dynamic occupancy and the velocity of the dynamic cells on the 32 x 20 m grid of bev_truth.py.
Trained on the truth BEV of the collected episodes. Its output is the BEV the avoidance Flow reads.

Input [IN, ROWS, COLS]: the evidence of now (8 channels, velocity 2), the evidence 0.4 s and 0.8 s ago moved into the
ego frame of now (motion of what moves, and the velocity it implies), the lot border (a map prior: no camera sees
the kerb as an obstacle) and the ego footprint. Output [4, ROWS, COLS]: static logit, dynamic logit, vx, vy (m/s, ego
axes, ground speed).

--memory: the evidence of HISTORY frames, back to 2 s,
moved into the ego frame of now and pooled, is read at the bottleneck of the U-Net - `conv` joins it there with
convolutions, `attention` with a transformer decoder (every cell of now looks at every cell of every remembered
frame). The same frames for both, so the two differ in how the memory is read, and both from the plain U-Net in how
far back they see (the strip beside the car no camera looks at hides a walker for longer than 0.8 s).

usage (Python 3.11, from scripts/):
  python structure_flow/bev_refiner.py train OUT ROOT [ROOT ...] [--epochs 12] [--memory conv|attention]
  python structure_flow/bev_refiner.py eval CHECKPOINT ROOT [--png DIR]
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bev_truth as T                                   # noqa: E402
import bev_evidence as E                                # noqa: E402

PAST = (2, 4)               # evidence frames back (0.4 s, 0.8 s at the 0.2 s evidence step)
EVIDENCE = len(E.CHANNELS)
IN = (EVIDENCE+2)+len(PAST)*EVIDENCE+2
COUNT_SCALE = 16.           # depth points per cell that count as full evidence
SPEED_SCALE = 2.            # m/s
VALIDATION_SLOTS = (19, 26, 34, 45)     # bev_flow.VALIDATION_SLOTS: episodes of these goal slots are held out
GROUP = 10                  # episodes in memory at a time
HISTORY = (1, 2, 3, 5, 7, 10)       # evidence frames back (0.2 .. 2 s) of a model with a memory
HISTORY_POOL = 4            # they are max-pooled to 0.4 m cells


# ---------------------------------------------------------------- data
class Episode:
    """One folder: evidence, velocity, the ego poses and the truth of its evidence frames."""

    def __init__(self, folder):
        data = np.load(os.path.join(folder, 'bev_evidence.npz'))
        scene = T.Scene(folder)
        self.folder, self.frames = folder, data['frames']
        self.evidence, self.velocity = data['evidence'], data['velocity']
        self.poses = np.array([scene.pose(i) for i in self.frames], np.float32)
        truth = np.stack([scene.render(i) for i in self.frames])
        self.static, self.border, self.dynamic = (truth[:, i] > .5 for i in (0, 1, 2))
        self.flow = truth[:, 3:5].astype(np.float16)
        self.walkers = [scene.walkers(i) for i in self.frames]

    def __len__(self):
        return len(self.frames)

    def sample(self, k, history=False):
        past = [max(0, k-p) for p in PAST]
        out = dict(evidence=self.evidence[k], velocity=self.velocity[k], past=self.evidence[past],
                   pose=self.poses[k], past_pose=self.poses[past], border=self.border[k],
                   static=self.static[k], dynamic=self.dynamic[k], flow=self.flow[k])
        if history:
            back = [max(0, k-h) for h in HISTORY]
            out.update(history=self.evidence[back], history_pose=self.poses[back])
        return out


def folders_of(roots):
    out = []
    for root in roots:
        out += sorted(str(p) for p in Path(root).iterdir() if (p/'bev_evidence.npz').exists() and (p/'meta.json').exists())
    return out


def split(folders):
    """The goal slots the avoidance Flow holds out are held out here too: its closed-loop test on those slots must
    read a BEV the refiner was not trained on (ep00045_s19_0: slot 19)."""
    def slot(folder):
        return int(os.path.basename(folder).split('_')[1][1:])
    return ([f for f in folders if slot(f) not in VALIDATION_SLOTS], [f for f in folders if slot(f) in VALIDATION_SLOTS])


def collate(samples, device):
    def stack(key, dtype):
        return torch.from_numpy(np.stack([s[key] for s in samples])).to(device=device, dtype=dtype)
    out = dict(evidence=stack('evidence', torch.float32), velocity=stack('velocity', torch.float32),
               past=stack('past', torch.float32), pose=stack('pose', torch.float32),
               past_pose=stack('past_pose', torch.float32), border=stack('border', torch.float32),
               static=stack('static', torch.float32), dynamic=stack('dynamic', torch.float32), flow=stack('flow', torch.float32))
    if 'history' in samples[0]:     # uint8 to the device, floats there (37 MB a batch)
        out['history'] = torch.from_numpy(np.stack([s['history'] for s in samples])).to(device).float()
        out['history_pose'] = stack('history_pose', torch.float32)
    return out


def groups(folders, rng):
    """Episode groups of an epoch, loaded one group ahead in a thread."""
    order = list(folders)
    rng.shuffle(order)
    chunks = [order[i:i+GROUP] for i in range(0, len(order), GROUP)]
    box = {}

    def load(index):
        box[index] = [Episode(f) for f in chunks[index]]

    thread = threading.Thread(target=load, args=(0,))
    thread.start()
    for index in range(len(chunks)):
        thread.join()
        episodes = box.pop(index)
        if index+1 < len(chunks):
            thread = threading.Thread(target=load, args=(index+1,))
            thread.start()
        yield episodes


# ---------------------------------------------------------------- input
_GRID = {}


def base_grid(device):
    if device not in _GRID:
        x, y = T.cell_centres()
        _GRID[device] = (torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device))
    return _GRID[device]


def warp(past, pose, past_pose):
    """past [B, C, ROWS, COLS] in the ego frame of past_pose [B, 3] (x, y, yaw deg) -> the ego frame of pose."""
    x, y = base_grid(past.device)
    yaw, old = torch.deg2rad(pose[:, 2]), torch.deg2rad(past_pose[:, 2])
    c, s, co, so = torch.cos(yaw), torch.sin(yaw), torch.cos(old), torch.sin(old)
    wx = pose[:, 0, None, None]+c[:, None, None]*x-s[:, None, None]*y
    wy = pose[:, 1, None, None]+s[:, None, None]*x+c[:, None, None]*y
    dx, dy = wx-past_pose[:, 0, None, None], wy-past_pose[:, 1, None, None]
    px = co[:, None, None]*dx+so[:, None, None]*dy
    py = -so[:, None, None]*dx+co[:, None, None]*dy
    grid = torch.stack((py/T.Y_HALF, -px/T.X_HALF), -1)             # (u over the columns, v over the rows)
    return F.grid_sample(past, grid, mode='bilinear', padding_mode='zeros', align_corners=False)


def scale_evidence(evidence):
    out = evidence/255.
    out[:, :3] = torch.clamp(evidence[:, :3]/COUNT_SCALE, max=1.)
    return out


_EGO = {}


def ego_mask(device):
    if device not in _EGO:
        x, y = base_grid(device)
        _EGO[device] = ((x.abs() <= 2.4) & (y.abs() <= 1.08)).float()
    return _EGO[device]


def make_input(batch):
    """[B, IN, ROWS, COLS] from a collated batch (evidence uint8 values as float)."""
    now = scale_evidence(batch['evidence'])
    velocity = torch.clamp(batch['velocity']/SPEED_SCALE, -1.5, 1.5)
    parts = [now, velocity]
    for k in range(len(PAST)):
        parts.append(warp(scale_evidence(batch['past'][:, k]), batch['pose'], batch['past_pose'][:, k]))
    ego = ego_mask(now.device)[None, None].expand(len(now), 1, -1, -1)
    return torch.cat(parts+[batch['border'][:, None], ego], 1)


def make_history(batch):
    """[B, len(HISTORY), EVIDENCE, ROWS / HISTORY_POOL, COLS / HISTORY_POOL]: the evidence of the remembered frames in
    the ego frame of now, max-pooled."""
    frames = []
    for k in range(len(HISTORY)):
        moved = warp(scale_evidence(batch['history'][:, k]), batch['pose'], batch['history_pose'][:, k])
        frames.append(F.max_pool2d(moved, HISTORY_POOL))
    return torch.stack(frames, 1)


def flip(batch_input, targets):
    """Mirror left-right (the columns): y and vy change sign."""
    x = batch_input.flip(-1).clone()
    x[:, EVIDENCE+1] = -x[:, EVIDENCE+1]
    out = {k: v.flip(-1) for k, v in targets.items()}
    out['flow'] = out['flow'].clone()
    out['flow'][:, 1] = -out['flow'][:, 1]
    return x, out


# ---------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
                                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())

    def forward(self, x):
        return self.body(x)


class Refiner(nn.Module):
    def __init__(self, width=(32, 64, 128, 192), memory=None):
        super().__init__()
        self.down = nn.ModuleList([Block(IN if i == 0 else width[i-1], w) for i, w in enumerate(width)])
        self.up = nn.ModuleList([Block(width[i]+width[i-1], width[i-1]) for i in range(len(width)-1, 0, -1)])
        self.head = nn.Conv2d(width[0], 4, 1)
        self.memory = memory
        top = width[-1]
        if memory:
            # a remembered frame (0.4 m cells) -> the grid of the bottleneck (0.8 m cells), marked with how old it is
            self.past = nn.Sequential(Block(EVIDENCE, 48), nn.MaxPool2d(2), Block(48, 96))
            self.when = nn.Parameter(torch.zeros(len(HISTORY), 96, 1, 1))
        if memory == 'conv':
            self.squeeze = nn.Conv2d(96, 32, 1)
            self.fuse = nn.Sequential(Block(top+32*len(HISTORY), top), Block(top, top))
        elif memory == 'attention':
            self.lift = nn.Conv2d(96, top, 1)
            self.rows = nn.Parameter(torch.zeros(1, top, T.ROWS//8, 1))
            self.cols = nn.Parameter(torch.zeros(1, top, 1, T.COLS//8))
            layer = nn.TransformerDecoderLayer(top, 4, 2*top, dropout=0., batch_first=True, norm_first=True)
            self.reader = nn.TransformerDecoder(layer, 2)

    def remember(self, x, history):
        """The bottleneck x [B, C, h, w] after reading the remembered frames (make_history)."""
        b, k = history.shape[:2]
        past = self.past(history.flatten(0, 1))
        past = past.reshape(b, k, *past.shape[1:])+self.when[None]
        if self.memory == 'conv':
            return self.fuse(torch.cat((x, self.squeeze(past.flatten(0, 1)).reshape(b, -1, *x.shape[-2:])), 1))
        place = self.rows+self.cols
        seen = (self.lift(past.flatten(0, 1))+place).reshape(b, k, x.shape[1], -1)             # [B, K, C, h w]
        seen = seen.permute(0, 1, 3, 2).reshape(b, -1, x.shape[1])                             # [B, K h w, C]
        now = (x+place).flatten(2).transpose(1, 2)
        return self.reader(now, seen).transpose(1, 2).reshape(x.shape)

    def forward(self, x, history=None):
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        if self.memory:
            x = self.remember(x, history)
        for block, skip in zip(self.up, reversed(skips[:-1])):
            x = block(torch.cat((F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip), 1))
        return self.head(x)


def range_weight(device):
    """1 within 12.5 m of the car, falling to 0.3 at the far ends of the grid (what monocular depth still sees)."""
    x, y = base_grid(device)
    return torch.clamp(1.-.7*(torch.hypot(x, y)-12.5)/4., .3, 1.)


def loss_of(output, targets):
    weight = range_weight(output.device)
    static = F.binary_cross_entropy_with_logits(output[:, 0], targets['static'], reduction='none')
    dynamic = F.binary_cross_entropy_with_logits(output[:, 1], targets['dynamic'], reduction='none',
                                                 pos_weight=torch.tensor(40., device=output.device))
    # velocity where something moves, a little around it too (the mask of the truth disc)
    near = F.max_pool2d(targets['dynamic'][:, None], 5, 1, 2)[:, 0]
    flow = (F.smooth_l1_loss(output[:, 2:], targets['flow']/SPEED_SCALE, reduction='none').sum(1)*near).sum()/near.sum().clamp(min=1.)
    return (static*weight).mean()+(dynamic*weight).mean()+flow, dict(static=float((static*weight).mean()),
                                                                      dynamic=float((dynamic*weight).mean()), flow=float(flow))


# ---------------------------------------------------------------- evaluation
def blobs(mask):
    import cv2
    count, _, stats, centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return [(centres[i][1], centres[i][0], stats[i, cv2.CC_STAT_AREA]) for i in range(1, count)]       # row, col, cells


def cell_to_ego(row, col):
    return T.X_HALF-(row+.5)*T.CELL, -T.Y_HALF+(col+.5)*T.CELL


class Meter:
    """Static IoU within 12.5 m and the moving people as objects."""

    def __init__(self):
        x, y = T.cell_centres()
        self.area = (np.hypot(x, y) < 12.5) & ~((np.abs(x) <= 2.45) & (np.abs(y) <= 1.05))
        self.inter, self.union, self.false = {}, {}, {}
        self.people = {}

    def static(self, name, pred, truth):
        p, t = pred & self.area, truth & self.area
        self.inter[name] = self.inter.get(name, 0)+int((p & t).sum())
        self.union[name] = self.union.get(name, 0)+int((p | t).sum())
        self.false[name] = self.false.get(name, [])+[int((p & ~t).sum())]

    def moving(self, name, pred, flow, walkers, pose):
        """pred: dynamic mask, flow [2, R, C] m/s, walkers [(x, y, vx, vy)] world."""
        row = self.people.setdefault(name, dict(truth=0, found=0, error=[], speed=[], false=0, frames=0, by_range={}))
        row['frames'] += 1
        found = blobs(pred)
        c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
        used = set()
        for x, y, vx, vy in walkers:
            ex, ey = c*(x-pose[0])+s*(y-pose[1]), -s*(x-pose[0])+c*(y-pose[1])
            if abs(ex) > T.X_HALF-.5 or abs(ey) > T.Y_HALF-.5 or math.hypot(vx, vy) < .3:
                continue
            row['truth'] += 1
            band = min(3, int(math.hypot(ex, ey)//4))
            stats = row['by_range'].setdefault(band, [0, 0])
            stats[0] += 1
            best, best_d = None, .8
            for k, (r, cc, cells) in enumerate(found):
                bx, by = cell_to_ego(r, cc)
                d = math.hypot(bx-ex, by-ey)
                if d < best_d and k not in used:
                    best, best_d = k, d
            if best is None:
                continue
            used.add(best)
            row['found'] += 1
            stats[1] += 1
            row['error'].append(best_d)
            r, cc = int(round(found[best][0])), int(round(found[best][1]))
            window = (slice(max(0, r-2), r+3), slice(max(0, cc-2), cc+3))
            m = pred[window]
            if m.any():
                v = np.array([flow[0][window][m].mean(), flow[1][window][m].mean()])
                row['speed'].append(float(np.hypot(v[0]-(c*vx+s*vy), v[1]-(-s*vx+c*vy))))
        row['false'] += sum(1 for k, b in enumerate(found) if k not in used and b[2] >= 6)

    def summary(self):
        out = {}
        for name in self.inter:
            out[name] = dict(static_iou=round(self.inter[name]/max(1, self.union[name]), 4),
                             false_cells=round(float(np.mean(self.false[name])), 1))
        for name, row in self.people.items():
            out.setdefault(name, {}).update(
                walker_recall=round(row['found']/max(1, row['truth']), 4), walkers=row['truth'],
                position_error=round(float(np.mean(row['error'])), 3) if row['error'] else None,
                velocity_error=round(float(np.median(row['speed'])), 3) if row['speed'] else None,
                false_per_frame=round(row['false']/max(1, row['frames']), 4),
                recall_by_range={'{}-{} m'.format(4*b, 4*b+4): round(f/max(1, t), 3) for b, (t, f) in sorted(row['by_range'].items())})
        return out


def baselines(batch, k):
    """Rule-based BEVs from the evidence of sample k, to compare the refiner with."""
    e = batch['evidence'][k].cpu().numpy()
    v = batch['velocity'][k].cpu().numpy()
    ddp = e[6] > 127
    objects = (e[3] > 0) | (e[5] > 0)
    unknown = (e[1]+e[2]) >= 3
    person = e[4] > 0
    moving = person & (np.hypot(v[0], v[1]) > .3)
    return dict(ddp=(ddp, np.zeros_like(ddp)), open=(objects | unknown, moving), fused=(ddp | (e[5] > 0), moving), flow=v)


def evaluate(model, folders, device, png=None, limit=None):
    meter = Meter()
    model.eval()
    shown = 0
    for folder in folders[:limit]:
        episode = Episode(folder)
        for start in range(0, len(episode), 16):
            ks = list(range(start, min(start+16, len(episode))))
            batch = collate([episode.sample(k, bool(model.memory)) for k in ks], device)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=device != 'cpu'):
                out = model(make_input(batch), make_history(batch) if model.memory else None).float()
            static = (out[:, 0] > 0).cpu().numpy()
            dynamic = (out[:, 1] > 0).cpu().numpy()
            flow = (out[:, 2:]*SPEED_SCALE).cpu().numpy()
            for j, k in enumerate(ks):
                truth_static = episode.static[k]
                base = baselines(batch, j)
                meter.static('refiner', static[j], truth_static)
                meter.moving('refiner', dynamic[j], flow[j], episode.walkers[k], episode.poses[k])
                for name in ('ddp', 'open', 'fused'):
                    meter.static(name, base[name][0], truth_static)
                    meter.moving(name, base[name][1], base['flow'], episode.walkers[k], episode.poses[k])
                if png and episode.walkers[k] and shown < 24 and k % 15 == 7:
                    save_sheet(os.path.join(png, '{}_{:03d}.png'.format(os.path.basename(folder), k)), episode, k,
                               static[j], dynamic[j], base)
                    shown += 1
    return meter.summary()


def save_sheet(path, episode, k, static, dynamic, base):
    from PIL import Image

    def panel(occ, moving):
        image = np.full((T.ROWS, T.COLS, 3), 255, np.uint8)
        image[episode.border[k]] = (225, 225, 225)
        image[occ] = (150, 150, 150)
        image[moving] = (220, 40, 40)
        x, y = T.cell_centres()
        image[(np.abs(x) <= 2.4) & (np.abs(y) <= 1.08)] = (60, 110, 230)
        return image
    gap = np.zeros((T.ROWS, 3, 3), np.uint8)
    sheet = np.concatenate([panel(episode.static[k], episode.dynamic[k]), gap, panel(static, dynamic), gap,
                            panel(*base['fused']), gap, panel(*base['open'])], 1)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(np.repeat(np.repeat(sheet, 2, 0), 2, 1)).save(path)


# ---------------------------------------------------------------- training
def train(args):
    device = 'cuda'
    torch.manual_seed(0)
    rng = random.Random(0)
    train_folders, val_folders = split(folders_of(args.roots))
    print('episodes: train', len(train_folders), 'validation', len(val_folders), flush=True)
    model = Refiner(memory=args.memory).to(device).to(memory_format=torch.channels_last)
    print('parameters', sum(p.numel() for p in model.parameters()), 'memory', args.memory, flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs*sum(1 for _ in train_folders)*150//args.batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=max(steps, 10), pct_start=.05)
    scaler = torch.amp.GradScaler('cuda')
    os.makedirs(args.out, exist_ok=True)
    history, best, done = [], None, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        logs, began = [], time.perf_counter()
        for episodes in groups(train_folders, rng):
            index = [(e, k) for e in range(len(episodes)) for k in range(len(episodes[e]))]
            rng.shuffle(index)
            for start in range(0, len(index)-args.batch+1, args.batch):
                batch = collate([episodes[e].sample(k, bool(args.memory)) for e, k in index[start:start+args.batch]], device)
                x = make_input(batch)
                remembered = make_history(batch) if args.memory else None
                targets = dict(static=batch['static'], dynamic=batch['dynamic'], flow=batch['flow'])
                if rng.random() < .5:
                    x, targets = flip(x, targets)
                    remembered = remembered.flip(-1) if remembered is not None else None
                if rng.random() < .3:           # a detector that misses: one of the object channels dropped
                    x[:, 3+rng.randrange(3)] = 0.
                with torch.autocast('cuda', dtype=torch.float16):
                    out = model(x.contiguous(memory_format=torch.channels_last), remembered)
                loss, parts = loss_of(out.float(), targets)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                scaler.step(optimizer)
                scaler.update()
                if done < steps-1:
                    scheduler.step()
                done += 1
                logs.append((float(loss), parts['static'], parts['dynamic'], parts['flow']))
        metrics = evaluate(model, val_folders, device, limit=args.val_limit)
        row = dict(epoch=epoch, loss=round(float(np.mean([l[0] for l in logs])), 4), static=round(float(np.mean([l[1] for l in logs])), 4),
                   dynamic=round(float(np.mean([l[2] for l in logs])), 4), flow=round(float(np.mean([l[3] for l in logs])), 4),
                   seconds=round(time.perf_counter()-began), validation=metrics)
        history.append(row)
        print(json.dumps(row), flush=True)
        r = metrics['refiner']
        score = r['static_iou']+r.get('walker_recall', 0.)-.2*r.get('false_per_frame', 0.)
        state = dict(model=model.state_dict(), epoch=epoch, metrics=metrics, channels=E.CHANNELS, past=PAST,
                     grid=(T.ROWS, T.COLS, T.CELL), memory=args.memory, history=HISTORY)
        torch.save(state, os.path.join(args.out, 'last.pt'))
        if best is None or score > best:
            best = score
            torch.save(state, os.path.join(args.out, 'best.pt'))
        with open(os.path.join(args.out, 'history.json'), 'w') as stream:
            json.dump(history, stream, indent=1)


def load(checkpoint, device='cuda'):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Refiner(memory=saved.get('memory')).to(device)
    model.load_state_dict(saved['model'])
    return model.eval()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('out')
    t.add_argument('roots', nargs='+')
    t.add_argument('--epochs', type=int, default=12)
    t.add_argument('--batch', type=int, default=12)
    t.add_argument('--lr', type=float, default=2e-3)
    t.add_argument('--val-limit', type=int, default=None)
    t.add_argument('--memory', choices=('conv', 'attention'), default=None,
                   help='read the evidence of HISTORY frames (2 s) at the bottleneck: with convolutions or a transformer')
    e = sub.add_parser('eval')
    e.add_argument('checkpoint')
    e.add_argument('roots', nargs='+')
    e.add_argument('--png')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    else:
        _, val_folders = split(folders_of(args.roots))
        print(json.dumps(evaluate(load(args.checkpoint), val_folders, 'cuda', png=args.png), indent=1))


if __name__ == '__main__':
    main()
