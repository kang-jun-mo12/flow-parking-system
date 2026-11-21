"""The arbiter: which of the two drives - the parking model or the avoidance Flow - learned, in
place of the rule that reads the simulator's walkers (relevance.RelevanceGate on the plan).

  perceived BEV [12, 320, 200] (bev_flow.bev_truth of the refiner's static mask and people) ── conv, stride 2 ─┐
  the car's boxes of the next 3 s along the parking plan, on the 0.2 m grid (time-coded, and as one mask) ─────┴─ CNN
  speed, reverse, a plan at all ── MLP ─────────────────────────────────────────────────────────────────────── head
  head ── P(a walker in the plan's way within 3 s), P(a prop in it)

It answers one question that has no memory in it - is the plan's way blocked - with the rule's verdict of the same step
as the label (bev_arbiter_data.py). Who drives now is not an input: the mode is nearly always the one of 0.2 s ago and
a model that reads it copies it. The 0.6 s before the Flow lets go, the staging
and the quiet time after a hand-back stay rules of time (Supervisor here, supervisor.FlowDriver.update at run time).

usage (Python 3.11, from scripts/):
  python structure_flow/bev_arbiter.py train OUT ROWS.npz [--epochs 12]
  python structure_flow/bev_arbiter.py eval CHECKPOINT ROWS.npz
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
import torch.nn as nn
import torch.nn.functional as NF

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bev_truth as T                                   # noqa: E402
import bev_flow as F                                    # noqa: E402
import bev_arbiter_data as D                            # noqa: E402

COARSE = 2                          # the corridor is drawn every second cell (0.2 m)
EVERY = 2                           # ... from every second forecast sample (0.2 s: the boxes overlap)
HALF = (D.DF.SPECS['half_length']+D.DF.GATE_CONFIG['gate_margin_m'], D.DF.SPECS['half_width']+D.DF.GATE_CONFIG['gate_margin_m'])
ON, OFF, CLEAR_STEPS = .5, .3, 3    # the supervisor: in above ON, out after 0.6 s (3 steps) below OFF
_GRID = {}


def coarse_cells(device):
    if device not in _GRID:
        rows, cols = T.ROWS//COARSE, T.COLS//COARSE
        x = T.X_HALF-(torch.arange(rows, device=device)+.5)*T.CELL*COARSE
        y = -T.Y_HALF+(torch.arange(cols, device=device)+.5)*T.CELL*COARSE
        _GRID[device] = (x[:, None].expand(rows, cols), y[None, :].expand(rows, cols))
    return _GRID[device]


def corridor(forecast, planned):
    """forecast [B, S, 3] (x, y, yaw of the car's centre, actor frame of now) -> [B, 2, ROWS/2, COLS/2]: the cells the
    car's box (with the gate's margin) covers, 1 at once falling to 0 at 3 s, and the same as one mask; a third
    channel of ones where there is no plan (the gate then forecasts straight on at the speed of now)."""
    x, y = coarse_cells(forecast.device)
    f = forecast[:, ::EVERY]
    fx, fy, yaw = f[..., 0, None, None], f[..., 1, None, None], f[..., 2, None, None]
    c, s = torch.cos(yaw), torch.sin(yaw)
    dx, dy = x-fx, y-fy
    inside = ((c*dx+s*dy).abs() <= HALF[0]) & ((-s*dx+c*dy).abs() <= HALF[1])        # [B, S, R, C]
    weight = 1.-torch.arange(f.shape[1], device=f.device, dtype=torch.float32)/f.shape[1]
    timed = (inside*weight[None, :, None, None]).amax(1)
    blind = (~planned).float()[:, None, None].expand_as(timed)
    return torch.stack((timed, inside.any(1).float(), blind), 1)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
                                 nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU())

    def forward(self, x):
        return self.net(x)


class Arbiter(nn.Module):
    def __init__(self, width=(32, 48, 96, 128)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(F.BEV_CHANNELS, width[0], 3, 2, 1, bias=False), nn.GroupNorm(8, width[0]), nn.SiLU())
        self.body = nn.Sequential(Block(width[0]+3, width[1], 2), Block(width[1], width[2], 2), Block(width[2], width[3], 2),
                                  Block(width[3], width[3]))
        self.vector = nn.Sequential(nn.Linear(3, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(2*width[3]+32, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, bev, lane, vector):
        h = self.body(torch.cat((self.stem(bev), lane), 1))
        return self.head(torch.cat((h.amax((2, 3)), h.mean((2, 3)), self.vector(vector)), 1))


def inputs(pose, goal, static, walkers, forecast, planned, speed, reverse, mirror=None):
    """The model's three inputs from a batch on the device. mirror [B] bool: those rows left-right mirrored."""
    bev = F.bev_truth(pose, goal, static, walkers)
    lane = corridor(forecast, planned)
    if mirror is not None and bool(mirror.any()):
        flipped = bev.flip(-1)
        flipped[:, 4] = -flipped[:, 4]              # vy
        flipped[:, 11] = -flipped[:, 11]            # the y coordinate channel
        bev = torch.where(mirror[:, None, None, None], flipped, bev)
        lane = torch.where(mirror[:, None, None, None], lane.flip(-1), lane)
    vector = torch.stack((speed/3., reverse.float(), planned.float()), 1)
    return bev, lane, vector


class Rows:
    def __init__(self, path, device):
        data = np.load(path, allow_pickle=True)
        self.episodes = json.loads(str(data['episodes']))
        self.device = device
        self.a = {k: data[k] for k in data.files if k != 'episodes'}
        slots = np.array([e['slot'] for e in self.episodes])[self.a['episode']]
        self.validation = np.isin(slots, D.VALIDATION_SLOTS)
        self.label = np.stack((self.a['walker'], self.a['prop']), 1).astype(np.float32)

    def batch(self, index):
        d, a = self.device, self.a
        t = lambda v, dtype=torch.float32: torch.as_tensor(v, dtype=dtype, device=d)       # noqa: E731
        static = np.unpackbits(a['static'][index], axis=1)[:, :T.ROWS*T.COLS].reshape(-1, T.ROWS, T.COLS).astype(bool)
        return dict(pose=t(a['pose'][index]), goal=t(a['goal'][index]), static=t(static, torch.bool),
                    walkers=t(a['walkers'][index]), forecast=t(a['forecast'][index]), planned=t(a['planned'][index], torch.bool),
                    speed=t(a['speed'][index]), reverse=t(a['reverse'][index], torch.bool), label=t(self.label[index]))


@torch.no_grad()
def predict(model, rows, index, batch=192):
    model.eval()
    out = []
    for start in range(0, len(index), batch):
        b = rows.batch(index[start:start+batch])
        with torch.autocast('cuda', dtype=torch.float16):
            logits = model(*inputs(b['pose'], b['goal'], b['static'], b['walkers'], b['forecast'], b['planned'], b['speed'],
                                   b['reverse']))
        out.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(out)


def auc(score, label):
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score)+1)
    pos = label > .5
    return float((rank[pos].sum()-pos.sum()*(pos.sum()+1)/2)/max(1, pos.sum()*(~pos).sum()))


def supervise(p, on=ON, off=OFF, clear=CLEAR_STEPS):
    """The supervisor's timing over the steps of one episode: active from the first step above `on`, until `clear`
    steps in a row below `off` (the rule: active with a conflict, released after 0.6 s without one)."""
    active, quiet, out = False, 0, np.zeros(len(p), bool)
    for i, value in enumerate(p):
        if value >= (off if active else on):
            active, quiet = True, 0
        elif active:
            quiet += 1
            if quiet >= clear:
                active = False
        out[i] = active
    return out


def onsets(mine, theirs):
    """The rule's take-overs (theirs: False -> True) and mine around them, in steps of 0.2 s: how many I have within
    2 steps, the mean lag of those (negative: earlier), and my take-overs with none of the rule's within 5 steps."""
    rule = np.nonzero(theirs[1:] & ~theirs[:-1])[0]+1
    own = np.nonzero(mine[1:] & ~mine[:-1])[0]+1
    found, lags = 0, []
    for k in rule:
        near = own[np.abs(own-k) <= 2]
        if len(near) or mine[max(0, k-2):k+3].all():
            found += 1
            lags.append(int(near[np.argmin(np.abs(near-k))]-k) if len(near) else 0)
    false = sum(1 for k in own if not len(rule) or np.abs(rule-k).min() > 5)
    return len(rule), found, lags, false


def evaluate(model, rows, index, baseline=True):
    """Per step: accuracy, precision, recall, AUC of the two outputs. Per episode, with the supervisor's timing: the
    agreement with the rule's `active`, its take-overs found, my false ones per minute. baseline: the rule itself on
    the perceived people (walkers only - the perceived BEV has no prop objects)."""
    p = predict(model, rows, index)
    a = rows.a
    out = {}
    for j, name in enumerate(('walker', 'prop')):
        y = rows.label[index, j] > .5
        hit = p[:, j] >= .5
        out[name] = dict(positives=int(y.sum()), accuracy=round(float((hit == y).mean()), 4),
                         precision=round(float((hit & y).sum()/max(1, hit.sum())), 4),
                         recall=round(float((hit & y).sum()/max(1, y.sum())), 4), auc=round(auc(p[:, j], y), 4))
    stats = dict(steps=0, same=0, rule=0, found=0, false=0, lags=[], minutes=0.)
    base = dict(same=0, found=0, false=0, lags=[], steps=0, hit=0, y=0, said=0)
    for e in np.unique(a['episode'][index]):
        part = np.nonzero(a['episode'][index] == e)[0]
        part = part[np.argsort(a['t'][index][part])]
        mine = supervise(p[part].max(1))
        theirs = a['active'][index][part]
        n, found, lags, false = onsets(mine, theirs)
        stats['steps'] += len(part)
        stats['same'] += int((mine == theirs).sum())
        stats['rule'] += n
        stats['found'] += found
        stats['false'] += false
        stats['lags'] += lags
        stats['minutes'] += len(part)*.2/60.
        if baseline:
            gate = D.DF.relevance.RelevanceGate(dict(D.DF.GATE_CONFIG), D.DF.SPECS)
            said = []
            for i in part:
                k = index[i]
                tracks = [dict(object_id='walker-{}'.format(w), class_name='pedestrian', x=float(v[0]), y=float(v[1]),
                               vx=float(v[2]), vy=float(v[3]), yaw=math.atan2(v[3], v[2]) if math.hypot(v[2], v[3]) > .1 else 0.,
                               length=.6, width=.6, age_s=0., hits=10, ever_moving=True)
                          for w, v in enumerate(a['walkers'][k]) if v[4] > .5]
                # the forecast kept in the row is the gate's own: the same boxes against the perceived people
                margin = D.DF.GATE_CONFIG['gate_margin_m']
                conflict = False
                for track in tracks:
                    for s, (fx, fy, fyaw) in enumerate(a['forecast'][k]):
                        ts = s*D.DF.GATE_CONFIG['gate_sample_dt_s']
                        ego = (fx, fy, fyaw, D.DF.SPECS['half_length']+margin, D.DF.SPECS['half_width']+margin)
                        other = (track['x']+track['vx']*ts, track['y']+track['vy']*ts, track['yaw'], .3, .3)
                        if D.DF.relevance.box_separation(ego, other) <= 0:
                            conflict = True
                            break
                    if conflict:
                        break
                said.append(conflict)
            said = np.array(said)
            y = rows.label[index[part], 0] > .5
            base['hit'] += int((said & y).sum())
            base['y'] += int(y.sum())
            base['said'] += int(said.sum())
            base['steps'] += len(part)
            base['same'] += int((said == y).sum())
    out['supervised'] = dict(agreement_with_the_rule=round(stats['same']/max(1, stats['steps']), 4), rule_takeovers=stats['rule'],
                             found_within_0_4_s=stats['found'], mean_lag_s=round(.2*float(np.mean(stats['lags'])), 3) if stats['lags'] else None,
                             late_by_0_4_s=int(sum(1 for v in stats['lags'] if v >= 2)),
                             false_takeovers=stats['false'], false_per_minute=round(stats['false']/max(1e-6, stats['minutes']), 3))
    if baseline:
        out['rule_on_perceived_people'] = dict(walker_accuracy=round(base['same']/max(1, base['steps']), 4),
                                               precision=round(base['hit']/max(1, base['said']), 4),
                                               recall=round(base['hit']/max(1, base['y']), 4))
    return out


def train(args):
    device = 'cuda'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rows = Rows(args.rows, device)
    train_index = np.nonzero(~rows.validation)[0]
    val_index = np.nonzero(rows.validation)[0]
    print('rows: train', len(train_index), 'validation', len(val_index), 'episodes', len(rows.episodes), flush=True)
    model = Arbiter().to(device)
    print('parameters', sum(p.numel() for p in model.parameters()), flush=True)
    positive = rows.label[train_index].mean(0)
    # a miss costs more than a needless take-over: the positives weigh MISS times their balance
    weight = torch.tensor(args.miss*(1.-positive)/np.maximum(positive, 1e-3), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs*(len(train_index)//args.batch)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=steps, pct_start=.1)
    scaler = torch.amp.GradScaler('cuda')
    os.makedirs(args.out, exist_ok=True)
    history, best = [], None
    for epoch in range(1, args.epochs+1):
        model.train()
        began, total, count = time.time(), 0., 0
        order = rng.permutation(train_index)
        for start in range(0, len(order)-args.batch+1, args.batch):
            b = rows.batch(np.sort(order[start:start+args.batch]))
            mirror = torch.rand(len(b['pose']), device=device) < .5
            with torch.no_grad():
                bev, lane, vector = inputs(b['pose'], b['goal'], b['static'], b['walkers'], b['forecast'], b['planned'],
                                           b['speed'], b['reverse'], mirror)
            with torch.autocast('cuda', dtype=torch.float16):
                logits = model(bev, lane, vector)
            loss = NF.binary_cross_entropy_with_logits(logits.float(), b['label'], pos_weight=weight)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            total += float(loss.detach())
            count += 1
        row = dict(epoch=epoch, seconds=round(time.time()-began), loss=round(total/max(1, count), 4))
        if epoch % args.every == 0 or epoch == args.epochs:
            row['validation'] = evaluate(model, rows, val_index, baseline=False)
            v = row['validation']
            score = v['walker']['auc']+v['prop']['auc']+v['supervised']['agreement_with_the_rule']
            state = dict(model=model.state_dict(), epoch=epoch, metrics=v, validation_slots=D.VALIDATION_SLOTS, rows=str(args.rows))
            torch.save(state, os.path.join(args.out, 'last.pt'))
            if best is None or score > best:
                best = score
                torch.save(state, os.path.join(args.out, 'best.pt'))
        history.append(row)
        print(json.dumps(row), flush=True)
        with open(os.path.join(args.out, 'history.json'), 'w', encoding='utf-8') as stream:
            json.dump(history, stream)


def load(checkpoint, device='cuda'):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = Arbiter().to(device)
    model.load_state_dict(saved['model'])
    return model.eval(), saved


class Runtime:
    """One decision at a time (system_server.py --arbiter): the perceived static mask and people of this step,
    the gate's forecast of the car along the plan (supervisor.FlowDriver sends it) -> (P walker, P prop)."""

    def __init__(self, checkpoint, device='cuda'):
        self.model, _ = load(str(checkpoint), device)
        self.device = device

    @torch.no_grad()
    def __call__(self, pose, goal, static, walkers, forecast, planned, speed, reverse):
        t = lambda v, dtype=torch.float32: torch.as_tensor(np.asarray(v), dtype=dtype, device=self.device)[None]    # noqa: E731
        with torch.autocast('cuda', dtype=torch.float16):
            logits = self.model(*inputs(t(pose), t(goal), t(static, torch.bool), t(walkers), t(forecast), t(planned, torch.bool),
                                        t(speed), t(reverse, torch.bool)))
        p = torch.sigmoid(logits.float())[0].cpu().numpy()
        return float(p[0]), float(p[1])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('out')
    t.add_argument('rows')
    t.add_argument('--epochs', type=int, default=12)
    t.add_argument('--batch', type=int, default=96)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--miss', type=float, default=2., help='weight of a missed conflict against a needless take-over')
    t.add_argument('--every', type=int, default=2)
    t.add_argument('--seed', type=int, default=0)
    e = sub.add_parser('eval')
    e.add_argument('checkpoint')
    e.add_argument('rows')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    else:
        rows = Rows(args.rows, 'cuda')
        model, saved = load(args.checkpoint)
        print(json.dumps(dict(epoch=saved['epoch'], validation=evaluate(model, rows, np.nonzero(rows.validation)[0])), indent=1))


if __name__ == '__main__':
    main()
