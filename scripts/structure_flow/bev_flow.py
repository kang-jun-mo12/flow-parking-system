"""BEV avoidance Flow: the obstacle-avoidance policy reads one BEV - static and moving obstacles
and the target slot on the 32 x 20 m grid of bev_truth.py - and the car's own state, and gives the next 0.8 s of
steer / speed / gear. Trained from scratch on the rule teacher's decisions (bev_flow_data.py).

  BEV [BEV_CHANNELS, 320, 200] ── CNN (stride 8) ── 40 x 25 tokens ─┐
  vector [VEC] ── MLP ── token ─ 8 queries read the BEV tokens ──────┴─ context
  context ── intent (8 manoeuvres: stage, wait, forward / back detour left / right, back away, forward away)
  context + intent ── rectified Flow over the chunk [4, 2] (steer, speed), gears [4, 3]
                   ── aids of training only: the car's poses of the next 3 s, the cells it will cover

The intent comes first: with it given the teacher's chunk is nearly one answer, so the Flow does not blend a left
and a right detour, and the candidates at run time are samples of the two likeliest intents.
BEV input: `truth` (drawn from the simulator's objects, here on the GPU) or `none` (the vector and the walkers
only: the comparison without the scene). The refined perception BEV takes the same channels.

usage (Python 3.11, from scripts/):
  python structure_flow/bev_flow.py train OUT ROWS.npz [--bev truth|none] [--epochs 40]
  python structure_flow/bev_flow.py eval CHECKPOINT ROWS.npz
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bev_truth as T                                   # noqa: E402
import bev_flow_data as D                               # noqa: E402

BEV_CHANNELS = 12           # static, border, dynamic, vx, vy, dynamic +1 s, +2 s, target, distance, ego, x, y
VALIDATION_SLOTS = (19, 26, 34, 45)        # episodes of these goal slots are held out
SPEED_SCALE = 2.            # m/s of the velocity channels
DISTANCE_CLIP = 3.          # m of the distance channel
INTENT_COUNT = len(D.INTENTS)
STEPS, ACTION = 4, 2
LOT = T.LOT_BOUND
EGO_HALF = (2.4, 1.08)
# inputs of the vector (bev_flow_data.vector) a model can be made blind to: the applied steer and the car's last
# poses carry the teacher's own last answers, and a model that reads them tends to copy the applied steer
BLIND = dict(steer=((1, 2),), history=((30, 45),), direction=((2, 3),))


# ---------------------------------------------------------------- BEV of a batch
_CELLS = {}


def cells(device):
    if device not in _CELLS:
        x, y = T.cell_centres()
        _CELLS[device] = (torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device))
    return _CELLS[device]


def rectangle(px, py, x, y, yaw, length, width):
    """Cells (world px, py [B, R, C]) inside the rectangles (per batch element: centre, yaw rad, size)."""
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    dx, dy = px-x[:, None, None], py-y[:, None, None]
    return ((c*dx+s*dy).abs() <= length/2) & ((-s*dx+c*dy).abs() <= width/2)


def distance_field(occupied):
    """Distance (m, 8-neighbour steps of a cell, clipped) to the nearest occupied cell, [B, R, C]."""
    d = torch.where(occupied, torch.zeros_like(occupied, dtype=torch.float32),
                    torch.full_like(occupied, DISTANCE_CLIP, dtype=torch.float32))[:, None]
    for _ in range(int(DISTANCE_CLIP/T.CELL)):
        d = torch.minimum(d, -F.max_pool2d(-d, 3, 1, 1)+T.CELL)
    return d[:, 0]


def bev_truth(pose, goal, static, walkers, noise=None):
    """[B, BEV_CHANNELS, ROWS, COLS] from the simulator's objects. pose [B, 3] (x, y, yaw deg), goal [B, 3]
    (x, y, yaw rad), static [B, ROWS, COLS] bool, walkers [B, W, 5] ego x, y, vx, vy, valid.
    noise: (position sigma m, velocity sigma m/s) on the walkers, for training."""
    device = pose.device
    x, y = cells(device)
    yaw = torch.deg2rad(pose[:, 2])
    c, s = torch.cos(yaw)[:, None, None], torch.sin(yaw)[:, None, None]
    px = pose[:, 0, None, None]+c*x-s*y
    py = pose[:, 1, None, None]+s*x+c*y
    border = ~((px > LOT[0]) & (px < LOT[1]) & (py > LOT[2]) & (py < LOT[3]))
    target = rectangle(px, py, goal[:, 0], goal[:, 1], goal[:, 2], T.SLOT_LENGTH, T.SLOT_WIDTH)
    w = walkers.clone()
    if noise is not None:
        w[..., 0:2] += noise[0]*torch.randn_like(w[..., 0:2])
        w[..., 2:4] += noise[1]*torch.randn_like(w[..., 2:4])
    valid = w[..., 4, None, None] > .5
    layers = []
    for ahead in (0., 1., 2.):
        wx = (w[..., 0]+ahead*w[..., 2])[..., None, None]
        wy = (w[..., 1]+ahead*w[..., 3])[..., None, None]
        layers.append((torch.hypot(x-wx, y-wy) <= T.WALKER_RADIUS) & valid)        # [B, W, R, C]
    dynamic = layers[0].any(1)
    vx = (layers[0]*w[..., 2, None, None]).sum(1)/layers[0].sum(1).clamp(min=1)
    vy = (layers[0]*w[..., 3, None, None]).sum(1)/layers[0].sum(1).clamp(min=1)
    return assemble(static, border, dynamic.float(), vx, vy, layers[1].any(1).float(), layers[2].any(1).float(), target)


def assemble(static, border, dynamic, vx, vy, future1, future2, target):
    """The Flow's BEV from its layers ([B, R, C]; occupancies 0..1, velocities m/s)."""
    device = static.device
    x, y = cells(device)
    occupied = (static > .5) | border.bool() | (dynamic > .5)
    distance = distance_field(occupied)/DISTANCE_CLIP
    batch = len(static)
    ego = ((x.abs() <= EGO_HALF[0]) & (y.abs() <= EGO_HALF[1])).float().expand(batch, -1, -1)
    cx, cy = (x/T.X_HALF).expand(batch, -1, -1), (y/T.Y_HALF).expand(batch, -1, -1)
    return torch.stack((static.float(), border.float(), dynamic, torch.clamp(vx/SPEED_SCALE, -1.5, 1.5),
                        torch.clamp(vy/SPEED_SCALE, -1.5, 1.5), future1, future2, target.float(), distance, ego, cx, cy), 1)


def mirror(bev, batch):
    """Left-right mirror of a batch: the scene, the vector and the labels (y, steer and yaw change sign)."""
    out = dict(batch)
    if bev is not None:
        bev = bev.flip(-1).clone()
        bev[:, 4] = -bev[:, 4]
        bev[:, 11] = -bev[:, 11]
    v = batch['vector'].clone()
    for i in (1, 3, 5, 7, 9, 10, 12, 14, 16):
        v[:, i] = -v[:, i]
    for k in range(D.HISTORY):
        v[:, 30+3*k+1] = -v[:, 30+3*k+1]
        v[:, 30+3*k+2] = -v[:, 30+3*k+2]
    v[:, 45:45+INTENT_COUNT] = batch['vector'][:, 45:45+INTENT_COUNT][:, list(D.MIRROR)]
    out['vector'] = v
    w = batch['walkers'].clone()
    w[..., 1], w[..., 3] = -w[..., 1], -w[..., 3]
    out['walkers'] = w
    a = batch['actions'].clone()
    a[..., 0] = -a[..., 0]
    out['actions'] = a
    t = batch['trajectory'].clone()
    t[..., 1], t[..., 2] = -t[..., 1], -t[..., 2]
    out['trajectory'] = t
    out['intent'] = torch.tensor(D.MIRROR, device=batch['intent'].device)[batch['intent']]
    return bev, out


# ---------------------------------------------------------------- model
class Res(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.a = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(8, cout), nn.SiLU(),
                               nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.GroupNorm(8, cout))
        self.skip = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        return F.silu(self.a(x)+self.skip(x))


class FlowNet(nn.Module):
    """Velocity of the rectified Flow: residual MLP blocks, each shifted and scaled by the condition and the time."""

    def __init__(self, condition, hidden=256, blocks=4):
        super().__init__()
        self.inp = nn.Linear(STEPS*ACTION, hidden)
        self.time = nn.Sequential(nn.Linear(16, hidden), nn.SiLU())
        self.cond = nn.Linear(condition, hidden)
        self.blocks = nn.ModuleList([nn.ModuleDict(dict(norm=nn.LayerNorm(hidden, elementwise_affine=False),
                                                        film=nn.Linear(hidden, 2*hidden),
                                                        body=nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                                                           nn.Linear(hidden, hidden)))) for _ in range(blocks)])
        self.out = nn.Linear(hidden, STEPS*ACTION)

    def forward(self, x, t, condition):
        freq = torch.exp(torch.arange(8, device=x.device)*math.log(1000.)/7.)
        phase = t[:, None]*freq[None]
        signal = F.silu(self.cond(condition))+self.time(torch.cat((torch.sin(phase), torch.cos(phase)), 1))
        h = self.inp(x.flatten(1))
        for block in self.blocks:
            scale, shift = block['film'](signal).chunk(2, 1)
            h = h+block['body'](block['norm'](h)*(1+scale)+shift)
        return self.out(h).view(-1, STEPS, ACTION)


class BevFlow(nn.Module):
    def __init__(self, bev=True, width=128, queries=8, previous=True, blind=()):
        super().__init__()
        self.bev = bev
        self.blind = tuple(blind)
        # previous False: the tail of the previous plan and the previous intent are not read. A student fed its own
        # last answer tends to keep answering `wait` after a walker has gone by
        self.previous = previous
        self.vector = nn.Sequential(nn.Linear(D.VEC+(0 if bev else 5*D.WALKERS), 256), nn.SiLU(), nn.Linear(256, width))
        if bev:
            self.encoder = nn.Sequential(nn.Conv2d(BEV_CHANNELS, 32, 5, 2, 2, bias=False), nn.GroupNorm(8, 32), nn.SiLU(),
                                         Res(32, 64, 2), Res(64, 96, 2), Res(96, width, 1))
            self.rows = nn.Parameter(torch.zeros(1, width, T.ROWS//8, 1))
            self.cols = nn.Parameter(torch.zeros(1, width, 1, T.COLS//8))
            self.queries = nn.Parameter(torch.randn(1, queries, width)*.02)
            layer = nn.TransformerDecoderLayer(width, 4, 2*width, dropout=.1, batch_first=True, norm_first=True)
            self.reader = nn.TransformerDecoder(layer, 2)
            self.context = nn.Sequential(nn.Linear((queries+1)*width, 256), nn.SiLU())
            self.footprint = nn.Conv2d(width, 1, 1)
            self.footprint_film = nn.Linear(256, width)
        else:
            self.context = nn.Sequential(nn.Linear(width, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU())
        self.intent = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, INTENT_COUNT))
        self.intent_embedding = nn.Embedding(INTENT_COUNT, 32)
        self.gear = nn.Sequential(nn.Linear(288, 128), nn.SiLU(), nn.Linear(128, STEPS*3))
        self.trajectory = nn.Sequential(nn.Linear(288, 256), nn.SiLU(), nn.Linear(256, D.HORIZON*3))
        self.flow = FlowNet(288)

    def encode(self, bev, vector, walkers=None):
        """context [B, 256] and the BEV feature map (None without a BEV)."""
        if not self.previous or self.blind:
            vector = vector.clone()
        if not self.previous:
            vector[:, 12:30] = 0.
            vector[:, 45:45+INTENT_COUNT] = 0.
        for name in self.blind:
            for first, last in BLIND[name]:
                vector[:, first:last] = 0.
        if not self.bev:
            return self.context(self.vector(torch.cat((vector, walkers.flatten(1)), 1))), None
        token = self.vector(vector)[:, None]
        feature = self.encoder(bev)
        memory = (feature+self.rows+self.cols).flatten(2).transpose(1, 2)
        read = self.reader(torch.cat((token, self.queries.expand(len(bev), -1, -1)+token), 1), memory)
        return self.context(read.flatten(1)), feature

    def condition(self, context, intent):
        return torch.cat((context, self.intent_embedding(intent)), 1)

    def footprint_logits(self, feature, context):
        return self.footprint(feature*(1+self.footprint_film(context)[:, :, None, None]))[:, 0]

    @torch.no_grad()
    def sample(self, condition, count=1, steps=8, noise=1.):
        """count chunks per row from the condition [B, 288]: steer -1..1, speed 0..1 [B, count, 4, 2]. noise 0: the
        chunk of the mean."""
        cond = condition.repeat_interleave(count, 0)
        x = noise*torch.randn(len(cond), STEPS, ACTION, device=cond.device)
        for k in range(steps):
            t = torch.full((len(cond),), k/steps, device=cond.device)
            x = x+self.flow(x, t, cond)/steps
        x = torch.stack((x[..., 0].clamp(-1., 1.), x[..., 1].clamp(0., 1.)), -1)
        return x.view(len(condition), count, STEPS, ACTION)


# ---------------------------------------------------------------- data
class Rows:
    def __init__(self, path, device):
        z = np.load(path, allow_pickle=True)
        self.episodes = json.loads(str(z['episodes']))
        self.device = device
        self.static = z['static']
        # the same scene as the perception gave it (bev_flow_data.py), on the rows that have it
        self.perceived = z['perceived'] if 'perceived' in z.files else np.zeros(len(self.static), bool)
        self.static_perceived = z['static_perceived'] if 'perceived' in z.files else None
        self.walkers_perceived = torch.from_numpy(z['walkers_perceived']) if 'perceived' in z.files else None
        self.data = {k: torch.from_numpy(z[k]) for k in ('pose', 'goal', 'walkers', 'vector', 'actions', 'gears', 'intent',
                                                         'trajectory', 'trajectory_mask', 'change')}
        self.episode = z['episode']
        slots = np.array([e['slot'] for e in self.episodes])[self.episode]
        held = np.isin(slots, VALIDATION_SLOTS)
        self.train, self.validation = np.flatnonzero(~held), np.flatnonzero(held)

    def batch(self, index, perceived=None):
        """perceived: bool per row of index - those rows come with the static mask and the people of the perception
        in place of the simulator's."""
        out = {k: v[index].to(self.device) for k, v in self.data.items()}
        packed = self.static[index]
        if perceived is not None and perceived.any():
            packed = np.where(perceived[:, None], self.static_perceived[index], packed)
            out['walkers'] = torch.where(torch.from_numpy(perceived)[:, None, None], self.walkers_perceived[index],
                                         self.data['walkers'][index]).to(self.device)
        bits = np.unpackbits(packed, axis=1)[:, :T.ROWS*T.COLS].reshape(len(index), T.ROWS, T.COLS)
        out['static'] = torch.from_numpy(bits).to(self.device).bool()
        return out


def footprint_target(trajectory, mask):
    """Cells (stride 8) the car covers within the horizon, in the ego frame of now: [B, R/8, C/8]."""
    device = trajectory.device
    x, y = cells(device)
    x, y = x[4::8, 4::8], y[4::8, 4::8]
    c, s = torch.cos(trajectory[..., 2])[..., None, None], torch.sin(trajectory[..., 2])[..., None, None]
    dx, dy = x-trajectory[..., 0, None, None], y-trajectory[..., 1, None, None]
    inside = ((c*dx+s*dy).abs() <= EGO_HALF[0]+.4) & ((-s*dx+c*dy).abs() <= EGO_HALF[1]+.4)
    return (inside & (mask[..., None, None] > .5)).any(1).float()


def losses(model, bev, batch, weights, previous_dropout=.5):
    vector = batch['vector']
    if previous_dropout:
        drop = torch.rand(len(vector), device=vector.device) < previous_dropout
        vector = vector.clone()
        vector[drop, 45:45+INTENT_COUNT] = 0.
    context, feature = model.encode(bev, vector, batch['walkers'])
    logits = model.intent(context)
    weight = weights[batch['intent']]*torch.where(batch['change'], 3., 1.)
    intent_loss = (F.cross_entropy(logits, batch['intent'], reduction='none')*weight).sum()/weight.sum()
    cond = model.condition(context, batch['intent'])
    x1 = batch['actions']
    x0 = torch.randn_like(x1)
    t = torch.sigmoid(torch.randn(len(x1), device=x1.device))
    xt = (1-t[:, None, None])*x0+t[:, None, None]*x1
    flow_loss = F.mse_loss(model.flow(xt, t, cond), x1-x0)
    gear_loss = F.cross_entropy(model.gear(cond).view(-1, 3), batch['gears'].view(-1))
    mask = batch['trajectory_mask']
    scale = torch.tensor([5., 5., 1.], device=x1.device)
    traj = model.trajectory(cond).view(-1, D.HORIZON, 3)
    traj_loss = (F.smooth_l1_loss(traj, batch['trajectory']/scale, reduction='none').sum(-1)*mask).sum()/mask.sum().clamp(min=1.)
    total = flow_loss+.5*gear_loss+.5*intent_loss+.5*traj_loss
    parts = dict(flow=float(flow_loss), gear=float(gear_loss), intent=float(intent_loss), trajectory=float(traj_loss))
    if feature is not None:
        cover = F.binary_cross_entropy_with_logits(model.footprint_logits(feature, context),
                                                   footprint_target(batch['trajectory'], mask))
        total = total+.2*cover
        parts['footprint'] = float(cover)
    return total, parts


@torch.no_grad()
def evaluate(model, rows, index, bev_mode, batch_size=128, perceived=False):
    model.eval()
    stats = dict(n=0, intent=0, change_n=0, change=0, intent_blind=0, gear=0, gear_pred=0, steer=0., speed=0., steer_pred=0.,
                 speed_pred=0., ade1=0., ade3=0., ade_n1=0., ade_n3=0., best_steer=0., top2=0, copy_steer=0., gear_change_n=0,
                 gear_change=0)
    per_class = np.zeros((INTENT_COUNT, 2))
    scale = torch.tensor([5., 5., 1.], device=rows.device)
    for start in range(0, len(index), batch_size):
        part = index[start:start+batch_size]
        batch = rows.batch(part, np.ones(len(part), bool) if perceived else None)
        bev = bev_truth(batch['pose'], batch['goal'], batch['static'], batch['walkers']) if bev_mode == 'truth' else None
        context, _ = model.encode(bev, batch['vector'], batch['walkers'])
        logits = model.intent(context)
        predicted = logits.argmax(1)
        blind = batch['vector'].clone()
        blind[:, 45:45+INTENT_COUNT] = 0.
        predicted_blind = model.intent(model.encode(bev, blind, batch['walkers'])[0]).argmax(1)
        truth = batch['intent']
        n = len(truth)
        stats['n'] += n
        stats['intent'] += int((predicted == truth).sum())
        stats['intent_blind'] += int((predicted_blind == truth).sum())
        stats['top2'] += int((logits.topk(2, 1).indices == truth[:, None]).any(1).sum())
        stats['change_n'] += int(batch['change'].sum())
        stats['change'] += int(((predicted == truth) & batch['change']).sum())
        # what holding the applied steer for the chunk would give, and the rows whose gear differs from the applied one
        stats['copy_steer'] += float((batch['vector'][:, 1, None]-batch['actions'][..., 0]).abs().mean(1).sum())
        applied = torch.where(batch['vector'][:, 2] < 0, 2, 0)
        switch = (batch['gears'][:, 0] != applied)
        stats['gear_change_n'] += int(switch.sum())
        gear_now = model.gear(model.condition(context, predicted)).view(-1, STEPS, 3).argmax(-1)[:, 0]
        stats['gear_change'] += int(((gear_now == batch['gears'][:, 0]) & switch).sum())
        for k in range(INTENT_COUNT):
            per_class[k, 0] += int((truth == k).sum())
            per_class[k, 1] += int(((truth == k) & (predicted == k)).sum())
        for name, intent in (('', truth), ('_pred', predicted)):
            cond = model.condition(context, intent)
            chunk = model.sample(cond, 1, noise=0.)[:, 0]
            stats['steer'+name] += float((chunk[..., 0]-batch['actions'][..., 0]).abs().mean(1).sum())
            stats['speed'+name] += float((chunk[..., 1]-batch['actions'][..., 1]).abs().mean(1).sum())
            stats['gear'+name] += int((model.gear(cond).view(-1, STEPS, 3).argmax(-1)[:, 0] == batch['gears'][:, 0]).sum())
            if not name:
                traj = model.trajectory(cond).view(-1, D.HORIZON, 3)*scale
                error = torch.hypot(traj[..., 0]-batch['trajectory'][..., 0], traj[..., 1]-batch['trajectory'][..., 1])
                mask = batch['trajectory_mask']
                stats['ade1'] += float((error[:, 4]*mask[:, 4]).sum())
                stats['ade_n1'] += float(mask[:, 4].sum())
                stats['ade3'] += float((error[:, 14]*mask[:, 14]).sum())
                stats['ade_n3'] += float(mask[:, 14].sum())
        # 8 candidates: 4 samples of each of the two likeliest intents - the closest to the teacher's chunk
        top = logits.topk(2, 1).indices
        candidates = torch.cat([model.sample(model.condition(context, top[:, k]), 4) for k in range(2)], 1)
        stats['best_steer'] += float((candidates[..., 0]-batch['actions'][:, None, :, 0]).abs().mean(2).min(1).values.sum())
    n = max(1, stats['n'])
    return dict(rows=stats['n'], intent_accuracy=round(stats['intent']/n, 4), intent_top2=round(stats['top2']/n, 4),
                intent_at_change=round(stats['change']/max(1, stats['change_n']), 4), changes=stats['change_n'],
                intent_without_previous=round(stats['intent_blind']/n, 4),
                intent_recall={name: round(float(per_class[k, 1]/max(1, per_class[k, 0])), 3) for k, name in enumerate(D.INTENTS)},
                gear_accuracy=round(stats['gear']/n, 4), gear_accuracy_predicted_intent=round(stats['gear_pred']/n, 4),
                gear_accuracy_where_it_changes=round(stats['gear_change']/max(1, stats['gear_change_n']), 4),
                gear_changes=stats['gear_change_n'], steer_error_holding_the_applied_steer=round(stats['copy_steer']/n, 4),
                steer_error=round(stats['steer']/n, 4), speed_error=round(stats['speed']/n, 4),
                steer_error_predicted_intent=round(stats['steer_pred']/n, 4),
                speed_error_predicted_intent=round(stats['speed_pred']/n, 4),
                best_of_8_steer_error=round(stats['best_steer']/n, 4),
                trajectory_error_1s=round(stats['ade1']/max(1., stats['ade_n1']), 3),
                trajectory_error_3s=round(stats['ade3']/max(1., stats['ade_n3']), 3))


def train(args):
    device = 'cuda'
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rows = Rows(args.rows, device)
    train_index = rows.train
    if args.fraction < 1.:          # learning curve: a share of the training episodes
        episodes = np.unique(rows.episode[train_index])
        keep = rng.permutation(episodes)[:max(1, int(round(args.fraction*len(episodes))))]
        train_index = train_index[np.isin(rows.episode[train_index], keep)]
    print('rows: train', len(train_index), 'validation', len(rows.validation), 'episodes', len(rows.episodes), flush=True)
    counts = np.bincount(rows.data['intent'][train_index].numpy(), minlength=INTENT_COUNT)
    weights = torch.tensor((counts.max()/np.maximum(counts, 1))**.5, dtype=torch.float32, device=device)
    model = BevFlow(bev=args.bev != 'none', previous=not args.no_previous, blind=args.blind).to(device)
    print('parameters', sum(p.numel() for p in model.parameters()), 'bev', args.bev, 'previous', not args.no_previous,
          'blind', args.blind, flush=True)
    average = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    per_epoch = len(train_index)//args.batch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.epochs*per_epoch, pct_start=.05)
    scaler = torch.amp.GradScaler('cuda')
    os.makedirs(args.out, exist_ok=True)
    history, best = [], None
    for epoch in range(1, args.epochs+1):
        model.train()
        order = rng.permutation(train_index)
        logs, began = [], time.perf_counter()
        for step in range(per_epoch):
            part = np.sort(order[step*args.batch:(step+1)*args.batch])
            batch = rows.batch(part, rows.perceived[part] & (rng.random(len(part)) < args.perceived) if args.perceived else None)
            bev = None
            if args.bev == 'truth':
                bev = bev_truth(batch['pose'], batch['goal'], batch['static'], batch['walkers'], noise=(.08, .12))
            if rng.random() < .5:
                bev, batch = mirror(bev, batch)
            with torch.autocast('cuda', dtype=torch.float16):
                loss, parts = losses(model, bev, batch, weights)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            with torch.no_grad():
                for a, p in zip(average.parameters(), model.parameters()):
                    a.lerp_(p, 1-args.ema)
            logs.append(parts)
        row = dict(epoch=epoch, seconds=round(time.perf_counter()-began),
                   train={k: round(float(np.mean([l[k] for l in logs])), 4) for k in logs[0]})
        if epoch % args.every == 0 or epoch == args.epochs:
            row['validation'] = evaluate(average, rows, rows.validation, args.bev)
            v = row['validation']
            score = v['steer_error_predicted_intent']+v['speed_error_predicted_intent']+(1-v['intent_accuracy'])
            p = None
            if args.perceived:      # the held-out rows that have a perceived BEV, read from it: the same three terms
                p = row['validation_perceived'] = evaluate(average, rows, rows.validation[rows.perceived[rows.validation]],
                                                           args.bev, perceived=True)
                score += p['steer_error_predicted_intent']+p['speed_error_predicted_intent']+(1-p['intent_accuracy'])
            state = dict(model=average.state_dict(), bev=args.bev, previous=not args.no_previous, blind=tuple(args.blind),
                         epoch=epoch, metrics=v, metrics_perceived=p,
                         intents=D.INTENTS,
                         validation_slots=VALIDATION_SLOTS, config=vars(args))
            torch.save(state, os.path.join(args.out, 'last.pt'))
            if best is None or score < best:
                best = score
                torch.save(state, os.path.join(args.out, 'best.pt'))
        history.append(row)
        print(json.dumps(row), flush=True)
        with open(os.path.join(args.out, 'history.json'), 'w') as stream:
            json.dump(history, stream, indent=1)


def load(checkpoint, device='cuda'):
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = BevFlow(bev=saved['bev'] != 'none', previous=saved.get('previous', True), blind=saved.get('blind', ())).to(device)
    model.load_state_dict(saved['model'])
    return model.eval(), saved


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train')
    t.add_argument('out')
    t.add_argument('rows')
    t.add_argument('--bev', choices=('truth', 'none'), default='truth')
    t.add_argument('--epochs', type=int, default=40)
    t.add_argument('--batch', type=int, default=64)
    t.add_argument('--lr', type=float, default=3e-4)
    t.add_argument('--ema', type=float, default=.999)
    t.add_argument('--every', type=int, default=4, help='validate every so many epochs')
    t.add_argument('--fraction', type=float, default=1., help='share of the training episodes (learning curve)')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--no-previous', action='store_true', help='the previous plan and intent are not inputs')
    t.add_argument('--blind', nargs='*', default=[], choices=sorted(BLIND), help='inputs of the vector the model does not read')
    t.add_argument('--perceived', type=float, default=0., help='share of the rows with a perceived BEV (bev_perceived.py) '
                   'that are trained on it in place of the truth')
    e = sub.add_parser('eval')
    e.add_argument('checkpoint')
    e.add_argument('rows')
    e.add_argument('--perceived', action='store_true', help='the held-out rows that have a perceived BEV, read from it')
    args = parser.parse_args()
    if args.command == 'train':
        train(args)
    else:
        model, saved = load(args.checkpoint)
        rows = Rows(args.rows, 'cuda')
        index = rows.validation[rows.perceived[rows.validation]] if args.perceived else rows.validation
        print(json.dumps(evaluate(model, rows, index, saved['bev'], perceived=args.perceived), indent=1))


if __name__ == '__main__':
    main()
