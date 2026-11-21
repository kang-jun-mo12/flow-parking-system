"""Closed-loop detour teacher on recorded parking tasks, in a kinematic world (no CARLA).

An obstacle is put on the expert's route. Every 1.6 s (the benchmark replan period) the teacher takes
the label of the nearest expert frame as its start, optimises it against the obstacle while tracking
the expert's remaining path (detour.py), and the car follows the first 1.6 s of the result exactly.
The episode succeeds when the car reaches the expert's final pose without touching the obstacle.
The driven 10 Hz trajectory is the material of the detour labels (window_label on it) and of the CARLA
replay that renders images for every model.

Parked cars: the DDP segmentation of all windows of the task (cached in the pack), voted into a world
grid. They limit the detour through the `vehicles` cost; touching them is reported, not a failure,
because label paths themselves graze the predicted mask.

usage: python scripts/structure_flow/detour_drive.py PACK OUT [--tasks 24] [--kinds cone,can,barrier,person,crossing,none]
Writes OUT/drive_summary.json, OUT/KIND/episodes.json, OUT/KIND/TASK.npz (driven trajectory), OUT/KIND/sheet.png.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structure_flow.config import DEFAULT as cfg
from structure_flow.detour import BODY_LENGTH, BODY_WIDTH, DetourTeacher, structure, to_centre
from structure_flow.detour_route import DISCS, RouteConfig, RouteTask, detour_route

REROUTE = RouteConfig(max_expansions=15000)     # re-planned every 1.6 s: a smaller search budget
from structure_flow.labels import rear_axle, window_label
from sync_flow.windows import load_task, to_ego, to_world

ROOTS = ('C:/CARLA/carlaparking/e2e_dataset', 'C:/CARLA/carlaparking/parkinglot_1000')
CELL = .1
REPLAN = 16                     # frames (1.6 s at 10 Hz)
SIZES = dict(cone=(.4, .4), can=(.6, .6), barrier=(1.6, .5), person=(.5, .5))
WALK, PERSON_RADIUS = 1.3, .25
CROSS_FROM_SWITCH = 2.          # m of route between a crossing point and any gear change
SAFE_GAP = .3                   # m from a mover that a go/wait decision keeps if it can
ONCOMING_MEET = 7.              # s: latest meeting time of an approaching walker and the expert car
# walkers heading at the car, degrees off head-on: 0 against the car, 90 from the side, 135 from behind
APPROACH = dict(oncoming=0., diagonal=45., side=90., diagonal_rear=135.)
PREDICT = 3.                    # s of a mover's predicted path drawn as an obstacle for the A* route
MOVER_CLEAR = 1.6               # m: predicted mover points this close to a body disc are left to the local layer


def pack_tasks(roots=ROOTS):
    """Task folders in pack order and the pack row of each task's first window."""
    paths = sorted(p.parent.parent for r in roots for p in Path(r).glob('*/task*/fused_bev/summary.json'))
    offsets, start = [], 0
    for path in paths:
        offsets.append(start)
        start += len(range(0, len(load_task(path).pose), cfg.stride_frames))
    return paths, np.array(offsets)


class WorldGrid:
    """A 0.1 m world raster around a task."""

    def __init__(self, poses, margin=14.):
        self.origin = poses[:, :2].min(0)-margin
        size = np.ceil((poses[:, :2].max(0)+margin-self.origin)/CELL).astype(int)
        self.shape = (size[0], size[1])

    def index(self, xy):
        return np.floor((np.asarray(xy)-self.origin)/CELL).astype(int)

    def lookup(self, grid, xy):
        ij = self.index(xy)
        inside = (ij[..., 0] >= 0) & (ij[..., 0] < self.shape[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < self.shape[1])
        out = np.zeros(ij.shape[:-1], grid.dtype)
        out[inside] = grid[ij[..., 0][inside], ij[..., 1][inside]]
        return out


EGO_CELLS = np.stack(np.meshgrid((np.arange(200)+.5-100)*CELL, (np.arange(200)+.5-100)*CELL, indexing='ij'), -1)


def vehicle_map(task, seg_rows, world):
    """Vote the DDP vehicle class of every window into the world grid."""
    seen = np.zeros(world.shape, np.int32)
    hits = np.zeros(world.shape, np.int32)
    for frame, seg in seg_rows:
        xy = to_world(task.pose[frame], np.concatenate((EGO_CELLS, np.zeros((200, 200, 1))), -1))[..., :2]
        ij = world.index(xy).reshape(-1, 2)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < world.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < world.shape[1])
        ij, value = ij[ok], (seg == 1).reshape(-1)[ok]
        np.add.at(seen, (ij[:, 0], ij[:, 1]), 1)
        np.add.at(hits, (ij[:, 0], ij[:, 1]), value.astype(np.int32))
    return (hits >= 2) & (hits >= .5*seen)


def local_mask(world, grid, centre):
    """World grid -> 200x200 ego-centre BEV of the body centre pose."""
    xy = to_world(centre, np.concatenate((EGO_CELLS, np.zeros((200, 200, 1))), -1))[..., :2]
    return world.lookup(grid, xy)


def body_cells(world, centre_pose):
    """World grid indices covered by the body grid of tools/eval_structure_flow.hits."""
    u, v = np.meshgrid(np.linspace(-.5, .5, 9)*BODY_LENGTH, np.linspace(-.5, .5, 5)*BODY_WIDTH)
    pts = to_world(centre_pose, np.stack((u.ravel(), v.ravel(), np.zeros(u.size)), -1))[:, :2]
    return pts


def rect_gap(centre_pose, point, radius):
    d = to_ego(centre_pose, np.array([point[0], point[1], 0.]))
    u, v = abs(d[0])-BODY_LENGTH/2, abs(d[1])-BODY_WIDTH/2
    return np.hypot(max(u, 0), max(v, 0))+min(max(u, v), 0)-radius


def place_obstacle(kind, task, rear, world, vehicles, rng):
    """World obstacle on the expert's route: (static world mask, movers [M,5] world at t=0, point) or None."""
    centre = task.pose
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(centre[:, :2], axis=0), axis=1))]
    if arc[-1] < 12:
        return None
    static = np.zeros(world.shape, bool)
    if kind == 'none':
        return static, np.zeros((0, 5), np.float32), None
    for _ in range(30):
        s = rng.uniform(5., arc[-1]-7.)
        i = int(np.searchsorted(arc, s))
        if np.linalg.norm(centre[i, :2]-centre[-1, :2]) < 6.:
            continue
        heading = centre[i, 2]
        normal = np.array([-np.sin(heading), np.cos(heading)])
        point = centre[i, :2]
        near = world.lookup(vehicles, point+np.stack(np.meshgrid(np.linspace(-1, 1, 9), np.linspace(-1, 1, 9)), -1).reshape(-1, 2))
        if near.any():
            continue
        if kind in APPROACH:
            # a walker heading at the car (APPROACH degrees off head-on, either side), meeting it at frame i;
            # within the first ONCOMING_MEET s, so that it starts from a short straight walk that crosses no
            # parked car
            switches = np.flatnonzero(np.diff(task.reverse.astype(int)) != 0)
            if abs(task.speed[i]) < .5 or i/10. > ONCOMING_MEET or \
                    (len(switches) and np.abs(arc[switches]-s).min() < CROSS_FROM_SWITCH):
                continue
            travel = np.sign(task.speed[i])*np.array([np.cos(heading), np.sin(heading)])
            meet = point+normal*rng.uniform(-.3, .3)
            angle = np.radians(APPROACH[kind])*rng.choice((-1, 1)) if APPROACH[kind] else 0.
            c, s_ = np.cos(angle), np.sin(angle)
            velocity = -np.array([c*travel[0]-s_*travel[1], s_*travel[0]+c*travel[1]])*WALK
            start = meet-velocity*i/10.
            line = start[None]+(meet-start)[None]*np.linspace(0, 1, 40)[:, None]
            if world.lookup(vehicles, line).any():
                continue
            return static, np.array([[*start, *velocity, PERSON_RADIUS]], np.float32), meet
        if kind == 'crossing':
            # only where the expert is under way: a walker crossing where the car stands to change gear would
            # walk through a stopped car, which no planner can prevent
            switches = np.flatnonzero(np.diff(task.reverse.astype(int)) != 0)
            if abs(task.speed[i]) < .5 or (len(switches) and np.abs(arc[switches]-s).min() < CROSS_FROM_SWITCH):
                continue
            side = rng.choice((-1, 1))
            velocity = -side*normal*WALK
            start = point-velocity*i/10.        # where the expert car is at frame i, so is the person
            return static, np.array([[*start, *velocity, PERSON_RADIUS]], np.float32), point
        where = point+normal*rng.uniform(-.4, .4)
        yaw = heading+np.pi/2+rng.uniform(-.3, .3) if kind == 'barrier' else rng.uniform(-np.pi, np.pi)
        length, width = SIZES[kind]
        u, v = np.meshgrid(np.arange(-length/2, length/2+1e-6, CELL/2), np.arange(-width/2, width/2+1e-6, CELL/2))
        pts = to_world((*where, yaw), np.stack((u.ravel(), v.ravel(), np.zeros(u.size)), -1))[:, :2]
        ij = world.index(pts)
        static[ij[:, 0], ij[:, 1]] = True
        return static, np.zeros((0, 5), np.float32), where
    return None


class Episode:
    """`task` is the expert; `route` (detour_route.RouteTask, default the task) is what the teacher tracks."""

    def __init__(self, name, task, world, vehicles, static, movers, route=None):
        self.name, self.task, self.world = name, task, world
        self.route = task if route is None else route
        self.vehicles, self.static, self.movers = vehicles, static, movers
        self.rear = rear_axle(self.route.pose)
        self.gear = np.where(self.route.reverse, -1, 1)
        self.state = self.rear[0].copy()            # rear axle x, y, yaw (world)
        self.speed, self.t, self.k = float(task.speed[0]), 0., 0
        self.poses, self.speeds, self.reverse = [task.pose[0].copy()], [self.speed], [bool(task.reverse[0])]
        self.done, self.success, self.reason = False, False, ''
        self.static_hit = self.vehicle_hit = False
        self.walked_into = self.hit_moving = False
        self.min_gap, self.waits, self.not_clear = 99., 0, 0
        self.limit = len(self.route.pose)/10.*2+20.
        self.held = 0
        self.mover_route, self.reroutes, self.reroute_seconds = False, 0, 0.
        self.mover_start = np.zeros(len(movers))     # s: a mover stands at its start point until then
        self.mover_stop = np.full(len(movers), np.inf)  # s: and stands again from then (blocked ahead)

    def movers_at(self, t, predicted=False):
        """[M, 5] x, y, vx, vy, r of the movers at time t. predicted: extrapolated from their state now, as a
        planner that does not know when a standing walker will set off would see them."""
        out = np.zeros((len(self.movers), 5))
        for m, (x, y, vx, vy, r) in enumerate(self.movers):
            when = self.t if predicted else t
            walked = min(max(when-self.mover_start[m], 0.), self.mover_stop[m]-self.mover_start[m])
            moving = self.mover_start[m] <= when < self.mover_stop[m]
            px, py = x+vx*walked, y+vy*walked
            vx_, vy_ = (vx, vy) if moving else (0., 0.)
            if predicted:
                px, py = px+vx_*(t-self.t), py+vy_*(t-self.t)
            out[m] = (px, py, vx_, vy_, r)
        return out

    def mover_mask(self):
        """The next PREDICT s of every mover's straight walk as obstacle cells, except near the body now."""
        mask = np.zeros(self.world.shape, bool)
        if not len(self.movers):
            return mask
        centre = self.centre()
        discs = centre[:2]+DISCS[:, None]*np.array([np.cos(centre[2]), np.sin(centre[2])])
        for m, r in enumerate(self.movers[:, 4]):
            n = int(np.ceil(r/CELL))
            for s in np.arange(0., PREDICT+1e-6, .1):
                p = self.movers_at(self.t+s, predicted=True)[m, :2]
                if np.linalg.norm(discs-p, axis=1).min() < MOVER_CLEAR:
                    continue
                i, j = self.world.index(p)
                mask[max(i-n, 0):max(i+n+1, 0), max(j-n, 0):max(j+n+1, 0)] = True
        return mask

    def reroute(self):
        """A* again from the current route frame when the predicted movers block the route ahead."""
        mask = self.mover_mask()
        if not mask.any():
            return
        k = self.progress()
        r = self.route
        if len(r.pose)-k < 10:
            return
        sub = RouteTask(r.pose[k:], r.speed[k:], r.reverse[k:], r.target, r.steer[k:])
        started = time.perf_counter()
        new, info = detour_route(sub, self.world, self.vehicles, self.static | mask, REROUTE)
        self.reroute_seconds += time.perf_counter()-started
        if new is None or not info.get('blocked'):
            return
        self.route = new
        self.rear = rear_axle(new.pose)
        self.gear = np.where(new.reverse, -1, 1)
        self.k = 0
        self.reroutes += 1

    def centre(self, rear=None):
        x, y, yaw = self.state if rear is None else rear
        return np.array([x+cfg.rear_offset_m*np.cos(yaw), y+cfg.rear_offset_m*np.sin(yaw), yaw])

    def progress(self):
        """Nearest expert frame ahead of the last one (position and heading)."""
        last = len(self.rear)-1
        window = np.arange(self.k, min(self.k+60, last)+1)
        d = np.linalg.norm(self.rear[window, :2]-self.state[:2], axis=1)+np.abs(np.angle(np.exp(1j*(self.rear[window, 2]-self.state[2]))))
        self.k = int(window[d.argmin()])
        return self.k

    def window(self):
        last = len(self.rear)-1
        k = self.progress()
        label = window_label(self.route, k, cfg)
        h, latent = int(label['h']), label['latent']
        gear, used, phases = structure([h], cfg)
        first_gear = gear[0, 0]
        speed = self.speed if np.sign(self.speed) in (0, first_gear) else 0.
        frames = np.arange(k, min(k+2*cfg.window_frames, last)+1)
        reference = to_ego(self.state, self.rear[frames])
        end = min(k+cfg.window_frames, last)
        final_gear = gear[0, phases[0]-1]
        if end == last:
            goal = frames >= last-1
        else:
            goal = (frames >= end-8) & (frames <= end+30) & (self.gear[frames] == final_gear)
            if not goal.any():
                goal = frames >= end
        centre = self.centre()
        movers = np.zeros((len(self.movers), 5), np.float32)
        for m, (x, y, vx, vy, r) in enumerate(self.movers_at(self.t)):
            p = to_ego(centre, np.array([x, y, 0.]))
            v = to_ego((0, 0, centre[2]), np.array([vx, vy, 0.]))
            movers[m] = (p[0], p[1], v[0], v[1], r)
        return dict(latent=latent, h=h, start_speed=abs(speed), reference=reference, track=np.ones(len(frames), bool),
                    goal=goal, vehicles=local_mask(self.world, self.vehicles, centre),
                    static=local_mask(self.world, self.static, centre), movers=movers)

    def future_gap(self, rears, hold=24):
        """Smallest mover gap if the car drives through `rears` (one per 0.1 s) and then stands for `hold` steps."""
        rears = list(rears)+[rears[-1]]*hold
        gap = 99.
        for n, rear in enumerate(rears, 1):
            centre, t = self.centre(rear), self.t+n/10.
            for x, y, vx, vy, r in self.movers_at(t, predicted=True):
                gap = min(gap, rect_gap(centre, (x, y), r))
        return gap

    def executed(self, path):
        t = path['time']
        at = np.arange(1, REPLAN+1)/10.
        index = np.minimum(np.searchsorted(t, at), len(t)-1)
        local = np.stack((path['x'][index], path['y'][index], path['yaw'][index]), -1)
        return to_world(self.state, local), index, at > t[-1]

    def hits_static(self, path):
        poses, _, _ = self.executed(path)
        return any(self.world.lookup(self.static, body_cells(self.world, self.centre(p))).any() for p in poses)

    def step(self, path, clear, last_plan):
        """Follow the plan for REPLAN frames. path: dict of numpy arrays from the decoder (window frame)."""
        if not clear:
            self.not_clear += 1
        moving = True
        if len(self.movers):
            # go (1.6 s of the plan, then standing 2.4 s) or wait (4 s here): the one that stays farther from
            # every mover, going when both keep the margin
            go = self.future_gap(self.executed(path)[0])
            wait = self.future_gap([self.state]*REPLAN)
            moving = go >= SAFE_GAP or go >= wait
        if moving and self.hits_static(path):     # the next 1.6 s would touch the obstacle: hold instead
            moving = False
            self.held += 1
        if not moving:                  # wait where the car is
            self.waits += 1
            poses = [self.state]*REPLAN
            speeds = [0.]*REPLAN
            reverse = [self.reverse[-1]]*REPLAN
        else:
            poses, index, stopped = self.executed(path)
            poses = list(poses)
            speeds = list(np.where(stopped, 0., path['speed'][index]))
            reverse = list(path['gear'][index] < 0)
        for i, rear in enumerate(poses):
            self.t += .1
            centre = self.centre(rear)
            pts = body_cells(self.world, centre)
            self.static_hit |= bool(self.world.lookup(self.static, pts).any())
            self.vehicle_hit |= bool(self.world.lookup(self.vehicles, pts).any())
            for x, y, vx, vy, r in self.movers_at(self.t):
                gap = rect_gap(centre, (x, y), r)
                self.min_gap = min(self.min_gap, gap)
                if gap <= 0:        # standing still for a second: the walker walked into the car
                    still = bool(len(self.poses) > 10 and np.linalg.norm(self.poses[-10][:2]-centre[:2]) < .05)
                    self.walked_into |= still
                    self.hit_moving |= not still
            self.poses.append(centre)
            self.speeds.append(float(speeds[i]) if moving else 0.)
            self.reverse.append(bool(reverse[i]))
        self.state = np.array(poses[-1], np.float64)
        self.speed = float(self.speeds[-1])
        goal = self.rear[-1]
        if np.linalg.norm(self.state[:2]-goal[:2]) < .3 and abs(np.angle(np.exp(1j*(self.state[2]-goal[2])))) < .09:
            self.done, self.success, self.reason = True, True, 'parked'
        elif self.t > self.limit:
            self.done, self.reason = True, 'timeout'
        if self.static_hit or self.min_gap <= 0:
            self.success = False
            if self.done or self.static_hit:
                self.done, self.reason = True, 'collision'


def gear_changes(speed):
    moving = np.sign(speed[np.abs(speed) > .1])
    return int((np.diff(moving) != 0).sum()) if len(moving) else 0


def path_quality(e):
    """Largest distance of the driven centre from the expert's centre path, length ratio, gear changes."""
    driven, expert = np.array(e.poses)[:, :2], e.task.pose[:, :2]
    dense = np.concatenate([np.linspace(a, b, 5, endpoint=False) for a, b in zip(expert[:-1], expert[1:])]+[expert[-1:]])
    gap = np.linalg.norm(driven[:, None]-dense[None], axis=-1).min(1)
    length = lambda p: float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    return dict(deviation=round(float(gap.max()), 2), length_ratio=round(length(driven)/max(length(expert), 1e-3), 2),
                gear_changes=gear_changes(np.array(e.speeds)), expert_gear_changes=gear_changes(e.task.speed))


def run(episodes, teacher, log=print):
    started = time.perf_counter()
    while True:
        active = [e for e in episodes if not e.done]
        if not active:
            break
        for e in active:
            if e.mover_route:
                e.reroute()
        windows = [e.window() for e in active]
        results = teacher.solve(windows)
        latents = torch.tensor(np.stack([r['latent'] for r in results]), dtype=torch.float32, device=teacher.device)
        h = np.array([w['h'] for w in windows])
        speed = torch.tensor([w['start_speed'] for w in windows], dtype=torch.float32, device=teacher.device)
        with torch.no_grad():
            path = teacher.decoder(latents, h, speed, teacher.detour.min_speed)
        path = {k: v.cpu().numpy() for k, v in path.items() if k != 'coefficients'}
        for i, (e, r) in enumerate(zip(active, results)):
            e.step({k: v[i] for k, v in path.items()}, r['clear'], r)
        log(f'  t {max(e.t for e in active):5.1f} s, active {len(active)}, {time.perf_counter()-started:.0f} s')


def draw(e, scale=4):
    world = e.world
    poses = np.array(e.poses)
    lo = np.minimum(poses[:, :2].min(0), e.task.pose[:, :2].min(0))-6
    hi = np.maximum(poses[:, :2].max(0), e.task.pose[:, :2].max(0))+6
    size = ((hi-lo)/CELL).astype(int)
    xs = lo[0]+(np.arange(size[0])+.5)*CELL
    ys = lo[1]+(np.arange(size[1])+.5)*CELL
    grid = np.stack(np.meshgrid(xs, ys, indexing='ij'), -1)
    image = np.full((size[0], size[1], 3), 255, np.uint8)
    image[world.lookup(e.vehicles, grid)] = (150, 170, 210)
    image[world.lookup(e.static, grid)] = (230, 120, 30)
    img = Image.fromarray(image.transpose(1, 0, 2)[::-1]).resize((size[0]*scale//2, size[1]*scale//2), Image.NEAREST)
    d = ImageDraw.Draw(img)

    def uv(p):
        return ((p[0]-lo[0])/CELL*scale/2, (hi[1]-p[1])/CELL*scale/2)
    d.line([uv(p) for p in e.task.pose[:, :2]], fill=(120, 120, 120), width=2)
    if e.route is not e.task:
        d.line([uv(p) for p in e.route.pose[:, :2]], fill=(170, 80, 220), width=2)
    d.line([uv(p) for p in poses[:, :2]], fill=(20, 140, 20) if e.success else (210, 20, 20), width=3)
    for x, y, vx, vy, r in e.movers:
        for t in np.arange(0, e.t, 1.):
            u, v = uv((x+vx*t, y+vy*t))
            rr = r/CELL*scale/2
            d.ellipse((u-rr, v-rr, u+rr, v+rr), outline=(220, 0, 0))
    d.text((4, 4), f'{e.name} {e.reason}', fill=(0, 0, 0))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pack', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--tasks', type=int, default=24)
    parser.add_argument('--kinds', default='cone,can,barrier,person,crossing,none')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=None)
    parser.add_argument('--route', type=int, default=1, help='1: Hybrid A* detour route first (detour_route.py)')
    parser.add_argument('--mover-route', type=int, default=0,
                        help='1: every 1.6 s, A* again around the predicted walker paths')
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    selection = json.loads((args.pack/'selection.json').read_text())
    val = {str(Path(p)) for p in selection['val']}
    paths, offsets = pack_tasks()
    seg_all = np.load(args.pack/'seg.npy', mmap_mode='r')
    candidates = [i for i, p in enumerate(paths) if str(p) in val]
    teacher = DetourTeacher(cfg)
    if args.iterations:
        from dataclasses import replace
        teacher = DetourTeacher(cfg, replace(teacher.detour, iterations=args.iterations))
    worlds = {}
    summary = {}
    for kind in args.kinds.split(','):
        out = args.out/kind
        out.mkdir(parents=True, exist_ok=True)
        episodes, no_route = [], []
        for i in rng.permutation(candidates):
            if len(episodes) == args.tasks:
                break
            task = load_task(paths[i])
            if i not in worlds:
                world = WorldGrid(task.pose)
                rows = [(frame, np.asarray(seg_all[offsets[i]+n])) for n, frame in enumerate(range(0, len(task.pose), cfg.stride_frames))]
                worlds[i] = (world, vehicle_map(task, rows, world))
            world, vehicles = worlds[i]
            placed = place_obstacle(kind, task, rear_axle(task.pose), world, vehicles, rng)
            if placed is None:
                continue
            static, movers, _ = placed
            name = f'{paths[i].parent.name}_{paths[i].name}'
            route, info = detour_route(task, world, vehicles, static) if args.route else (None, {})
            if args.route and route is None:
                no_route.append(name)
                continue
            episodes.append(Episode(name, task, world, vehicles, static, movers, route))
            episodes[-1].route_info = info
            episodes[-1].mover_route = bool(args.mover_route)
        print(kind, len(episodes), 'episodes,', len(no_route), 'without an A* route', flush=True)
        run(episodes, teacher, log=lambda s: print(s, flush=True))
        rows = []
        for e in episodes:
            quality = path_quality(e)
            rows.append(dict(name=e.name, success=e.success, reason=e.reason, time=round(e.t, 1), **quality,
                             expert_time=round(len(e.task.pose)/10, 1), static_hit=e.static_hit, vehicle_hit=e.vehicle_hit,
                             mover_gap=round(e.min_gap, 2), walked_into=e.walked_into, hit_moving=e.hit_moving, waits=e.waits, not_clear=e.not_clear, held=e.held,
                             reroutes=e.reroutes, reroute_seconds=round(e.reroute_seconds, 1),
                             route={k: int(v) for k, v in getattr(e, 'route_info', {}).items()}))
            np.savez(out/f'{e.name}.npz', pose=np.array(e.poses), speed=np.array(e.speeds), reverse=np.array(e.reverse),
                     target=e.task.target, static=np.argwhere(e.static), movers=e.movers, origin=e.world.origin)
        (out/'episodes.json').write_text(json.dumps(rows, indent=1))
        summary[kind] = dict(episodes=len(rows), no_route=len(no_route), success=float(np.mean([r['success'] for r in rows])),
                             deviation_median=float(np.median([r['deviation'] for r in rows])),
                             deviation_p90=float(np.percentile([r['deviation'] for r in rows], 90)),
                             length_ratio=float(np.median([r['length_ratio'] for r in rows])),
                             extra_gear_changes=float(np.mean([r['gear_changes']-r['expert_gear_changes'] for r in rows])),
                             collision=float(np.mean([r['static_hit'] or r['mover_gap'] <= 0 for r in rows])),
                             timeout=float(np.mean([r['reason'] == 'timeout' for r in rows])),
                             hit_while_moving=float(np.mean([r['hit_moving'] for r in rows])),
                             walked_into_stopped=float(np.mean([r['walked_into'] and not r['hit_moving'] for r in rows])),
                             vehicle_touch=float(np.mean([r['vehicle_hit'] for r in rows])),
                             time_ratio=float(np.median([r['time']/r['expert_time'] for r in rows])))
        print(kind, json.dumps(summary[kind]), flush=True)
        tiles = [draw(e) for e in episodes[:8]]
        if tiles:
            w = max(t.width for t in tiles)
            hgt = max(t.height for t in tiles)
            sheet = Image.new('RGB', (4*w, 2*hgt), 'white')
            for n, t in enumerate(tiles):
                sheet.paste(t, ((n % 4)*w, (n//4)*hgt))
            sheet.save(out/'sheet.png')
    (args.out/'drive_summary.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
