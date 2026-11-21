"""Scene set-up and the parking plan (Python 3.11, torch) for drive.py. Run from scripts.

  episode   the scene of one episode: a route from the start into the slot, the props on it, the walkers
  state     every 1.6 s: the parking plan from the real state. With --model-port the plan is the parking model's
            (model_server.py: its sampled paths per gear structure): the sample that passes the check against the
            true scene (parked cars, props, poles: model_choice) is taken; if none passes, that sample re-optimised
            by the detour optimiser, and failing that the optimiser's own plan on the A* route.
The file is also the teacher the training labels were made with; what follows describes that.

Detour-teacher data collection in the benchmark lot, teacher side (Python 3.11, torch). Run from scripts.

drive.py (Python 3.7, CARLA 0.9.11) spawns the parked cars of an episode and sends their boxes, the start
and the goal slot. This server makes the scenario: a Hybrid A* route from the start into the slot (searched
from the slot outwards, so that it ends exactly in the slot), cone-sized props on it (detour_props.json,
use_final), the A* detour around them (detour_route) and a walker, and replies with what the client spawns.
Then it re-optimises the detour every 1.6 s from the real state the client sends (detour.py)
and saves every plan with its latent coefficients as the labels.

Walkers walk a straight line: they stand at their start until `mover_start`, walk at their speed and stand
again at `mover_stop`, before a parked car, a prop or the lot border; in CARLA they also stop while the car is in
their way (drive.car_ahead). The teacher gets their present state from CARLA with every re-plan and
predicts them at constant velocity (a standing walker is predicted to stand).

usage: python structure_flow/scene_server.py OUT --port 6301
       python structure_flow/scene_server.py OUT --selftest 40      (scenarios only, no CARLA)
Writes OUT/NAME/teacher.npz and OUT/NAME/scenario.json.
"""
import argparse
import json
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structure_flow.config import DEFAULT as cfg
from dataclasses import replace
from multiprocessing.connection import Client
from structure_flow.detour import DetourConfig, DetourTeacher, structure
from structure_flow.detour_route import Clearance, RouteConfig, RouteTask, detour_route, hybrid_astar, time_detour
from structure_flow.detour_drive import APPROACH, CELL, REPLAN, SAFE_GAP, PERSON_RADIUS, Episode, WorldGrid
from structure_flow.labels import is_stop, rear_axle
from sync_flow.windows import to_world

BOUND = (264., 304., -241., -178.)      # x_min, x_max, y_min, y_max of the DDP evaluator
PROPS = [p for p in json.loads((Path(__file__).resolve().parents[2]/'data'/'detour_props.json').read_text())['props'] if p.get('use_final')]
WALKER_KINDS = ('crossing', 'oncoming', 'side', 'diagonal', 'diagonal_rear', 'slot_exit')
PROP_COUNTS = ((0, .2), (1, .5), (2, .3))
FLOW_PASS_WIDTH, FLOW_PASS_ALONG = 3.1, 4.     # m free beside a Flow scene's prop (2.2 m body + clearances), along
FLOW_PROP_OFF_CENTRE = .5     # the Flow scenes' props: this far off the aisle centre line, all on one side
FLOW_PROP_CLEAR = 5.       # m before the slot: no prop nearer in the Flow scenes (the planner's half-circle)
NO_WALKER = .35
GOAL_MOUTH = 3.5        # m along the aisle from the goal slot's centre line kept free of props
MIN_STRUCTURE = .05     # as plan_server: the runner-up structure can be one with no training data
# street-light poles of Town04_Opt inside the lot (get_environment_objects, Poles): the column is ~0.3 m, the
# lamp arms of the 5.7 m box are overhead, so a 0.7 m square is painted as a parked car would be
POLES = ((276.3, -234.3), (294.8, -186.4), (294.7, -210.7), (294.7, -234.3), (276.2, -210.7), (276.3, -186.4))
AISLE_X, AISLE_HALF = 285.45, 2.4     # the aisle between the slot rows at x 290.9 and 280.0
LOT_ROUTE = RouteConfig(max_expansions=80000)    # enough for the Flow scenes' start 16 m before the slot
DETOUR = RouteConfig(max_expansions=20000)


def lot_world():
    return WorldGrid(np.array([[BOUND[0], BOUND[2], 0.], [BOUND[1], BOUND[3], 0.]]), margin=0.)


def box_points(x, y, yaw, length, width, pad=0.):
    u, v = np.meshgrid(np.arange(-length/2-pad, length/2+pad+1e-6, CELL/2), np.arange(-width/2-pad, width/2+pad+1e-6, CELL/2))
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((x+c*u.ravel()-s*v.ravel(), y+s*u.ravel()+c*v.ravel()), -1)


def paint(grid, world, points):
    ij = world.index(points)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < world.shape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < world.shape[1])
    grid[ij[ok, 0], ij[ok, 1]] = True


def inside(xy, pad=0.):
    xy = np.atleast_2d(xy)
    return (xy[:, 0] > BOUND[0]+pad) & (xy[:, 0] < BOUND[1]-pad) & (xy[:, 1] > BOUND[2]+pad) & (xy[:, 1] < BOUND[3]-pad)


def lot_route(world, vehicles, start, goal, static=None):
    """RouteTask (centre poses at 10 Hz) from the start centre pose into the goal centre pose, or None.
    A* runs from the goal outwards to the start, then the path is turned around."""
    clearance = Clearance(world, vehicles, np.zeros(world.shape, bool) if static is None else static, LOT_ROUTE)
    # searched backwards in time, so its forward moves are the car's reverse moves and pay the reverse cost
    found = hybrid_astar(rear_axle(goal), 1, rear_axle(start)[None], clearance, LOT_ROUTE, reverse_direction=1)
    if found is None:
        return None, None
    poses, gears, _, expansions = found
    if len(poses) < 2:          # already at the goal (a stall right in front of the slot)
        return None, None
    real = np.empty(len(gears), int)
    real[1:] = -gears[::-1][:-1]
    real[0] = real[1]
    poses = poses[::-1].copy()
    samples, g, v = time_detour(poses, real, 0., 0., LOT_ROUTE)
    centre = samples.copy()
    centre[:, 0] += cfg.rear_offset_m*np.cos(samples[:, 2])
    centre[:, 1] += cfg.rear_offset_m*np.sin(samples[:, 2])
    centre[:, 2] = np.unwrap(centre[:, 2])
    switches = int((np.diff(g) != 0).sum())
    info = dict(expansions=int(expansions), switches=switches, route_seconds=round(len(samples)/10., 1),
                start_error=round(float(np.linalg.norm(rear_axle(start)[:2]-poses[0, :2])), 2))
    return RouteTask(centre, g*v, g < 0, np.asarray(goal, float)), info


def place_props(task, world, vehicles, count, rng, approach_only=False):
    """`count` props of PROPS on the route where it runs in the aisle, at least 3 m apart along it, clear of
    the start, the mouth of the goal slot and the parked cars: (static mask, list of prop dicts) or None.
    approach_only (the Flow scenes): only between the start and the slot, FLOW_PROP_CLEAR or more before the slot -
    the half-circle back of the parking planner, which does not know them, stays clear."""
    centre = task.pose
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(centre[:, :2], axis=0), axis=1))]
    static = np.zeros(world.shape, bool)
    props = []
    for _ in range(120):
        if len(props) == count:
            break
        s = rng.uniform(3., arc[-1])
        if any(abs(s-p['s']) < 3. for p in props):
            continue
        i = min(int(np.searchsorted(arc, s)), len(arc)-1)
        if abs(centre[i, 0]-AISLE_X) > AISLE_HALF or np.linalg.norm(centre[i, :2]-centre[0, :2]) < 3.5:
            continue
        if abs(centre[i, 1]-task.target[1]) < GOAL_MOUTH:      # in front of the goal slot it bars the way in
            continue
        if approach_only and ((centre[i, 1]-task.target[1])*(centre[0, 1]-task.target[1]) <= 0
                              or abs(centre[i, 1]-task.target[1]) < FLOW_PROP_CLEAR):
            continue
        heading = centre[i, 2]
        normal = np.array([-np.sin(heading), np.cos(heading)])
        # the Flow scenes: towards one side of the aisle, so that the car can pass on the other with room
        offset = rng.choice((-1., 1.))*rng.uniform(.6, 1.) if approach_only else rng.uniform(-.8, .8)
        x, y = centre[i, :2]+normal*offset
        # the Flow scenes, measured on the aisle (the route bends): every prop 0.5 m or more off the centre line and
        # all on one side of it - the other side open all the way (props on both sides are a slalom the car cannot
        # drive in the room)
        if approach_only and (abs(x-AISLE_X) < FLOW_PROP_OFF_CENTRE or any((x-AISLE_X)*(p['x']-AISLE_X) < 0 for p in props)):
            continue
        prop = PROPS[rng.integers(len(PROPS))]
        yaw = rng.uniform(-np.pi, np.pi)
        if world.lookup(vehicles, box_points(x, y, yaw, prop['length'], prop['width'], pad=.4)).any() or not inside((x, y), 1.).all():
            continue
        if approach_only:
            # the Flow scenes: FLOW_PASS_WIDTH free on the open side of the prop, FLOW_PASS_ALONG before and after it
            # (the nose of a long parked car across the aisle can leave less than the body's width beside the prop)
            side = 1. if x < AISLE_X else -1.
            edge = x+side*.5*float(np.hypot(prop['length'], prop['width']))
            across = edge+side*np.arange(.1, FLOW_PASS_WIDTH+1e-6, .1)
            along = y+np.arange(-FLOW_PASS_ALONG, FLOW_PASS_ALONG+1e-6, .5)
            passage = np.stack(np.meshgrid(across, along, indexing='ij'), -1).reshape(-1, 2)
            if world.lookup(vehicles, passage).any() or not inside(passage).all():
                continue
        paint(static, world, box_points(x, y, yaw, prop['length'], prop['width']))
        props.append(dict(blueprint=prop['blueprint'], x=float(x), y=float(y), yaw=float(yaw), length=prop['length'],
                          width=prop['width'], height=prop['height'], s=round(float(s), 2), offset=round(float(offset), 2)))
    return (static, props) if len(props) == count else None


def walk_until_blocked(start, velocity, world, vehicles, static, longest=14.):
    """Seconds a walker from `start` walks at `velocity` before a parked car, a prop or the lot border."""
    speed = np.linalg.norm(velocity)
    direction = velocity/speed
    ahead = start[None]+direction[None]*np.arange(.1, longest, .1)[:, None]
    lead = ahead+direction[None]*(PERSON_RADIUS+.25)
    blocked = world.lookup(vehicles | static, lead) | ~inside(lead, .5)
    first = np.flatnonzero(blocked)
    travel = longest if not len(first) else (first[0]+1)*.1
    return travel/speed


def place_walker(kind, route, world, vehicles, static, rng):
    """One walker: (movers [1, 5], start time, stop time, info) or None. It stands until it sets off a few
    metres before the point where it meets the route at the route's timing, give or take a second or two."""
    centre = route.pose
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(centre[:, :2], axis=0), axis=1))]
    switches = np.flatnonzero(np.diff(route.reverse.astype(int)) != 0)
    speed = rng.uniform(1., 1.5)
    if kind == 'slot_exit':
        goal = centre[-1]
        out = np.array([np.cos(goal[2]), np.sin(goal[2])])        # the parked car faces the aisle
        side = np.array([-out[1], out[0]])
        start = goal[:2]-out*rng.uniform(.5, 2.)+side*rng.uniform(-.4, .4)
        angle = np.radians(rng.uniform(-25., 25.))
        c, s = np.cos(angle), np.sin(angle)
        velocity = np.array([c*out[0]-s*out[1], s*out[0]+c*out[1]])*speed
        switch = (switches[-1]+1)/10. if len(switches) else len(centre)/10.
        t0 = max(0., switch+rng.uniform(-4., 1.))
        walk = walk_until_blocked(start, velocity, world, vehicles, static)
        # it sets off when the car comes within `trigger` m of the slot (drive.py), so that it meets the car
        # at the slot whatever the car's timing; t0 is only the fallback
        return np.array([[*start, *velocity, PERSON_RADIUS]]), t0, t0+walk, dict(kind=kind, speed=round(speed, 2),
                                                                                trigger=round(float(rng.uniform(5., 9.)), 2))
    for _ in range(60):
        s_ = rng.uniform(4., arc[-1]-5.)
        i = int(np.searchsorted(arc, s_))
        if abs(route.speed[i]) < .5 or (len(switches) and np.abs(arc[switches]-s_).min() < 2.):
            continue
        heading = centre[i, 2]
        normal = np.array([-np.sin(heading), np.cos(heading)])
        travel = np.sign(route.speed[i])*np.array([np.cos(heading), np.sin(heading)])
        meet = centre[i, :2]+normal*rng.uniform(-.3, .3)
        if kind == 'crossing':
            velocity = -rng.choice((-1, 1))*normal*speed
        else:
            angle = np.radians(APPROACH[kind])*rng.choice((-1, 1)) if APPROACH[kind] else 0.
            c, s = np.cos(angle), np.sin(angle)
            velocity = -np.array([c*travel[0]-s*travel[1], s*travel[0]+c*travel[1]])*speed
        meet_time = i/10.+rng.uniform(-1.5, 1.5)
        walk = min(max(meet_time, 0.), rng.uniform(2.5, 6.)/speed)
        start = meet-velocity*walk
        line = start[None]+(meet-start)[None]*np.linspace(0, 1, 30)[:, None]
        if world.lookup(vehicles | static, line).any() or not inside(start, .5).all():
            continue
        t0 = max(meet_time-walk, 0.)
        stop = t0+walk_until_blocked(start, velocity, world, vehicles, static)
        return np.array([[*start, *velocity, PERSON_RADIUS]]), t0, stop, dict(kind=kind, speed=round(speed, 2), s=round(float(s_), 2))
    return None


def scenario(parked, start, goal, seed, plain=False, prop_count=None):
    """Episode, scenario dict for the client, or None, reason. plain: no walker of the teacher's (the client
    brings its own walkers, the dynamic-Flow training scenes, which the teacher does not see); the props as usual."""
    rng = np.random.default_rng(seed)
    world = lot_world()
    vehicles = np.zeros(world.shape, bool)
    for x, y, yaw, length, width in parked:
        paint(vehicles, world, box_points(x, y, yaw, length, width))
    for x, y in POLES:
        paint(vehicles, world, box_points(x, y, 0., .7, .7))
    began = time.perf_counter()
    task, route_info = lot_route(world, vehicles, np.asarray(start, float), np.asarray(goal, float))
    if task is None:
        return None, 'no route'
    if route_info['switches'] == 0 and not plain:     # a reverse-only swing into the slot; the teacher crawls on these
        # (plain: the Flow scenes, where a start past the slot - one half-circle back - is the point)
        return None, 'route without a gear change'
    count = int(rng.choice([c for c, _ in PROP_COUNTS], p=[p for _, p in PROP_COUNTS]))
    kind = None if rng.random() < NO_WALKER else str(rng.choice(WALKER_KINDS))
    if count == 0 and kind is None:
        kind = str(rng.choice(WALKER_KINDS))
    # the drawn combination first; where it does not fit (a short route, a blocked aisle) fewer props, then
    # the other walker kinds, so that every slot and start cell still gets its episode
    plans = [(c, kind) for c in range(count, -1, -1)]
    plans += [(c, k) for c in range(count, -1, -1) for k in WALKER_KINDS if k != kind]
    plans = [(c, k) for c, k in plans if c or k]
    if plain:
        plans = [(c, None) for c in range(count, -1, -1)]
        if prop_count is not None:      # a fixed Flow case: exactly that many props, else no episode
            plans = [(int(prop_count), None)]
    fails = []
    found_all = None
    for c, k in plans:
        for attempt in range(4):
            placed = place_props(task, world, vehicles, c, rng, approach_only=plain)
            if placed is None:
                fails.append(f'{c}:props')
                break
            static, props = placed
            if plain:       # the Flow goes round them; the teacher's route and map leave them out (a detour
                # must exist all the same: checked with the detour optimiser, its route not used)
                if c and detour_route(task, world, vehicles, static, DETOUR)[0] is None:
                    fails.append(f'{c}:detour')
                    continue
                route, detour = task, dict(blocked=False, props_left_to_flow=bool(c))
            else:
                route, detour = detour_route(task, world, vehicles, static, DETOUR) if c else (task, dict(blocked=False))
            if route is None:
                fails.append(f'{c}:detour')
                continue
            movers, t0, stop, walker = np.zeros((0, 5)), [], [], None
            if k:
                found = place_walker(k, route, world, vehicles, static, rng)
                if found is None:
                    fails.append(f'{k}:walker')
                    continue
                movers, t0, stop, walker = found
                t0, stop = [t0], [stop]
            found_all = static, props, route, detour, movers, t0, stop, walker
            break
        if found_all:
            break
    if found_all is None:
        return None, f'no placement: ' + ','.join(fails)
    static, props, route, detour, movers, t0, stop, walker = found_all
    drawn = dict(props=count, walker=kind)
    episode = Episode(f'seed{seed}', task, world, vehicles, np.zeros_like(static) if plain else static,
                      movers.astype(np.float32), route)
    episode.mover_start = np.array(t0, float)
    episode.mover_stop = np.array(stop, float)
    episode.k_log, episode.last_hold, episode.last_reroute, episode.stall_reroutes = [], False, -99., 0
    detour = {k: (int(v) if isinstance(v, (np.integer, bool, np.bool_)) else v) for k, v in detour.items()}
    setup = dict(seed=int(seed), props=props, movers=movers.tolist(), mover_start=list(map(float, t0)),
                 mover_stop=list(map(float, stop)), walker=walker, route=route.pose[:, :2].tolist(),
                 lot_route=task.pose[:, :2].tolist(), route_info=route_info, detour=detour, drawn=drawn, limit=float(episode.limit),
                 scenario_seconds=round(time.perf_counter()-began, 2))
    return episode, setup


EGO_LENGTH, EGO_WIDTH = 4.8, 2.2        # vehicle.tesla.model3 bounding box, rounded up (the detour body is 4.4 x 1.9)
SLOT_MOUTH = 2.5                        # goal frame: centres with along < this are inside the slot row
# along, side, yaw of a rest pose inside the slot row: the window often ends at rest part way into the slot,
# so only the side offset and the heading tell a wrong slot
REST_TOLERANCE = np.inf, .6, np.radians(15.)


def wrong_rest(episode, path):
    """True if the decoded window `path` comes to rest inside the slot row but not in the goal slot (a
    neighbouring slot, or the goal slot too far to the side): the model's own idea of the goal, which the
    obstacle checks cannot catch and the detour optimiser would keep, as it ends where the model ends."""
    if abs(float(np.asarray(path['speed'])[-1])) > .05:
        return False            # still moving at the window's end; a later replan sees where it stops
    rear = to_world(episode.state, np.array([[path['x'][-1], path['y'][-1], path['yaw'][-1]]], float))[0]
    centre = rear[:2]+cfg.rear_offset_m*np.array([np.cos(rear[2]), np.sin(rear[2])])
    gx, gy, gyaw = episode.route.pose[-1]
    dx, dy = centre[0]-gx, centre[1]-gy
    along, side = dx*np.cos(gyaw)+dy*np.sin(gyaw), -dx*np.sin(gyaw)+dy*np.cos(gyaw)
    dyaw = abs(np.angle(np.exp(1j*(rear[2]-gyaw))))
    if along >= SLOT_MOUTH:
        return False            # a stop in the aisle (gear change)
    return not (abs(along) <= REST_TOLERANCE[0] and abs(side) <= REST_TOLERANCE[1] and dyaw <= REST_TOLERANCE[2])


def body_hits(episode, path, first_phase=True):
    """True if the real ego body, driven along the decoded window `path` (rear axle, frame of episode.state),
    touches a parked car, prop or pole on the world grid. Only the current gear by default, as plan_server."""
    x, y, yaw = (np.asarray(path[k], float) for k in ('x', 'y', 'yaw'))
    if first_phase:
        gear = np.asarray(path['gear'])
        n = int(np.argmax(gear != gear[0])) if (gear != gear[0]).any() else len(gear)
        x, y, yaw = x[:n], y[:n], yaw[:n]
    rear = to_world(episode.state, np.stack((x, y, yaw), -1))
    centre = rear.copy()
    centre[:, 0] += cfg.rear_offset_m*np.cos(rear[:, 2])
    centre[:, 1] += cfg.rear_offset_m*np.sin(rear[:, 2])
    u, v = np.meshgrid(np.linspace(-.5, .5, 13)*EGO_LENGTH, np.linspace(-.5, .5, 7)*EGO_WIDTH)
    u, v = u.ravel(), v.ravel()
    c, s = np.cos(centre[:, 2:3]), np.sin(centre[:, 2:3])
    px, py = centre[:, :1]+c*u-s*v, centre[:, 1:2]+s*u+c*v
    return bool(episode.world.lookup(episode.vehicles | episode.static, np.stack((px, py), -1)).any())


def model_choice(teacher, checker, base, reply, speed, episode):
    """The planner's own output where the true scene lets it pass: per likely structure, the medoid (by end
    point) of its samples that clear the props, poles and walkers (and preferably the parked cars); if none
    does, the medoid re-optimised by the detour teacher (its spline coefficients, ending where the model's
    path ends). None when that fails too. `base` is a window (masks, movers) whose latent is replaced."""
    latents, hs, probabilities = np.asarray(reply['latents'], np.float32), np.asarray(reply['h']), reply['probabilities']
    base = {k: v for k, v in base.items() if k not in ('latent', 'h', 'start_speed', 'reference', 'track', 'goal')}
    # parked cars are as hard as props here (the route teacher only grazes-checks them, as the DDP mask it was made for)
    base['static'] = base['static'] | base['vehicles']
    for h in dict.fromkeys(hs.tolist()):
        if probabilities[h] < MIN_STRUCTURE:
            continue
        group = latents[hs == h]
        gear, _, _ = structure([h], cfg)
        start_speed = abs(speed) if np.sign(speed) in (0, gear[0, 0]) else 0.
        if all(is_stop(latent, h, cfg) for latent in group):
            return dict(latent=group[0], h=int(h), start_speed=start_speed, clear=True, source='model_stop')
        with torch.no_grad():
            path = teacher.decoder(torch.tensor(group, device=teacher.device), np.full(len(group), h),
                                   torch.full((len(group),), float(start_speed), device=teacher.device), teacher.detour.min_speed)
        ends = torch.stack((path['x'][:, -1], path['y'][:, -1]), -1).cpu().numpy()
        checks = checker.solve([dict(base, latent=latent, h=int(h), start_speed=start_speed) for latent in group], search=False)
        numpy_path = {k: v.cpu().numpy() for k, v in path.items() if k in ('x', 'y', 'yaw', 'gear', 'speed')}
        sample = [{k: v[i] for k, v in numpy_path.items()} for i in range(len(group))]
        aimed = [i for i in range(len(group)) if not wrong_rest(episode, sample[i])]
        if not aimed:
            continue            # every sample parks off the goal: no correction of its spline helps, the route does
        good = [i for i in aimed if checks[i]['clear'] and not body_hits(episode, sample[i])]
        pool = good or aimed
        medoid = pool[int(np.linalg.norm(ends[pool][:, None]-ends[pool][None], axis=-1).sum(1).argmin())]
        if good:
            return dict(latent=group[medoid], h=int(h), start_speed=start_speed, clear=True, source='model')
        r = teacher.solve([dict(base, latent=group[medoid], h=int(h), start_speed=start_speed)])[0]
        with torch.no_grad():
            p = teacher.decoder(torch.tensor(r['latent'][None], device=teacher.device), np.array([h]),
                                torch.tensor([float(start_speed)], device=teacher.device), teacher.detour.min_speed)
        opt = {k: v[0].cpu().numpy() for k, v in p.items() if k in ('x', 'y', 'yaw', 'gear', 'speed')}
        if r['clear'] and not body_hits(episode, opt) and not wrong_rest(episode, opt):
            return dict(latent=r['latent'], h=int(h), start_speed=start_speed, clear=True, source='model_opt')
    return None


def plan(teacher, episode, rear, speed, t, movers=None, model=None):
    """One re-optimisation from the real state: reply dict and the label record.
    movers [M, 4]: the walkers' present position and velocity from CARLA, which the prediction starts from."""
    episode.state, episode.speed, episode.t = np.asarray(rear, float), float(speed), float(t)
    if movers is not None and len(episode.movers):
        episode.movers = np.c_[np.asarray(movers, float), episode.movers[:, 4]].astype(np.float32)
        episode.mover_start = np.full(len(movers), float(t))
        episode.mover_stop = np.full(len(movers), np.inf)
    goal = episode.rear[-1]
    near = np.linalg.norm(episode.state[:2]-goal[:2]) < .3 and abs(np.angle(np.exp(1j*(episode.state[2]-goal[2])))) < .09
    k_log = episode.k_log
    standing = episode.k >= len(episode.rear)-3 or (len(k_log) >= 2 and k_log[-1] == k_log[-2] == episode.k)
    if not near and abs(speed) < .05 and standing:
        # standing at the end of the route (or no longer advancing on it, e.g. the model's own fine approach
        # after a reroute) but outside the tight box: done if inside the benchmark's goal box
        centre = episode.centre()
        goal_centre = episode.route.pose[-1]
        near = bool(abs(centre[0]-goal_centre[0]) <= 1. and abs(centre[1]-goal_centre[1]) <= .6 and
                    abs(np.angle(np.exp(1j*(centre[2]-goal_centre[2])))) <= np.radians(10.))
    # stalled: the car stands (or crawls) off the route, typically after waiting for a walker, and the local
    # optimiser finds no way back to the route from there; A* again from where the car is, around the props
    if (not near and not episode.last_hold and abs(speed) < .15 and len(k_log) >= 3 and k_log[-1] == k_log[-3]
            and t-episode.last_reroute > 6.):
        episode.last_reroute = t
        task, info = lot_route(episode.world, episode.vehicles, episode.centre(), episode.route.pose[-1], episode.static)
        if task is not None:
            episode.route, episode.rear = task, rear_axle(task.pose)
            episode.gear = np.where(task.reverse, -1, 1)
            episode.k = 0
            episode.stall_reroutes += 1
            print(f'  t {t:.1f} stalled at k {k_log[-1]}: new route {info}', flush=True)
    w = episode.window()
    k_log.append(episode.k)
    r, source = None, 'route'
    if model is not None:       # (checker, reply of model_server): the planner's output, corrected if blocked
        choice = model_choice(teacher, model[0], w, model[1], speed, episode)
        if choice is not None and choice['source'] == 'model_stop' and not near:
            choice = None       # the model believes it is parked but the car is outside the goal box: finish on the route
        if choice is not None:
            w = dict(w, latent=choice['latent'], h=choice['h'], start_speed=choice['start_speed'])
            r, source = dict(latent=choice['latent'], clear=choice['clear']), choice['source']
    if r is None:               # no model, or nothing of it passes: the detour teacher on the A* route
        r = teacher.solve([w])[0]
    lat = torch.tensor(r['latent'][None], dtype=torch.float32, device=teacher.device)
    sp = torch.tensor([w['start_speed']], dtype=torch.float32, device=teacher.device)
    with torch.no_grad():
        p = teacher.decoder(lat, np.array([w['h']]), sp, teacher.detour.min_speed)
    p = {k: v[0].cpu().numpy() for k, v in p.items() if k != 'coefficients'}
    gear = p['gear']
    phase = np.r_[0, np.cumsum(np.diff(gear) != 0)]
    rows = np.stack((p['x'], p['y'], p['yaw'], np.abs(p['speed'])*np.sign(gear), phase), -1)
    go = wait = 99.
    moving = True
    if len(episode.movers):
        go = episode.future_gap(episode.executed(p)[0])
        wait = episode.future_gap([episode.state]*REPLAN)
        moving = go >= SAFE_GAP or go >= wait
    if moving and episode.hits_static(p):
        moving = False
    episode.last_hold = not moving
    record = dict(t=float(t), rear=episode.state.copy(), reroutes=episode.stall_reroutes, source=source, speed=float(speed), k=int(episode.k), h=int(w['h']),
                  latent=np.asarray(r['latent'], np.float32), start_speed=float(w['start_speed']), clear=bool(r['clear']),
                  go=float(go), wait=float(wait), hold=not moving, near=bool(near), rows=rows.astype(np.float32),
                  movers=np.asarray(episode.movers, np.float32).copy())
    reply = dict(rows=rows.tolist(), world=to_world(episode.state, rows[:, :3])[:, :2].tolist(), done=bool(near),
                 clear=bool(r['clear']), hold=not moving, go=float(go), wait=float(wait), source=source)
    return reply, record


def save(out, name, setup, records, result):
    folder = out/name
    folder.mkdir(parents=True, exist_ok=True)
    h = np.array([r['h'] for r in records])
    np.savez(folder/'teacher.npz', t=np.array([r['t'] for r in records]), rear=np.array([r['rear'] for r in records]),
             speed=np.array([r['speed'] for r in records]), k=np.array([r['k'] for r in records]), h=h,
             latent=np.stack([r['latent'] for r in records]) if records else np.zeros((0,)),
             start_speed=np.array([r['start_speed'] for r in records]), clear=np.array([r['clear'] for r in records]),
             go=np.array([r['go'] for r in records]), wait=np.array([r['wait'] for r in records]),
             hold=np.array([r['hold'] for r in records]), reroutes=np.array([r['reroutes'] for r in records]), source=np.array([r['source'] for r in records]), movers=np.array([r['movers'] for r in records]), rows=np.stack([r['rows'] for r in records]) if records else np.zeros((0,)))
    (folder/'scenario.json').write_text(json.dumps(dict(setup, result=result)))


def serve(args):
    teacher = DetourTeacher(cfg)
    checker = DetourTeacher(cfg, replace(DetourConfig(), iterations=0))      # exact checks of model samples
    model_link = Client(('127.0.0.1', args.model_port), authkey=b'lot-model') if args.model_port else None
    with Listener(('127.0.0.1', args.port), authkey=b'lot-collect') as listener:
        print('listening', args.port, flush=True)
        while True:
            with listener.accept() as link:
                episode, records, setup, name = None, [], None, None
                while True:
                    try:
                        message = json.loads(link.recv_bytes().decode())
                    except EOFError:
                        break
                    if message['type'] == 'episode':
                        name = message['name']
                        episode, setup = scenario(message['parked'], message['start'], message['goal'], message['seed'],
                                                  plain=bool(message.get('plain')), prop_count=message.get('prop_count'))
                        if episode is None:
                            print(name, 'skip', setup, flush=True)
                            link.send_bytes(json.dumps(dict(ok=False, reason=setup)).encode())
                            continue
                        records = []
                        print(name, 'props', len(setup['props']), 'walker', setup['walker'], 'route', setup['route_info'],
                              'detour', setup['detour'].get('blocked'), f"{setup['scenario_seconds']}s", flush=True)
                        link.send_bytes(json.dumps(dict(ok=True, **setup)).encode())
                    elif message['type'] == 'state':
                        began = time.perf_counter()
                        model = None
                        if message.get('images'):           # the 6 cameras follow; ask the shared model
                            images = link.recv_bytes()
                            if model_link is not None:
                                model_link.send_bytes(json.dumps(message['model_state']).encode())
                                model_link.send_bytes(images)
                                model = (checker, json.loads(model_link.recv_bytes().decode()))
                        reply, record = plan(teacher, episode, message['rear'], message['speed'], message['t'], message.get('movers'),
                                             model)
                        if model is not None:
                            record['model_seconds'] = model[1]['seconds']
                        record['seconds'] = time.perf_counter()-began
                        records.append(record)
                        link.send_bytes(json.dumps(reply).encode())
                    elif message['type'] == 'end':
                        save(args.out, name, dict(setup, client=message.get('meta')), records, message['result'])
                        print(name, 'saved', message['result'], flush=True)
                        link.send_bytes(b'{}')


def selftest(args):
    """Scenario generation on random DDP-style parked sets, without CARLA."""
    import random
    rows, slot_y = (298.5, 290.9, 280.0, 272.5), (-235.73, -232.73, -229.53, -226.43, -223.43, -220.23, -217.23, -214.03,
                                                 -210.73, -207.30, -204.23, -201.03, -198.03, -194.90, -191.53, -188.20)
    spots = [(x, y) for x in rows for y in slot_y]
    stats = []
    for n in range(args.selftest):
        rnd = random.Random(n)
        index = rnd.choice(list(range(16, 48)))
        goal_xy = spots[index]
        count = rnd.randint(len(spots)//3, len(spots)-1)
        order = list(spots)
        rnd.shuffle(order)
        parked = [(x, y, rnd.choice((0., np.pi)), 4.8, 2.1) for x, y in order[:count] if (x, y) != goal_xy]
        start_index = rnd.randrange(6)
        start = (285.6+rnd.uniform(-.4, .4), goal_xy[1]-8+3.2*start_index+rnd.uniform(-1.2, 1.2),
                 np.radians((90. if start_index < 3 else -90.)+rnd.uniform(-5, 5)))
        goal = (goal_xy[0], goal_xy[1], np.pi if index <= 31 else 0.)
        began = time.perf_counter()
        episode, setup = scenario(parked, start, goal, 1000+n)
        took = time.perf_counter()-began
        if episode is None:
            print(n, index, start_index, 'FAIL', setup, f'{took:.1f}s', flush=True)
            stats.append(dict(ok=False, seconds=took))
            continue
        print(n, index, start_index, 'props', len(setup['props']), (setup['walker'] or {}).get('kind'), setup['route_info'],
              'blocked', setup['detour'].get('blocked'), f'{took:.1f}s', flush=True)
        stats.append(dict(ok=True, seconds=took, **setup['route_info']))
    ok = [s for s in stats if s['ok']]
    print('ok', len(ok), '/', len(stats), 'median s', np.median([s['seconds'] for s in stats]).round(1),
          'max s', np.max([s['seconds'] for s in stats]).round(1),
          'switches', np.bincount([s['switches'] for s in ok]).tolist(), 'route s', np.median([s['route_seconds'] for s in ok]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--port', type=int, default=6301)
    parser.add_argument('--selftest', type=int, default=0)
    parser.add_argument('--model-port', type=int, default=0, help='model_server.py port: plan from the model output')
    args = parser.parse_args()
    selftest(args) if args.selftest else serve(args)
