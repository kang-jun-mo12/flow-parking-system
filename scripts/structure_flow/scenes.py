"""The walkers of the test scenes (CARLA client, Python 3.7): who walks where and when, around the car that parks.

FLOW_CASES names the cases a run goes through (drive.py --walkers cases): walkers that come at the car head-on, from
behind or across the aisle and do not yield (NonYieldingWalkers), with zero to two props on the car's way. The
geometry of the routes and their triggers is scene_contract.py with data/town04_hardcore_v7.json (HardcoreWalkers);
crossing_walkers / front_clear_walker are two older scenes kept for comparison.
"""
import json
import math
import os
import random

import numpy as np

import scene_contract as contract
from supervisor import FRONT_CLEAR, LOT_BOUND, to_local

DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data'))


def crossing_overrides(cfg):
    """five_dynamic_scenarios.through_crossing_overrides on the DynaPark config (the original file, which also holds
    the count weights and crossing endpoints), with the scene's own changes."""
    with open(os.path.join(DATA, 'dynapark_sfm_reconstruction.json'), encoding='utf-8') as f:
        cfg = dict(cfg, **json.load(f))
    cfg.update(crossing_exit_endpoints_x=[262., 306.], crossing_exit_boundary_x=[264., 304.],
               walk_bounds_xy=[261., 307., -239., -180.], stalled_crossing_escape=False,
               detour_ego_max_speed_mps=100., detour_retry_s=.5, relaxation_s=.35)
    return cfg


def blueprint_group(blueprint):
    def text(name):
        return blueprint.get_attribute(name).as_str().strip().lower() if blueprint.has_attribute(name) else 'unknown'
    return ('child' if text('age') == 'child' else 'adult')+'_'+('female' if text('gender') == 'female' else 'male')


def crossing_walkers(world, blueprints, rng, goal, grid, cfg, spawn):
    """dynapark_paper_scenarios.PaperPedestrianScenario.begin (mode ped_sfm) with the ped_crossing changes: 1-4
    walkers (count weights of the config), each crossing along world x at the target slot's y, from inside the
    target or the opposite bay to the far lot edge. Returns walker dicts in drive's format, or None."""
    low, high = cfg['crossing_endpoints_x']
    exit_low, exit_high = cfg['crossing_exit_endpoints_x']
    count = rng.choices((1, 2, 3, 4), weights=cfg['pedestrian_count_weights'], k=1)[0]
    walkers, starts = [], []
    for index in range(count):
        route = None
        for attempt in range(200):
            y = goal[1]+rng.uniform(-.25, .25)
            upward = rng.random() < .5
            start_x = rng.uniform(low, low+2.6) if upward else rng.uniform(high-2.6, high)
            start, end = (start_x, y), ((exit_high if upward else exit_low), y)
            if any(math.hypot(start[0]-p[0], start[1]-p[1]) < 1.2 for p in starts):
                continue
            if grid.segment_clear(start, end):
                route = [start, end]
                break
        if route is None:
            return None
        blueprint = rng.choice(blueprints)
        mean, sd, lower, upper = cfg['speed_groups'][blueprint_group(blueprint)]
        speed = min(upper, max(lower, rng.gauss(mean, sd)))
        speed = max(1.4, min(1.7, speed*1.2))           # ped_crossing: brisk walkers, max 1.9 m/s
        actor = spawn(blueprint, route[0], math.degrees(math.atan2(route[1][1]-route[0][1], route[1][0]-route[0][0])))
        if actor is None:
            return None
        starts.append(route[0])
        walkers.append(dict(actor=actor, id=actor.id, x=route[0][0], y=route[0][1], vx=0., vy=0., trigger=None, t0=0.,
                            stop=1e9, position=route[0], velocity=(0., 0.), route=route, waypoint=1, goal=route[1],
                            preferred_speed=speed, max_speed=1.9, arrived=False, waiting_ticks=0, detour_count=0,
                            navigation_status='following_independent_route',
                            crossing_direction=1 if route[1][0] > route[0][0] else -1, scene='ped_crossing'))
    return walkers


def front_clear_walker(world, blueprints, rng, goal, row_x, aisle_x, spawn):
    """five_dynamic_scenarios ped_front_clear: standing in front of the target slot (3.6 m from the row line
    towards the aisle, x 287.3 / 283.6 for the rows at 290.9 / 280.0), leaving along the aisle (+8 m in y)
    once the car has stood within 5.5 m for 0.6 s."""
    x = row_x-3.6 if row_x > aisle_x else row_x+3.6
    position = (x, goal[1])
    actor = spawn(rng.choice(blueprints), position, 0.)
    if actor is None:
        return None
    end = (aisle_x, goal[1]+8.)
    return [dict(actor=actor, id=actor.id, x=x, y=goal[1], vx=0., vy=0., trigger=None, t0=1e9, stop=1e9,
                 position=position, velocity=(0., 0.), route=[position, end], waypoint=1, goal=end,
                 preferred_speed=1.6, max_speed=1.9, arrived=False, waiting_ticks=0, detour_count=0,
                 navigation_status='standing_in_front_of_target', crossing_direction=0, scene='ped_front_clear',
                 front_wait=0., front_last=None)]


def front_clear_release(walker, t, ego_centre, ego_speed_kmh):
    """StoppedEncounter, simplified: time the car stands (<= 0.5 km/h) within 5.5 m of the walker; True to go."""
    near = math.hypot(walker['position'][0]-ego_centre[0], walker['position'][1]-ego_centre[1]) <= FRONT_CLEAR['trigger_m']
    if near and ego_speed_kmh <= FRONT_CLEAR['stopped_kmh'] and walker['front_last'] is not None:
        walker['front_wait'] += t-walker['front_last']
    elif not near:
        walker['front_wait'] = 0.
    walker['front_last'] = t
    return walker['front_wait'] >= FRONT_CLEAR['wait_s']


# ---------------------------------------------------------------- the Hardcore benchmark walkers

HARDCORE_CONFIG = os.path.join(DATA, 'town04_hardcore_v7.json')


def box_gaps(sweep, obstacles):
    """hardcore_spatial_safety.swept_clear's SAT separations: boxes (x, y, cos, sin, half length, half width)."""
    a = np.asarray(sweep, np.float64)[:, None, :]
    b = np.asarray(obstacles, np.float64)[None, :, :]
    dx, dy = b[..., 0]-a[..., 0], b[..., 1]-a[..., 1]
    separation = np.full(dx.shape, -np.inf)
    for ux, uy in ((a[..., 2], a[..., 3]), (-a[..., 3], a[..., 2]), (b[..., 2], b[..., 3]), (-b[..., 3], b[..., 2])):
        ra = a[..., 4]*abs(ux*a[..., 2]+uy*a[..., 3])+a[..., 5]*abs(-ux*a[..., 3]+uy*a[..., 2])
        rb = b[..., 4]*abs(ux*b[..., 2]+uy*b[..., 3])+b[..., 5]*abs(-ux*b[..., 3]+uy*b[..., 2])
        separation = np.maximum(separation, abs(dx*ux+dy*uy)-ra-rb)
    return separation


def swept_clear(sweep, obstacles, margin):
    return not len(sweep) or not len(obstacles) or bool(np.all(box_gaps(sweep, obstacles) >= margin))


def actor_box(actor, future_s=0.):
    """npc_deadlock_recovery.box_pose(body(actor)): the bounding box, moved on by its velocity for future_s."""
    t, b, v = actor.get_transform(), actor.bounding_box, actor.get_velocity()
    yaw = math.radians(t.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = t.location.x+v.x*future_s, t.location.y+v.y*future_s
    return (x+b.location.x*c-b.location.y*s, y+b.location.x*s+b.location.y*c, c, s, b.extent.x, b.extent.y)


def walker_box(point, radius):
    return (point[0], point[1], 1., 0., radius, radius)


def slot_id(slot):
    """Our SPOTS index (rows of 16 from x 298.5) -> the Hardcore 'row-column' id."""
    return '{}-{}'.format(slot//16+1, slot % 16+1)


class HardcoreWalkers:
    """The four pedestrians of hardcore_parking_scenario.HardcoreEpisode (town04_hardcore_v7), without its moving
    cars: the routes of scene_contract.pedestrian_routes, the independent spatial triggers of
    spatial_trigger_specs (reverse_slot_crossing only once the car reverses 0.65 m into the bay), the physical
    admission of update_spatial_triggers (no start into the ego's reaction + braking stop, 0.35 m), the
    walk_toward_ego approach that aims ahead of the car, constant paces (approach 1.4, burst 2.8, crossing 1.8,
    slot 1.6 m/s), waypoints at 0.22 m, and yielding to a standing ego. The scene frame is the benchmark's (v away
    from the lot's middle, by column) and so is the car's start (start())."""

    def __init__(self, slot, seed, head_on=False):
        with open(HARDCORE_CONFIG, encoding='utf-8-sig') as f:
            self.cfg = dict(json.load(f), seed=int(seed))
        self.frame = frame = contract.SceneFrame(slot_id(slot))
        local = contract.pedestrian_routes(frame, self.cfg['seed'])
        self.paths = {name: [frame.world(*p) for p in route] for name, route in local.items()}
        self.triggers = contract.spatial_trigger_specs(frame, self.cfg, local, self.cfg['seed'])
        self.order = list(contract.PEDESTRIAN_EVENTS)
        random.Random(self.cfg['seed']+1291).shuffle(self.order)
        self.head_on = head_on
        if head_on:
            # a test, not the benchmark: walk_toward_ego alone, on the aisle centre line straight at the car's front
            # (from 9 m past the slot back to 10 m behind the car's start), set off at the start, never yielding
            aisle = self.cfg['aisle_u_m']
            self.paths = {'walk_toward_ego': [frame.world(aisle, 9.), frame.world(aisle, self.cfg['ego_start_v_m']-10.)]}
            self.triggers = {'walk_toward_ego': dict(center_uv=[aisle, self.cfg['ego_start_v_m']], radius_m=3.)}
            self.order = ['walk_toward_ego']
        self.walkers, self.events = [], []
        self.requested, self.next_check = {}, {}

    def start(self, rng):
        """The benchmark's ego start (ego_start_u_m, ego_start_v_m = aisle centre, 6 m before the slot, heading along
        +v), jittered by up to 0.3 m across, 0.8 m along and 3 degrees: x, y, yaw in degrees."""
        f, cfg = self.frame, self.cfg
        start_v = getattr(self, 'start_v', cfg['ego_start_v_m'])
        x, y = f.world(cfg['ego_start_u_m']+rng.uniform(-.3, .3), start_v+rng.uniform(-.8, .8))
        return x, y, (90. if f.v_sign > 0 else -90.)+rng.uniform(-3., 3.)

    def corridor_points(self, spacing=.25):
        """All route points (the benchmark keeps parked cars off the pedestrian corridors)."""
        points = []
        for route in self.paths.values():
            for a, b in zip(route, route[1:]):
                n = max(1, int(math.hypot(b[0]-a[0], b[1]-a[1])/spacing))
                points += [(a[0]+(b[0]-a[0])*i/n, a[1]+(b[1]-a[1])*i/n) for i in range(n+1)]
        return points

    def spawn(self, spawn_fn, blueprints, rng):
        """Spawns the four walkers at their route starts; the name of one that failed, else None."""
        for name in self.paths:
            start = self.paths[name][0]
            actor = spawn_fn(rng.choice(blueprints), start, 0.)
            if actor is None:
                return name
            self.walkers.append(dict(actor=actor, id=actor.id, name=name, scene='hardcore', x=start[0], y=start[1],
                                     vx=0., vy=0., t0=1e9, stop=1e9, position=start, velocity=(0., 0.),
                                     route=list(self.paths[name]), waypoint=1, goal=self.paths[name][-1], arrived=False,
                                     waiting_ticks=0, detour_count=0, navigation_status='waiting', phase='waiting'))
        return None

    def pace(self, name):
        cfg = self.cfg
        return (cfg['approach_speed_mps'] if name == 'walk_toward_ego' else cfg['crossing_speed_mps'] if name == 'aisle_crossing'
                else cfg['burst_speed_mps'] if name == 'occluded_burst' else cfg['slot_walker_speed_mps'])

    def stopping_boxes(self, ego):
        box = actor_box(ego)
        v = ego.get_velocity()
        speed = math.hypot(v.x, v.y)
        distance = speed*self.cfg['trigger_reaction_s']+speed*speed/(2.*self.cfg['trigger_braking_mps2'])
        boxes = [box]
        if speed > .05:
            n = max(1, int(math.ceil(distance/.2)))
            boxes += [(box[0]+v.x/speed*distance*i/n, box[1]+v.y/speed*distance*i/n)+box[2:] for i in range(1, n+1)]
        return boxes

    def approach_route(self, w, ego):
        """hardcore approach_route without its walking-grid detour: aim at the far aisle edge just ahead of the car."""
        f = self.frame
        location, v = ego.get_location(), ego.get_velocity()
        speed = math.hypot(v.x, v.y)
        ev = f.local(location.x, location.y)[1]
        lead = max(3.6, ego.bounding_box.extent.x+speed*self.cfg['trigger_reaction_s']
                   + speed**2/(2.*self.cfg['trigger_braking_mps2'])+.8)
        start_v = f.local(*w['route'][0])[1]
        side = 1. if start_v >= ev else -1.
        aim_v = ev+side*min(lead, .5*abs(start_v-ev))
        return [w['route'][0], f.world(7.4, aim_v), f.world(8., aim_v), self.paths['walk_toward_ego'][-1]]

    def admitted(self, w, route, ego):
        radius = self.cfg['walker_radius_m']
        sweep = []
        for a, b in zip(route, route[1:]):
            n = max(1, int(math.hypot(b[0]-a[0], b[1]-a[1])/.2))
            sweep += [walker_box((a[0]+(b[0]-a[0])*i/n, a[1]+(b[1]-a[1])*i/n), radius) for i in range(n+1)]
        if not swept_clear(sweep, self.stopping_boxes(ego), self.cfg['trigger_actor_clearance_m']):
            return False
        return swept_clear(sweep, [actor_box(o['actor']) for o in self.walkers if o is not w], .18)

    def step(self, t, ego, reversing):
        """One tick: triggers, admission, walking. Returns [x, y, vx, vy] per walker."""
        f, cfg = self.frame, self.cfg
        for w in self.walkers:
            location, velocity = w['actor'].get_location(), w['actor'].get_velocity()
            w['position'], w['velocity'] = (location.x, location.y), (velocity.x, velocity.y)
        transform = ego.get_transform()
        u, v = f.local(transform.location.x, transform.location.y)
        heading = math.cos(math.radians(transform.rotation.yaw))*f.sign
        box = actor_box(ego)
        box_u = f.local(box[0], box[1])[0]
        depth = 2.5-(box_u-abs(box[2])*box[4]-abs(box[3])*box[5])
        ev = ego.get_velocity()
        ready = contract.reverse_entry_ready(u, v, heading, reversing, math.hypot(ev.x, ev.y), depth,
                                            cfg['reverse_ped_entry_depth_m'])
        by_name = {w['name']: w for w in self.walkers}
        for name in self.order:
            w = by_name.get(name)
            if w is None or w['phase'] != 'waiting':
                continue
            if name not in self.requested:
                if not contract.inside_trigger(self.triggers[name], u, v) or (name == 'reverse_slot_crossing' and not ready):
                    continue
                self.requested[name] = t
            if t < self.next_check.get(name, 0.):
                continue
            self.next_check[name] = t+.3
            route = self.approach_route(w, ego) if name == 'walk_toward_ego' and not self.head_on else w['route']
            if not self.head_on and not self.admitted(w, route, ego):
                continue
            w.update(route=list(route), waypoint=1, phase='walking_toward_ego' if name == 'walk_toward_ego' else 'crossing',
                     t0=t, navigation_status='triggered')
            self.events.append(dict(name=name, t=round(t, 2), requested=round(self.requested[name], 2), ego_uv=[u, v]))
        standing = math.hypot(ev.x, ev.y) < .10
        for w in self.walkers:
            command = (0., 0.)
            if w['phase'] not in ('waiting', 'cleared'):
                point = w['position']
                while w['waypoint'] < len(w['route']) and math.hypot(point[0]-w['route'][w['waypoint']][0],
                                                                     point[1]-w['route'][w['waypoint']][1]) < .22:
                    w['waypoint'] += 1
                if w['waypoint'] >= len(w['route']):
                    w['phase'], w['arrived'], w['navigation_status'] = 'cleared', True, 'cleared'
                else:
                    target = w['route'][w['waypoint']]
                    dx, dy = target[0]-point[0], target[1]-point[1]
                    norm = max(.001, math.hypot(dx, dy))
                    pace = self.pace(w['name'])
                    ahead = [walker_box((point[0]+dx/norm*pace*s, point[1]+dy/norm*pace*s), cfg['walker_radius_m'])
                             for s in (0., .25, .5)]
                    # a walker yields to a standing car (the benchmark's rule); a moving car has to yield to it
                    if standing and not self.head_on and not swept_clear(ahead, [actor_box(ego, s) for s in (0., .35)], .20):
                        w['waiting_ticks'] += 1
                    else:
                        command = (dx/norm*pace, dy/norm*pace)
            walk(w['actor'], command)
        return [[w['position'][0], w['position'][1], w['velocity'][0], w['velocity'][1]] for w in self.walkers]


def walk(actor, velocity):
    import carla
    speed = math.hypot(*velocity)
    direction = (velocity[0]/speed, velocity[1]/speed) if speed > 1e-9 else (0., 0.)
    actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=direction[0], y=direction[1], z=0.),
                                            speed=speed, jump=False))


# ---------------------------------------------------------------- walkers that do not yield (Flow training data)

NONYIELD_KINDS = ('cross', 'along')
# The fixed cases of the Flow collection: name -> (walkers, props). Walkers: 'front' along the aisle towards the car,
# 'front_after' the same once the first has left, 'cross' straight across the aisle ahead, 'behind' along the aisle
# from behind in the car's direction (beside its line), 'toward_back' from behind straight at a car that starts past
# the slot.
FLOW_CASES = dict(head_on=(['front'], 0),
                  head_on_two=(['front', 'front_after'], 0),
                  cross_props=(['cross'], 2),
                  behind_cross=(['behind', 'cross'], 0),
                  prop_cross=(['cross'], 1),
                  prop_head_on=(['front'], 1),
                  behind_reverse=(['toward_back'], 0),    # the behind_demo scenes
                  head_on_close=(['front_close'], 0))     # the walker stands ahead and sets off close, so the car
#                                                           backs off first, then goes round it
CASE_ORDER = ('head_on', 'head_on_two', 'cross_props', 'behind_cross', 'prop_cross', 'prop_head_on', 'behind_reverse',
              'head_on_close')
# Both directions: flip reverses the scene frame's v, so the car comes at the slot from the lot's end instead of its
# middle. The car starts 16 m before the slot, nearer where the lot ends sooner (a margin kept to LOT_BOUND, more for a
# walker from behind); a case that then lacks its run-up (props go 3.5 m past the start ..
# scene_server.FLOW_PROP_CLEAR before the slot, 3 m apart) is not scheduled.
CASE_RUNWAY = {0: 9., 1: 10., 2: 12.}       # props -> least metres from the car's start to the slot


def lot_back(frame, aisle):
    """v (<= 0) where the aisle centre line, from the slot back along -v, leaves LOT_BOUND."""
    for back in np.arange(0., 60., .5):
        x, y = frame.world(aisle, -back)
        if not (LOT_BOUND[0] < x < LOT_BOUND[1] and LOT_BOUND[2] < y < LOT_BOUND[3]):
            return -float(back)
    return -60.


def case_start_v(frame, aisle, case):
    """The car's start v of a fixed case (behind_reverse: past the slot, set by its walkers), None if it lacks room."""
    walkers, props = FLOW_CASES[case]
    # 4 m: the car's rear (2.4 m) and the start's 0.8 m jitter inside the lot - with the rear over the lot's edge
    # every move overlaps the mask outside it and the teacher waits (prop_wait)
    # props: 5 m more behind, room to back up for a detour
    # ... and so for a walker coming at the front: the car backs off first
    # ... 7 m for a walker at the front: 1.65 m aside at 15 degrees takes 6.2 m of backing
    room = -lot_back(frame, aisle)-(6.5 if 'behind' in walkers else 4.)-(
        7. if any(k.startswith('front') for k in walkers) else 5. if props else 0.)
    runway = min(16., room)
    return -runway if runway >= CASE_RUNWAY[props] else None


def case_fits(slot, flip, case):
    """Whether the case can be run at the slot in that direction (drive schedules only these)."""
    frame = contract.SceneFrame(slot_id(slot))
    if flip:
        frame.v_sign = -frame.v_sign
    with open(HARDCORE_CONFIG, encoding='utf-8-sig') as f:
        aisle = json.load(f)['aisle_u_m']
    if case == 'behind_reverse':
        # the car starts past the slot, with lot enough ahead of it and behind the slot for the walker and the
        # planner's turn
        back = -lot_back(frame, aisle)
        frame.v_sign = -frame.v_sign
        ahead = -lot_back(frame, aisle)
        # 8.5 m from the car (9 m past the slot at least) to the lot's end, as its start rule wants
        return ahead >= 17.5 and back >= 12.
    if case == 'behind_cross':
        # the walker from behind walks on past the slot: with the lot's end near, the car turning there for the
        # reverse is caught between it and the end
        frame.v_sign = -frame.v_sign
        ahead = -lot_back(frame, aisle)
        frame.v_sign = -frame.v_sign
        if ahead < 13.:
            return False
    return case_start_v(frame, aisle, case) is not None
STUCK_SECONDS_WALKER = 2.       # a walker held up by a parked car this long is taken out of the scene


class NonYieldingWalkers(HardcoreWalkers):
    """Scripted walkers that never react to the car, for the training data of the Flow. Every walker walks one
    straight line and leaves the scene at its end. Scene frame and car start are the Hardcore benchmark's (u from
    the slot towards the aisle, aisle centre u = 5.45, the parked rows' bumpers at u = 2.4 and 8.5, the car starting
    at v = -6 heading +v, then reversing into the slot at v = 0). One walker per episode, a second one with p = .3:
      cross   straight across the aisle (along u), bumper line to bumper line (u 8.2 <-> 2.7), either way, at v in
              [-2, 10] once the car is 4-8 m before that line; with p = .3 across the slot mouth (v +-1.5) once the
              car reverses into the slot;
      along   straight along the aisle (along v) on one side of it (1.3-2.0 m off the centre line, as people keep near the bumpers; on the line
              itself a 5.9 m aisle leaves no room to pass), towards the car from 12-20 m ahead of its start once it
              passes v in [-6, -2], or with p = .3 from 6-10 m behind it in its direction, at once.
    Speeds 0.9-1.8 m/s (from behind 1.5-1.9). The futures for the Oracle are exact: the line at constant pace."""

    def __init__(self, slot, seed, behind_demo=False, case=None, flip=False):
        HardcoreWalkers.__init__(self, slot, seed)
        if flip:        # the other direction: coming at the slot from the lot's end (the routes below are in v)
            self.frame.v_sign = -self.frame.v_sign
        rng = random.Random(int(seed)*7+13)
        f, cfg = self.frame, self.cfg
        aisle, start_v = cfg['aisle_u_m'], cfg['ego_start_v_m']
        kinds = [rng.choice(NONYIELD_KINDS)]
        if rng.random() < .3:
            kinds.append(rng.choice(NONYIELD_KINDS))
        self.start_v = start_v = -16.     # room on the approach for props (scene_server.FLOW_PROP_CLEAR .. start)
        self.case = case
        self.back = lot_back(f, aisle)     # no walker route point beyond the lot's end behind the car ...
        f.v_sign = -f.v_sign
        self.front = -lot_back(f, aisle)   # ... nor ahead (else a walker walks on past the lot's end into the car
        f.v_sign = -f.v_sign               # turning there)
        if case is not None:
            kinds = list(FLOW_CASES[case][0])
            behind_demo = case == 'behind_reverse'
            if not behind_demo:
                self.start_v = start_v = case_start_v(f, aisle, case)
                assert start_v is not None, ('no room for the case', slot, flip, case)
        if behind_demo:
            # the car starts 12 m past the slot, so it has to back up to where the reverse parking starts, and a
            # walker comes from behind along the aisle straight at it: the backward detour
            self.start_v = 12.
            # ... but with room ahead to draw away from the walker before the backward detour: at least 8.5 m from
            # the car's centre to the lot's end, and never nearer than 9 m past the slot
            ends = [v for v in np.arange(0., 30., .5) if not (LOT_BOUND[0] < f.world(aisle, v)[0] < LOT_BOUND[1]
                                                              and LOT_BOUND[2] < f.world(aisle, v)[1] < LOT_BOUND[3])]
            if ends:
                self.start_v = max(9., min(12., ends[0]-8.5))
            kinds = ['toward_back']
        self.kinds, self.paths, self.triggers, self.order, self.paces, self.plan = kinds, {}, {}, [], {}, {}
        self.leads = {}         # name -> (lane u, metres ahead of the car where it appears): set at its trigger
        for i, kind in enumerate(kinds):
            name = '{}_{}'.format(kind, i)
            pace = rng.uniform(.9, 1.8)
            if case is not None and kind != 'toward_back':
                # the fixed cases: walking pace (1.0-1.5 m/s), one walker coming at a time
                pace = rng.uniform(1., 1.5)
                side = rng.choice((-1., 1.))
                if kind == 'cross':
                    # straight across the aisle 4-8 m ahead of the car (never at the slot mouth while it parks)
                    ends = (8.2, 2.7) if side > 0 else (2.7, 8.2)
                    v = min(rng.uniform(-2., 10.), self.front-2.)     # not past the lot's end ahead (slots 16, 32)
                    route, trigger = [(ends[0], v), (ends[1], v)], dict(kind='v_at_least', v=v-rng.uniform(4., 8.))
                elif kind == 'behind':
                    # from behind in the car's direction, near the bumpers (beside the car's line)
                    u = aisle+side*rng.uniform(1.6, 2.0)
                    route, trigger = [(u, start_v-rng.uniform(6., 10.)), (u, 14.)], dict(kind='at_once')
                    pace = rng.uniform(1.2, 1.5)
                else:
                    # towards the car along the aisle, set down 12-16 m ahead of it when triggered; the second one
                    # only after the first has left and while the car is still on its approach (no walker coming
                    # at a car that parks)
                    # on the car's way (0.6-1.3 m off the centre line): farther out it passes beside the car and
                    # the Flow does not take over
                    u = aisle+side*rng.uniform(.6, 1.3)
                    if kind == 'front_after' and 'front_0' in self.leads:
                        # the second walker follows the first on its side of the aisle, within 0.3 m of its line
                        # (on the other side it would come down the lane the car has moved onto for the first)
                        first = self.leads['front_0'][0]-aisle
                        u = aisle+math.copysign(min(1.3, max(.6, abs(first)+rng.uniform(-.3, .3))), first)
                    if kind == 'front_close':
                        # standing in view in the aisle ahead, the walker sets off towards the car once the car is
                        # 5-8 m from it - too close to go round at once, the car backs off first
                        v0 = start_v+rng.uniform(12., 16.)
                        v0 = min(v0, self.front-1.)
                        route, trigger = [(u, v0), (u, max(start_v-12., self.back+.5))], \
                            dict(kind='v_at_least', v=v0-rng.uniform(5., 8.))
                        self.paths[name] = [f.world(*p) for p in route]
                        self.plan[name] = trigger
                        self.paces[name] = pace
                        self.order.append(name)
                        self.triggers[name] = dict(center_uv=[aisle, start_v], radius_m=0.)
                        continue
                    self.leads[name] = (u, rng.uniform(12., 16.))
                    route = [(u, start_v+14.), (u, start_v-12.)]
                    trigger = dict(kind='v_at_least', v=rng.uniform(-14., -10.)) if kind == 'front' else \
                        dict(kind='after', name='front_0', max_v=-3.)
            elif kind == 'cross':
                ends = (8.2, 2.7) if rng.random() < .5 else (2.7, 8.2)
                if rng.random() < .3:
                    v = rng.uniform(-1.5, 1.5)
                    trigger = dict(kind='reversing_in')
                else:
                    v = rng.uniform(-2., 10.)
                    trigger = dict(kind='v_at_least', v=v-rng.uniform(4., 8.))
                route = [(ends[0], v), (ends[1], v)]
            elif kind == 'toward_back':
                # 1.3-2.0 m off the centre line: nearer to it the walker corners the car at the lot's end
                u = aisle+rng.choice((-1., 1.))*rng.uniform(1.3, 2.0)
                route = [(u, self.start_v-rng.uniform(13., 16.)), (u, self.start_v+16.)]
                trigger = dict(kind='at_once')
                pace = rng.uniform(1.2, 1.5)
            elif rng.random() < .3:
                u = aisle+rng.choice((-1., 1.))*rng.uniform(1.3, 2.0)
                route = [(u, start_v-rng.uniform(6., 10.)), (u, 14.)]
                trigger = dict(kind='at_once')
                pace = rng.uniform(1.5, 1.9)
            else:
                u = aisle+rng.choice((-1., 1.))*rng.uniform(1.3, 2.0)
                route = [(u, start_v+rng.uniform(12., 20.)), (u, start_v-12.)]
                trigger = dict(kind='v_at_least', v=rng.uniform(-6., -2.))
            route = [(pu, min(max(pv, self.back+.5), self.front-1.)) for pu, pv in route]
            self.paths[name] = [f.world(*p) for p in route]
            self.plan[name] = trigger
            self.paces[name] = pace
            self.order.append(name)
            self.triggers[name] = dict(center_uv=[aisle, start_v], radius_m=0.)     # for the meta only

    def pace(self, name):
        return self.paces[name]

    HIDDEN_X = 232.             # waiting walkers stand here, off the lot (the removed ones at x 240)
    APPEAR_CLEAR = 3.4          # m from the car's centre: a walker is not set down closer than this

    def spawn(self, spawn_fn, blueprints, rng):
        """Spawned at their route starts as before, then put out of the scene until their trigger (a walker waiting
        at the slot mouth would stand in the reverse parking's way)."""
        failed = HardcoreWalkers.spawn(self, spawn_fn, blueprints, rng)
        if failed is None:
            for w in self.walkers:
                if not w['name'].startswith('front_close'):     # that one stands in view until it sets off
                    self.hide(w)
        return failed

    def hide(self, w):
        import carla
        w['actor'].set_simulate_physics(False)
        w['actor'].set_transform(carla.Transform(carla.Location(x=self.HIDDEN_X, y=-200.+3.*self.walkers.index(w), z=.5)))
        w['position'] = (self.HIDDEN_X, -200.+3.*self.walkers.index(w))

    def appear(self, w, ego_xy):
        """Set the walker down at its route start; a crossing walker starts from the end farther from the car.
        False (stays hidden, tried again next tick) if the start is within APPEAR_CLEAR of the car."""
        import carla
        if w['name'] in self.leads:
            u, lead = self.leads[w['name']]
            ev = self.frame.local(*ego_xy)[1]
            w['route'] = [self.frame.world(u, min(ev+lead, 14., getattr(self, 'front', 60.)-1.)),
                          self.frame.world(u, max(self.start_v-12., getattr(self, 'back', -60.)+.5))]
        options = [w['route']]
        if w['name'].startswith('cross'):
            options.append(w['route'][::-1])
        start, route = max(((math.hypot(r[0][0]-ego_xy[0], r[0][1]-ego_xy[1]), r) for r in options), key=lambda o: o[0])
        if start < self.APPEAR_CLEAR:
            return False
        w['route'] = list(route)
        w['actor'].set_transform(carla.Transform(carla.Location(x=route[0][0], y=route[0][1], z=1.2)))
        w['actor'].set_simulate_physics(True)
        w['position'] = tuple(route[0])
        return True

    def remove(self, w, status):
        """Out of the scene (walked off, or held up by a parked car): else it stands in the aisle for good and the
        car parks round it."""
        import carla
        w['phase'], w['arrived'], w['navigation_status'] = 'cleared', True, status
        w['actor'].set_simulate_physics(False)
        w['actor'].set_transform(carla.Transform(carla.Location(x=240., y=-200.+3.*self.walkers.index(w), z=.5)))

    def step(self, t, ego, reversing):
        # the second walker of head_on_two waits until the first has walked 1 s and is behind the car: just set down,
        # the first still reads its hidden place off the lot for a tick
        f = self.frame
        for w in self.walkers:
            location, velocity = w['actor'].get_location(), w['actor'].get_velocity()
            w['position'], w['velocity'] = (location.x, location.y), (velocity.x, velocity.y)
        transform = ego.get_transform()
        ego_xy = (transform.location.x, transform.location.y)
        u, v = f.local(transform.location.x, transform.location.y)
        heading = math.cos(math.radians(transform.rotation.yaw))*f.sign
        box = actor_box(ego)
        depth = 2.5-(f.local(box[0], box[1])[0]-abs(box[2])*box[4]-abs(box[3])*box[5])
        ev = ego.get_velocity()
        ready = contract.reverse_entry_ready(u, v, heading, reversing, math.hypot(ev.x, ev.y), depth, .3)
        for w in self.walkers:
            if w['phase'] != 'waiting':
                continue
            plan = self.plan[w['name']]
            if (plan['kind'] == 'at_once' or (plan['kind'] == 'v_at_least' and v >= plan['v'])
                    or (plan['kind'] == 'reversing_in' and ready)
                    or (plan['kind'] == 'after' and v <= plan.get('max_v', 1e9)
                        and any(o['name'] == plan['name'] and (o['phase'] == 'cleared' or (o['phase'] == 'walking'
                                and t-o.get('t0', t) > 1. and f.local(*o['position'])[1] < v-3.)) for o in self.walkers))) \
                    and self.appear(w, ego_xy):
                w.update(phase='walking', waypoint=1, t0=t, navigation_status='triggered', anchor=w['position'], moved_at=t)
                self.events.append(dict(name=w['name'], t=round(t, 2), ego_uv=[u, v]))
        for w in self.walkers:
            command = (0., 0.)
            if w['phase'] == 'walking':
                point = w['position']
                if math.hypot(point[0]-w['anchor'][0], point[1]-w['anchor'][1]) > .2:
                    w['anchor'], w['moved_at'] = point, t
                while w['waypoint'] < len(w['route']) and math.hypot(point[0]-w['route'][w['waypoint']][0],
                                                                     point[1]-w['route'][w['waypoint']][1]) < .22:
                    w['waypoint'] += 1
                if w['waypoint'] >= len(w['route']):
                    self.remove(w, 'left_the_scene')
                elif t-w['moved_at'] > STUCK_SECONDS_WALKER:
                    self.remove(w, 'held_up_removed')
                else:
                    target = w['route'][w['waypoint']]
                    dx, dy = target[0]-point[0], target[1]-point[1]
                    norm = max(.001, math.hypot(dx, dy))
                    command = (dx/norm*self.pace(w['name']), dy/norm*self.pace(w['name']))
            if w['phase'] != 'cleared':
                walk(w['actor'], command)
        return [[w['position'][0], w['position'][1], w['velocity'][0], w['velocity'][1]] for w in self.walkers]

    def futures(self, transform, steps=40, dt=.1):
        """Each walker's next `steps` x dt s (and now) in the car's actor frame: the route at constant pace from
        where it is; a waiting walker stands. [dict(positions [steps+1, 2], half_length, half_width)]."""
        out = []
        for w in self.walkers:
            position = np.array(w['position'], float)
            points = [position.copy()]
            if w['phase'] == 'walking':
                route = [np.array(p, float) for p in w['route'][w['waypoint']:]]
                pace = self.pace(w['name'])
                for _ in range(steps):
                    left = pace*dt
                    while route and left > 1e-9:
                        d = np.linalg.norm(route[0]-position)
                        if d <= left:
                            position, left = route[0].copy(), left-d
                            route.pop(0)
                        else:
                            position = position+(route[0]-position)/d*left
                            left = 0.
                    points.append(position.copy())
            else:
                points += [position.copy()]*steps
            local = [to_local(transform, p[0], p[1])[:2] for p in points]
            out.append(dict(positions=[list(p) for p in local], half_length=PEDESTRIAN_HALF_M, half_width=PEDESTRIAN_HALF_M))
        return out


PEDESTRIAN_HALF_M = .35
