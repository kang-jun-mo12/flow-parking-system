"""The driving loop (CARLA 0.9.11 client, Python 3.7): one episode after the other in the Town04 parking lot.

Per episode: the parked cars, the car at its start, the goal slot; scene_server.py gives the props and the walkers
of the scene (scenes.py walks them) and, every 1.6 s, the parking plan (the parking model's, checked there), which
the benchmark tracker follows. Every 0.2 s the 6 cameras go to system_server.py and the supervisor
(supervisor.FlowDriver) decides who drives; while the avoidance Flow does, its commands are the control.
An episode ends parked, in a collision, standing for STUCK_SECONDS or at the time limit; OUT/NAME/meta.json holds the
result and OUT/NAME/frames.json the 10 Hz record (state, control, who drove, the arbiter's verdict).

The loop is also the recorder the training data were made with (--images 1 --depth 1 writes the camera frames in
the CAA / DDP task layout); what follows describes that.

Detour-teacher data collection in the benchmark lot, CARLA side (Python 3.7, CARLA 0.9.11 egg).

Every episode: parked cars as the DDP evaluator spawns them (its init_static_npc, other seeds than the
benchmark's 66+slot), the ego at a start of the benchmark scheme (aisle x 285.6, goal y-8..+8 in 6 steps,
heading towards the slot) jittered by up to 0.4 m across, 1.2 m along and 5 degrees, the goal slot of the
episode (rows 2 and 3, the 16 benchmark slots and the 16 between them), the 'same' weather. scene_server.py
makes the props, the walker and the route, and re-optimises the plan every 1.6 s; the benchmark tracker
(schedule speed, MPC steering) drives it. Episodes go round the slot x start cells in a fixed shuffled order,
so any prefix of the schedule covers the cells evenly; worker w of W takes episodes w, w+W, ...

Saved per episode, OUT/NAME/, in the CAA / DDP task layout at 10 Hz: the 6 benchmark cameras (camera_*/NNNN.png),
the 6 depth cameras on the same mounts (depth_*/NNNN.png, CARLA depth encoding; --depth 0 leaves them out),
measurements/NNNN.json (the CAA generator's keys: pose, speed in km/h, Throttle/Steer/Brake/Reverse/Gear as
applied, IMU, GNSS) and parking_goal/0001.json; besides those frames.json (10 Hz state, control, walkers) and
meta.json (scenario, actors with boxes, result, timing). The topdown BEV is not rendered here; it can be drawn
later from meta.json and frames.json. Failed episodes keep only meta.json. The teacher server writes teacher.npz and scenario.json into the same folder.
usage: CARLA_EGG=... BENCH_DIR=... python37 drive.py OUT --port 2000 --server-port 6301 --worker 0 --workers 3 --count 2
"""
import argparse
import json
import math
import os
import queue
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.connection import Client

sys.path.insert(0, os.environ['BENCH_DIR'])
import benchmark
from benchmark import carla, np, Tracker, WEATHERS, MOUNTS, CAMERAS, SPOTS, SPOT_Z, EGO_X, EGO_Z, REAR_OFFSET
from social_force import SFM_CONFIG, WalkingGrid, acceleration_terms, integrate_velocity, update_detour
import scenes
import supervisor

FPS = 30
REPLAN_TICKS = 48           # 1.6 s
WARMUP_TICKS = 30
SETTLE_TICKS = 30           # standing in the slot after the teacher's done
STUCK_SECONDS = 20.
SLOTS = list(range(16, 48))     # rows 2 (x 290.9) and 3 (x 280.0); the benchmark uses the even ones
STARTS = range(6)
POLES = ((276.3, -234.3), (294.8, -186.4), (294.7, -210.7), (294.7, -234.3), (276.2, -210.7), (276.3, -186.4))  # scene_server.POLES
DEPTH = {c: c.replace('camera', 'depth') for c in CAMERAS}     # camera_front -> depth_front, as CAA names them
CAMERA_OF = {d: c for c, d in DEPTH.items()}


def schedule(n, slots):
    """Episode n: (slot index, start index, seed). Round r = n // cells goes through all cells in a shuffled order."""
    cells = [(s, k) for s in slots for k in STARTS]
    r, i = divmod(n, len(cells))
    order = list(range(len(cells)))
    random.Random(7919*r+17).shuffle(order)
    slot, start = cells[order[i]]
    return slot, start, 100000+n


_CASE_CELLS = {}


def case_schedule(n, slots, attempt=0, out=None):
    """--walkers cases, episode n: (slot, flip, case, seed). Every (slot, direction, case) that fits
    (scenes.case_fits) once per round, in a shuffled order; attempt > 0: the same cell again with another seed
    after a skip (no route). The cells are kept in OUT/schedule.json: a later change of case_fits must not move
    the episode numbers of a folder being resumed."""
    key = tuple(slots)
    if key not in _CASE_CELLS:
        path = None if out is None else os.path.join(out, 'schedule.json')
        if path and os.path.exists(path):
            with open(path) as stream:
                cells = [tuple(c) for c in json.load(stream)['cells']]
        else:
            cells = [(s, flip, c) for s in slots for flip in (0, 1) for c in scenes.CASE_ORDER
                     if scenes.case_fits(s, flip, c)]
            if path:
                with open(path+'.{}.tmp'.format(os.getpid()), 'w') as stream:
                    json.dump(dict(cells=cells), stream)
                os.replace(path+'.{}.tmp'.format(os.getpid()), path)
        _CASE_CELLS[key] = cells
    cells = _CASE_CELLS[key]
    r, i = divmod(n, len(cells))
    order = list(range(len(cells)))
    random.Random(7919*r+17).shuffle(order)
    slot, flip, case = cells[order[i]]
    return slot, flip, case, 100000+n+1000000*attempt


def start_pose(slot, start_index, rng):
    spot = SPOTS[slot]
    y = spot[1]-8+3.2*start_index+rng.uniform(-1.2, 1.2)
    yaw = (90. if start_index < 3 else -90.)+rng.uniform(-5., 5.)
    return EGO_X+rng.uniform(-.4, .4), y, yaw


def goal_pose(slot):
    x, y = SPOTS[slot]
    return x, y, math.pi if slot < 32 else 0.     # the car stands in the slot facing the aisle


class Collector:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client('127.0.0.1', args.port)
        self.client.set_timeout(120.)
        world = self.client.get_world()
        if 'Town04_Opt' not in world.get_map().name:
            world = self.client.load_world('Town04_Opt')
        for pattern in ('sensor.*', 'vehicle.*', 'walker.*', 'static.prop.*', 'controller.*'):
            for actor in world.get_actors().filter(pattern):
                actor.destroy()
        world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        settings = world.get_settings()
        settings.fixed_delta_seconds, settings.synchronous_mode = 1./FPS, True
        world.apply_settings(settings)
        for _ in range(20):     # the actors of a killed run are gone only some ticks later; spawning into them fails
            world.tick(120.)
        cloud, azimuth, altitude, fog, wet, rain, deposits = WEATHERS[args.weather]
        world.set_weather(carla.WeatherParameters(cloudiness=cloud, precipitation=rain, precipitation_deposits=deposits,
                                                  sun_azimuth_angle=azimuth, sun_altitude_angle=altitude, fog_density=fog,
                                                  wetness=wet))
        self.world, self.library = world, world.get_blueprint_library()
        self.ego = world.spawn_actor(self.library.find('vehicle.tesla.model3'),
                                     carla.Transform(carla.Location(x=EGO_X, y=-243.73, z=EGO_Z), carla.Rotation(yaw=90.)))
        self.queues, self.sensors = {}, []
        # the CAA / DDP data layout: camera_* RGB, depth_* CARLA depth (same mounts), imu and gnss for measurements
        self.image_names = list(CAMERAS)+([DEPTH[c] for c in CAMERAS] if args.depth else [])
        for name in self.image_names+['imu', 'gnss']:
            if name in ('imu', 'gnss'):
                bp, transform = self.library.find('sensor.other.'+name), carla.Transform()
            else:
                bp = self.library.find('sensor.camera.depth' if name.startswith('depth') else 'sensor.camera.rgb')
                bp.set_attribute('image_size_x', '400')
                bp.set_attribute('image_size_y', '300')
                bp.set_attribute('fov', '100')
                x, y, z, yaw = MOUNTS[CAMERA_OF.get(name, name)]
                transform = carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw))
            sensor = world.spawn_actor(bp, transform, attach_to=self.ego)
            self.queues[name] = queue.Queue()
            sensor.listen(self.queues[name].put)
            self.sensors.append(sensor)
        self.collisions = []
        sensor = world.spawn_actor(self.library.find('sensor.other.collision'), carla.Transform(), attach_to=self.ego)
        sensor.listen(lambda e: self.collisions.append(e.other_actor.type_id))
        self.sensors.append(sensor)
        self.vehicle_bps = [b for b in self.library.filter('vehicle') if int(b.get_attribute('number_of_wheels')) == 4]
        self.walker_bps = list(self.library.filter('walker.pedestrian.*'))
        self.link = Client(('127.0.0.1', args.server_port), authkey=b'lot-collect')
        self.walk_cfg = SFM_CONFIG
        self.flow = supervisor.FlowDriver(args.flow_port, args.flow_authkey) if args.flow_port else None
        self.pool = ThreadPoolExecutor(6)

    def tick(self, images_needed=True):
        """World tick; the camera images of this frame only when needed (the others are dropped later)."""
        frame = self.world.tick(120.)
        images = {}
        if not images_needed:
            return images
        for name, q in self.queues.items():
            item = q.get(timeout=60.)
            while item.frame < frame:
                item = q.get(timeout=60.)
            images[name] = item
        return images

    def ask(self, message, payload=None):
        self.link.send_bytes(json.dumps(message).encode())
        if payload is not None:
            self.link.send_bytes(payload)
        return json.loads(self.link.recv_bytes().decode())

    def spawn_parked(self, seed, slot, keep_free=()):
        """init_static_npc of the DDP world (benchmark.Bench.spawn_npcs) with this seed; `keep_free` spots stay
        empty as the goal does (the crossing walkers' column)."""
        random.seed(seed)
        spots = list(SPOTS)
        goal = spots[slot]
        count = random.randint(len(spots)//3, len(spots)-1)
        order = list(spots)
        random.shuffle(order)
        actors = []
        free = {goal} | {spots[i] for i in keep_free}
        for spot in order[:count]:
            if spot in free:
                continue
            transform = carla.Transform(carla.Location(x=spot[0], y=spot[1], z=SPOT_Z), carla.Rotation(yaw=random.choice((180., 0.))))
            actor = self.world.try_spawn_actor(random.choice(self.vehicle_bps), transform)
            if actor is not None:
                actor.set_simulate_physics(False)
                actors.append(actor)
        self.parked_wanted = sum(spot not in free for spot in order[:count])
        return actors

    def box(self, actor):
        t, b = actor.get_transform(), actor.bounding_box
        c = t.transform(b.location)
        return dict(type=actor.type_id, x=c.x, y=c.y, z=c.z, yaw=t.rotation.yaw, length=2*b.extent.x, width=2*b.extent.y,
                    height=2*b.extent.z)

    def episode(self, n, attempt=0):
        args = self.args
        flip, case = 0, None
        if args.walkers == 'cases':     # start_index: 0 from the lot's middle, 1 from its end (flip)
            slot, flip, case, seed = case_schedule(n, args.slots, attempt, args.out)
            start_index = flip
            if not scenes.case_fits(slot, flip, case):
                return None         # a cell of the folder's schedule that a later rule dropped: passed over
        else:
            slot, start_index, seed = schedule(n, args.slots)
        name = 'ep{:05d}_s{}_{}'.format(n, slot, start_index)+('_a{}'.format(attempt) if attempt else '')
        folder = os.path.join(args.out, name)
        if os.path.exists(os.path.join(folder, 'meta.json')):
            return None
        began = time.time()
        rng = random.Random(seed)
        sx, sy, syaw = start_pose(slot, start_index, rng)
        # walkers: the teacher's scenario, or a scene the dynamic Flow was trained on (scenes.py)
        scene = args.walkers if args.walkers != 'training' else random.Random(seed+7).choice(('ped_crossing', 'ped_front_clear'))
        keep_free = [i for i, spot in enumerate(SPOTS) if spot[1] == SPOTS[slot][1]] if scene == 'ped_crossing' else []
        # 'cases': the fixed Flow collection cases in turn (scenes.FLOW_CASES), the slot from the schedule
        hard = (scenes.NonYieldingWalkers(slot, seed, behind_demo=scene == 'behind_demo', case=case, flip=flip)
                if scene in ('nonyield', 'behind_demo', 'cases') else
                scenes.HardcoreWalkers(slot, seed, head_on=scene == 'head_on') if scene in ('hardcore', 'head_on') else None)
        if hard:
            # the benchmark's start: aisle centre line 6 m before the slot, coming from the middle of the lot (its
            # triggers sit on that approach), jittered as ours; no parked car on a pedestrian corridor (box 4.8 x
            # 2.2 m plus the walker's 0.3 m)
            sx, sy, syaw = hard.start(rng)
            corridor = np.array(hard.corridor_points())
            keep_free = [i for i, (x, y) in enumerate(SPOTS)
                         if (np.all(np.abs(corridor-(x, y)) <= (2.7, 1.4), axis=1)).any()]
        start = carla.Transform(carla.Location(x=sx, y=sy, z=EGO_Z), carla.Rotation(yaw=syaw))
        self.ego.set_transform(start)
        self.ego.apply_control(carla.VehicleControl(brake=1.))
        actors = self.spawn_parked(seed, slot, keep_free)
        self.tick()
        parked = [self.box(a) for a in actors]
        goal = goal_pose(slot)
        if len(actors) < .7*self.parked_wanted:   # most spawn normally (as in the DDP world); far fewer means
                                                  # ghosts of a killed run hold spots the teacher would not know
            meta = dict(name=name, n=n, slot=slot, start_index=start_index, flip=flip, seed=seed,
                        result=dict(outcome='skipped', reason='parked spawn failed {}/{}'.format(len(actors), self.parked_wanted)),
                        timing=dict(setup=time.time()-began))
            self.finish(folder, meta, actors, None)
            return meta
        setup = self.ask(dict(type='episode', name=name, seed=seed, start=[sx, sy, math.radians(syaw)], goal=list(goal),
                              parked=[[p['x'], p['y'], math.radians(p['yaw']), p['length'], p['width']] for p in parked],
                              plain=scene != 'teacher',
                              prop_count=None if case is None else scenes.FLOW_CASES[case][1]))
        meta = dict(name=name, n=n, slot=slot, start_index=start_index, flip=flip, seed=seed, start=[sx, sy, syaw], goal=list(goal),
                    weather=args.weather, benchmark_slot=slot in benchmark.GOAL_INDICES, parked=parked,
                    port=args.port, renderer=args.renderer, walker_scene=scene, flow=bool(self.flow), case=case)
        timing = dict(setup=time.time()-began)
        if not setup['ok']:
            meta.update(result=dict(outcome='skipped', reason=setup['reason']), timing=timing)
            self.finish(folder, meta, actors, None)
            return meta
        props, walkers = [], []
        for p in setup['props']:
            actor = self.world.try_spawn_actor(self.library.find(p['blueprint']), carla.Transform(
                carla.Location(x=p['x'], y=p['y'], z=.23), carla.Rotation(yaw=math.degrees(p['yaw']))))
            if actor is not None:
                actors.append(actor)
                props.append(actor)
        # walkers: the scenario gives start, line, speed and start time; the social force model (social_force.py,
        # the SAC work's DynaPark calibration) walks them to the end of that line through CARLA's walker physics,
        # pushed off parked cars, props, poles, each other and the ego, and routed round a car that stands
        static = [sfm_box(p['x'], p['y'], math.radians(p['yaw']), p['length'], p['width']) for p in parked]
        static += [sfm_box(p['x'], p['y'], p['yaw'], 2*a.bounding_box.extent.x, 2*a.bounding_box.extent.y)
                   for p, a in zip(setup['props'], props)]
        static += [sfm_box(x, y, 0., .4, .4) for x, y in POLES]
        self.walk_cfg = scenes.crossing_overrides(SFM_CONFIG) if scene == 'ped_crossing' else SFM_CONFIG
        walk_grid = WalkingGrid(static, self.walk_cfg) if setup['movers'] or scene != 'teacher' else None
        if scene != 'teacher':
            def spawn(bp, position, heading):
                if bp.has_attribute('is_invincible'):
                    bp.set_attribute('is_invincible', 'true')
                for z in (.35, .55, .85, 1.):
                    actor = self.world.try_spawn_actor(bp, carla.Transform(carla.Location(x=position[0], y=position[1], z=z),
                                                                           carla.Rotation(yaw=heading)))
                    if actor is not None:
                        actors.append(actor)
                        return actor
                return None
            if hard:
                missing = hard.spawn(spawn, self.walker_bps, rng)
                found = None if missing else hard.walkers
            elif scene == 'ped_crossing':
                found = scenes.crossing_walkers(self.world, self.walker_bps, rng, goal, walk_grid, self.walk_cfg, spawn)
            else:
                found = scenes.front_clear_walker(self.world, self.walker_bps, rng, goal, SPOTS[slot][0], EGO_X, spawn)
            if found is None:
                meta.update(result=dict(outcome='skipped', reason='{} walkers failed'.format(scene)), timing=timing)
                self.ask(dict(type='end', result=meta['result'], meta=None))
                self.finish(folder, meta, actors, None)
                return meta
            walkers.extend(found)
        for m, (x, y, vx, vy, r) in enumerate(setup['movers']):
            bp = rng.choice(self.walker_bps)
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'true')   # as in the SAC setup: no ragdoll; the ego's sensor still logs contact
            heading = math.degrees(math.atan2(vy, vx))
            actor = self.world.try_spawn_actor(bp, carla.Transform(carla.Location(x=x, y=y, z=1.), carla.Rotation(yaw=heading)))
            if actor is not None:
                actors.append(actor)
                trigger = (setup.get('walker') or {}).get('trigger')
                v, walk_time = math.hypot(vx, vy), setup['mover_stop'][m]-setup['mover_start'][m]
                end = (x+vx*walk_time, y+vy*walk_time)
                walkers.append(dict(actor=actor, id=actor.id, x=x, y=y, vx=vx, vy=vy, trigger=trigger,
                                    t0=1e9 if trigger else setup['mover_start'][m], stop=setup['mover_stop'][m],
                                    position=(x, y), velocity=(0., 0.), route=[(x, y), end], waypoint=1, goal=end,
                                    preferred_speed=v, max_speed=v+.4, arrived=False, waiting_ticks=0,
                                    detour_count=0, navigation_status='waiting_to_start'))
        if len(props) < len(setup['props']) or len(walkers) < len(setup['movers']):
            meta.update(result=dict(outcome='skipped', reason='spawn failed'), timing=timing)
            self.ask(dict(type='end', result=meta['result'], meta=None))
            self.finish(folder, meta, actors, None)
            return meta
        for name_ in self.image_names+['measurements']:
            os.makedirs(os.path.join(folder, name_), exist_ok=True)
        tracker = Tracker(max_steer=1., steer_rate=.1, speed_mode='schedule', steering='mpc')
        flow, flow_speed = self.flow, benchmark.SpeedControl()
        if flow:
            prop_rows = [[p['x'], p['y'], p['yaw'], 2*a.bounding_box.extent.x, 2*a.bounding_box.extent.y]
                         for p, a in zip(setup['props'], props)]
            flow.begin(goal, [[p['x'], p['y'], math.radians(p['yaw']), p['length'], p['width']] for p in parked] +
                       prop_rows + [[x, y, 0., .4, .4] for x, y in POLES], prop_rows if scene != 'teacher' else ())
        flowing, replan_now, track_steer, flow_ticks = False, False, 0., 0
        del self.collisions[:]
        frames, futures, plans, sources = [], [], 0, []
        step, done_at, outcome = -WARMUP_TICKS, None, None
        teacher_seconds = tick_seconds = perceive_seconds = 0.
        limit = setup['limit']
        moved_at, moved_from = 0., None      # abort an episode where the car has not moved 1 m in STUCK_SECONDS
        timing['spawn'] = time.time()-began-timing['setup']
        while True:
            t0 = time.time()
            images = self.tick(step >= 0 and step % 3 == 0)
            tick_seconds += time.time()-t0
            transform, velocity = self.ego.get_transform(), self.ego.get_velocity()
            applied = self.ego.get_control()     # the control that drove into this frame (what CAA records)
            yaw = math.radians(transform.rotation.yaw)
            centre = (transform.location.x, transform.location.y)
            rear = [centre[0]-REAR_OFFSET*math.cos(yaw), centre[1]-REAR_OFFSET*math.sin(yaw), yaw]
            speed = velocity.x*math.cos(yaw)+velocity.y*math.sin(yaw)
            if step < 0:
                self.ego.set_transform(start)
                self.ego.set_target_velocity(carla.Vector3D())
                self.ego.apply_control(carla.VehicleControl(brake=1.))
                for w in walkers:
                    walk(w['actor'], (0., 0.))
                step += 1
                if step == 0:
                    del self.collisions[:]
                continue
            t = step/FPS
            if hard:
                walker_now = hard.step(t, self.ego, bool(applied.reverse))
            else:
                walker_now = self.walk_step(walkers, static, walk_grid, transform, velocity, centre, goal, t)
            if args.watch:
                self.show(transform, walkers, goal, flowing)
            if flow and flow.perceives and done_at is None and step % supervisor.DECISION_TICKS == 0:
                # the perception in the loop: the cameras of every decision step, whoever drives
                perceive_seconds += flow.perceive(t, transform, np.stack(
                    [np.frombuffer(images[cam].raw_data, np.uint8).reshape(300, 400, 4)[:, :, 2::-1] for cam in CAMERAS]),
                    tracker, speed, applied)
            if flow and done_at is None and step % supervisor.DECISION_TICKS == 0:
                # the Flow runtime's hand-over: the gate on the teacher's plan; the Flow drives while it holds,
                # the teacher plans again from where the car is once it lets go
                was = flowing
                flowing = flow.update(t, transform, speed, applied, walkers, tracker, track_steer,
                                      futures=hard.futures(transform) if isinstance(hard, scenes.NonYieldingWalkers) else None)
                if was and not flowing:
                    replan_now = True
                    flow_speed = benchmark.SpeedControl()
            if done_at is None and not flowing and (step % REPLAN_TICKS == 0 or replan_now):
                replan_now = False
                t1 = time.time()
                # the cameras and what the planner reads (the benchmark's state: goal in the vehicle frame from the
                # true pose, signed speed, applied steer and gear); the teacher plans from the model's output
                c, s = math.cos(yaw), math.sin(yaw)
                dx, dy = goal[0]-centre[0], goal[1]-centre[1]
                model_state = dict(target=[dx*c+dy*s, -dx*s+dy*c, (math.degrees(goal[2])-transform.rotation.yaw+180.) % 360.-180.],
                                   speed=speed, steer=applied.steer, reverse=bool(applied.reverse))
                cameras = np.stack([np.frombuffer(images[cam].raw_data, np.uint8).reshape(300, 400, 4)[:, :, 2::-1]
                                    for cam in CAMERAS])
                reply = self.ask(dict(type='state', rear=rear, speed=speed, t=t, movers=walker_now, images=True,
                                      model_state=model_state), np.ascontiguousarray(cameras).tobytes())
                sources.append(reply.get('source'))
                teacher_seconds += time.time()-t1
                plans += 1
                if reply['done']:
                    done_at = t
                    tracker.clear()
                elif reply['hold']:
                    tracker.clear()
                else:
                    tracker.set_path(np.array(rear), reply['rows'])
                    if args.watch:      # the teacher's plan, blue forward / red reverse, until the next replan
                        path, gears = tracker.path, tracker.gears
                        for a, b, g in zip(path[:-1], path[1:], gears[1:]):
                            self.world.debug.draw_line(carla.Location(x=a[0], y=a[1], z=.3), carla.Location(x=b[0], y=b[1], z=.3),
                                                       thickness=.06, color=carla.Color(0, 90, 255) if g > 0 else carla.Color(255, 40, 40),
                                                       life_time=REPLAN_TICKS/FPS)
            if done_at is not None:
                control = carla.VehicleControl(brake=1., reverse=True, manual_gear_shift=True, gear=-1)
            elif flowing:
                control = flow.control(carla, flow_speed, speed)
                flow_ticks += 1
            else:
                control = tracker.control(np.array(rear), abs(speed))
                track_steer = control.steer
            self.ego.apply_control(control)
            if step % 3 == 0:
                index = step//3
                for cam in (self.image_names if args.images else ()):
                    item = images[cam]
                    # RGB as is; depth in CARLA's raw R+256G+65536B encoding, which the CAA / DDP loaders decode
                    a = np.frombuffer(item.raw_data, np.uint8).reshape(item.height, item.width, 4)[:, :, 2::-1].copy()
                    futures.append(self.pool.submit(save_png, a, os.path.join(folder, cam, '%04d.png' % index)))
                futures.append(self.pool.submit(save_json, measurement(transform, velocity, applied, images['imu'], images['gnss']),
                                                os.path.join(folder, 'measurements', '%04d.json' % index)))
                frames.append(dict(index=index, t=round(t, 3), x=transform.location.x, y=transform.location.y,
                                   z=transform.location.z, yaw=transform.rotation.yaw, pitch=transform.rotation.pitch,
                                   roll=transform.rotation.roll, vx=velocity.x, vy=velocity.y, speed=speed,
                                   steer=control.steer, throttle=control.throttle, brake=control.brake,
                                   reverse=bool(control.reverse), plan=plans-1, walkers=walker_now,
                                   controller='flow' if flowing else 'teacher',
                                   flow_action=flow.action if flowing else None,
                                   gate=getattr(flow, 'gate_log', None) if flow else None))
            step += 1
            if self.collisions:
                outcome = 'collision'
                break
            if done_at is not None and t >= done_at+SETTLE_TICKS/FPS:
                outcome = 'parked'
                break
            if t > limit:
                outcome = 'timeout'
                break
            if moved_from is None or math.hypot(centre[0]-moved_from[0], centre[1]-moved_from[1]) > 1.:
                moved_at, moved_from = t, centre
            elif done_at is None and t-moved_at > STUCK_SECONDS:
                outcome = 'stuck'
                break
        for f in futures:
            f.result()
        if flow:
            flow.end(outcome)
            flow_rows = flow.finish(folder, outcome)
        transform = self.ego.get_transform()
        error = dict(x=transform.location.x-goal[0], y=transform.location.y-goal[1],
                     yaw=(transform.rotation.yaw-math.degrees(goal[2])+180.) % 360.-180.)
        meta.update(result=dict(outcome=outcome, time=done_at, sim_seconds=round(step/FPS, 2), collisions=sorted(set(self.collisions)),
                                final_error=error, frames=len(frames), plans=plans,
                                sources={k: sources.count(k) for k in set(sources)},
                                flow=dict(activations=flow.activations, decisions=flow.decisions,
                                          seconds=round(flow_ticks/FPS, 2), rows=flow_rows) if flow else None),
                    props=[self.box(a) for a in props], walkers=[dict(self.box(w['actor']), t0=w['t0'], stop=w['stop'], x0=w['x'], y0=w['y'], vx=w['vx'], vy=w['vy'],
                                   goal=list(w['goal']), arrived=w['arrived'], waiting_ticks=w['waiting_ticks'],
                                   detour_count=w['detour_count'], navigation_status=w['navigation_status'],
                                   scene=w.get('scene', 'teacher'), model='social_force '+SFM_CONFIG['version']) for w in walkers],
                    scenario=dict(walker=setup['walker'], route_info=setup['route_info'], detour=setup['detour'],
                                  scenario_seconds=setup['scenario_seconds'],
                                  hardcore=dict(events=hard.events, triggers=hard.triggers, order=hard.order,
                                                v_sign=hard.frame.v_sign, kept_free=len(keep_free)) if hard else None))
        timing.update(drive=time.time()-began-timing['setup']-timing['spawn'], ticks=tick_seconds, teacher=teacher_seconds)
        if perceive_seconds:
            timing['perceive'] = perceive_seconds
        self.ask(dict(type='end', result=meta['result'], meta=dict(name=name)))
        timing['total'] = time.time()-began
        meta['timing'] = {k: round(v, 2) for k, v in timing.items()}
        self.finish(folder, meta, actors, frames)
        return meta

    def show(self, transform, walkers, goal, flowing=False):
        """--watch: top-down spectator over the car and the goal, each walker's SFM route (green) and goal (yellow),
        run at real time."""
        x, y = transform.location.x, transform.location.y
        cx, cy = (x+goal[0])/2, (y+goal[1])/2
        height = max(18., 1.3*math.hypot(x-goal[0], y-goal[1])+12.)
        self.world.get_spectator().set_transform(carla.Transform(carla.Location(x=cx, y=cy, z=height),
                                                                 carla.Rotation(pitch=-90., yaw=0.)))
        life = 2./FPS
        if flowing:     # the dynamic Flow has the car
            self.world.debug.draw_string(carla.Location(x=x, y=y, z=3.), 'FLOW', color=carla.Color(255, 0, 255), life_time=life)
            self.world.debug.draw_point(carla.Location(x=x, y=y, z=2.), size=.35, color=carla.Color(255, 0, 255), life_time=life)
        for w in walkers:
            route = [w['position']]+list(w['route'][w['waypoint']:])
            for a, b in zip(route[:-1], route[1:]):
                self.world.debug.draw_line(carla.Location(x=a[0], y=a[1], z=.25), carla.Location(x=b[0], y=b[1], z=.25),
                                           thickness=.05, color=carla.Color(0, 220, 0), life_time=life)
            self.world.debug.draw_point(carla.Location(x=w['goal'][0], y=w['goal'][1], z=.3), size=.15,
                                        color=carla.Color(255, 220, 0), life_time=life)
        now = time.time()
        wait = getattr(self, 'shown_at', 0.)+1./FPS-now
        if wait > 0:
            time.sleep(wait)
        self.shown_at = time.time()

    def walk_step(self, walkers, static, grid, transform, velocity, centre, goal, t):
        """One SFM step of every walker (as dynapark_paper_scenarios.update: snapshot all, then forces, then
        controls). Returns the walkers' [x, y, vx, vy] of this frame for the teacher and frames.json."""
        cfg = self.walk_cfg
        for w in walkers:
            location, v = w['actor'].get_location(), w['actor'].get_velocity()
            w['position'], w['velocity'] = (location.x, location.y), (v.x, v.y)
        yaw = math.radians(transform.rotation.yaw)
        box = self.ego.bounding_box
        c = transform.transform(box.location)
        ego = sfm_box(c.x, c.y, yaw, 2*box.extent.x, 2*box.extent.y, velocity.x, velocity.y)
        commands = []
        for w in walkers:
            if w['trigger'] and w['t0'] > t and math.hypot(centre[0]-goal[0], centre[1]-goal[1]) < w['trigger']:
                w['t0'] = t             # the walker in the slot sets off as the car comes near
            if w.get('scene') == 'ped_front_clear' and w['t0'] > t and scenes.front_clear_release(
                    w, t, centre, 3.6*math.hypot(velocity.x, velocity.y)):
                w['t0'], w['progress_time'] = t, t-2.1      # as five_dynamic_scenarios: route check at once
                w['navigation_status'] = 'walking_away_from_ego'
            position = w['position']
            exit_x = cfg.get('crossing_exit_boundary_x')
            if (w.get('crossing_direction') and exit_x and not w['arrived'] and
                    (position[0] < exit_x[0] if w['crossing_direction'] < 0 else position[0] > exit_x[1])):
                w['arrived'], w['navigation_status'] = True, 'exited_parking_area'
            if t < w['t0'] or w['arrived']:
                commands.append((0., 0.))
                continue
            if math.hypot(position[0]-w['goal'][0], position[1]-w['goal'][1]) <= cfg['arrival_radius_m']:
                w['arrived'], w['navigation_status'] = True, 'arrived'
                commands.append((0., 0.))
                continue
            if w['navigation_status'] == 'waiting_to_start':
                w['navigation_status'] = 'walking'
            update_detour(w, ego, grid, cfg, t, others=walkers)
            route, k = w['route'], w['waypoint']
            if (k < len(route)-1 and math.hypot(position[0]-route[k][0], position[1]-route[k][1]) < .55
                    and w.get('detour_grid', grid).segment_clear(position, route[k+1])):
                w['waypoint'] += 1
            terms = acceleration_terms(w, route[w['waypoint']], static, walkers, ego, cfg)
            command = integrate_velocity(w['velocity'], terms, 1./FPS, w['max_speed'], cfg)
            w['waiting_ticks'] += math.hypot(*command) < .15
            commands.append(command)
        for w, command in zip(walkers, commands):
            walk(w['actor'], command)
        return [[w['position'][0], w['position'][1], w['velocity'][0], w['velocity'][1]] for w in walkers]

    def finish(self, folder, meta, actors, frames):
        for a in actors:
            a.destroy()
        self.ego.apply_control(carla.VehicleControl(brake=1.))
        self.tick()
        if not os.path.isdir(folder):
            os.makedirs(folder)
        if frames is not None:      # the 10 Hz trace also of a failed episode (small; the images go)
            with open(os.path.join(folder, 'frames.json'), 'w') as f:
                json.dump(frames, f)
        if meta['result']['outcome'] != 'parked':
            for name in self.image_names+['measurements']:
                shutil.rmtree(os.path.join(folder, name), ignore_errors=True)
        elif frames is not None:
            # CAA parking_goal: the slot centre and the car's final heading (data_generator.check_goal)
            os.makedirs(os.path.join(folder, 'parking_goal'), exist_ok=True)
            save_json(dict(x=meta['goal'][0], y=meta['goal'][1], yaw=self.ego.get_transform().rotation.yaw),
                      os.path.join(folder, 'parking_goal', '0001.json'))
        with open(os.path.join(folder, 'meta.json'), 'w') as f:
            json.dump(meta, f, indent=1)

    def close(self):
        # the server stays in synchronous mode: it then waits for a client instead of rendering freely, which
        # would take the GPU from the teachers of the other workers
        for a in self.sensors+[self.ego]:
            a.destroy()
        self.link.close()
        if self.flow:
            self.flow.close()


def sfm_box(x, y, yaw, length, width, vx=0., vy=0.):
    """A box as social_force takes it (yaw in radians, half sizes, velocity for the ego)."""
    return dict(x=x, y=y, yaw=yaw, half_length=length/2., half_width=width/2., vx=vx, vy=vy)


def walk(actor, velocity):
    speed = math.hypot(*velocity)
    direction = (velocity[0]/speed, velocity[1]/speed) if speed > 1e-9 else (0., 0.)
    actor.apply_control(carla.WalkerControl(direction=carla.Vector3D(x=direction[0], y=direction[1], z=0.),
                                            speed=speed, jump=False))


def measurement(transform, velocity, control, imu, gnss):
    """One measurements/NNNN.json of the CAA data generator (data_generator.save_unit_data): speeds in km/h,
    controls as applied, IMU and GNSS readings."""
    return {'x': transform.location.x, 'y': transform.location.y, 'z': transform.location.z,
            'pitch': transform.rotation.pitch, 'yaw': transform.rotation.yaw, 'roll': transform.rotation.roll,
            'speed': 3.6*math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2),
            'speed_x': 3.6*velocity.x, 'speed_y': 3.6*velocity.y, 'speed_z': 3.6*velocity.z,
            'Throttle': control.throttle, 'Steer': control.steer, 'Brake': control.brake, 'Reverse': control.reverse,
            'Hand brake': control.hand_brake, 'Manual': control.manual_gear_shift,
            'Gear': {-1: 'R', 0: 'N'}.get(control.gear, control.gear),
            'acc_x': imu.accelerometer.x, 'acc_y': imu.accelerometer.y, 'acc_z': imu.accelerometer.z,
            'gyr_x': imu.gyroscope.x, 'gyr_y': imu.gyroscope.y, 'gyr_z': imu.gyroscope.z, 'compass': imu.compass,
            'lat': gnss.latitude, 'lon': gnss.longitude}


def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)


def save_png(array, path):
    from PIL import Image
    Image.fromarray(array).save(path, compress_level=6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--server-port', type=int, default=6301)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--first', type=int, default=0, help='first episode number of the schedule')
    parser.add_argument('--count', type=int, default=1, help='episodes this worker runs')
    parser.add_argument('--slots', default='all', choices=('all', 'train', 'benchmark'),
                        help='all 32 slots of rows 2 and 3, the 16 between the benchmark slots, or the 16 benchmark slots')
    parser.add_argument('--weather', default='same', choices=sorted(WEATHERS))
    parser.add_argument('--renderer', default='opengl')
    parser.add_argument('--depth', type=int, default=1, help='also save the 6 depth cameras (depth_*) the CAA / DDP depth loss uses')
    parser.add_argument('--watch', action='store_true', help='demo: top-down view over the car in the server window, real time')
    parser.add_argument('--walkers', default='teacher', choices=('teacher', 'ped_crossing', 'ped_front_clear', 'training', 'hardcore', 'head_on', 'nonyield', 'behind_demo', 'cases'),
                        help="the teacher's walker scenario, or a pedestrian scene of the dynamic Flow's training (training: "
                             "either, per episode); the latter two come without props and the teacher plans without them")
    parser.add_argument('--flow-port', type=int, default=0, help='dynamic-avoidance Flow worker (supervisor.py); 0: none')
    parser.add_argument('--flow-authkey', default='lot-flow')
    parser.add_argument('--images', type=int, default=1, help='0: the camera frames are not written')
    parser.add_argument('--last', type=int, default=None, help='stop before this episode number (resuming a round: '
                        'episodes with a meta.json are passed over, so --count alone would run on into the next round)')
    args = parser.parse_args()
    args.slots = dict(all=SLOTS, train=[s for s in SLOTS if s % 2], benchmark=[s for s in SLOTS if s % 2 == 0])[args.slots]
    os.makedirs(args.out, exist_ok=True)
    collector = Collector(args)
    try:
        n, attempt, done, skips = args.first+args.worker, 0, 0, 0
        while done < args.count and (args.last is None or n < args.last):
            meta = collector.episode(n, attempt)
            if args.walkers == 'cases' and meta is not None and meta['result']['outcome'] == 'skipped' and attempt < 2:
                attempt += 1        # the cell again with another seed: the schedule stays even
            else:
                n, attempt = n+args.workers, 0
            if meta is None:
                continue
            skipped = meta['result']['outcome'] == 'skipped'
            done += not skipped                     # --count episodes that were driven
            print(meta['name'], meta['result']['outcome'], meta['result'].get('sim_seconds'), meta.get('timing'), flush=True)
            # a few skips are normal (no route); a run of them means a broken server, and spinning through
            # hundreds of spawn-destroy cycles in a minute is what breaks the CARLA server further
            # ... only skips that point at the server (not a scene without room for its props or a route)
            reason = str(meta['result'].get('reason'))
            skips = skips+1 if skipped and not reason.startswith(('no placement', 'no route')) else 0
            if skips >= 5:
                raise RuntimeError('5 skipped episodes in a row, last: {}'.format(meta['result'].get('reason')))
    finally:
        collector.close()


if __name__ == '__main__':
    main()
