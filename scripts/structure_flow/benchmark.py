"""Static Town04_Opt parking benchmark of the DDP evaluator, driven by plan_server.

Runs under Python 3.7 with the CARLA 0.9.11 egg. The protocol is that of
carla_parking_eva.py, data_generation/network_evaluator.py, world.py and
agent/parking_agent.py of the DDP release: a fresh map without the ParkedVehicles layer,
16 goal slots (index 16, 18, ... of parking_vehicle_locations_Town04), 6 starts each at
goal y-8..+8, seed 66 plus the slot number for the parked cars, 30 Hz, a 1 s hold, a 30 s
limit, out of bounds for 10 s, success after two stopped reverse frames in the goal, and
the three weathers. The frame order is the one of game_loop: the agent acts on the sensor
frame of the previous tick. The target is the one the DDP agent makes, with the position
dead reckoned at 10 Hz from velocity and IMU, and a goal yaw of 180 in row 2 and 0 in row 3.
"""
from __future__ import print_function
import argparse
import csv
import json
from multiprocessing.connection import Client
import math
import os
import queue
import random
import sys
import time
import numpy as np

EGG = os.environ.get('CARLA_EGG', 'carla-0.9.11-py3.7-win-amd64.egg')
if not os.path.isfile(EGG):
    # A missing egg would otherwise fall through to any carla package pip installed.
    sys.exit('CARLA_EGG must point to the CARLA 0.9.11 egg, no such file: {}'.format(EGG))
sys.path.insert(0, EGG)
import carla

CARLA_VERSION = '0.9.11'

CAMERAS = ('camera_front', 'camera_front_left', 'camera_front_right',
           'camera_back', 'camera_back_left', 'camera_back_right')
MOUNTS = dict(camera_front=(2.36, 0., 1.5, 0.), camera_front_left=(2.36, -.792, 1.5, -55.),
              camera_front_right=(2.36, .792, 1.5, 55.), camera_back=(-2.36, 0., 1.55, -180.),
              camera_back_left=(-2.36, -.792, 1.55, -110.), camera_back_right=(-2.36, .792, 1.55, 110.))
ROWS = (298.5, 290.9, 280.0, 272.5)
SLOT_Y = (-235.73, -232.73, -229.53, -226.43, -223.43, -220.23, -217.23, -214.03,
          -210.73, -207.30, -204.23, -201.03, -198.03, -194.90, -191.53, -188.20)
SPOTS = [(x, y) for x in ROWS for y in SLOT_Y]
GOAL_INDICES = list(range(16, 48, 2))
SLOT_NAMES = ['{}-{}'.format(row, index) for row in (2, 3) for index in range(1, 16, 2)]
BOUND = dict(x_min=264., x_max=304., y_min=-241., y_max=-178.)
EGO_X, EGO_Z = 285.600006, .32682
FPS = 30
HOLD_FRAMES, TOTAL_FRAMES, OUT_FRAMES, NEAR_FRAMES, IN_GOAL_FRAMES = 30, 30*FPS, 10*FPS, 2*FPS, 1
X_TOL, Y_TOL, YAW_TOL = 1., .6, 10.
SPOT_Z = .3
PROCESS_FRAMES, TRACK_DT = 3, .1    # the agent reads its sensors at 10 Hz
TICK_SECONDS = 120.     # wait per world tick; the 10 s default is too short for OpenGL servers under WSL
VELOCITY_POINT = .335       # metres ahead of the actor origin, a fitted value
# cloudiness, sun azimuth, sun altitude, fog, wetness, precipitation, deposits
WEATHERS = dict(same=(15., 0., 75., 0., 0., 0., 0.), mild=(35., 20., 25., 2., 20., 30., 20.),
                large=(45., 30., 5., 3., 30., 45., 30.))
REAR_OFFSET = 1.5
STEP_SECONDS = .4       # one model step: 4 frames at 10 Hz
# Steering command against driven curvature, measured on e2e_dataset.
CURVATURE = (0., .072, .151, .24, .345, .47)
STEER = (0., .2, .4, .6, .8, 1.)


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class SpeedControl:
    """Throttle and brake for a target speed, for slow driving of vehicle.tesla.model3 (CARLA 0.9.11).

    Measured on Town04_Opt: from rest the car needs throttle about .25 forward and .15 in reverse to move,
    and once rolling forward even .12 settles near 1 m/s, so slow forward driving is held with the brake.
    One signed command: positive is throttle, negative is brake. A launch kick starts the car from rest; it
    grows by LAUNCH_RAMP per second while the car does not move, since after a stop the car sometimes stands
    for seconds under the plain kick."""
    LAUNCH = {1: .15, -1: .15}
    LAUNCH_RAMP, LAUNCH_MAX = .4, .75
    HOLD = {1: .07, -1: .07}
    STOP = .05

    def __init__(self, kp=.3, ki=.35, brake_gain=.8):
        self.kp, self.ki, self.brake_gain = kp, ki, brake_gain
        self.integral, self.stuck = 0., 0

    def __call__(self, target, speed, gear):
        """Returns (throttle, brake) for the unsigned target and measured speeds in m/s."""
        if target < self.STOP:
            self.integral, self.stuck = 0., 0
            return 0., 1.
        error = target-speed
        if speed < .02 and error > 0:
            self.stuck += 1
            return float(min(self.LAUNCH[gear]+self.LAUNCH_RAMP*self.stuck/FPS, self.LAUNCH_MAX)), 0.
        self.stuck = 0
        self.integral = float(np.clip(self.integral+error/FPS, -.4, .4))
        command = self.HOLD[gear]+self.kp*error+self.ki*self.integral
        if command >= 0:
            return float(min(command, .75)), 0.
        return 0., float(min(1., -command*self.brake_gain))


class Tracker:
    """Pure pursuit on the rear axle, with gear changes only at rest.

    The steering limits are those of the DDP controller (agents/navigation/controller.py,
    VehiclePIDController with the defaults parking_agent.py keeps): every 30 Hz tick the command
    moves at most `steer_rate` from the last one, then is clamped to +-`max_steer`.
    max_steer=1 with steer_rate=0 gives an unlimited tracker.

    speed_mode 'legacy' is the planned speed with a `min_speed` floor
    and a fixed .3 throttle offset, which drives a slow plan at 0.5 to 0.7 m/s. 'schedule' follows the plan's
    timing instead: the reference arc length at the time since the plan arrived (the plan's speeds, at least
    0.1 m/s as in the decoder), the target speed is the planned one plus LAG_GAIN times the lag behind it,
    and SpeedControl turns it into throttle or brake."""
    LAG_GAIN = .8

    WHEELBASE = 2.875       # vehicle.tesla.model3
    STANLEY_GAIN, STANLEY_SOFT = 1., .5
    MPC_STEPS, MPC_DT = 20, .1
    MPC_WEIGHTS = dict(lateral=1., heading=.5, curvature=.05, change=1.)

    def __init__(self, lookahead=1.2, stop_decel=.8, min_speed=.45, max_steer=.8, steer_rate=.1, speed_mode='legacy',
                 steering='pursuit'):
        self.lookahead, self.stop_decel, self.min_speed = lookahead, stop_decel, min_speed
        self.max_steer, self.steer_rate = max_steer, steer_rate
        self.speed_mode, self.steering = speed_mode, steering
        self.last_curvature = 0.
        self.speed_control = SpeedControl()
        self.path = None
        self.gear, self.integral, self.past_steer = 1, 0., 0.
        self.ticks = 0

    def set_path(self, rear, rows):
        """rows: rear-axle path in the frame of `rear` (x, y, yaw, speed, phase)."""
        rows = np.asarray(rows, float)
        c, s = math.cos(rear[2]), math.sin(rear[2])
        x = rear[0]+c*rows[:, 0]-s*rows[:, 1]
        y = rear[1]+s*rows[:, 0]+c*rows[:, 1]
        self.path = np.stack((x, y), 1)
        self.speed, self.phase = np.abs(rows[:, 3]), rows[:, 4].astype(int)
        self.gears = np.sign(rows[:, 3])
        for phase in set(self.phase.tolist()):
            part = self.phase == phase
            sign = np.sign(self.gears[part].sum()) or 1
            self.gears[part] = sign
        step = np.linalg.norm(np.diff(np.vstack((rear[:2], self.path)), axis=0), axis=1)
        self.arc = np.cumsum(step)
        self.index = 0
        middle = np.r_[self.speed[0], (self.speed[1:]+self.speed[:-1])/2]
        self.times = np.cumsum(step/np.maximum(middle, .1))
        self.ticks = 0
        # path heading (world) and the rear-axle curvature with yaw rate = signed speed * curvature
        self.yaw = rear[2]+rows[:, 2]
        turn = np.r_[0., np.arctan2(np.sin(np.diff(self.yaw)), np.cos(np.diff(self.yaw)))]
        ds = np.r_[1., np.diff(self.arc)]
        kappa = np.where(ds > 1e-3, turn/np.maximum(ds, 1e-3), 0.)*self.gears
        kernel = np.ones(5)/5
        self.kappa = np.convolve(np.pad(kappa, 2, mode='edge'), kernel, mode='valid')

    def clear(self):
        self.path = None

    def errors(self, point, yaw, index):
        """Signed lateral offset of `point` from the path at `index` (CARLA y side positive) and heading error."""
        py = self.yaw[index]
        dx, dy = point[0]-self.path[index, 0], point[1]-self.path[index, 1]
        return -math.sin(py)*dx+math.cos(py)*dy, wrap(yaw-py)

    def stanley(self, rear, speed, last):
        """Stanley at the front axle when driving forward. In reverse the same law runs at the rear axle, which
        leads, with the heading turned around; that virtual car steers at its back, so the angle flips sign."""
        gear = self.gear
        if gear > 0:
            front = rear[:2]+self.WHEELBASE*np.array([math.cos(rear[2]), math.sin(rear[2])])
            ahead = self.path+self.WHEELBASE*np.stack((np.cos(self.yaw), np.sin(self.yaw)), 1)
            window = np.arange(self.index, last+1)
            index = int(window[np.argmin(np.linalg.norm(ahead[window]-front, axis=1))])
            py = self.yaw[index]
            dx, dy = front[0]-ahead[index, 0], front[1]-ahead[index, 1]
            lateral, heading = -math.sin(py)*dx+math.cos(py)*dy, wrap(rear[2]-py)
            angle = -heading-math.atan2(self.STANLEY_GAIN*lateral, self.STANLEY_SOFT+speed)
        else:
            # without the path curvature as feed-forward the reverse arc into the slot leaves about L*kappa/k
            # (1 m) of lateral error
            lateral, heading = self.errors(rear[:2], rear[2], self.index)
            angle = heading+math.atan2(self.STANLEY_GAIN*-lateral, self.STANLEY_SOFT+speed)
            return float(self.kappa[self.index])+math.tan(float(np.clip(angle, -1.2, 1.2)))/self.WHEELBASE
        return math.tan(float(np.clip(angle, -1.2, 1.2)))/self.WHEELBASE

    def mpc(self, rear, speed, last):
        """Linear MPC on the rear-axle path errors: e' = v*heading, heading' = v*(curvature - path curvature),
        v signed. Least squares over MPC_STEPS steps of MPC_DT on curvature, its change and the errors;
        the first curvature is applied (the steering clamp and rate are applied after, in control)."""
        n, dt, w = self.MPC_STEPS, self.MPC_DT, self.MPC_WEIGHTS
        v = self.gear*max(speed, .3)
        lateral, heading = self.errors(rear[:2], rear[2], self.index)
        s = self.arc[self.index]+abs(v)*dt*np.arange(n)
        ref = np.interp(np.minimum(s, self.arc[last]), self.arc[:last+1], self.kappa[:last+1])
        # states after each step are affine in the inputs u: x_k = F_k x0 + G_k u + H_k ref
        rows, target = [], []
        e_lin, h_lin = np.zeros(n), np.zeros(n)     # dependence on u, built step by step
        de, dh = np.zeros((n, n)), np.zeros((n, n))
        e, h = lateral, heading
        ce, ch = np.zeros(n), np.zeros(n)
        for k in range(n):
            # e_{k+1} = e_k + v dt h_k ; h_{k+1} = h_k + v dt (u_k - ref_k)
            ce, ch = ce+v*dt*ch, ch.copy()
            ch[k] += v*dt
            e, h = e+v*dt*h, h-v*dt*ref[k]
            rows.append(math.sqrt(w['lateral'])*ce)
            target.append(-math.sqrt(w['lateral'])*e)
            rows.append(math.sqrt(w['heading'])*ch)
            target.append(-math.sqrt(w['heading'])*h)
        eye = np.eye(n)
        rows.extend(math.sqrt(w['curvature'])*eye)
        target.extend(math.sqrt(w['curvature'])*ref)
        change = eye-np.eye(n, k=-1)
        rows.extend(math.sqrt(w['change'])*change)
        first = np.zeros(n)
        first[0] = self.last_curvature
        target.extend(math.sqrt(w['change'])*first)
        u = np.linalg.lstsq(np.array(rows), np.array(target), rcond=None)[0]
        return float(u[0])

    def remaining(self):
        return 0. if self.path is None else float(self.arc[-1]-self.arc[self.index])

    def control(self, rear, speed):
        """Returns a VehicleControl for the rear-axle pose and the unsigned speed in m/s, called once per tick."""
        command = self.pursue(rear, speed)
        self.ticks += 1
        if self.speed_mode == 'schedule':
            # CARLA's automatic gearbox waits in neutral for about 1.9 s at a standstill, then drops into gear
            # at 2 m/s; a fixed gear starts the car at once and smoothly
            command.manual_gear_shift = True
            command.gear = -1 if self.gear < 0 else 1
        steer = command.steer
        if self.steer_rate > 0:
            steer = min(max(steer, self.past_steer-self.steer_rate), self.past_steer+self.steer_rate)
        steer = min(max(steer, -self.max_steer), self.max_steer)
        self.past_steer = command.steer = steer
        return command

    def pursue(self, rear, speed):
        hold = carla.VehicleControl(throttle=0., brake=1., steer=0., reverse=self.gear < 0)
        if self.path is None:
            return hold
        phase = self.phase[self.index]
        last = int(np.flatnonzero(self.phase == phase)[-1])
        window = np.arange(self.index, min(self.index+60, last+1))
        self.index = int(window[np.argmin(np.linalg.norm(self.path[window]-rear[:2], axis=1))])
        wanted = int(self.gears[self.index])
        left = float(self.arc[last]-self.arc[self.index])
        final_phase = last == len(self.path)-1
        rolling_end = final_phase and self.speed[-1] > .3
        if left < .08 and not rolling_end:
            if not final_phase and speed < .05:
                self.index = last+1
                self.integral = 0.
                return hold
            if final_phase:
                self.path = None
            return hold
        if wanted != self.gear:
            if speed > .05:
                return hold
            self.gear, self.integral = wanted, 0.
        if self.steering == 'stanley':
            curvature = self.stanley(rear, speed, last)
        elif self.steering == 'mpc':
            curvature = self.mpc(rear, speed, last)
        else:
            reach = self.arc[self.index]+self.lookahead
            point = int(min(np.searchsorted(self.arc, reach), last))
            dx, dy = self.path[point]-rear[:2]
            c, s = math.cos(rear[2]), math.sin(rear[2])
            ahead, side = c*dx+s*dy, -s*dx+c*dy
            distance = max(ahead*ahead+side*side, .25)
            curvature = 2*side/distance
        self.last_curvature = curvature
        steer = math.copysign(float(np.interp(abs(curvature), CURVATURE, STEER)), curvature)
        if self.speed_mode == 'schedule':
            elapsed = self.ticks/FPS
            planned = min(float(np.interp(elapsed, self.times, self.arc)), float(self.arc[last]))
            target = float(np.interp(elapsed, self.times, self.speed))+self.LAG_GAIN*(planned-float(self.arc[self.index]))
            target = min(max(target, 0.), float(self.speed[self.index:last+1].max())+.3)
            if not rolling_end:
                target = min(target, math.sqrt(2*self.stop_decel*left)+.05)
            throttle, brake = self.speed_control(target, speed, self.gear)
            return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer, reverse=self.gear < 0)
        target = max(float(self.speed[self.index]), self.min_speed)
        if not rolling_end:
            target = min(target, math.sqrt(2*self.stop_decel*left)+.1)
        error = target-speed
        self.integral = float(np.clip(self.integral+error/FPS, -.5, .5))
        drive = .3+.5*error+.4*self.integral
        if error < -.25:
            return carla.VehicleControl(throttle=0., brake=float(min(1., -.6*error)), steer=steer, reverse=self.gear < 0)
        return carla.VehicleControl(throttle=float(np.clip(drive, 0., .75)), brake=0., steer=steer, reverse=self.gear < 0)


class Bench:
    def __init__(self, args):
        self.args = args
        self.client = carla.Client('127.0.0.1', args.port)
        self.client.set_timeout(60.)
        versions = self.client.get_client_version(), self.client.get_server_version()
        if versions != (CARLA_VERSION, CARLA_VERSION):
            sys.exit('The benchmark needs CARLA {}, found client {} and server {}'.format(CARLA_VERSION, *versions))
        self.world = self.client.load_world(args.map)
        # The map bakes in parked cars, two of them inside goal slots 2-7 and 3-3.
        self.world.unload_map_layer(carla.MapLayer.ParkedVehicles)
        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 1./FPS
        settings.synchronous_mode = True
        self.world.apply_settings(settings)
        library = self.world.get_blueprint_library()
        self.library = library
        start = carla.Transform(carla.Location(x=EGO_X, y=-243.729996, z=EGO_Z), carla.Rotation(yaw=90.))
        self.ego = self.world.spawn_actor(library.find('vehicle.tesla.model3'), start)
        self.sensors, self.queues = [], {}
        self.collided, self.other = False, None
        collision = self.world.spawn_actor(library.find('sensor.other.collision'), carla.Transform(), attach_to=self.ego)
        collision.listen(self.on_collision)
        self.sensors.append(collision)
        for name in CAMERAS:
            blueprint = library.find('sensor.camera.rgb')
            blueprint.set_attribute('image_size_x', '400')
            blueprint.set_attribute('image_size_y', '300')
            blueprint.set_attribute('fov', '100')
            x, y, z, yaw = MOUNTS[name]
            self.attach(name, blueprint, carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(yaw=yaw)))
        self.attach('imu', library.find('sensor.other.imu'), carla.Transform())
        self.npcs = []
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(carla.Transform(carla.Location(x=283.825165, y=-210.039487, z=35.), carla.Rotation(pitch=-90.)))
        cloud, azimuth, altitude, fog, wet, rain, deposits = WEATHERS[args.weather]
        self.world.set_weather(carla.WeatherParameters(
            cloudiness=cloud, precipitation=rain, precipitation_deposits=deposits, sun_azimuth_angle=azimuth,
            sun_altitude_angle=altitude, fog_density=fog, wetness=wet))
        self.data = None
        self.link = Client(('127.0.0.1', args.server_port), authkey=b'structure-flow-benchmark')

    def attach(self, name, blueprint, mount):
        sensor = self.world.spawn_actor(blueprint, mount, attach_to=self.ego)
        self.queues[name] = queue.Queue()
        sensor.listen(self.queues[name].put)
        self.sensors.append(sensor)

    def on_collision(self, event):
        self.collided = True
        self.other = event.other_actor.type_id

    def next_slot(self, seed, goal_index, start):
        """start_next_slot of the DDP evaluator: the ego leaves the old goal before the new cars spawn."""
        for actor in self.npcs:
            actor.destroy()
        self.npcs = []
        self.ego.set_transform(start)
        free = self.spawn_npcs(seed, goal_index)
        self.ego.set_transform(start)
        self.ego.apply_control(carla.VehicleControl())
        self.ego.set_target_velocity(carla.Vector3D())
        return free

    def spawn_npcs(self, seed, goal_index):
        """init_static_npc of the DDP world, in the same random order."""
        random.seed(seed)
        spots = list(SPOTS)
        goal = spots[goal_index]
        count = random.randint(len(spots)//3, len(spots)-1)
        order = list(spots)
        random.shuffle(order)
        blueprints = [b for b in self.library.filter('vehicle') if int(b.get_attribute('number_of_wheels')) == 4]
        free = []
        for spot in order[:count]:
            if spot == goal:
                free.append(spot)
                continue
            transform = carla.Transform(carla.Location(x=spot[0], y=spot[1], z=SPOT_Z),
                                        carla.Rotation(yaw=random.choice((180., 0.))))
            actor = self.world.try_spawn_actor(random.choice(blueprints), transform)
            if actor is None:
                free.append(spot)
            else:
                actor.set_simulate_physics(False)
                self.npcs.append(actor)
        return free+order[count:]

    def read(self, frame):
        """The frame of the evaluator: vehicle state and the sensors of this very tick."""
        data = dict(transform=self.ego.get_transform(), velocity=self.ego.get_velocity(), control=self.ego.get_control(),
                    spin=self.ego.get_angular_velocity())
        for name, stream in self.queues.items():
            item = stream.get(timeout=20.)
            while item.frame < frame:
                item = stream.get(timeout=20.)
            data[name] = item
        velocity = data['velocity']
        data['speed'] = math.sqrt(velocity.x**2+velocity.y**2+velocity.z**2)
        return data

    def track(self, goal, tracked):
        """Target in the vehicle frame as the DDP agent makes it: position dead reckoned at 10 Hz, yaw measured."""
        data = self.data
        location, velocity, accelerometer = data['transform'].location, data['velocity'], data['imu'].accelerometer
        heading = data['transform'].rotation.yaw
        if not tracked:
            tracked[:] = [location.x, location.y]*2
        yaw = math.radians(heading)
        c, s = math.cos(yaw), math.sin(yaw)
        tracked[0] += velocity.x*TRACK_DT+.5*(accelerometer.x*c+accelerometer.y*s)*TRACK_DT**2
        tracked[1] += velocity.y*TRACK_DT+.5*(-accelerometer.x*s+accelerometer.y*c)*TRACK_DT**2
        # get_velocity belongs to a point ahead of the actor origin, so a turn shifts the plain sum by about 0.3 m.
        turn = math.radians(data['spin'].z)*VELOCITY_POINT
        tracked[2] += (velocity.x+turn*s)*TRACK_DT
        tracked[3] += (velocity.y-turn*c)*TRACK_DT
        self.motion.append((location.x, location.y, heading, velocity.x, velocity.y, accelerometer.x, accelerometer.y,
                            data['spin'].z, tracked[0], tracked[1], tracked[2], tracked[3]))
        known = dict(gt=[location.x, location.y], corrected=tracked[2:], tracking=tracked)[self.args.target]
        self.pose = [known[0], known[1], heading]      # what the agent believes, for trackers in the planner
        dx, dy = goal['x']-known[0], goal['y']-known[1]
        difference = goal['yaw']-heading
        difference = difference-360 if difference > 180 else difference+360 if difference < -180 else difference
        return [dx*c+dy*s, -dx*s+dy*c, difference]

    def plan(self, target):
        data = self.data
        images = np.stack([np.frombuffer(data[name].raw_data, np.uint8).reshape(300, 400, 4)[:, :, 2::-1]
                           for name in CAMERAS])
        control = data['control']
        state = dict(target=target, speed=-data['speed'] if control.reverse else data['speed'],
                     steer=control.steer, reverse=bool(control.reverse),
                     episode=self.label, time=self.step/FPS, pose=self.pose)
        self.link.send_bytes(json.dumps(state).encode())
        self.link.send_bytes(np.ascontiguousarray(images).tobytes())
        return json.loads(self.link.recv_bytes().decode())

    def episode(self, start, goal, free, name):
        """One pass of the DDP loop per frame: world tick, agent on the previous frame, evaluator on this one."""
        args = self.args
        tracker, tracked = Tracker(max_steer=args.max_steer, steer_rate=args.steer_rate, speed_mode=args.speed_mode,
                                   steering=args.steering), []
        counters = dict(in_goal=0, near_goal=0, near_other=0, outside=0)
        trace, plans, seconds = [], [], []
        next_plan, outcome, step, total = HOLD_FRAMES, None, -1, 0
        self.other, self.motion, self.label = None, [], name
        while outcome is None:
            frame = self.world.tick(TICK_SECONDS)
            step += 1
            self.step = step
            if step < HOLD_FRAMES:
                self.ego.apply_control(carla.VehicleControl())
                self.ego.set_transform(start)
            else:
                if step % PROCESS_FRAMES == 0:
                    target = self.track(goal, tracked)
                    if step >= next_plan or (tracker.path is not None and tracker.remaining() < .1 and tracker.speed[-1] > .3):
                        reply = self.plan(target)
                        seconds.append(reply['seconds'])
                        rear = rear_pose(self.data['transform'])
                        plans.append(dict(frame=step, h=reply.get('h'), stop=reply['stop'], blocked=reply['blocked'],
                                          safe=reply['safe'], p=[round(p, 3) for p in reply['probabilities']], target=target,
                                          mask=reply.get('mask'), open=reply.get('open')))
                        if args.save_paths:
                            plans[-1].update(rear=rear.tolist(), path=reply.get('path'))
                        if reply['stop']:
                            tracker.clear()
                            next_plan = step+FPS
                        else:
                            tracker.set_path(rear, reply['path'])
                            next_plan = step+int(args.replan_steps*STEP_SECONDS*FPS)
                speed = self.ego.get_velocity()
                speed = math.sqrt(speed.x**2+speed.y**2+speed.z**2)
                if tracker.path is None and step+1 < next_plan and speed < .02:
                    next_plan = min(next_plan, step+FPS)
                self.ego.apply_control(tracker.control(rear_pose(self.ego.get_transform()), speed))
            total += 1
            self.data = self.read(frame)
            outcome, final = self.judge(self.data, goal, free, counters, total)
            location, control = self.data['transform'].location, self.data['control']
            trace.append((total, location.x, location.y, self.data['transform'].rotation.yaw,
                          -self.data['speed'] if control.reverse else self.data['speed'], control.steer, control.throttle, control.brake))
        np.save(os.path.join(args.out, name+'.npy'), np.array(trace, np.float32))
        if args.save_motion:
            np.save(os.path.join(args.out, name+'.motion.npy'), np.array(self.motion, np.float64))
        with open(os.path.join(args.out, name+'.json'), 'w') as stream:
            json.dump(dict(outcome=outcome, plans=plans, final=final, other=self.other), stream)
        return outcome, total, final, seconds

    def judge(self, data, goal, free, counters, total):
        """tick and eva_check_goal of the DDP evaluator. Distances are 3D to the spots at z = 0.3."""
        location, yaw, control = data['transform'].location, data['transform'].rotation.yaw, data['control']
        x, y = location.x, location.y
        distance_to = lambda p: math.sqrt((x-p[0])**2+(y-p[1])**2+(location.z-SPOT_Z)**2)
        closest = min(free, key=distance_to)
        yaw_error = min(abs(yaw), 180-abs(yaw))
        final = dict(x=x, y=y, z=location.z, yaw=yaw, distance=distance_to(closest), yaw_error=yaw_error,
                     goal_distance=math.hypot(x-goal['x'], y-goal['y']))
        if self.collided:
            self.collided = False
            return 'collision', final
        if total > TOTAL_FRAMES:
            return 'timeout', final
        outside = x < BOUND['x_min'] or x > BOUND['x_max'] or y < BOUND['y_min'] or y > BOUND['y_max']
        counters['outside'] = counters['outside']+1 if outside else 0
        if counters['outside'] > OUT_FRAMES:
            return 'outbound', final
        if not (control.throttle == 0. and data['speed']*3.6 < 1e-3 and control.reverse):
            counters.update(in_goal=0, near_goal=0, near_other=0)
            return None, final
        target = closest[0] == goal['x'] and closest[1] == goal['y']
        dx, dy = abs(x-closest[0]), abs(y-closest[1])
        if dx <= X_TOL and dy <= Y_TOL and yaw_error <= YAW_TOL:
            counters['in_goal'] += 1
        if counters['in_goal'] > IN_GOAL_FRAMES:
            return ('target_success' if target else 'non_target_success'), final
        if X_TOL < dx <= 2*X_TOL or Y_TOL < dy <= 2*Y_TOL or YAW_TOL < yaw_error <= 2*YAW_TOL:
            counters['near_goal' if target else 'near_other'] += 1
        if counters['near_goal'] > NEAR_FRAMES:
            return 'target_fail', final
        if counters['near_other'] > NEAR_FRAMES:
            return 'non_target_fail', final
        return None, final

    def close(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)
        for actor in self.sensors+self.npcs+[self.ego]:
            actor.destroy()
        self.link.close()


def rear_pose(transform):
    yaw = math.radians(transform.rotation.yaw)
    location = transform.location
    return np.array([location.x-REAR_OFFSET*math.cos(yaw), location.y-REAR_OFFSET*math.sin(yaw), yaw])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--slots', default='', help='comma separated names, all 16 by default')
    parser.add_argument('--starts', default='0,1,2,3,4,5')
    parser.add_argument('--weather', choices=sorted(WEATHERS), default='same')
    parser.add_argument('--replan-steps', type=float, default=4)
    parser.add_argument('--max-steer', type=float, default=.8, help='steering clamp of the DDP controller')
    parser.add_argument('--steer-rate', type=float, default=.1,
                        help='largest steering change per tick of the DDP controller, 0 for none')
    parser.add_argument('--speed-mode', choices=('legacy', 'schedule'), default='schedule',
                        help='legacy: planned speed with a 0.45 m/s floor; schedule: follow the plan timing (Tracker)')
    parser.add_argument('--steering', choices=('pursuit', 'stanley', 'mpc'), default='mpc',
                        help='the steering law (Tracker): pure pursuit, Stanley or linear MPC')
    parser.add_argument('--save-paths', action='store_true', help='keep every planned path in the episode json')
    parser.add_argument('--save-motion', action='store_true', help='keep the 10 Hz dead reckoning inputs of every episode')
    parser.add_argument('--target', choices=('tracking', 'gt', 'corrected'), default='tracking',
                        help='tracking and gt are target types of the DDP agent, corrected removes the velocity point offset')
    parser.add_argument('--seed', type=int, default=66)
    parser.add_argument('--map', default='Town04_Opt')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--server-port', type=int, default=6100)
    args = parser.parse_args()
    if not os.path.isdir(args.out):
        os.makedirs(args.out)
    wanted = [name for name in args.slots.split(',') if name] or SLOT_NAMES
    starts = [int(value) for value in args.starts.split(',')]
    bench = Bench(args)
    rows_path = os.path.join(args.out, 'episodes.csv')
    fields = ('epoch', 'slot', 'start', 'outcome', 'seconds', 'position_error', 'yaw_error', 'goal_distance', 'inference_ms')
    done = set()
    if os.path.exists(rows_path):
        with open(rows_path) as stream:
            done = {(int(r['epoch']), r['slot'], int(r['start'])) for r in csv.DictReader(stream)}
    try:
        with open(rows_path, 'a', newline='') as stream:
            writer = csv.DictWriter(stream, fields)
            if not done:
                writer.writeheader()
            for epoch in range(args.epochs):
                for number, (name, index) in enumerate(zip(SLOT_NAMES, GOAL_INDICES)):
                    left = [s for s in starts if (epoch, name, s) not in done]
                    if name not in wanted or not left:
                        continue
                    spot = SPOTS[index]
                    goal = dict(x=spot[0], y=spot[1], yaw=180. if index <= 30 else 0.)    # the eva_parking_goal property
                    free = None
                    for start_index in left:
                        y = spot[1]-8+3.2*start_index
                        start = carla.Transform(carla.Location(x=EGO_X, y=y, z=EGO_Z),
                                                carla.Rotation(yaw=90. if start_index < 3 else -90.))
                        if free is None:
                            free = bench.next_slot(args.seed+number, index, start)
                        else:
                            bench.ego.set_transform(start)
                            bench.ego.apply_control(carla.VehicleControl())
                        began = time.time()
                        label = 'e{}_{}_{}'.format(epoch, name, start_index)
                        outcome, frames, final, seconds = bench.episode(start, goal, free, label)
                        writer.writerow(dict(epoch=epoch, slot=name, start=start_index, outcome=outcome,
                                             seconds=round(frames/FPS, 2), position_error=round(final['distance'], 3),
                                             yaw_error=round(final['yaw_error'], 2),
                                             goal_distance=round(final['goal_distance'], 3),
                                             inference_ms=round(1000*float(np.mean(seconds)), 1) if seconds else ''))
                        stream.flush()
                        print(label, outcome, 'sim {:.1f}s wall {:.0f}s goal {:.2f}m yaw {:.1f}'.format(
                            frames/FPS, time.time()-began, final['goal_distance'], final['yaw_error']), flush=True)
    finally:
        bench.close()


if __name__ == '__main__':
    main()
