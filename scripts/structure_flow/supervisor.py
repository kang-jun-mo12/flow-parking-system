"""The supervisor on the car's side (CARLA client, Python 3.7): who drives, the parking model's plan or the
avoidance Flow, and the Flow's commands as vehicle control.

Every 0.2 s (DECISION_TICKS at 30 Hz) FlowDriver
  perceive()  sends the 6 camera images and the car's forecast along the parking plan (relevance.forecast_path: its
              centre for 3 s every 0.1 s) to the server (system_server.py) and gets the arbiter's two probabilities;
  update()    turns them into the hand-over: the Flow takes over at ARBITER_ON, a conflict lasts while above
              ARBITER_OFF, the Flow lets go 0.6 s after the last one (gate_clear_s) - and then leads the car on to
              where the reverse parking starts (staged(), at most STAGE_SECONDS) before the parking model drives
              again; after a hand-back props do not call the Flow back for REENTRY_QUIET_S (walkers do). While the
              Flow drives, one decision of the server per step;
  control()   the Flow's [steer, speed gate, direction] as a VehicleControl (direction changes only at rest).
The rule the arbiter learned from (relevance.RelevanceGate on the simulator's walkers and props) still runs beside
it: its verdict goes into the log of every decision step (gate_log), not into the driving.
For the safety check on the simulator's objects the decision message also carries the static mask painted from the
parked cars, props and poles, and the walkers' true way of the next seconds.
"""
import json
import math
from multiprocessing.connection import Client

import numpy as np

import relevance

GATE_CONFIG = dict(gate_horizon_s=3., gate_sample_dt_s=.1, gate_margin_m=.25, gate_clear_s=.6, gate_track_max_age_s=.6,
                   gate_vehicle_motion_m=.45, gate_vehicle_speed_mps=.55, k4_forward_kmh=12., k4_reverse_kmh=10.)
# vehicle.tesla.model3 as the Flow runtime describes it (live_cvt_hybrid_astar.VehicleConfig), frame of the actor
SPECS = dict(half_length=4.7917795/2, half_width=2.16345/2, rear_axle_to_center=1.41563697, rear_axle_lateral_offset=0.)
DECISION_TICKS = 6          # 0.2 s at 30 Hz, the Flow's action step
EPISODE_SECONDS = 90.       # the worker's elapsed-fraction scale (five_scenario_contract episode_timeout_s)
AISLE_FROM_SLOT = 5.45      # slot centre to aisle centre line (the lot's geometry)
STAGE_PAST = 5.             # m past the slot centre (safety.STAGE_PAST): the CAA / DDP reverse start
STAGE_MAX = 7.5             # ... to this (safety.STAGE_MAX)
STAGE_SECONDS = 25.
ARBITER_ON, ARBITER_OFF = .5, .3     # the learned arbiter (bev_arbiter.py): in above ON, a conflict still above OFF
REENTRY_QUIET_S = 3.        # after a hand-back props do not call the Flow back for this long (walkers do)
FRONT_CLEAR = dict(trigger_m=5.5, wait_s=.6, stopped_kmh=.5)
LOT_BOUND = (264., 304., -247., -178.)     # x_min, x_max, y_min, y_max (scene_server.BOUND, y_min 6 m past the lot's
                                           # end: the DDP evaluator's starts stand on the walkway there): beyond it is
#                                            vegetation and kerb, painted into the static mask


def to_local(transform, x, y, vx=0., vy=0.):
    """World point and velocity into the actor frame (x forward, y right, CARLA's)."""
    yaw = math.radians(transform.rotation.yaw)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x-transform.location.x, y-transform.location.y
    return c*dx+s*dy, -s*dx+c*dy, c*vx+s*vy, -s*vx+c*vy


class FlowDriver:
    """Hand-over, observation and control around one connection to the system server."""

    def __init__(self, port, authkey, hand_back_at_start=True):
        self.hand_back_at_start = hand_back_at_start
        self.link = Client(('127.0.0.1', port), authkey=authkey.encode())
        self.link.send(dict(cmd='ping'))
        self.status = self.link.recv()
        if not self.status.get('ok'):
            raise RuntimeError('Flow worker: {}'.format(self.status))
        # a collection server with the perception in the loop (system_server.py --perceive) wants the cameras
        self.perceives = bool(self.status.get('perceive'))
        # ... and one with the learned arbiter (--arbiter) gives its verdict with the perception's reply
        self.arbitrates = bool(self.status.get('arbiter'))

    def begin(self, goal, statics, props=()):
        """New episode: goal (x, y, yaw rad), the static boxes [x, y, yaw rad, length, width] (the props among them)
        and the props alone (same rows)."""
        self.goal, self.statics = goal, np.asarray(statics, float).reshape(-1, 5)
        self.props = [list(map(float, p)) for p in props]
        self.gate = relevance.RelevanceGate(dict(GATE_CONFIG), SPECS)
        self.active, self.started = False, False
        self.clear_since = None
        self.hand_back = False
        self.quiet_until = -1.
        self.action, self.direction, self.steer = None, 1, 0.
        self.decisions = self.activations = 0
        self.active_ticks = 0
        self.unsafe = 0
        self.poses = []             # the car's pose at the last decision steps (0.2 s apart), newest last
        self.viewed = False
        self.gate_log = None        # the gate's verdict of the latest decision step (drive keeps it per frame)
        self.arbiter_p, self.arbiter_active, self.arbiter_clear = None, False, None

    def staged(self, transform):
        """safety.staged in the world: on the aisle centre line (5.45 m from the slot centre on the side the
        parked car faces) within 0.4 m, parallel to it within 8 degrees, STAGE_PAST..STAGE_MAX past the slot along
        the car's heading: where the CAA / DDP data start their half-circle reverse."""
        gx, gy, gyaw = self.goal
        cx, cy = gx+AISLE_FROM_SLOT*math.cos(gyaw), gy+AISLE_FROM_SLOT*math.sin(gyaw)
        yaw = math.radians(transform.rotation.yaw)
        ax, ay = -math.sin(gyaw), math.cos(gyaw)
        if ax*math.cos(yaw)+ay*math.sin(yaw) < 0:
            ax, ay = -ax, -ay
        dx, dy = transform.location.x-cx, transform.location.y-cy
        s, lateral = ax*dx+ay*dy, -ay*dx+ax*dy
        heading = abs(math.atan2(-ay*math.cos(yaw)+ax*math.sin(yaw), ax*math.cos(yaw)+ay*math.sin(yaw)))
        return STAGE_PAST <= s <= STAGE_MAX and abs(lateral) <= .4 and heading <= math.radians(8.)

    def stage_inside(self, transform):
        """True if the car can stand where the reverse parking starts (staged()) inside the lot: its front at
        STAGE_PAST past the slot along the car's heading lies within LOT_BOUND."""
        gx, gy, gyaw = self.goal
        cx, cy = gx+AISLE_FROM_SLOT*math.cos(gyaw), gy+AISLE_FROM_SLOT*math.sin(gyaw)
        yaw = math.radians(transform.rotation.yaw)
        ax, ay = -math.sin(gyaw), math.cos(gyaw)
        if ax*math.cos(yaw)+ay*math.sin(yaw) < 0:
            ax, ay = -ax, -ay
        reach = STAGE_PAST+SPECS['half_length']
        x, y = cx+reach*ax, cy+reach*ay
        return LOT_BOUND[0] <= x <= LOT_BOUND[1] and LOT_BOUND[2] <= y <= LOT_BOUND[3]

    def prop_tracks(self, transform):
        """The props as standing objects for the gate (class 'vehicle', relevant at once): the Flow takes over when
        the planner's path runs into one."""
        rows = []
        for i, (x, y, yaw, length, width) in enumerate(self.props):
            lx, ly, _, _ = to_local(transform, x, y)
            rows.append(dict(object_id='prop-{}'.format(i), class_name='vehicle', x=lx, y=ly, vx=0., vy=0.,
                             yaw=yaw-math.radians(transform.rotation.yaw), length=length, width=width, age_s=0.,
                             hits=10, ever_moving=True))
        return rows

    def prop_futures(self, transform):
        """The props for the rule teacher: standing, with their size."""
        out = []
        for x, y, yaw, length, width in self.props:
            lx, ly, _, _ = to_local(transform, x, y)
            half = .5*math.hypot(length, width)
            out.append(dict(positions=[[lx, ly]]*41, half_length=half, half_width=half, static=True))
        return out

    def tracks(self, transform, walkers):
        rows = []
        for w in walkers:
            location, velocity, box = w['actor'].get_location(), w['actor'].get_velocity(), w['actor'].bounding_box
            x, y, vx, vy = to_local(transform, location.x, location.y, velocity.x, velocity.y)
            yaw = math.atan2(vy, vx) if math.hypot(vx, vy) > .1 else 0.
            rows.append(dict(object_id='walker-{}'.format(w['id']), class_name='pedestrian', x=x, y=y, vx=vx, vy=vy,
                             yaw=yaw, length=max(.6, 2*box.extent.x), width=max(.6, 2*box.extent.y), age_s=0., hits=10,
                             ever_moving=True))
        return rows

    def path_states(self, transform, tracker):
        """The teacher's plan left to drive, as the gate's K4 states [x, y, yaw, direction, 0, speed km/h] in the
        actor frame (rear axle points), from the plan point nearest to the car."""
        if tracker.path is None or len(tracker.path) < 2:
            return []
        yaw0 = math.radians(transform.rotation.yaw)
        rear = (transform.location.x-SPECS['rear_axle_to_center']*math.cos(yaw0),
                transform.location.y-SPECS['rear_axle_to_center']*math.sin(yaw0))
        start = max(0, tracker.index-2)
        near = start+int(np.argmin(np.hypot(tracker.path[start:, 0]-rear[0], tracker.path[start:, 1]-rear[1])))
        rows = []
        for (x, y), yaw, gear, speed in zip(tracker.path[near:], tracker.yaw[near:], tracker.gears[near:], tracker.speed[near:]):
            lx, ly, _, _ = to_local(transform, x, y)
            rows.append([lx, ly, (yaw-yaw0+math.pi) % (2*math.pi)-math.pi, 1 if gear >= 0 else -1, 0., 3.6*speed])
        return rows

    def static_mask(self, transform):
        """200 x 200 occupancy at 0.1 m around the actor (row r: x = 10-(r+.5)/10, column c: y = -10+(c+.5)/10),
        the frame of the worker's ExactRasterFootprintCollisionChecker."""
        cx, cy = transform.location.x, transform.location.y
        near = self.statics[np.hypot(self.statics[:, 0]-cx, self.statics[:, 1]-cy) < 18.]
        u = 10.-(np.arange(200)+.5)*.1
        v = -10.+(np.arange(200)+.5)*.1
        U, V = np.meshgrid(u, v, indexing='ij')
        yaw = math.radians(transform.rotation.yaw)
        c, s = math.cos(yaw), math.sin(yaw)
        X, Y = cx+c*U-s*V, cy+s*U+c*V
        # the lot's edge: without it the car, drawing away from a walker, could leave the lot
        mask = (X < LOT_BOUND[0]) | (X > LOT_BOUND[1]) | (Y < LOT_BOUND[2]) | (Y > LOT_BOUND[3])
        if not len(near):
            return mask
        for bx, by, byaw, length, width in near:
            dx, dy = X-bx, Y-by
            a, b = math.cos(byaw), math.sin(byaw)
            mask |= (np.abs(a*dx+b*dy) <= length/2) & (np.abs(-b*dx+a*dy) <= width/2)
        return mask

    def observation(self, transform, speed, applied, base_steer, progress, t, tracks):
        gx, gy, _, _ = to_local(transform, self.goal[0], self.goal[1])
        dyaw = self.goal[2]-math.radians(transform.rotation.yaw)
        vector = np.zeros(68, np.float32)
        vector[4] = applied.steer
        vector[5], vector[6] = np.clip(gx/20., -1., 1.), np.clip(gy/20., -1., 1.)
        vector[10], vector[11] = math.cos(dyaw), math.sin(dyaw)
        vector[15] = min(1., t/EPISODE_SECONDS)
        vector[16] = base_steer
        vector[21] = progress
        speed_kmh = 3.6*abs(speed)
        info = dict(speed_kmh=speed_kmh, control_direction=self.direction, direction=self.direction,
                    base_steer_normalized=base_steer, unified_k4_feasible=True, progress=progress,
                    goal_distance_m=math.hypot(gx, gy), dynamic_state=dict(tracks=tracks),
                    gear_shift_speed_kmh=.25, gear_hold_remaining_s=0., gear_shift_settle_remaining_s=0.,
                    gear_request_confirm_s=0., gear_shift_settle_s=0., gear_min_hold_s=0., candidate_direction=0,
                    candidate_confirm_progress=0., unified_policy_active=True)
        observation = dict(vector=vector.tolist(), flow_static_physical_mask=self.static_mask(transform).tolist(),
                           flow_boundary_polygon_xy=None)
        return observation, info

    def perceive(self, t, transform, cameras, tracker=None, speed=0., applied=None):
        """The 6 cameras of this decision step (RGB uint8 [6, 300, 400, 3], the benchmark's order) to the server's
        perception - every decision step of the episode, before update(): its tracker and its refiner read the
        sequence. Returns the seconds it took there."""
        message = dict(cmd='perceive', t=t, first=not self.viewed,
                       pose=[transform.location.x, transform.location.y, transform.rotation.yaw],
                       goal=[float(g) for g in self.goal])
        if tracker is not None:
            # what the gate forecasts in update(): the car's centre along the plan left, 3 s every 0.1 s - the arbiter
            # reads the same (and the server keeps it with the episode)
            reverse = bool(applied is not None and applied.reverse)
            signed = 3.6*speed if not reverse else -3.6*abs(speed)
            states = self.path_states(transform, tracker)
            forecast = relevance.forecast_path(states, 0, GATE_CONFIG, SPECS, signed)
            message.update(forecast=[[x, y, yaw] for _, x, y, yaw in forecast], planned=bool(states),
                           speed=signed/3.6, reverse=reverse)
        self.link.send(message)
        self.link.send_bytes(np.ascontiguousarray(cameras, np.uint8).tobytes())
        reply = self.link.recv()
        if not reply.get('ok'):
            raise RuntimeError('Flow worker perception: {}'.format(reply))
        self.viewed = True
        self.arbiter_p = reply.get('arbiter')
        return float(reply.get('seconds', 0.))

    def update(self, t, transform, speed, applied, walkers, tracker, base_steer, futures=None):
        """Every DECISION_TICKS: run the gate; while it holds, one Flow decision. True while the Flow drives."""
        tracks = self.tracks(transform, walkers)
        pose = [transform.location.x, transform.location.y, transform.rotation.yaw]
        history, self.poses = list(self.poses), (self.poses+[pose])[-6:]
        report = self.gate.update(t, tracks+self.prop_tracks(transform), self.path_states(transform, tracker), 0,
                                  3.6*speed if not applied.reverse else -3.6*abs(speed))
        walker_conflict = any(not str(r.get('object_id', '')).startswith('prop') for r in report.get('tracks', []))
        # the gate's own verdict, before the staging and quiet rules below change who drives (the label of an arbiter)
        self.gate_log = dict(t=round(float(t), 3), active=bool(report['active']), walker=bool(walker_conflict),
                             conflicts=[str(r.get('object_id', '')) for r in report.get('tracks', [])])
        if self.arbitrates and self.arbiter_p is not None:
            # the learned arbiter in the rule's place: is the plan's way blocked, by a walker, by a prop - with the
            # rule's own timing (active with a conflict, let go after gate_clear_s without one)
            p_walker, p_prop = self.arbiter_p
            level = ARBITER_OFF if self.arbiter_active else ARBITER_ON
            walker_conflict = p_walker >= level
            if walker_conflict or p_prop >= level:
                self.arbiter_active, self.arbiter_clear = True, None
            elif self.arbiter_active:
                if self.arbiter_clear is None:
                    self.arbiter_clear = t
                if t-self.arbiter_clear+1e-6 >= GATE_CONFIG['gate_clear_s']:
                    self.arbiter_active = False
            self.gate_log.update(arbiter=[round(float(p_walker), 4), round(float(p_prop), 4)],
                                 arbiter_active=bool(self.arbiter_active))
            report = dict(report, active=self.arbiter_active)
        if self.active and self.hand_back and self.hand_back_at_start:
            # the Flow is done (at the reverse parking's start, or no way there): hand back whatever the gate says
            # - it forecasts the planner's old path, which may still run through a prop
            self.active, self.hand_back = False, False
            self.link.send(dict(cmd='reposition_train_transition', terminated=True, info=dict(outcome='handback')))
            self.link.recv()
            self.quiet_until = t+REENTRY_QUIET_S
            return False
        if not self.active and t < self.quiet_until and not walker_conflict:
            return False            # just handed back: the planner's new path gets a moment before props count
        if report['active']:
            self.clear_since = None
        elif self.active and self.hand_back_at_start:
            # the walkers are clear; the Flow leads the car on until it is past the slot on the aisle centre line,
            # parallel to it (the parking planner reverses in from there), at most STAGE_SECONDS
            if self.clear_since is None:
                self.clear_since = t
            # (where that start lies past the lot's end, the parking model drives again once the walkers are clear)
            if (not self.staged(transform) and self.stage_inside(transform) and not self.hand_back
                    and t-self.clear_since < STAGE_SECONDS):
                report = dict(report, active=True)
        if not report['active']:
            if self.active:
                self.active = False
                self.link.send(dict(cmd='reposition_train_transition', terminated=True, info=dict(outcome='handback')))
                self.link.recv()
            return False
        self.direction = -1 if applied.reverse else 1
        progress = float(tracker.arc[min(tracker.index, len(tracker.arc)-1)]/max(tracker.arc[-1], 1e-3)) \
            if tracker.path is not None and len(tracker.path) > 1 else 0.
        # the props go into the policy's input as well (its 6 actor slots, class 'vehicle', standing, with their
        # size): the Flow goes round them, so it has to see them (and the worker's dynamic check keeps .12 m off)
        observation, info = self.observation(transform, speed, applied, base_steer, progress, t,
                                             tracks+self.prop_tracks(transform))
        cmd ='reposition_train_transition' if self.active else 'reposition_train_reset'
        if not self.active:
            self.activations += 1
        self.active = True
        message = dict(cmd=cmd, observation=observation, info=info, terminated=False, truncated=False, t=t)
        if futures is not None:     # a collection server (system_server.py): the Oracle's inputs as well
            gx, gy, _, _ = to_local(transform, self.goal[0], self.goal[1])
            dyaw = (self.goal[2]-math.radians(transform.rotation.yaw)+math.pi) % (2*math.pi)-math.pi
            message.update(futures=futures+self.prop_futures(transform), goal_local=[gx, gy, dyaw])
            # for the BEV Flow (bev_flow.py), which draws its own BEV: the car and the scene in the world, the control
            # applied up to now and the poses of the last decision steps
            message['world'] = dict(pose=pose, goal=[float(g) for g in self.goal], speed=float(speed),
                                    steer=float(applied.steer), reverse=bool(applied.reverse), history=history,
                                    statics=self.statics.tolist())
        self.link.send(message)
        reply = self.link.recv()
        if not reply.get('ok', True) or reply.get('action') is None:
            raise RuntimeError('Flow worker reply: {}'.format(reply))
        self.action = [float(a) for a in reply['action']]
        self.hand_back = bool(reply.get('hand_back'))
        self.decisions += 1
        self.unsafe = int(reply.get('fallbacks', 0))
        return True

    def control(self, carla, speed_control, speed):
        """The latest Flow action as a VehicleControl (the worker's speed gate -> km/h as its rollout reads it)."""
        steer, gate, request = self.action
        wanted = 0 if request == 0. or gate <= -1. else (1 if request > 0 else -1)
        kmh = min(1., max(0., (gate+1.)*.6))*(12. if wanted > 0 else 10.) if wanted else 0.
        if wanted and wanted != self.direction:
            if abs(speed) > .25/3.6:
                kmh = 0.                # stop first; the worker's preview switches only at rest
            else:
                self.direction = wanted
        self.steer += float(np.clip(steer-self.steer, -.1, .1))
        throttle, brake = speed_control(kmh/3.6, abs(speed), self.direction)
        return carla.VehicleControl(throttle=throttle, brake=brake, steer=float(np.clip(self.steer, -1., 1.)),
                                    reverse=self.direction < 0, manual_gear_shift=True, gear=-1 if self.direction < 0 else 1)

    def finish(self, folder, outcome):
        """A collection server writes the episode's decision rows to FOLDER/flow_decisions.npz."""
        if self.status.get('collector'):
            self.link.send(dict(cmd='collect_save', folder=folder, outcome=outcome))
            reply = self.link.recv()
            if not reply.get('ok'):
                raise RuntimeError('Flow collection server: {}'.format(reply))
            return reply.get('rows', 0)
        return 0

    def end(self, outcome):
        if self.active:
            self.link.send(dict(cmd='reposition_train_transition', terminated=True, info=dict(outcome=outcome)))
            self.link.recv()
            self.active = False

    def close(self):
        try:
            self.link.send(dict(cmd='close'))
            self.link.recv()
        except (EOFError, OSError):
            pass
        self.link.close()
