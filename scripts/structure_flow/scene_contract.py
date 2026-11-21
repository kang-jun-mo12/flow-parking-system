"""Geometry and event rules for our own mixed Town04 benchmark (no CARLA import)."""
import json
import math
import random
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parents[2] / 'data' / 'town04_hardcore_v7.json'
PEDESTRIAN_EVENTS = ('walk_toward_ego', 'occluded_burst', 'reverse_slot_crossing', 'aisle_crossing')
EVENTS = ('adjacent_parking', 'parking_reverse') + PEDESTRIAN_EVENTS + ('adjacent_departure',)
TARGET_SLOTS = tuple('{}-{}'.format(r, c) for r in (2, 3) for c in range(1, 17))
ROW_X = (298.5, 290.9, 280.0, 272.5)
SLOT_Y = (-235.73, -232.73, -229.53, -226.43, -223.43, -220.23,
          -217.23, -214.03, -210.73, -207.30, -204.23, -201.03,
          -198.03, -194.90, -191.53, -188.20)


def load_config(path=CONFIG_PATH):
    cfg = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if cfg['fps'] != 30 or cfg['timeout_s'] <= 0:
        raise ValueError('Use 30 Hz and a positive simulator timeout')
    if not 0 <= cfg['static_density'] <= 1:
        raise ValueError('static_density must be in [0, 1]')
    for key in ('walk_speed_mps', 'approach_speed_mps', 'burst_speed_mps',
                'slot_walker_speed_mps', 'walker_radius_m', 'npc_speed_kmh',
                'actor_fault_stall_s', 'slot_occupancy_limit_s', 'crossing_speed_mps',
                'parking_reverse_speed_kmh', 'parking_reverse_warning_s', 'ego_yield_distance_m'):
        if cfg[key] <= 0:
            raise ValueError('Positive value required: ' + key)
    if not 0.8 <= cfg['approach_speed_mps'] <= 1.6:
        raise ValueError('Approaching pedestrian must use ordinary walking speed')
    return cfg


class SceneFrame:
    """u points from the goal towards the aisle; v follows the approach direction."""
    def __init__(self, slot_id):
        row, column = map(int, slot_id.split('-'))
        if row not in (2, 3) or not 1 <= column <= 16:
            raise ValueError('Hardcore supports all 32 goals in rows 2 and 3')
        self.row, self.column, self.slot_id = row, column, slot_id
        self.x, self.y = ROW_X[row - 1], SLOT_Y[column - 1]
        self.sign = 1 if row == 3 else -1
        self.v_sign = -1 if column <= 8 else 1
        adjacent = column + self.v_sign
        if not 1 <= adjacent <= 16:
            adjacent = column - self.v_sign
        departure = column - self.v_sign
        if departure == adjacent:
            departure = column - 2 * self.v_sign
        self.adjacent_v = (SLOT_Y[adjacent - 1] - self.y) * self.v_sign
        self.adjacent_id = '{}-{}'.format(row, adjacent)
        self.departure_v = (SLOT_Y[departure - 1] - self.y) * self.v_sign
        self.departure_id = '{}-{}'.format(row, departure)
        gaps = [(((a + b) / 2. - self.y) * self.v_sign, (i, i + 1))
                for i, (a, b) in enumerate(zip(SLOT_Y, SLOT_Y[1:]), 1)]
        self.burst_v, self.occluder_columns = min(gaps, key=lambda p: abs(p[0] + 1.5))
        self.sprint_exit_v = min(gaps, key=lambda p: abs(p[0] + 4.9))[0]
        self.gaps = gaps

    def world(self, u, v):
        return self.x + self.sign * u, self.y + self.v_sign * v

    def local(self, x, y):
        return (x - self.x) * self.sign, (y - self.y) * self.v_sign

    def yaw(self, canonical_yaw):
        return math.atan2(self.v_sign * math.sin(canonical_yaw), self.sign * math.cos(canonical_yaw))

    def route(self, points):
        return [list(self.world(p[0], p[1])) + [self.yaw(p[2]), p[3], p[4] * self.sign * self.v_sign]
                for p in points]

    def slots(self):
        return {'{}-{}'.format(r, c): (x, y) for r, x in enumerate(ROW_X, 1)
                for c, y in enumerate(SLOT_Y, 1)}


def line(start, end, yaw, gear, spacing=.10):
    count = max(1, int(math.ceil(math.hypot(end[0] - start[0], end[1] - start[1]) / spacing)))
    return [[start[0] + (end[0] - start[0]) * i / count,
             start[1] + (end[1] - start[1]) * i / count, yaw, gear, 0.]
            for i in range(count + 1)]


def turn(start, end_yaw, radius, wheelbase, gear):
    x, y, yaw = start
    delta = end_yaw - yaw
    turn_sign = 1 if delta * gear > 0 else -1
    count = max(1, int(math.ceil(abs(delta) * radius / .10)))
    result = [[x, y, yaw, gear, 0.]]
    steer = turn_sign * math.atan(wheelbase / radius)
    for i in range(1, count + 1):
        angle = yaw + delta * i / count
        result.append([x + radius / turn_sign * (math.sin(angle) - math.sin(yaw)),
                       y - radius / turn_sign * (math.cos(angle) - math.cos(yaw)),
                       angle, gear, steer])
    return result


def vehicle_routes(frame, cfg, ego_specs, npc_specs):
    aisle = cfg['aisle_u_m']
    ego_rx = ego_specs['rear_axle_x']
    radius = max(4.8, ego_specs['min_turning_radius'] * 1.08)
    if radius > aisle - ego_rx - .25:
        raise ValueError('Ego turning radius does not fit the reserved reverse approach')
    start_u = ego_start_u(frame, cfg)
    start_v = cfg['ego_start_v_m'] + ego_rx
    join_radius = max(6., ego_specs['min_turning_radius'] * 1.2)
    offset = start_u - aisle
    if not 0. <= offset <= 2. * join_radius:
        raise ValueError('Invalid lateral room for initial ego aisle join')
    angle = math.acos(1. - offset / (2. * join_radius))
    approach = [[start_u, start_v, math.pi/2, 1, 0.]]
    if offset > 1e-8:
        approach = turn((start_u, start_v, math.pi/2), math.pi/2 + angle,
                        join_radius, ego_specs['wheel_base'], 1)
        approach += turn(approach[-1][:3], math.pi/2, join_radius, ego_specs['wheel_base'], 1)[1:]
    if approach[-1][1] > radius:
        raise ValueError('Initial aisle join extends past reverse-parking setup')
    approach += line(approach[-1][:2], (aisle, radius), math.pi / 2, 1)[1:]
    reverse = turn((aisle, radius, math.pi / 2), 0., radius, ego_specs['wheel_base'], -1)
    reverse += line(reverse[-1][:2], (ego_rx, 0.), 0., -1)[1:]
    npc_rx = npc_specs['rear_axle_x']
    npc_radius = min(4.2, aisle + npc_rx - .18)
    if npc_radius < npc_specs['min_turning_radius'] * 1.02:
        raise ValueError('NPC cannot turn into the adjacent bay with this vehicle')
    npc = line((aisle, cfg['npc_start_v_m'] + npc_rx),
               (aisle, frame.adjacent_v - npc_radius), math.pi / 2, 1)
    npc += turn((aisle, frame.adjacent_v - npc_radius, math.pi / 2), math.pi,
                npc_radius, npc_specs['wheel_base'], 1)[1:]
    npc += line(npc[-1][:2], (-npc_rx, frame.adjacent_v), math.pi, 1)[1:]
    yielding = line((start_u, start_v), (start_u, start_v - cfg['ego_yield_distance_m']), math.pi/2, -1)
    return {'ego_approach': frame.route(approach), 'ego_reverse': frame.route(reverse),
            'ego_yield_reverse': frame.route(yielding),
            'npc_park': frame.route(npc)}


def ego_start_u(frame, cfg):
    if cfg.get('ego_spawn_policy') == 'middle_aisle_centerline':
        center_x = (ROW_X[1] + ROW_X[2]) / 2.
        return round((center_x - frame.x) * frame.sign, 9)
    return cfg['ego_end_slot_start_u_m'] if frame.column in (1, 16) else cfg['ego_start_u_m']


def pedestrian_routes(frame, seed=910260, start_backset_m=0.):
    # Sample real inter-bay gaps, rather than arbitrary points inside parked cars.
    rng = random.Random(seed + 48271)
    cross_nominal = rng.choice((-4.9, 1.65, 4.9))
    candidate_gaps = [p for p in frame.gaps if p[0] * cross_nominal > 0.]
    crossing_v = (min(candidate_gaps, key=lambda p: abs(p[0] - cross_nominal))[0]
                  if candidate_gaps else cross_nominal + .35) + rng.uniform(.02, .08)
    exit_nominal = rng.choice((-4.9, -8.0))
    exit_v = min(frame.gaps, key=lambda p: abs(p[0] - exit_nominal))[0]
    start_u = rng.uniform(7.75, 8.05)
    # Keep the waiting walker ahead of the closer ego and the car counterturn.
    start_v = rng.choice((rng.uniform(-4.5, -2.0), rng.uniform(8.0, 9.0))) - start_backset_m
    burst_gap = rng.choice([g for g in frame.gaps if -5.6 < g[0] < -.8])
    burst_v = burst_gap[0] + rng.uniform(-.035, .035)
    side = rng.choice((-1., 1.))
    entry_u = rng.uniform(-1.25, -.95)
    entry_v = rng.uniform(.85, 1.05)
    refuge_v = rng.uniform(1.6, 1.8)
    approach_exit_v = min(12., max(frame.local(frame.x, y)[1] for y in (-245.2, -174.8)))
    return {
        'walk_toward_ego': [(start_u, start_v), (7.4, -2.0), (8.0, -2.0), (8.0, approach_exit_v)],
        'occluded_burst': [(rng.uniform(11., 11.5), burst_v), (5.45, burst_v),
                           (-3., burst_v), (-3., burst_v - 3.)],
        'reverse_slot_crossing': [(-3., side*refuge_v), (entry_u, side*entry_v), (entry_u, -side*entry_v),
                                   (-3., -side*refuge_v), (-4.8, -side*refuge_v)],
        'aisle_crossing': [(rng.uniform(11., 11.5), crossing_v), (5.45, crossing_v), (3.1, crossing_v),
                           (3.1, 9. if crossing_v > 0. else -12.)],
    }


def spatial_trigger_specs(frame, cfg, local_paths, seed):
    """Independent encounter regions; no event order or completion inputs."""
    rng = random.Random(seed + 7919)
    result = {}
    for name in ('walk_toward_ego', 'occluded_burst', 'aisle_crossing'):
        v = local_paths[name][0][1]
        # Trigger centres sit on the reachable approach, so far-side origins
        # do not require driving into parked cars to activate a walker.
        lead = rng.uniform(2.7, 4.5)
        center_v = max(-6.5, min(3.5, v - lead))
        result[name] = dict(center_uv=[cfg['aisle_u_m'], center_v], radius_m=rng.uniform(2.3, 3.4))
    result['adjacent_parking'] = dict(center_uv=[cfg['aisle_u_m'], rng.uniform(-5.5, -3.8)],
                                     radius_m=rng.uniform(2.4, 3.3))
    parking = result['adjacent_parking']
    # The initial car already constrains the lane. Its range must be reached
    # before ego would have to cross that parked body to activate it.
    initial_distance = math.hypot(ego_start_u(frame, cfg)-parking['center_uv'][0],
                                  cfg['ego_start_v_m']-parking['center_uv'][1])
    parking['radius_m'] = max(parking['radius_m'], initial_distance+.15)
    result['adjacent_departure'] = dict(center_uv=[cfg['aisle_u_m'],
        rng.uniform(.7, 1.4) if frame.column in (1, 16) else rng.uniform(-6., -4.)],
        radius_m=rng.uniform(1.4, 2.2) if frame.column in (1, 16) else rng.uniform(2.5, 3.5))
    result['reverse_slot_crossing'] = dict(center_uv=[2., 0.], radius_m=rng.uniform(2.8, 3.5),
        minimum_rear_entry_depth_m=cfg['reverse_ped_entry_depth_m'])
    return result


def inside_trigger(spec, u, v):
    return math.hypot(u-spec['center_uv'][0], v-spec['center_uv'][1]) <= spec['radius_m']


def cvt_goal_coverage(frame, ego_pose, bounds, margin=.25):
    """All four bay corners in the actual actor-centred CVT coordinate frame."""
    x, y, yaw = ego_pose
    c, s = math.cos(yaw), math.sin(yaw)
    corners = []
    for u, v in ((-2.5, -1.5), (2.5, -1.5), (2.5, 1.5), (-2.5, 1.5)):
        wx, wy = frame.world(u, v)
        dx, dy = wx-x, wy-y
        corners.append([c*dx+s*dy, -s*dx+c*dy])
    clearance = min(min(p[0]-bounds['x_min'], bounds['x_max']-p[0],
                        p[1]-bounds['y_min'], bounds['y_max']-p[1]) for p in corners)
    return dict(full_target_inside=clearance >= margin, corners_ego_xy=corners,
                minimum_edge_clearance_m=clearance, required_margin_m=margin,
                bounds=dict(bounds), check='geometric_coverage_not_network_detection')


def sample_polyline(points, spacing=.12):
    result = []
    for a, b in zip(points, points[1:]):
        result += [p[:2] for p in line(a, b, 0., 1, spacing)]
    return result


def inside_goal(u, v, padding=0.):
    return abs(u) <= 2.5 + padding and abs(v) <= 1.5 + padding


def reverse_entry_ready(u, v, outward_heading_cosine, reversing, speed_mps,
                        rear_entry_depth_m=0., minimum_depth_m=.65):
    # Require real penetration of the goal bay by the vehicle footprint.
    return (reversing and speed_mps > .06 and 0. <= u <= 5.
            and abs(v) < 1.5 and outward_heading_cosine > .85
            and rear_entry_depth_m >= minimum_depth_m)


def classify_result(collision, outbound, timed_out, goal_reached, fired, cleared, faults):
    if collision:
        return 'collision'
    if faults:
        return 'scenario_invalid'
    if outbound:
        return 'outbound'
    if goal_reached and set(EVENTS).issubset(fired) and set(EVENTS).issubset(cleared):
        return 'target_success'
    if timed_out:
        return 'timeout'
    return 'running'
