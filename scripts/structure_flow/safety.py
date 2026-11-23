"""Safety check of the avoidance Flow's candidates, and the vehicle model it rolls them through.

A candidate is a chunk of 4 x 0.2 s [steer (normalized, x 0.981 rad), speed fraction (x 12 km/h forward, x 10 km/h
reverse)] with gears F 0 / STOP 1 / R 2, in the car's actor frame (x forward, y right). safe() drives it through the
vehicle model for its 0.8 s and 0.5 s of braking and asks:
  - static_ok(): does the body stay off every occupied cell of the 0.1 m mask (200 x 200 around the car; row r:
    x = 10 - (r + .5) / 10, column c: y = -10 + (c + .5) / 10) - or, where the car already stands a little inside a
    painted box, does it come no deeper (the escape rule);
  - does it keep COLLISION_GAP from every walker on the walker's way of the next seconds.
The server takes the first candidate that passes. What the check reads is the server's choice: the simulator's
objects, or the perceived world state (perceived_scene()). staged(): the car stands where the reverse parking starts.
"""
import math

import numpy as np

# ---------------------------------------------------------------- the vehicle model
DT = .1
STEPS = 40                  # 4 s
CHUNK_STEPS = 8             # the 4 x 0.2 s actions
SAMPLES = 12
COLLISION_GAP = .20
PEDESTRIAN_HALF = .35
PASS_SPEED, PASS_STEER, PASS_STEPS = .35, .6, 12     # 4.2 km/h, steer out 1.2 s, back 1.2 s, then straight
ASIDE_SPEED, ASIDE_STEER, ASIDE_STEPS = .4, .6, 15   # into the side space (an empty bay), 1.5 s, then stop
MAX_STEER_RAD, WHEELBASE_M = .9809930780518399, 3.004647


def commands(actions, gears, continuation):
    """Per 0.1 s step (gear, speed fraction, steer) over STEPS: the chunk (each action 0.2 s), then the
    continuation: 'brake' stop, 'hold' the last command, 'straighten' last gear and speed with the wheel
    back to 0, 'counter' the opposite steer for as long as the chunk steered, then 0 (a lane change), 'resume'
    forward at 6 km/h straight, 'pass_left' / 'pass_right' forward at 6 km/h with the wheel 1 s out to that side
    and 1 s back (stopping first and changing gear when the chunk ends in reverse): the second half of
    'back off a little, then go round', which a single chunk never shows."""
    out = []
    chunk_steer = float(np.mean(actions[:, 0]))
    for step in range(STEPS):
        if step < CHUNK_STEPS:
            i = step//2
            out.append((int(gears[i]), float(actions[i, 1]), float(actions[i, 0])))
            continue
        g, v, w = int(gears[3]), float(actions[3, 1]), float(actions[3, 0])
        k = step-CHUNK_STEPS
        if continuation == 'brake':
            out.append((1, 0., w))
        elif continuation == 'hold':
            out.append((g, v, w))
        elif continuation == 'straighten':
            out.append((g, v, 0.))
        elif continuation == 'counter':
            out.append((g, v, -chunk_steer if k < CHUNK_STEPS else 0.))
        elif continuation == 'resume':
            out.append((0, PASS_SPEED, 0.))
        elif continuation.startswith('aside'):
            # pull out of the walker's line into the side space and wait there, as a driver lets someone pass
            side = -1. if continuation.endswith('left') else 1.
            gear = 2 if '_back_' in continuation else 0
            out.append((gear, ASIDE_SPEED, side*ASIDE_STEER) if k < ASIDE_STEPS else (1, 0., 0.))
        else:
            side = -1. if continuation == 'pass_left' else 1.      # negative steer turns left (y < 0)
            out.append((0, PASS_SPEED, side*PASS_STEER if k < PASS_STEPS else -side*PASS_STEER if k < 2*PASS_STEPS else 0.))
    return out


def rollout(actions, gears, state, continuation):
    """The car's poses over STEPS [x, y, yaw, speed, distance driven, gear changes] in the frame it starts in: the
    chunk, then the continuation (commands()), through the vehicle model - accelerating at 2, braking at 4 m/s^2,
    the wheel turning at 2 /s, a gear change only at rest."""
    x = y = yaw = 0.
    steer = float(state.get('base_steer_normalized', 0.))
    speed = abs(float(state.get('speed_kmh', 0.)))/3.6
    direction = 1 if int(state.get('control_direction', 1)) > 0 else -1
    hold = max(0., float(state.get('gear_hold_remaining_s', 0.)))
    settle = max(0., float(state.get('gear_shift_settle_remaining_s', 0.)))
    confirm = max(0., float(state.get('gear_request_confirm_s', .2)))
    shift_wait, poses, distance, changes = 0., [], 0., 0
    for gear, fraction, wanted in commands(actions, gears, continuation):
        requested = 1 if gear == 0 else -1 if gear == 2 else 0
        desired = fraction*(12./3.6 if gear == 0 else 10./3.6)
        braking = requested == 0 or requested != direction or settle > 0.
        if requested and requested != direction and speed <= .25/3.6 and hold <= 0. and settle <= 0.:
            shift_wait += DT
            if shift_wait >= confirm-1e-8:
                direction, changes, shift_wait = requested, changes+1, 0.
                hold = max(0., float(state.get('gear_min_hold_s', .7)))
                settle = max(0., float(state.get('gear_shift_settle_s', .1)))
        elif requested == direction or requested == 0:
            shift_wait = 0.
        old_speed = speed
        if braking:
            speed = max(0., speed-4.*DT)
        else:
            speed = max(0., speed+float(np.clip(desired-speed, -4.*DT, 2.*DT)))
        steer += float(np.clip(wanted-steer, -2.*DT, 2.*DT))
        ds = direction*(old_speed+speed)*.5*DT
        # the deployed vehicle (live_cvt_hybrid_astar.VehicleConfig: normalized steer x 0.981 rad bicycle angle,
        # wheelbase 3.005 m, as the worker and the bridge use it)
        curvature = math.tan(steer*MAX_STEER_RAD)/WHEELBASE_M
        yaw0 = yaw
        yaw += ds*curvature
        if abs(curvature) < 1e-8:
            x += ds*math.cos(yaw0)
            y += ds*math.sin(yaw0)
        else:
            x += (math.sin(yaw)-math.sin(yaw0))/curvature
            y += (-math.cos(yaw)+math.cos(yaw0))/curvature
        distance += abs(ds)
        poses.append((x, y, yaw, speed, distance, changes))
        hold, settle = max(0., hold-DT), max(0., settle-DT)
    return np.asarray(poses)


def future_samples(actors, seed=0):
    """actors: [dict(positions [STEPS+1, 2] at 0.1 s from now, half_length, half_width)] in the car frame.
    Returns boxes [SAMPLES, STEPS, n, 6] (x, y, cos, sin, half length, half width): the exact future and 11 copies
    with a per-actor drift (sigma 0.24 m) growing until 1.4 s, then constant, and a small jitter; each box covers the motion since the
    previous 0.1 s."""
    rng = np.random.default_rng(int(seed))
    n = len(actors)
    out = np.zeros((SAMPLES, STEPS, n, 6))
    if not n:
        return out
    p = np.stack([np.asarray(a['positions'], float)[:STEPS+1] for a in actors], 1)          # [STEPS+1, n, 2]
    half0 = np.array([(a.get('half_length', PEDESTRIAN_HALF), a.get('half_width', PEDESTRIAN_HALF)) for a in actors])
    t = (np.arange(1, STEPS+1)*DT)[:, None, None]
    for k in range(SAMPLES):
        drift = rng.normal(0., .24, (1, n, 2)) if k else np.zeros((1, n, 2))
        jitter = rng.normal(0., 1., (STEPS+1, n, 2))*(.025*np.minimum(np.arange(STEPS+1), 14)/2.)[:, None, None] if k else 0.
        q = p+drift*np.minimum(np.r_[0., np.arange(1, STEPS+1)*DT][:, None, None]/1.4, 1.)+jitter
        a, b = q[:-1], q[1:]
        out[k, :, :, 0:2] = (a+b)/2.
        out[k, :, :, 2] = 1.
        out[k, :, :, 4:6] = np.abs(b-a)/2.+half0[None]
    del t
    return out


def sat_gaps(poses, margins, boxes):
    """Gap between the car body (4.7 x 2.0 m box) and boxes, by separating axes: poses [S, >=3], margins [S],
    boxes [S, M, 6] -> [S, M] in metres (negative: overlap)."""
    x, y, yaw = poses[:, 0:1], poses[:, 1:2], poses[:, 2:3]
    c, s = np.cos(yaw), np.sin(yaw)
    m = margins[:, None]
    dx, dy = boxes[..., 0]-x, boxes[..., 1]-y
    bc, bs, bl, bw = boxes[..., 2], boxes[..., 3], boxes[..., 4], boxes[..., 5]
    result = np.full(dx.shape, -np.inf)
    for ax, ay in ((c, s), (-s, c), (bc, bs), (-bs, bc)):
        radius = (2.35+m)*np.abs(c*ax+s*ay)+(1.+m)*np.abs(-s*ax+c*ay)+bl*np.abs(bc*ax+bs*ay)+bw*np.abs(-bs*ax+bc*ay)
        result = np.maximum(result, np.abs(dx*ax+dy*ay)-radius)
    return result


# ---------------------------------------------------------------- the body against the static mask
HALF_LENGTH, HALF_WIDTH = 2.4, 1.1          # model3, rounded up
CHECK_STEPS = 13                             # 0.8 s chunk + 0.5 s braking
_OCCUPIED = [None, None]     # (mask, its occupied cell centres in the car frame [n, 2]) of the last mask seen


def occupied_points(mask200):
    """Centres of the occupied cells of the 0.1 m mask (car frame), cached for the mask object of this decision."""
    if _OCCUPIED[0] is not mask200:
        rows, cols = np.nonzero(mask200)
        _OCCUPIED[0], _OCCUPIED[1] = mask200, np.stack((10.-(rows+.5)*.1, -10.+(cols+.5)*.1), 1)
    return _OCCUPIED[1]


def body_free(mask200, poses, every=2, skip=None):
    """No occupied mask cell inside the car body (HALF_LENGTH x HALF_WIDTH box) at every `every`-th pose; skip
    (centre, radius): cells of that obstacle are ignored (a prop being gone round is checked against its box with
    the clearance instead). The body as a box, not 15 points on it: the points, 1.1-1.2 m apart, let a cone or a
    bin through."""
    points = occupied_points(np.asarray(mask200, bool))
    poses = np.asarray(poses, float)[::every]
    if not len(points) or not len(poses):
        return True
    if skip is not None:
        points = points[np.hypot(points[:, 0]-skip[0][0], points[:, 1]-skip[0][1]) > skip[1]+.1]
    reach = math.hypot(HALF_LENGTH, HALF_WIDTH)+.1
    near = (points[:, 0] > poses[:, 0].min()-reach) & (points[:, 0] < poses[:, 0].max()+reach) & \
        (points[:, 1] > poses[:, 1].min()-reach) & (points[:, 1] < poses[:, 1].max()+reach)
    points = points[near]
    if not len(points):
        return True
    c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    dx, dy = points[None, :, 0]-poses[:, 0, None], points[None, :, 1]-poses[:, 1, None]
    inside = (np.abs(c*dx+s*dy) <= HALF_LENGTH) & (np.abs(-s*dx+c*dy) <= HALF_WIDTH)
    return not bool(inside.any())


ESCAPE_DEPTH, ESCAPE_SLACK = .15, .02


def body_depth(mask200, poses):
    """How deep the occupied mask cells reach into the car body box at each pose, m (0: none inside)."""
    points = occupied_points(np.asarray(mask200, bool))
    poses = np.atleast_2d(np.asarray(poses, float))[:, :3]        # rollout rows carry more than x, y, yaw
    if not len(points) or not len(poses):
        return np.zeros(len(poses))
    reach = math.hypot(HALF_LENGTH, HALF_WIDTH)+.1
    near = (points[:, 0] > poses[:, 0].min()-reach) & (points[:, 0] < poses[:, 0].max()+reach) &         (points[:, 1] > poses[:, 1].min()-reach) & (points[:, 1] < poses[:, 1].max()+reach)
    points = points[near]
    if not len(points):
        return np.zeros(len(poses))
    c, s = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    dx, dy = points[None, :, 0]-poses[:, 0, None], points[None, :, 1]-poses[:, 1, None]
    depth = np.minimum(HALF_LENGTH-np.abs(c*dx+s*dy), HALF_WIDTH-np.abs(-s*dx+c*dy))
    return np.clip(depth, 0., None).max(1)


def static_ok(actions, gears, state, mask200):
    """The chunk and 0.5 s of braking, through the vehicle model, keep the body off every occupied cell of the
    0.1 m mask.
    A car that stands with cells inside its body box already (the box is the body rounded up; it stopped some cm
    into the painted box of a prop beside it) may move where they come no deeper: without this every chunk fails
    there, the rule says wait and the student stands for good."""
    poses = rollout(actions, gears, state, 'brake')[:CHECK_STEPS]
    if body_free(mask200, poses, every=1):
        return True
    start = float(body_depth(mask200, np.zeros((1, 3)))[0])
    if not 0. < start <= ESCAPE_DEPTH or int(np.asarray(gears)[0]) == 1:
        return False
    return float(body_depth(mask200, poses).max()) <= start+ESCAPE_SLACK


# ---------------------------------------------------------------- where the reverse parking starts
AISLE_FROM_SLOT = 5.45                       # slot centre to aisle centre line, m (the lot's geometry)
STAGE_LATERAL, STAGE_HEADING = .4, math.radians(8.)
STAGE_PAST = 5.                              # m past the slot centre along the aisle: where the CAA / DDP data start
STAGE_MAX = 7.5                              # ... and not further: past it the Flow backs up to it


def aisle_frame(goal):
    """The aisle at the goal slot in the car frame: the centre line's point abreast of the slot, the aisle
    direction closer to the car's heading, and the car's position along (s, negative before the slot) and across
    (lateral) it, and its heading error."""
    gx, gy, gyaw = goal
    centre = np.array([gx, gy])+AISLE_FROM_SLOT*np.array([math.cos(gyaw), math.sin(gyaw)])   # parked cars face it
    a = np.array([-math.sin(gyaw), math.cos(gyaw)])
    if a[0] < 0:
        a = -a
    n = np.array([-a[1], a[0]])
    s = float(a @ -centre)
    lateral = float(n @ -centre)
    heading = math.atan2(a[1], a[0])       # the car's x axis is (1, 0): the error is the aisle's angle
    return centre, a, n, s, lateral, heading


def staged(goal):
    """True where the parking planner takes over well: where the CAA / DDP data start their
    half-circle reverse - STAGE_PAST..STAGE_MAX past the slot in the car's direction, on the aisle centre line,
    parallel to it."""
    _, _, _, s, lateral, heading = aisle_frame(goal)
    return STAGE_PAST <= s <= STAGE_MAX and abs(lateral) <= STAGE_LATERAL and abs(heading) <= STAGE_HEADING


# ---------------------------------------------------------------- the check of one candidate
LOT_BOUND = (264., 304., -247., -178.)          # supervisor.LOT_BOUND: beyond it the mask is occupied


def perceived_scene(seen, pose):
    """--screen perceived: what the safety check of the student's candidates reads, from the perception in place of
    the simulator - the 200 x 200 mask of the worker (the central 200 rows of the refiner's static mask, and the lot's
    edge from the map as supervisor.FlowDriver.static_mask paints it) and the people going on as they go (constant velocity,
    4 s), in the form of the true futures."""
    static = np.asarray(seen['static'], bool)
    first = (static.shape[0]-200)//2
    mask = static[first:first+200].copy()
    u = 10.-(np.arange(200)+.5)*.1
    v = -10.+(np.arange(200)+.5)*.1
    U, V = np.meshgrid(u, v, indexing='ij')
    c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
    X, Y = pose[0]+c*U-s*V, pose[1]+s*U+c*V
    mask |= (X < LOT_BOUND[0]) | (X > LOT_BOUND[1]) | (Y < LOT_BOUND[2]) | (Y > LOT_BOUND[3])
    steps = np.arange(41)[:, None]*.1
    people = [dict(positions=(np.array([x, y])[None]+steps*np.array([vx, vy])[None]).tolist(),
                   half_length=PEDESTRIAN_HALF, half_width=PEDESTRIAN_HALF)
              for x, y, vx, vy, valid in np.asarray(seen['walkers'], float) if valid > .5]
    return mask, people


def safe(actions, gears, state, mask200, walkers):
    """A candidate of the avoidance Flow for its chunk and 0.5 s of braking: the body off every occupied cell of the
    0.1 m mask (static_ok) and COLLISION_GAP from every walker on its way of that time. walkers: [dict(positions
    [41, 2] every 0.1 s from now in the car frame, half_length, half_width)]."""
    actions, gears = np.asarray(actions, np.float64), np.asarray(gears, np.int64)
    if not static_ok(actions, gears, state, mask200):
        return False
    if not walkers:
        return True
    poses = rollout(actions, gears, state, 'brake')[:CHECK_STEPS]
    boxes = future_samples(walkers)[0][:len(poses)]
    return float(sat_gaps(poses, np.zeros(len(poses)), boxes).min()) >= COLLISION_GAP


def clip(value, low=-1.0, high=1.0):
    return max(float(low), min(float(high), float(value)))


def first_action(actions, gears):
    """A chunk's first step as the reply to the car: [steer, speed gate (-1 stop .. 1 full), direction request
    (1 forward, -1 reverse, 0 stand)]."""
    steer, speed = float(actions[0, 0]), float(actions[0, 1])
    if int(gears[0]) == 1:
        return np.array([clip(steer), -1., 0.], np.float32)
    return np.array([clip(steer), clip(speed/.6-1.), 1. if int(gears[0]) == 0 else -1.], np.float32)
