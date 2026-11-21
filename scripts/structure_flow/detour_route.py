"""Global layer of the detour teacher: a Hybrid A* detour that leaves the expert route before a new
obstacle and rejoins it after, in the world grid of detour_drive.py.

The body is five discs along its length (radius 1.05 m covers the 4.4 x 1.9 m box). A disc may enter the
predicted vehicle mask by `vehicle_tolerance` (negative: keep that margin) and must keep `static_margin` from the
new obstacle. Motion primitives are 0.6 m arcs at five curvatures, forward or reverse; reversing and gear
changes cost extra. The heuristic is an obstacle-aware 2D distance from the rejoin poses.

The route is then: expert frames up to the leave point, the A* detour timed at a cruise speed with stops
at gear changes, and the expert frames from the rejoin point on, all at 10 Hz. It is a Task, so
window_label() gives the start latents the local optimiser refines.
"""
import heapq
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from .config import DEFAULT as cfg
from .detour import BODY_LENGTH, signed_distance

DISCS = (np.arange(5)+.5)/5*BODY_LENGTH-BODY_LENGTH/2       # disc centres along the body, from its centre
RADIUS = float(np.hypot(BODY_LENGTH/10, 1.9/2))             # 1.05 m


@dataclass(frozen=True)
class RouteConfig:
    static_margin: float = .3
    vehicle_tolerance: float = -.1   # negative: a margin kept from the predicted vehicles
    skip_cost: float = 2.            # per metre of expert route skipped by rejoining later
    step: float = .6
    curvatures: tuple = (-.45, -.22, 0., .22, .45)
    reverse_cost: float = 1.5
    switch_cost: float = 4.
    steer_cost: float = .3
    xy_bin: float = .3
    yaw_bins: int = 72
    heuristic_weight: float = 1.6
    max_expansions: int = 80000
    goal_distance: float = .35
    goal_heading: float = .2
    leave_before: float = 7.         # m of expert route kept free before the first blocked frame
    rejoin_after: float = 3.         # m after the last blocked frame where the detour may rejoin
    cruise: float = 1.2              # m/s on the detour
    accel: float = .6                # m/s^2 into and out of stops


class RouteTask:
    """Task-shaped route (what window_label reads)."""

    def __init__(self, pose, speed, reverse, target, steer=None):
        self.pose, self.speed, self.reverse, self.target = pose, speed, reverse, target
        self.steer = np.zeros(len(pose)) if steer is None else steer

    def __len__(self):
        return len(self.pose)


class Clearance:
    """Disc clearance of rear-axle poses in the world grid."""

    def __init__(self, world, vehicles, static, rc):
        self.world, self.rc = world, rc
        self.sdf_v = signed_distance(vehicles)
        self.sdf_s = signed_distance(static)

    def discs(self, pose):
        pose = np.atleast_2d(pose)
        c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
        along = cfg.rear_offset_m+DISCS[None]
        return np.stack((pose[:, :1]+c[:, None]*along, pose[:, 1:2]+s[:, None]*along), -1)     # [N, 5, 2]

    def margins(self, pose):
        """Per pose: worst disc clearance to the vehicles (minus tolerance) and to the static obstacle (minus margin)."""
        xy = self.discs(pose)
        ij = self.world.index(xy)
        inside = (ij[..., 0] >= 0) & (ij[..., 0] < self.world.shape[0]) & (ij[..., 1] >= 0) & (ij[..., 1] < self.world.shape[1])
        i = np.clip(ij[..., 0], 0, self.world.shape[0]-1)
        j = np.clip(ij[..., 1], 0, self.world.shape[1]-1)
        v = np.where(inside, self.sdf_v[i, j], -1.)-RADIUS+self.rc.vehicle_tolerance
        s = np.where(inside, self.sdf_s[i, j], -1.)-RADIUS-self.rc.static_margin
        return v.min(1), s.min(1)

    def free(self, pose):
        v, s = self.margins(pose)
        return (v >= 0) & (s >= 0)


def heuristic_grid(clearance, goals, cell=.3):
    """Dijkstra distance (m) from the goal positions over cells a car centre line can pass."""
    world = clearance.world
    stride = int(round(cell/.1))
    free = (np.minimum(clearance.sdf_v+clearance.rc.vehicle_tolerance, clearance.sdf_s-clearance.rc.static_margin) > .7)
    free = free[::stride, ::stride]
    dist = np.full(free.shape, np.inf)
    heap = []
    for g in goals:
        ij = tuple(((g[:2]-world.origin)/cell).astype(int))
        if 0 <= ij[0] < free.shape[0] and 0 <= ij[1] < free.shape[1] and dist[ij] > 0:
            dist[ij] = 0.
            heap.append((0., ij))
    heapq.heapify(heap)
    moves = [(a, b, cell*np.hypot(a, b)) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]
    while heap:
        d, (i, j) = heapq.heappop(heap)
        if d > dist[i, j]:
            continue
        for a, b, w in moves:
            u, v = i+a, j+b
            if 0 <= u < free.shape[0] and 0 <= v < free.shape[1] and free[u, v] and d+w < dist[u, v]:
                dist[u, v] = d+w
                heapq.heappush(heap, (d+w, (u, v)))
    return dist, cell


def hybrid_astar(start, start_gear, goals, clearance, rc=RouteConfig(), goal_cost=None, reverse_direction=-1):
    """Rear-axle poses [N, 3] and gears [N] from start to a pose near one of `goals` [G, 3], or None.
    goal_cost [G] is added when the search ends at that goal (the expert route it skips); the search stops
    when such a finished path is the cheapest open entry, so a late goal is only taken if it is worth it.
    reverse_direction: the search direction that pays reverse_cost (+1 when searching backwards in time)."""
    goal_cost = np.zeros(len(goals)) if goal_cost is None else np.asarray(goal_cost, float)
    tree = cKDTree(goals[:, :2])
    dist, cell = heuristic_grid(clearance, goals)
    world = clearance.world

    def h(p):
        ij = ((p[:2]-world.origin)/cell).astype(int)
        if 0 <= ij[0] < dist.shape[0] and 0 <= ij[1] < dist.shape[1] and np.isfinite(dist[ij[0], ij[1]]):
            return dist[ij[0], ij[1]]
        return 50.

    def key(p, gear):
        return (int(np.floor(p[0]/rc.xy_bin)), int(np.floor(p[1]/rc.xy_bin)),
                int(np.floor((p[2] % (2*np.pi))/(2*np.pi)*rc.yaw_bins)), gear)

    substeps = 4
    ds = rc.step/substeps
    nodes = [(np.asarray(start, float), start_gear, 0., -1, None)]       # pose, gear, g, parent, segment
    heap = [(rc.heuristic_weight*h(nodes[0][0]), 0)]
    closed = set()
    expansions = 0
    finished = {}
    while heap and expansions < rc.max_expansions:
        _, n = heapq.heappop(heap)
        if n in finished:
            index = finished[n]
            segments, gears, m = [], [], n
            while m > 0:
                segments.append(nodes[m][4])
                gears.append(np.full(len(nodes[m][4]), nodes[m][1]))
                m = nodes[m][3]
            out = np.concatenate([np.atleast_2d(start)]+segments[::-1])
            return out, np.concatenate([[start_gear]]+gears[::-1]), int(index), expansions
        pose, gear, g, _, _ = nodes[n]
        k = key(pose, gear)
        if k in closed:
            continue
        closed.add(k)
        expansions += 1
        near = [j for j in tree.query_ball_point(pose[:2], rc.goal_distance)
                if abs(np.angle(np.exp(1j*(pose[2]-goals[j, 2])))) < rc.goal_heading]
        if near:
            j = min(near, key=lambda j: goal_cost[j])
            finished[n] = j
            heapq.heappush(heap, (g+goal_cost[j], n))
            continue
        for direction in (1, -1):
            for kappa in rc.curvatures:
                x, y, yaw = pose
                seg = []
                for _ in range(substeps):
                    turn = direction*kappa*ds
                    mid = yaw+turn/2
                    x, y, yaw = x+direction*ds*np.cos(mid), y+direction*ds*np.sin(mid), yaw+turn
                    seg.append((x, y, yaw))
                seg = np.array(seg)
                if not clearance.free(seg).all():
                    continue
                cost = rc.step*(rc.reverse_cost if direction == reverse_direction else 1.)+rc.steer_cost*abs(kappa)/.45
                if direction != gear:
                    cost += rc.switch_cost
                nodes.append((seg[-1], direction, g+cost, n, seg))
                heapq.heappush(heap, (g+cost+rc.heuristic_weight*h(seg[-1]), len(nodes)-1))
    return None


def time_detour(poses, gears, start_speed, end_speed, rc):
    """10 Hz samples along an A* detour: cruise speed, accelerating out of and braking into each gear change."""
    step = np.r_[0., np.linalg.norm(np.diff(poses[:, :2], axis=0), axis=1)]
    speed = np.zeros(len(poses))
    edges = np.r_[0, np.flatnonzero(np.diff(gears) != 0)+1, len(poses)]
    for n, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        s = np.cumsum(step[a:b])-step[a]
        v0 = start_speed if n == 0 else 0.
        v1 = end_speed if n == len(edges)-2 else 0.
        speed[a:b] = np.minimum.reduce([np.full(b-a, rc.cruise), np.sqrt(v0**2+2*rc.accel*s),
                                        np.sqrt(v1**2+2*rc.accel*(s[-1]-s))])
    speed = np.maximum(speed, .15)
    time = np.r_[0., np.cumsum(step[1:]/((speed[1:]+speed[:-1])/2))]
    at = np.arange(0., time[-1], .1)
    yaw = np.unwrap(poses[:, 2])
    out = np.stack((np.interp(at, time, poses[:, 0]), np.interp(at, time, poses[:, 1]), np.interp(at, time, yaw)), -1)
    g = gears[np.minimum(np.searchsorted(time, at), len(gears)-1)]
    return out, g, np.interp(at, time, speed)


def detour_route(task, world, vehicles, static, rc=RouteConfig()):
    """RouteTask around the static obstacle, the task itself if nothing is blocked, or None if A* fails."""
    from .labels import rear_axle
    if not static.any():
        return RouteTask(task.pose, task.speed, task.reverse, task.target, task.steer), dict(blocked=False)
    clearance = Clearance(world, vehicles, static, rc)
    rear = rear_axle(task.pose)
    _, static_margin = clearance.margins(rear)
    blocked = np.flatnonzero(static_margin < 0)
    if not len(blocked):
        return RouteTask(task.pose, task.speed, task.reverse, task.target, task.steer), dict(blocked=False)
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(rear[:, :2], axis=0), axis=1))]
    leave = int(np.searchsorted(arc, arc[blocked[0]]-rc.leave_before))
    leave = max(0, min(leave, blocked[0]-1))
    vehicle_margin, _ = clearance.margins(rear)
    after = np.flatnonzero((arc >= arc[blocked[-1]]+rc.rejoin_after) & (static_margin >= 0) & (vehicle_margin >= -.3))
    if not len(after):
        after = np.array([len(rear)-1])
    gear = np.where(task.reverse, -1, 1)
    found = hybrid_astar(rear[leave], int(gear[leave]), rear[after], clearance, rc,
                         goal_cost=rc.skip_cost*(arc[after]-arc[after[0]]))
    if found is None:
        return None, dict(blocked=True, leave=leave)
    poses, gears, index, expansions = found
    rejoin = int(after[index])
    samples, g, v = time_detour(poses, gears, abs(task.speed[leave]), abs(task.speed[rejoin]), rc)
    centre = samples.copy()
    centre[:, 0] += cfg.rear_offset_m*np.cos(samples[:, 2])
    centre[:, 1] += cfg.rear_offset_m*np.sin(samples[:, 2])
    pose = np.concatenate((task.pose[:leave], centre, task.pose[rejoin:]))
    speed = np.concatenate((task.speed[:leave], g*v, task.speed[rejoin:]))
    reverse = np.concatenate((task.reverse[:leave], g < 0, task.reverse[rejoin:]))
    pose[:, 2] = np.unwrap(pose[:, 2])
    return RouteTask(pose, speed, reverse, task.target), dict(blocked=True, leave=leave, rejoin=rejoin,
                                                               expansions=expansions, detour_frames=len(samples))
