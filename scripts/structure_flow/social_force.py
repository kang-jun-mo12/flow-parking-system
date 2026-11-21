"""Social force walkers (SFM): copied unchanged from carla-cvt-hybrid-autopark/scripts/pedestrian_social_force.py
(the DynaPark-Ped reconstruction used for the SAC work), with its calibration
(config/dynapark_sfm_reconstruction.json, v5) as SFM_CONFIG. Original docstring below."""

"""Explicitly calibrated four-term SFM and static walking routes.

The paper publishes the sum-of-forces/Euler structure, not the component
functions. All amplitudes below are accelerations (force divided by mass).
Crossing escape selects a goal away from the current ego after a standstill.
No teleport or post-arrival retargeting.
"""
import heapq
import math


def unit(x, y):
    n = math.hypot(x, y)
    return (x / n, y / n) if n > 1e-9 else (0., 0.)


def obstacle_gap(position, box, radius=0.):
    """Signed distance to an expanded OBB and outward normal in world axes."""
    c, s = math.cos(box['yaw']), math.sin(box['yaw'])
    dx, dy = position[0] - box['x'], position[1] - box['y']
    x, y = c * dx + s * dy, -s * dx + c * dy
    hx, hy = box['half_length'] + radius, box['half_width'] + radius
    ox, oy = max(abs(x) - hx, 0.), max(abs(y) - hy, 0.)
    gap = math.hypot(ox, oy)
    if gap > 1e-9:
        nx, ny = math.copysign(ox / gap, x), math.copysign(oy / gap, y)
    elif hx - abs(x) < hy - abs(y):
        gap, nx, ny = -(hx - abs(x)), math.copysign(1., x), 0.
    else:
        gap, nx, ny = -(hy - abs(y)), 0., math.copysign(1., y)
    return gap, (c * nx - s * ny, s * nx + c * ny)


def acceleration_terms(ped, goal, obstacles, others, ego, cfg):
    x, y = ped['position']; vx, vy = ped['velocity']
    radius = cfg['pedestrian_radius_m']
    ux, uy = unit(goal[0] - x, goal[1] - y)
    distance = math.hypot(goal[0] - x, goal[1] - y)
    desired = min(ped['preferred_speed'], distance / cfg['relaxation_s'])
    terms = {'drive': ((desired * ux - vx) / cfg['relaxation_s'],
                       (desired * uy - vy) / cfg['relaxation_s']),
             'obstacle': [0., 0.], 'pedestrian': [0., 0.], 'vehicle': [0., 0.]}
    def repel(name, gap, normal, amplitude, decay, gain=1.):
        magnitude = amplitude * gain * math.exp(-max(-.5, gap) / decay)
        terms[name][0] += magnitude * normal[0]
        terms[name][1] += magnitude * normal[1]
    for box in obstacles:
        gap, normal = obstacle_gap((x, y), box, radius)
        if gap < 3.:
            repel('obstacle', gap, normal, cfg['static_acceleration_mps2'], cfg['static_decay_m'])
    for other in others:
        if other['id'] == ped['id']:
            continue
        dx, dy = x - other['position'][0], y - other['position'][1]
        gap = math.hypot(dx, dy) - 2. * radius
        if gap < 3.:
            normal = unit(dx, dy)
            if normal == (0., 0.):
                normal = (1. if ped['id'] > other['id'] else -1., 0.)
            repel('pedestrian', gap, normal, cfg['pedestrian_acceleration_mps2'], cfg['pedestrian_decay_m'])
    if ego is not None:
        gap, normal = obstacle_gap((x, y), ego, radius)
        closing = max(0., (ego.get('vx', 0.) - vx) * normal[0]
                      + (ego.get('vy', 0.) - vy) * normal[1])
        # Relative-motion anticipation changes repulsion, not the pedestrian goal.
        anticipated_gap = gap - min(2., closing * cfg['vehicle_anticipation_s'])
        if anticipated_gap < 5.:
            repel('vehicle', anticipated_gap, normal, cfg['vehicle_acceleration_mps2'],
                  cfg['vehicle_decay_m'], 1. + cfg['relative_closing_gain'] * min(2., closing))
    return terms


def integrate_velocity(velocity, terms, dt, max_speed, cfg):
    ax = sum(v[0] for v in terms.values()); ay = sum(v[1] for v in terms.values())
    n = math.hypot(ax, ay)
    scale = min(1., cfg['max_acceleration_mps2'] / max(n, 1e-9))
    vx, vy = velocity[0] + ax * scale * dt, velocity[1] + ay * scale * dt
    scale = min(1., max_speed / max(math.hypot(vx, vy), 1e-9))
    return vx * scale, vy * scale


def update_detour(ped, ego, static_grid, cfg, elapsed, others=()):
    """Escape a force equilibrium; optionally abandon a blocked crossing.

    A stationary ego is included in a temporary walking graph. The first
    connector may leave its soft repulsion margin, but never its physical body.
    Keep the selected route rather than choosing alternating sides each tick.
    """
    pos = ped['position']
    anchor = ped.setdefault('progress_anchor',pos)
    if math.hypot(pos[0]-anchor[0],pos[1]-anchor[1]) >= .30:
        ped['progress_anchor'] = pos
        ped['progress_time'] = elapsed
    since = ped.setdefault('progress_time',elapsed)
    if elapsed-since < cfg['detour_stall_s'] or elapsed < ped.get('detour_retry_at',0.):
        return
    escape = cfg.get('stalled_crossing_escape',False) and 'crossing_direction' in ped
    ped['detour_retry_at'] = elapsed+(min(.5,cfg['detour_retry_s']) if escape else cfg['detour_retry_s'])
    was_pending=bool(ped.get('escape_route_pending',False))
    if escape:
        ped['escape_route_pending'] = True
    if escape:
        sign = -1 if ego['y']>pos[1] else 1
        ped.setdefault('original_crossing_goal',ped['goal'])
        # A successful route stays fixed while the walker makes progress.
        # If it gets stuck again, choose the side from THIS encounter, not
        # from the first ego pose or from the original crossing direction.
        if sign!=ped.get('escape_direction') or not was_pending:
            ped.setdefault('escape_events',[]).append(dict(time=elapsed,
                pedestrian_xy=list(pos),ego_xy=[ego['x'],ego['y']],direction=sign))
        ped['escape_direction'] = sign
        ped['escape_ego_xy'] = [ego['x'],ego['y']]
        ped['goal'] = (cfg['escape_aisle_x'],cfg['escape_endpoints_y'][1 if sign>0 else 0])
        # End-line completion for the original X crossing no longer applies.
        ped['crossing_direction'] = 0
        ped['navigation_status'] = 'escape_requested_away_from_ego'
    if not escape and math.hypot(ego.get('vx',0.),ego.get('vy',0.)) > cfg['detour_ego_max_speed_mps']:
        ped['navigation_status'] = 'waiting_for_moving_vehicle'
        return
    route_cfg = dict(cfg)
    if escape:
        bounds = list(cfg['walk_bounds_xy'])
        bounds[2],bounds[3] = cfg['escape_endpoints_y'][0]-1.,cfg['escape_endpoints_y'][1]+1.
        route_cfg['walk_bounds_xy'] = bounds
    # A walker already waiting at an exit must not become another stationary
    # obstacle that everyone tries to reach. Route around stopped people and
    # reserve distinct free destinations on the same away-side exit line.
    radius=cfg['pedestrian_radius_m']
    stopped_people=[dict(x=p['position'][0],y=p['position'][1],yaw=0.,
        half_length=radius,half_width=radius) for p in others if p['id']!=ped['id']
        and (p.get('arrived') or math.hypot(*p.get('velocity',(0.,0.)))<.15)]
    boxes=static_grid.obstacles+[ego]+stopped_people
    grid = WalkingGrid(boxes,route_cfg)
    if escape and grid.nodes:
        exit_y=ped['goal'][1];aisle=cfg['escape_aisle_x']
        reserved=[p['goal'] for p in others if p['id']!=ped['id'] and p.get('escape_direction')]
        goals=[(aisle+offset,exit_y-sign*back) for back in (0.,1.6,3.2) for offset in (0.,-1.6,1.6)]
        goals=[g for g in goals if grid.point_clear(g)
               and all(math.hypot(g[0]-r[0],g[1]-r[1])>=2.*radius+.5 for r in reserved)]
        if not goals:
            ped['navigation_status']='waiting_for_clear_goal';return
        ped['goal']=goals[0]
    if not grid.nodes or not grid.point_clear(ped['goal']):
        ped['navigation_status'] = 'waiting_for_clear_goal'
        return
    # The walker can already be inside the soft margin when a standstill is
    # detected. Only permit outward motion there, with physical clearance.
    def connector_clear(end):
        count = max(1,int(math.ceil(math.hypot(end[0]-pos[0],end[1]-pos[1])/.10)))
        previous = [obstacle_gap(pos,box,cfg['pedestrian_radius_m'])[0] for box in boxes]
        for i in range(count+1):
            point = (pos[0]+(end[0]-pos[0])*i/count,pos[1]+(end[1]-pos[1])*i/count)
            for index,box in enumerate(boxes):
                gap = obstacle_gap(point,box,cfg['pedestrian_radius_m'])[0]
                # A walker stopped against a parked car may already overlap
                # the nominal pedestrian-radius margin. Rejecting the FIRST
                # sample traps it forever, including routes going away from
                # that car. Permit only monotonically outward movement until
                # the positive margin is restored; never enter another body.
                if gap < -.10 or (gap < .05 and (previous[index] >= .05 or gap < previous[index]-1e-6)):
                    return False
                previous[index] = gap
        return True
    candidates = sorted(grid.nodes,key=lambda n:math.hypot(grid.point(n)[0]-pos[0],grid.point(n)[1]-pos[1]))
    for node in candidates[:32]:
        start = grid.point(node)
        if not connector_clear(start):
            continue
        route = grid.route(start,ped['goal'])
        if route is None or not grid.segment_clear(route[-1],ped['goal']):
            continue
        # Avoid stopping at a nearest grid node inside the repulsion field.
        # Join the farthest visible waypoint with the same physical/outward
        # connector check, so the walking force points along the escape route.
        route = route+[ped['goal']]
        join = len(route)-1
        while join>0 and not connector_clear(route[join]):
            join -= 1
        ped['route'] = [pos]+route[join:]
        ped['escape_route_pending'] = False
        ped['waypoint'] = 1
        ped['detour_grid'] = grid
        ped['detour_count'] = ped.get('detour_count',0)+1
        ped['navigation_status'] = 'escaping_away_from_ego' if escape else 'detouring_to_original_goal'
        ped['progress_anchor'],ped['progress_time'] = pos,elapsed
        return
    ped['navigation_status'] = 'waiting_for_clear_route'


class WalkingGrid:
    """Static-only 8-connected walking graph; built once at episode reset."""
    def __init__(self, obstacles, cfg):
        self.obstacles = obstacles
        self.bounds = cfg['walk_bounds_xy']; self.step = cfg['route_grid_m']
        self.margin = cfg['pedestrian_radius_m'] + cfg['route_clearance_m']
        xmin, xmax, ymin, ymax = self.bounds
        self.free = set()
        for i in range(int((xmax - xmin) / self.step) + 1):
            for j in range(int((ymax - ymin) / self.step) + 1):
                node = (i, j)
                if self.point_clear(self.point(node)):
                    self.free.add(node)
        self.nodes = sorted(self.free)

    def point(self, node):
        return self.bounds[0] + node[0] * self.step, self.bounds[2] + node[1] * self.step

    def point_clear(self, point):
        return all(obstacle_gap(point, box, self.margin)[0] >= 0. for box in self.obstacles)

    def closest(self, point):
        return min(self.nodes, key=lambda node: math.hypot(self.point(node)[0] - point[0], self.point(node)[1] - point[1]))

    def segment_clear(self, a, b):
        count = max(1, int(math.ceil(math.hypot(b[0]-a[0], b[1]-a[1]) / .15)))
        return all(self.point_clear((a[0]+(b[0]-a[0])*i/count, a[1]+(b[1]-a[1])*i/count)) for i in range(count+1))

    def route(self, start, goal):
        start, goal = self.closest(start), self.closest(goal)
        queue = [(0., start)]; cost = {start: 0.}; parent = {}
        while queue:
            _, current = heapq.heappop(queue)
            if current == goal:
                nodes = [current]
                while current in parent:
                    current = parent[current]; nodes.append(current)
                points = [self.point(n) for n in reversed(nodes)]
                # Greedy line-of-sight simplification checks the entire segment.
                path = [points[0]]; index = 0
                while index < len(points)-1:
                    nxt = len(points)-1
                    while nxt > index+1 and not self.segment_clear(points[index], points[nxt]):
                        nxt -= 1
                    path.append(points[nxt]); index = nxt
                return path
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)):
                nxt = (current[0]+dx, current[1]+dy)
                if nxt not in self.free:
                    continue
                if dx and dy and ((current[0]+dx,current[1]) not in self.free or (current[0],current[1]+dy) not in self.free):
                    continue
                distance = cost[current] + math.hypot(dx,dy)
                if distance >= cost.get(nxt,float('inf')):
                    continue
                if not self.segment_clear(self.point(current),self.point(nxt)):
                    continue
                cost[nxt] = distance; parent[nxt] = current
                heapq.heappush(queue,(distance+math.hypot(nxt[0]-goal[0],nxt[1]-goal[1]),nxt))
        return None


# config/dynapark_sfm_reconstruction.json of carla-cvt-hybrid-autopark (v5), without its notes
SFM_CONFIG = {
    "version": "dynapark-sfm-v5-two-second-escape",
    "detour_stall_s": 2.0,
    "detour_retry_s": 3.0,
    "detour_ego_max_speed_mps": 0.3,
    "walk_bounds_xy": [
        266.0,
        302.0,
        -239.0,
        -180.0
    ],
    "route_grid_m": 0.8,
    "pedestrian_radius_m": 0.3,
    "route_clearance_m": 0.8,
    "relaxation_s": 0.5,
    "static_acceleration_mps2": 4.0,
    "static_decay_m": 0.5,
    "pedestrian_acceleration_mps2": 2.5,
    "pedestrian_decay_m": 0.4,
    "vehicle_acceleration_mps2": 4.0,
    "vehicle_decay_m": 0.8,
    "vehicle_anticipation_s": 0.5,
    "relative_closing_gain": 0.7,
    "max_acceleration_mps2": 6.0,
    "arrival_radius_m": 0.35,
    "speed_groups": {
        "adult_male": [
            1.4,
            0.2,
            0.75,
            1.9
        ],
        "adult_female": [
            1.3,
            0.2,
            0.75,
            1.85
        ],
        "child_male": [
            1.15,
            0.23,
            0.6,
            1.65
        ],
        "child_female": [
            1.1,
            0.23,
            0.6,
            1.6
        ]
    },
    "crossing_endpoints_x": [
        276.3,
        294.6
    ],
    "route_policy": "target_slot_world_x_crossing_reserved_opposite_bay",
    "stalled_crossing_escape": True,
    "escape_aisle_x": 285.45,
    "escape_endpoints_y": [
        -237.0,
        -182.0
    ]
}
