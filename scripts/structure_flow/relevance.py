"""Is the planned path blocked: the car is forecast along its plan (rear-axle path with forward / reverse
segments and stops) and tested, time step by time step, against every object moved on at its velocity.
Never changes throttle or steering. The supervisor sends forecast_path() to the arbiter; RelevanceGate is the
rule the arbiter learned from (kept in the log beside the arbiter's verdict)."""
import math
import numpy as np


def box_separation(a, b):
    """Full four-axis 2-D OBB SAT; positive means disjoint footprints."""
    dx, dy = b[0]-a[0], b[1]-a[1]
    ca, sa, cb, sb = math.cos(a[2]), math.sin(a[2]), math.cos(b[2]), math.sin(b[2])
    gaps = []
    for x, y in ((ca,sa),(-sa,ca),(cb,sb),(-sb,cb)):
        ra = a[3]*abs(x*ca+y*sa) + a[4]*abs(-x*sa+y*ca)
        rb = b[3]*abs(x*cb+y*sb) + b[4]*abs(-x*sb+y*cb)
        gaps.append(abs(dx*x+dy*y)-ra-rb)
    return max(gaps)


def forecast_path(states, start, cfg, specs, signed_speed_kmh=0.):
    """Follow curved rear-axle path, including F/R segments and stop intervals."""
    horizon = cfg['gate_horizon_s']; dt = cfg['gate_sample_dt_s']
    samples = np.arange(0., horizon + dt*.5, dt)
    rows = np.asarray(states, dtype=float)
    if rows.ndim != 2 or len(rows) < 2:
        return [(float(t), signed_speed_kmh/3.6*t, 0., 0.) for t in samples]
    rows = rows[max(0,min(int(start),len(rows)-2)):]
    knots = [(0., rows[0][0], rows[0][1], rows[0][2])]
    at = 0.
    for previous, point in zip(rows, rows[1:]):
        distance = math.hypot(point[0]-previous[0], point[1]-previous[1])
        if len(point) > 3 and point[3]*previous[3] < 0:
            at += .3; knots.append((at, previous[0], previous[1], previous[2]))
        nominal = cfg['k4_reverse_kmh'] if len(point)>3 and point[3]<0 else cfg['k4_forward_kmh']
        if len(point)>5 and abs(point[5])>.05: nominal = min(nominal, abs(point[5]))
        speed = max(.6, nominal/3.6)
        at += distance/speed
        knots.append((at, point[0], point[1], point[2]))
        if at >= horizon: break
    knots = np.asarray(knots); yaw = np.unwrap(knots[:,3])
    xs, ys, angles = (np.interp(samples, knots[:,0], values) for values in (knots[:,1],knots[:,2],yaw))
    centre = float(specs.get('rear_axle_to_center',-float(specs.get('rear_axle_x', -1.4156))))
    lateral = float(specs.get('rear_axle_lateral_offset',-float(specs.get('rear_axle_y', 0.))))
    return [(float(t), float(x+centre*math.cos(a)-lateral*math.sin(a)),
             float(y+centre*math.sin(a)+lateral*math.cos(a)), float(a))
            for t,x,y,a in zip(samples,xs,ys,angles)]


class RelevanceGate:
    def __init__(self, cfg, specs):
        self.cfg, self.specs = cfg, specs
        self.active = False; self.clear_since = None; self.last_time = None

    def classify(self, track):
        if track.get('age_s', 0.) > self.cfg['gate_track_max_age_s']: return False
        if track.get('class_name') == 'pedestrian': return True
        if track.get('class_name') != 'vehicle': return False
        if track.get('ever_moving', False): return True
        return (track.get('hits', 0) >= 3 and track.get('motion_evidence', 0) >= 2
                and track.get('motion_displacement_m', 0.) >= self.cfg['gate_vehicle_motion_m']
                and track.get('motion_straightness', 0.) >= .6
                and math.hypot(track.get('vx',0.),track.get('vy',0.)) >= self.cfg['gate_vehicle_speed_mps'])

    def update(self, now, tracks, states, start=0, signed_speed_kmh=0.):
        if self.last_time is not None and now <= self.last_time:
            raise ValueError('Relevance gate needs a new synchronized CVT observation')
        self.last_time = now
        trajectory = forecast_path(states,start,self.cfg,self.specs,signed_speed_kmh)
        margin = self.cfg['gate_margin_m']; relevant = []; minimum = 10.
        candidates = [t for t in tracks if self.classify(t)]
        for track in candidates:
            first = None; gap = 10.
            vx,vy = float(track.get('vx',0.)),float(track.get('vy',0.))
            speed = math.hypot(vx,vy); cap = 4. if track['class_name']=='pedestrian' else 8.
            if speed > cap: vx,vy = vx*cap/speed,vy*cap/speed
            for t,x,y,yaw in trajectory:
                ego = (x,y,yaw,self.specs['half_length']+margin,self.specs['half_width']+margin)
                other = (track['x']+vx*t,track['y']+vy*t,track.get('yaw',0.),
                         max(.15,track.get('length',.6)/2),max(.15,track.get('width',.6)/2))
                separation = box_separation(ego,other); gap=min(gap,separation)
                if separation <= 0 and first is None: first=t
            minimum=min(minimum,gap)
            if first is not None:
                relevant.append(dict(object_id=track['object_id'],class_name=track['class_name'],
                    conflict_time_s=first,minimum_gap_m=gap,x=track['x'],y=track['y'],vx=vx,vy=vy,
                    age_s=track.get('age_s',0.),ever_moving=track.get('ever_moving',False)))
        if relevant:
            self.active=True;self.clear_since=None
        elif self.active:
            if self.clear_since is None: self.clear_since=now
            if now-self.clear_since+1.e-6 >= self.cfg['gate_clear_s']: self.active=False
        clear_age=0. if self.clear_since is None else now-self.clear_since
        return dict(active=self.active,raw_conflict=bool(relevant),
                    reason='path_conflict' if relevant else 'clear_confirmation' if self.active else 'clear',
                    conflict_time_s=min((r['conflict_time_s'] for r in relevant),default=self.cfg['gate_horizon_s']),
                    minimum_gap_m=minimum,clear_age_s=clear_age,
                    candidate_count=len(candidates),relevant_count=len(relevant),tracks=relevant,
                    input_source='CVT_tracks_ego_odometry_K4',action_speed_override=False)
