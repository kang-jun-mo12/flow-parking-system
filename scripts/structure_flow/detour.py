"""Detour teacher: re-optimise a window's spline latent so the path clears new obstacles.

The label latent (2 phases x 12 coefficients x [log length, curvature, speed]) is decoded with a
differentiable copy of labels.decode, and all 50 used values move together: lengths, curvature
(the detour) and speed (waiting for something that moves). The structure h stays the label's.

Cost, per window:
  end       the end pose (position and heading) to the nearest allowed goal pose: the end of the label
            for a fixed window, or any pose far enough along the expert's reference when driving
  track     each sample's distance to the reference beyond a 0.15 m band, so the detour returns to it
  static    summed over body points: points inside the signed distance field of the new obstacles (margin `static_margin`)
  vehicles  the same against the DDP vehicle mask, with a small tolerance: label paths graze it
  dynamic   body rectangle against moving discs at the time the path reaches each sample
  curvature beyond the largest curvature in the labels, speed outside [0, max]
  smooth    second differences of the curvature and speed coefficients
  prior     distance to the label latent, so an unblocked window keeps its label
Several starts per window (the label, shifted and bent to either side, a slow copy) run as one
batch; the cheapest start that passes the exact checks is kept.

Frames: paths are rear-axle poses in the rear-axle frame of the window start; the BEV and the
obstacles are in the ego-centre frame (x forward, y right, row = 100+10x, col = 100+10y).
"""
from dataclasses import dataclass
import time
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from .config import DEFAULT
from .labels import basis

BODY_LENGTH, BODY_WIDTH = 4.4, 1.9      # the swept body of tools/eval_structure_flow.hits


@dataclass(frozen=True)
class DetourConfig:
    iterations: int = 250
    lr: float = .03
    max_curvature: float = .48          # 1/m, the largest |curvature| in the training labels
    static_margin: float = .3           # m kept from new obstacles
    dynamic_margin: float = .4          # m kept from moving discs, at the predicted time
    vehicle_tolerance: float = .05      # m the body may enter the DDP vehicle mask
    min_speed: float = .1               # m/s, floor of the speed used for the arrival time (dwell at stops)
    w_end: float = 4.
    w_track: float = 2.
    w_static: float = 20.             # summed over body points and samples, not averaged
    w_vehicles: float = 5.
    w_dynamic: float = 20.
    w_curvature: float = 100.
    w_speed: float = 20.
    w_smooth: float = .5
    w_prior: float = .05
    bend: float = .6                    # curvature coefficient amplitude of the side starts
    end_tolerance: float = .5           # m, exact check of the end position
    end_heading: float = .175           # rad (10 deg)


def body_offsets(cols=11, rows=5, device='cpu'):
    u, v = torch.meshgrid(torch.linspace(-.5, .5, cols)*BODY_LENGTH, torch.linspace(-.5, .5, rows)*BODY_WIDTH,
                          indexing='ij')
    return torch.stack((u.flatten(), v.flatten()), -1).to(device)        # [P, 2]


def structure(h, cfg=DEFAULT):
    """Per window: gear of each phase (+1/-1) and whether the phase is used. h is [B] ints."""
    h = np.asarray(h)
    phases = h % cfg.max_phases+1
    first = np.where(h < cfg.max_phases, 1., -1.)
    gear = np.stack([first*(-1.)**i for i in range(cfg.max_phases)], 1)
    used = np.stack([phases > i for i in range(cfg.max_phases)], 1)
    return gear, used, phases


class Decoder:
    """labels.decode in torch over a batch. Returns rear-axle x, y, yaw, signed speed, time [B, phases*S]."""

    def __init__(self, cfg=DEFAULT, device='cuda'):
        self.cfg, self.device = cfg, device
        self.grid = torch.as_tensor(basis(cfg.coefficients, cfg.samples), dtype=torch.float32, device=device)

    def speed_coefficients(self, latent, start_speed, phases):
        """The imposed speeds of labels.decode: the current speed first, zero around a gear change."""
        cfg, rows = self.cfg, []
        for index in range(cfg.max_phases):
            c = latent[:, index, :, 2]
            first = c[:, :1]
            if index == 0:
                first = torch.where(torch.isnan(start_speed)[:, None], first, (start_speed.abs()/cfg.max_speed_mps)[:, None])
            else:
                first = torch.zeros_like(first)
            last = c[:, -1:]
            gear_change = torch.as_tensor(phases-1 > index, device=c.device)[:, None]
            last = torch.where(gear_change, torch.zeros_like(last), last)
            rows.append(torch.cat((first, c[:, 1:-1], last), 1))
        return torch.stack(rows, 1)

    def __call__(self, latent, h, start_speed, min_speed=.3):
        cfg = self.cfg
        gear, used, phases = structure(h, cfg)
        gear_t = torch.as_tensor(gear, dtype=torch.float32, device=latent.device)
        used_t = torch.as_tensor(used, dtype=torch.float32, device=latent.device)
        coefficients = self.speed_coefficients(latent, start_speed, phases)
        x0 = torch.zeros(len(latent), device=latent.device)
        y0, yaw0, t0 = x0.clone(), x0.clone(), x0.clone()
        out = dict(x=[], y=[], yaw=[], speed=[], time=[], curvature=[], gear=[])
        for index in range(cfg.max_phases):
            g = gear_t[:, index:index+1]
            length = torch.exp(latent[:, index, 0, 0])*used_t[:, index]
            curvature = latent[:, index, :, 1]@self.grid.T*cfg.curvature_scale          # [B, S]
            speed = (coefficients[:, index]@self.grid.T).clamp(0, 1)*cfg.max_speed_mps
            step = (length/cfg.samples)[:, None]
            turn = g*curvature*step
            yaw = yaw0[:, None]+torch.cumsum(turn, 1)
            middle = yaw-turn/2
            x = x0[:, None]+torch.cumsum(g*step*torch.cos(middle), 1)
            y = y0[:, None]+torch.cumsum(g*step*torch.sin(middle), 1)
            t = t0[:, None]+torch.cumsum(step/speed.clamp(min=min_speed), 1)
            for key, value in (('x', x), ('y', y), ('yaw', yaw), ('speed', g*speed), ('time', t),
                               ('curvature', curvature*used_t[:, index:index+1]), ('gear', g.expand_as(x))):
                out[key].append(value)
            x0, y0, yaw0, t0 = x[:, -1], y[:, -1], yaw[:, -1], t[:, -1]
        out = {key: torch.cat(value, 1) for key, value in out.items()}
        out['coefficients'] = coefficients
        return out


def to_centre(x, y, yaw, cfg=DEFAULT):
    """Rear-axle pose in the rear-axle start frame -> body centre in the ego-centre frame."""
    off = cfg.rear_offset_m
    return x-off+off*torch.cos(yaw), y+off*torch.sin(yaw)


def signed_distance(mask):
    """Metres from each cell centre to the mask edge, negative inside. All-free masks give +10 m."""
    if not mask.any():
        return np.full(mask.shape, 10., np.float32)
    outside = distance_transform_edt(~mask)*.1
    inside = distance_transform_edt(mask)*.1
    return (outside-inside).astype(np.float32)


def sample_field(field, px, py):
    """field [B, 200, 200] metres; px, py [B, N] ego metres -> [B, N] bilinear values (border outside)."""
    row = 100+10*px-.5
    col = 100+10*py-.5
    grid = torch.stack((col/199*2-1, row/199*2-1), -1)[:, None]           # [B, 1, N, 2]
    return torch.nn.functional.grid_sample(field[:, None], grid, align_corners=True, padding_mode='border')[:, 0, 0]


def rectangle_distance(cx, cy, yaw, ox, oy):
    """Distance from points (ox, oy) to the body rectangle at (cx, cy, yaw); negative inside."""
    dx, dy = ox-cx, oy-cy
    c, s = torch.cos(yaw), torch.sin(yaw)
    u, v = (c*dx+s*dy).abs()-BODY_LENGTH/2, (-s*dx+c*dy).abs()-BODY_WIDTH/2
    outside = torch.sqrt(u.clamp(min=0)**2+v.clamp(min=0)**2+1e-9)
    return outside+torch.minimum(torch.maximum(u, v), torch.zeros_like(u))


def bend_shapes(coefficients):
    """Curvature coefficient shapes of the side starts: `shift` (+, - lobes) leaves the path moved sideways,
    `bend` (+, -, + lobes) moves it out and back. Both bring the heading back to where it was."""
    k = np.arange(coefficients)/(coefficients-1)
    shift = np.sin(2*np.pi*k)*np.sin(np.pi*k)**.5
    bend = np.sin(3*np.pi*k)*np.sin(np.pi*k)
    return shift/np.abs(shift).max(), bend/np.abs(bend).max()


class DetourTeacher:
    """Windows are dicts:
    latent [2, K, 3], h, start_speed (m/s, NaN = free), vehicles / static [200, 200] bool (ego-centre BEV),
    movers [M, 5] (x, y, vx, vy, radius in the ego-centre frame at the window start),
    optional reference [R, 3] rear-axle poses in the window frame with masks track [R] (the path should stay
    near these) and goal [R] (the path should end on one of these, heading included). Without a reference the
    goal is the end of the decoded label, the test of a fixed window."""

    def __init__(self, cfg=DEFAULT, detour=DetourConfig(), device='cuda'):
        self.cfg, self.detour, self.device = cfg, detour, device
        self.decoder = Decoder(cfg, device)
        self.body = body_offsets(device=device)

    def starts(self, latent):
        """[6, 2, K, 3] start latents from the label latent [2, K, 3]: the label, shifted and bent to either
        side, and a slow copy."""
        d = self.detour
        shift, bend = bend_shapes(self.cfg.coefficients)
        out = [latent.copy()]
        for shape in (shift, bend):
            for sign in (1, -1):
                one = latent.copy()
                one[0, :, 1] += sign*d.bend*shape
                out.append(one)
        slow = latent.copy()
        slow[:, 1:, 2] *= .4
        out.append(slow)
        return np.stack(out)

    def body_points(self, cx, cy, yaw):
        c, s = torch.cos(yaw), torch.sin(yaw)
        u, v = self.body[:, 0], self.body[:, 1]
        px = (cx[..., None]+c[..., None]*u-s[..., None]*v).flatten(1)
        py = (cy[..., None]+s[..., None]*u+c[..., None]*v).flatten(1)
        return px, py

    @staticmethod
    def mover_gap(cx, cy, yaw, time, movers, hold=3.):
        """[B, N+H, M] distance from the body to each mover disc at the time the path gets there, then with the
        car standing at the end of the path for `hold` more seconds (a car may not stop in a walker's way)."""
        extra = torch.linspace(.5, hold, 6, device=time.device)
        cx = torch.cat((cx, cx[:, -1:].expand(-1, len(extra))), 1)
        cy = torch.cat((cy, cy[:, -1:].expand(-1, len(extra))), 1)
        yaw = torch.cat((yaw, yaw[:, -1:].expand(-1, len(extra))), 1)
        time = torch.cat((time, time[:, -1:]+extra), 1)
        t = time[..., None]
        ox = movers[:, None, :, 0]+movers[:, None, :, 2]*t
        oy = movers[:, None, :, 1]+movers[:, None, :, 3]*t
        gap = rectangle_distance(cx[..., None], cy[..., None], yaw[..., None], ox, oy)-movers[:, None, :, 4]
        return torch.where(movers[:, None, :, 4] > 0, gap, torch.full_like(gap, 99.))

    @staticmethod
    def goal_distance(path, reference, goal):
        """[B] squared distance of the end pose (position and heading) to the nearest goal pose."""
        dx = path['x'][:, -1:]-reference[..., 0]
        dy = path['y'][:, -1:]-reference[..., 1]
        dyaw = path['yaw'][:, -1:]-reference[..., 2]
        cost = dx**2+dy**2+4*(1-torch.cos(dyaw))
        return torch.where(goal, cost, torch.full_like(cost, 1e4)).amin(1)

    def costs(self, latent, h, start_speed, scene):
        d, cfg = self.detour, self.cfg
        path = self.decoder(latent, h, start_speed, d.min_speed)
        cx, cy = to_centre(path['x'], path['y'], path['yaw'], cfg)
        px, py = self.body_points(cx, cy, path['yaw'])
        terms = dict(end=d.w_end*self.goal_distance(path, scene['reference'], scene['goal']))
        reference = scene['reference']
        gap = (path['x'][..., None]-reference[:, None, :, 0])**2+(path['y'][..., None]-reference[:, None, :, 1])**2
        gap = torch.where(scene['track'][:, None], gap, torch.full_like(gap, 1e4)).amin(2)
        terms['track'] = d.w_track*((gap+1e-9).sqrt()-.15).clamp(min=0).pow(2).mean(1)
        terms['static'] = d.w_static*((d.static_margin-sample_field(scene['static_field'], px, py)).clamp(min=0)**2).sum(1)
        terms['vehicles'] = d.w_vehicles*((-d.vehicle_tolerance-sample_field(scene['vehicle_field'], px, py)).clamp(min=0)**2).sum(1)
        if scene['movers'].shape[1]:
            gap = self.mover_gap(cx, cy, path['yaw'], path['time'], scene['movers'])
            terms['dynamic'] = d.w_dynamic*((d.dynamic_margin-gap).clamp(min=0)**2).sum((1, 2))
        else:
            terms['dynamic'] = torch.zeros_like(terms['end'])
        terms['curvature'] = d.w_curvature*((path['curvature'].abs()-d.max_curvature).clamp(min=0)**2).mean(1)
        raw = latent[..., 2]
        terms['speed'] = d.w_speed*((raw-1).clamp(min=0)**2+(-raw).clamp(min=0)**2).mean((1, 2))
        bends = latent[..., 1]
        terms['smooth'] = d.w_smooth*(((bends[..., 2:]-2*bends[..., 1:-1]+bends[..., :-2])**2).sum((1, 2)) +
                                      ((raw[..., 2:]-2*raw[..., 1:-1]+raw[..., :-2])**2).sum((1, 2)))
        return terms, path, (cx, cy)

    def scene(self, windows, owner, expert_paths):
        """Batched tensors of the windows, one row per start."""
        fields_s = {i: signed_distance(windows[i]['static']) for i in set(owner.tolist())}
        fields_v = {i: signed_distance(windows[i]['vehicles']) for i in set(owner.tolist())}
        refs = []
        for row, i in enumerate(owner):
            w = windows[i]
            if w.get('reference') is not None:
                refs.append((np.asarray(w['reference'], np.float32), np.asarray(w['track'], bool),
                             np.asarray(w['goal'], bool)))
            else:
                p = expert_paths[row]
                goal = np.zeros(len(p), bool)
                goal[-1] = True
                refs.append((p, np.ones(len(p), bool), goal))
        R = max(len(r[0]) for r in refs)
        reference = np.zeros((len(owner), R, 3), np.float32)
        track = np.zeros((len(owner), R), bool)
        goal = np.zeros((len(owner), R), bool)
        for row, (r, t, g) in enumerate(refs):
            reference[row, :len(r)], track[row, :len(r)], goal[row, :len(r)] = r, t, g
        most = max([len(windows[i]['movers']) for i in owner]+[0])
        movers = np.zeros((len(owner), most, 5), np.float32)
        for row, i in enumerate(owner):
            if len(windows[i]['movers']):
                movers[row, :len(windows[i]['movers'])] = windows[i]['movers']

        def tensor(a):
            return torch.tensor(np.asarray(a), device=self.device)
        return dict(reference=tensor(reference), track=tensor(track), goal=tensor(goal), movers=tensor(movers),
                    static_field=tensor(np.stack([fields_s[i] for i in owner])),
                    vehicle_field=tensor(np.stack([fields_v[i] for i in owner])),
                    static=tensor(np.stack([windows[i]['static'] for i in owner])),
                    vehicles=tensor(np.stack([windows[i]['vehicles'] for i in owner])))

    def solve(self, windows, search=True):
        """One dict per window: the chosen latent, whether it passed the exact checks, and the checks.
        search=False keeps only the label start (with iterations=0, a check of the label itself)."""
        d, cfg, dev = self.detour, self.cfg, self.device
        started = time.perf_counter()
        rows, owner = [], []
        for index, w in enumerate(windows):
            for start in (self.starts(w['latent']) if search else w['latent'][None]):
                rows.append(start)
                owner.append(index)
        owner = np.array(owner)
        latent = torch.tensor(np.stack(rows), dtype=torch.float32, device=dev)
        expert = torch.tensor(np.stack([windows[i]['latent'] for i in owner]), dtype=torch.float32, device=dev)
        h = np.array([windows[i]['h'] for i in owner])
        start_speed = torch.tensor([windows[i]['start_speed'] for i in owner], dtype=torch.float32, device=dev)
        with torch.no_grad():
            p = self.decoder(expert, h, start_speed, d.min_speed)
            expert_paths = torch.stack((p['x'], p['y'], p['yaw']), -1).cpu().numpy()
        scene = self.scene(windows, owner, expert_paths)
        _, used, _ = structure(h, cfg)
        free = torch.tensor(used, dtype=torch.float32, device=dev)[:, :, None, None].expand_as(latent).clone()
        free[:, :, 1:, 0] = 0          # log length lives at knot 0 only
        latent.requires_grad_(True)
        optimiser = torch.optim.Adam([latent], lr=d.lr)
        for _ in range(d.iterations):
            terms, _, _ = self.costs(latent, h, start_speed, scene)
            prior = d.w_prior*(((latent-expert)*free)**2).sum((1, 2, 3))
            total = sum(terms.values())+prior
            optimiser.zero_grad()
            total.sum().backward()
            latent.grad *= free
            optimiser.step()
        with torch.no_grad():
            terms, path, centre = self.costs(latent, h, start_speed, scene)
            prior = d.w_prior*(((latent-expert)*free)**2).sum((1, 2, 3))
            total = (sum(terms.values())+prior).cpu().numpy()
            checks = self.check(path, centre, scene)
        final = latent.detach().cpu().numpy()
        results = []
        for index in range(len(windows)):
            mine = np.flatnonzero(owner == index)
            passing = [r for r in mine if checks['ok'][r]] or [r for r in mine if checks['clear'][r]]
            best = min(passing or mine, key=lambda r: total[r])
            results.append(dict(latent=final[best], ok=bool(checks['ok'][best]), clear=bool(checks['clear'][best]),
                                start=int(best-mine[0]), cost=float(total[best]),
                                **{k: float(v[best]) for k, v in checks.items() if k not in ('ok', 'clear')},
                                terms={k: float(v[best]) for k, v in terms.items()}))
        seconds = time.perf_counter()-started
        for r in results:
            r['seconds'] = seconds/len(windows)
        return results

    def check(self, path, centre, scene):
        """Exact tests: body grid inside the static mask, distance to movers, curvature, end to the goal."""
        d = self.detour
        cx, cy = centre
        px, py = self.body_points(cx, cy, path['yaw'])
        row, col = torch.floor(100+10*px).long(), torch.floor(100+10*py).long()
        inside = (row >= 0) & (row < 200) & (col >= 0) & (col < 200)
        row, col = row.clamp(0, 199), col.clamp(0, 199)
        batch = torch.arange(len(px), device=px.device)[:, None]
        static_hit = (scene['static'][batch, row, col] & inside).any(1)
        vehicle_hit = (scene['vehicles'][batch, row, col] & inside).any(1)
        if scene['movers'].shape[1]:
            gap = self.mover_gap(cx, cy, path['yaw'], path['time'], scene['movers']).amin((1, 2))
        else:
            gap = torch.full_like(cx[:, 0], 99.)
        dx = path['x'][:, -1:]-scene['reference'][..., 0]
        dy = path['y'][:, -1:]-scene['reference'][..., 1]
        distance = torch.where(scene['goal'], torch.hypot(dx, dy), torch.full_like(dx, 1e4))
        nearest = distance.argmin(1)
        end_error = distance.amin(1)
        goal_yaw = scene['reference'][torch.arange(len(nearest), device=px.device), nearest, 2]
        heading_error = torch.atan2(torch.sin(path['yaw'][:, -1]-goal_yaw), torch.cos(path['yaw'][:, -1]-goal_yaw)).abs()
        curvature = path['curvature'].abs().amax(1)
        clear = ~static_hit & (gap > 0) & (curvature < d.max_curvature+.02)
        ok = clear & (end_error < d.end_tolerance) & (heading_error < d.end_heading)
        out = dict(ok=ok, clear=clear, static_hit=static_hit, vehicle_hit=vehicle_hit, mover_gap=gap,
                   end_error=end_error, heading_error=heading_error, max_curvature=curvature)
        return {k: v.cpu().numpy() for k, v in out.items()}
