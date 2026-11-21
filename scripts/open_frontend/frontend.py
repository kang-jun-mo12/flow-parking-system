"""Training-free BEV front end: YOLO-World boxes + metric depth -> objects -> BEV layers -> DDP-style segmentation.

  6 RGB images ─┬─ detector (YOLO-World, text prompts) ──── boxes, category, score
                └─ depth (Depth Anything V2 metric) ─ ground scale check ─ z-depth maps
                         │
     per box: pixels inside -> nearest depth cluster -> ego points -> footprint (min-area box grown to the prior)
     every pixel: ego points 0.3..2.2 m above ground -> unknown obstacle cells (what the detector misses)
                         │
     merge boxes seen by two cameras -> tracker (world frame, velocity) -> predicted sweep of moving objects
                         │
  layers: vehicle, person, obstacle, unknown, future, target  ->  seg (0 free, 1 occupied, 2 target slot)

`seg` keeps the classes, grid and orientation of the DDP segmentation, so the planner side
(`hits`, the seg-conditioned Flow, `target_mask_centre`) reads it with no change.
"""
import time
import numpy as np

from . import geometry as G
from .models import CATEGORY_NAMES, MAX_SIZE, MOVABLE, OBSTACLE, PERSON, PERSON_HEIGHT, PRIOR_SIZE, VEHICLE
from .tracker import Tracker

LAYERS = ('vehicle', 'person', 'obstacle', 'unknown', 'future', 'target')


class Config:
    max_range = 12.5            # metres, the depth range of the DDP frustum (d_bound 0.5..12.5)
    min_height, max_height = .3, 2.2
    unknown_min_points = 3      # points per 0.1 m cell before an undetected obstacle counts
    unknown_stride = 2          # pixel stride of the dense obstacle cloud
    box_shrink = .1             # keep the inner 80 % of a box, its edges hold background
    cluster_cell = .2           # metres; points in touching cells of this size are one object
    min_object_points = 15
    merge_distance = {VEHICLE: 2.2, PERSON: .8, OBSTACLE: .6}
    person_margin = 0.          # extra radius around people; their motion goes into the predicted sweep instead
    horizon, horizon_step = 2.0, .25     # seconds of the predicted sweep of moving objects
    scale_check = True          # rescale depth so road pixels match the known camera height
    scale_limits = (.1, 3.)


class OpenFrontend:
    def __init__(self, detector, depth, config=None):
        self.detector, self.depth = detector, depth
        self.cfg = config or Config()
        self.tracker = Tracker()
        self.ground = [G.ground_depth(camera) for camera in range(len(G.CAMERAS))]
        self.ground_sparse = [G.ground_depth(camera, self.cfg.unknown_stride) for camera in range(len(G.CAMERAS))]
        self.ego = G.ego_body()
        self.last_scales = None

    def reset(self):
        self.tracker.reset()

    # ------------------------------------------------------------------ depth
    def check_scale(self, depth, boxes):
        """Monocular metric depth drifts in scale. Pixels in the lower third, outside every box and whose depth
        agrees with the ground within 30 %, are road; their median ratio to the ground depth rescales the map."""
        scales = []
        for camera, (dmap, ground) in enumerate(zip(depth, self.ground)):
            road = np.zeros(dmap.shape, bool)
            road[2*G.HEIGHT//3:] = True
            for x1, y1, x2, y2, *_ in boxes[camera]:
                road[int(y1):int(y2)+1, int(x1):int(x2)+1] = False
            road &= np.isfinite(ground) & (ground < self.cfg.max_range)
            ratio = ground[road]/np.maximum(dmap[road], 1e-3)
            scale = 1.
            if len(ratio) > 200:
                # Metric heads assume the focal length of their training camera (KITTI-like); the 100 degree rig
                # comes out about 4x too far, evenly. Median, then the median of the ratios within 20 % of it.
                first = np.median(ratio)
                inner = ratio[np.abs(ratio/first-1) < .2]
                scale = float(np.median(inner)) if len(inner) > 100 else float(first)
            elif self.last_scales is not None:
                scale = self.last_scales[camera]
            scale = float(np.clip(scale, *self.cfg.scale_limits))
            depth[camera] *= scale
            scales.append(scale)
        self.last_scales = scales
        return depth, scales

    # ------------------------------------------------------------------ objects
    def lift_box(self, camera, box, dmap):
        cfg = self.cfg
        x1, y1, x2, y2, score, category = box
        category = int(category)
        dx, dy = (x2-x1)*cfg.box_shrink, (y2-y1)*cfg.box_shrink
        u1, u2 = int(max(x1+dx, 0)), int(min(x2-dx, G.WIDTH-1))+1
        v1, v2 = int(max(y1+dy, 0)), int(min(y2-dy, G.HEIGHT-1))+1
        if u2 <= u1 or v2 <= v1:
            return None
        patch = dmap[v1:v2, u1:u2]
        valid = np.isfinite(patch) & (patch > .3) & (patch < cfg.max_range+3) & self.smooth[camera][v1:v2, u1:u2]
        # The bottom centre of the box stands on the ground. A box cut by the image bottom shows no contact.
        foot = G.ground_point((x1+x2)/2, y2, camera) if y2 < G.HEIGHT-3 else None
        if foot is None and category == PERSON and y1 > 2:
            # Feet below the image (a person close to the car): the head sits about PERSON_HEIGHT up.
            foot = G.height_point((x1+x2)/2, y1, camera, PERSON_HEIGHT)
        if foot is not None and category != VEHICLE and np.hypot(*foot[:2]) < cfg.max_range+3:
            # People and small things: the ground contact beats monocular depth on a flat lot.
            # The contact is the near edge of the feet; the centre lies half a body further along the ray.
            ray = foot[:2]-G.EGO_FROM_OPTICAL[camera][:2, 3]
            centre = foot[:2]+ray/(np.linalg.norm(ray)+1e-6)*PRIOR_SIZE[category][1]/2
            return dict(camera=camera, category=category, score=float(score), xy=centre[None])
        v, u = np.nonzero(valid)
        rays = np.stack((u+u1, v+v1, np.ones_like(u)), -1)@G.K_INV.T
        matrix = G.EGO_FROM_OPTICAL[camera]
        points = (rays*patch[valid][:, None])@matrix[:3, :3].T+matrix[:3, 3]
        keep = (points[:, 2] > G.GROUND_Z+.1) & (points[:, 2] < G.GROUND_Z+3.5)
        points, u = points[keep, :2], u[keep]
        # Depth errors run along the viewing ray and smear the cloud into a streak. Each image column keeps its
        # nearest point, which leaves the outline of the visible faces.
        if len(points):
            ranges = np.hypot(points[:, 0], points[:, 1])
            order = np.lexsort((ranges, u))
            first = np.r_[True, u[order][1:] != u[order][:-1]]
            points = points[order[first]]
        anchor = foot[:2] if foot is not None else np.zeros(2)
        points = self.main_cluster(points, anchor)
        if len(points) < cfg.min_object_points:
            if foot is None or np.hypot(*foot[:2]) > cfg.max_range+3:
                return None
            points = foot[None, :2]
        return dict(camera=camera, category=category, score=float(score), xy=points)

    def main_cluster(self, xy, anchor):
        """The box also holds what is behind the object. Points are grouped by touching 0.2 m cells; among the
        groups holding at least 10 % of the points, the one whose nearest point is closest to the anchor wins."""
        import cv2
        if len(xy) < self.cfg.min_object_points:
            return xy
        cell = self.cfg.cluster_cell
        low = xy.min(0)
        index = np.floor((xy-low)/cell).astype(int)
        grid = np.zeros(index.max(0)+1, np.uint8)
        grid[index[:, 0], index[:, 1]] = 1
        grid = cv2.dilate(grid, np.ones((3, 3), np.uint8))
        count, labels = cv2.connectedComponents(grid, connectivity=8)
        if count <= 2:
            return xy
        label = labels[index[:, 0], index[:, 1]]
        sizes = np.bincount(label, minlength=count).astype(float)
        sizes[0] = 0
        big = sizes >= max(self.cfg.min_object_points, .1*len(xy))
        if not big.any():
            big = sizes == sizes.max()
        distance = np.array([np.linalg.norm(xy[label == k]-anchor, axis=1).min() if big[k] else np.inf
                             for k in range(count)])
        return xy[label == int(np.argmin(distance))]

    def footprint(self, xy, category, camera_xy):
        """Rectangle of the category prior around the visible points (L-shape fitting).

        Cameras only see the near faces. The heading is the angle at which the points hug two edges of their
        bounding rectangle best (search every 3 degrees); the prior size is then laid out from the visible
        edges away from the camera, so the hidden part lands behind what was seen."""
        prior_long, prior_short = PRIOR_SIZE[category]
        xy = np.asarray(xy, float)
        best = (0., 0.)       # small things and single ground points keep the ego axes
        for angle in np.radians(np.arange(0, 90, 3)) if len(xy) >= 3 and prior_long >= 1. else ():
            c, s = np.cos(angle), np.sin(angle)
            a, b = xy@np.array([c, s]), xy@np.array([-s, c])
            da = np.minimum(a-a.min(), a.max()-a)
            db = np.minimum(b-b.min(), b.max()-b)
            cost = np.minimum(da, db).mean()
            if angle == 0 or cost < best[0]:
                best = (cost, angle)
        angle = best[1]
        units = [np.array([np.cos(angle), np.sin(angle)]), np.array([-np.sin(angle), np.cos(angle)])]
        # 2-98 % of the points along each axis: depth smear at object edges leaves stray points.
        coords = [np.clip(xy@unit, *np.percentile(xy@unit, (2, 98))) for unit in units]
        extents = [k.max()-k.min() for k in coords]
        ray = xy.mean(0)-camera_xy
        ray = ray/(np.linalg.norm(ray)+1e-6)
        if max(extents) > prior_short+.4:
            long_axis = int(np.argmax(extents))
        else:   # one short face only: the hidden length runs along the viewing ray
            long_axis = int(np.argmax([abs(unit@ray) for unit in units]))
        want = [prior_short, prior_short]
        want[long_axis] = prior_long
        centre, size = np.zeros(2), [0., 0.]
        for axis, (unit, k) in enumerate(zip(units, coords)):
            cap = MAX_SIZE[category][0 if axis == long_axis else 1]
            length = min(max(want[axis], k.max()-k.min()), cap)
            camera = camera_xy@unit
            if len(xy) == 1:              # a ground contact (feet): centred on it
                low = k.mean()-length/2
            elif camera <= k.min():       # camera on the low side: the object extends to higher values
                low = k.min()
            elif camera >= k.max():
                low = k.max()-length
            else:                         # seen face-on along this axis: centre on the points
                low = (k.min()+k.max())/2-length/2
            centre += unit*(low+length/2)
            size[axis] = length
        spread = (extents[long_axis], extents[1-long_axis])
        yaw = float(np.arctan2(units[long_axis][1], units[long_axis][0]))
        size = (size[long_axis], size[1-long_axis])
        return dict(centre=centre, size=size, yaw=yaw, spread=spread, corners=G.box_corners(centre, size, yaw))

    def fit(self, parts):
        """Footprint of one object from the points of every box that shows it."""
        xy = np.concatenate([p['xy'] for p in parts])
        main = max(parts, key=lambda p: len(p['xy']))
        obj = dict(category=main['category'], score=max(p['score'] for p in parts), points=len(xy),
                   cameras=sorted({p['camera'] for p in parts}), xy=xy)
        # The six cameras surround the ego centre, so it is the viewpoint: an object cut by an image edge in one
        # camera would otherwise be grown towards the wrong side.
        obj.update(self.footprint(xy, main['category'], np.zeros(2)))
        return obj

    def merge(self, parts):
        """Boxes of two cameras showing the same thing become one object: same category, their points touch
        (0.3 m) and the joined footprint stays within the size cap. Repeats until nothing joins."""
        objects = [self.fit([p]) for p in parts]
        groups = [[p] for p in parts]
        joined = True
        while joined:
            joined = False
            for i in range(len(objects)):
                for j in range(i+1, len(objects)):
                    a, b = objects[i], objects[j]
                    if a['category'] != b['category']:
                        continue
                    near = (a['category'] != VEHICLE and
                            np.linalg.norm(a['centre']-b['centre']) < self.cfg.merge_distance[a['category']])
                    if not (near or self.touch(a['xy'], b['xy'])):
                        continue
                    union = self.fit(groups[i]+groups[j])
                    cap = MAX_SIZE[a['category']]
                    if union['spread'][0] > cap[0]+.3 or union['spread'][1] > cap[1]+.3:
                        continue    # two neighbours, not one object
                    groups[i] += groups.pop(j)
                    objects.pop(j)
                    objects[i] = union
                    joined = True
                    break
                if joined:
                    break
        return objects

    @staticmethod
    def touch(a, b, gap=.3):
        """True if any point of a lies within gap of a point of b (checked on gap-sized cells)."""
        code = lambda c: c[..., 0]*100003+c[..., 1]
        cells = np.unique(code(np.floor(a/gap).astype(np.int64)))
        near = np.floor(b/gap).astype(np.int64)[:, None]+np.array([[dx, dy] for dx in (-1, 0, 1) for dy in (-1, 0, 1)])
        return bool(np.isin(code(near), cells).any())

    @staticmethod
    def not_edge(dmap, jump=.08):
        """Pixels away from depth edges. Monocular depth blends the object and the background across an edge,
        and those pixels float between the two."""
        import cv2
        local = cv2.medianBlur(dmap.astype(np.float32), 5)
        return np.abs(dmap-local) < jump*local

    # ------------------------------------------------------------------ dense
    def unknown_cells(self, depth):
        cfg = self.cfg
        clouds = []
        for camera, dmap in enumerate(depth):
            points = G.backproject(dmap, camera, cfg.unknown_stride).reshape(-1, 3)
            ranges = np.hypot(points[:, 0], points[:, 1])
            keep = ((points[:, 2] > G.GROUND_Z+cfg.min_height) & (points[:, 2] < G.GROUND_Z+cfg.max_height)
                    & (ranges < cfg.max_range) & np.isfinite(points).all(1))
            clouds.append(points[keep, :2])
        return G.raster_points(np.concatenate(clouds), cfg.unknown_min_points) & ~self.ego

    # ------------------------------------------------------------------ main
    def __call__(self, images, target, stamp=None, pose=None):
        """images [6, 300, 400, 3] RGB uint8 in rig.CAMERAS order; target (x m, y m, yaw deg) in the ego frame;
        stamp seconds; pose (x, y, yaw deg) of the ego in the world, for the tracker."""
        cfg = self.cfg
        timing, started = {}, time.perf_counter()
        boxes = self.detector(images)
        timing['detect'] = time.perf_counter()-started
        depth = np.array(self.depth(images), np.float32)
        timing['depth'] = time.perf_counter()-started-timing['detect']
        scales = None
        if cfg.scale_check:
            depth, scales = self.check_scale(depth, boxes)

        self.smooth = [self.not_edge(d) for d in depth]
        objects = [o for camera in range(len(images)) for box in boxes[camera]
                   if (o := self.lift_box(camera, box, depth[camera])) is not None]
        objects = self.merge(objects)
        stamp = time.time() if stamp is None else stamp
        objects = self.tracker.update(objects, stamp, pose)

        layers = {name: np.zeros((G.BEV_SIZE, G.BEV_SIZE), bool) for name in LAYERS}
        for obj in objects:
            name = CATEGORY_NAMES[obj['category']]
            corners = obj['corners']
            if obj['category'] == PERSON:
                corners = G.box_corners(obj['centre'], np.add(obj['size'], 2*cfg.person_margin), obj['yaw'])
            G.raster_polygon(layers[name], corners)
            if obj.get('dynamic'):
                for t in np.arange(cfg.horizon_step, cfg.horizon+1e-6, cfg.horizon_step):
                    G.raster_polygon(layers['future'], corners+obj['velocity']*t)
        layers['unknown'] = self.unknown_cells(depth)
        layers['target'] = G.target_slot(target)
        for name in ('vehicle', 'person', 'obstacle', 'future'):
            layers[name] &= ~self.ego

        occupied = layers['vehicle'] | layers['person'] | layers['obstacle'] | layers['unknown'] | layers['future']
        seg = np.zeros((G.BEV_SIZE, G.BEV_SIZE), np.uint8)
        seg[layers['target']] = 2
        seg[occupied] = 1           # an obstacle inside the slot wins over the slot
        timing['total'] = time.perf_counter()-started
        return dict(seg=seg, layers=layers, objects=objects, depth=depth, boxes=boxes, scales=scales, timing=timing)


def fuse_with_ddp(ddp_seg, open_out, use=('person', 'obstacle', 'future', 'vehicle')):
    """Keep the DDP segmentation (the planner was trained on it) and add the open layers as vehicles.
    Only adds occupied cells, never removes what DDP sees."""
    seg = ddp_seg.copy()
    extra = np.zeros_like(seg, bool)
    for name in use:
        extra |= open_out['layers'][name]
    seg[extra] = 1
    return seg
