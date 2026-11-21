"""Evidence BEV of a collected lot episode from its camera images: what the frozen perception says, laid out on the
grid of bev_truth.py (32 x 20 m, 0.1 m) for the BEV refiner to turn into occupancy and velocity.

  6 images ─┬─ DDP front end (frozen) ───────────── P(vehicle), P(target slot) on its 20 x 20 m grid
            ├─ YOLO-World (CARLA vocabulary) ─┐
            └─ metric depth (UniDepth V2-S) ──┴─ open front end: objects (footprint, category, track velocity)
                                              └─ every depth pixel as an ego point: counts per cell and height band

Channels, uint8 [len(CHANNELS), ROWS, COLS] (probabilities and scores x 255, counts clipped at 255):
  ground       depth points within GROUND_BAND of the ground: the cell was seen and is free
  low, high    depth points 0.3-1.0 m and 1.0-2.2 m above the ground (away from depth edges): something stands there
  vehicle, person, obstacle     footprints of the detected objects, the detector score
  ddp_vehicle, ddp_slot         the DDP segmentation probabilities (0 beyond its grid)
and float16 `velocity` [2, ROWS, COLS]: the tracker's velocity (ego axes, m/s) on the cells of moving objects.
The objects themselves (category, score, centre, size, yaw, velocity, dynamic) are kept per frame as well.

Every STEP-th recorded frame (0.2 s, the Flow's decision step), in order: the tracker needs the sequence.
Output: FOLDER/bev_evidence.npz (frames, evidence, velocity, scales, timing) and FOLDER/bev_objects.json.
usage (Python 3.11, from scripts/): python structure_flow/bev_evidence.py ROOT [--worker i --workers n] [--check OUT.png]
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bev_truth as T                                   # noqa: E402

CHANNELS = ('ground', 'low', 'high', 'vehicle', 'person', 'obstacle', 'ddp_vehicle', 'ddp_slot')
STEP = 2                    # recorded frames (10 Hz) per evidence frame
GROUND_BAND = .2
LOW, HIGH = (.3, 1.), (1., 2.2)
DEPTH = 'unidepth-s'
DDP_WEIGHTS = str(Path(__file__).resolve().parents[2]/'weights'/'ddp_frontend.pt')      # the Release asset (README)
MAX_RANGE = 13.5            # open front end: ground-contact objects up to MAX_RANGE + 3 m (the 16 m of the grid)


def cells(x, y):
    """Ego x, y arrays -> row, column and the mask of points on the grid (bev_truth convention)."""
    row = np.floor((T.X_HALF-x)/T.CELL).astype(np.int64)
    col = np.floor((y+T.Y_HALF)/T.CELL).astype(np.int64)
    return row, col, (row >= 0) & (row < T.ROWS) & (col >= 0) & (col < T.COLS)


def fill(corners, value, layer):
    """Ego-frame polygon [N, 2] -> the larger of layer and value on its cells."""
    import cv2
    points = np.stack(((corners[:, 1]+T.Y_HALF)/T.CELL, (T.X_HALF-corners[:, 0])/T.CELL), -1)      # (col, row)
    mask = np.zeros(layer.shape, np.uint8)
    cv2.fillPoly(mask, [np.round(points-.5).astype(np.int32)], 1)
    np.maximum(layer, np.where(mask > 0, value, 0).astype(layer.dtype), out=layer)
    return mask > 0


class Evidence:
    def __init__(self, depth=DEPTH, device='cuda'):
        import torch
        from ddp_frontend import rig
        from ddp_frontend.model import DdpFrontend
        from open_frontend import OpenFrontend
        from open_frontend import geometry as G
        from open_frontend.depth_models import make_depth
        from open_frontend.frontend import Config
        from open_frontend.models import CARLA_PROMPTS, CARLA_WEIGHTS, YoloWorld
        self.torch, self.rig, self.G, self.device = torch, rig, G, device
        self.ddp = DdpFrontend.load(DDP_WEIGHTS).to(device).eval()
        self.intrinsics, self.extrinsics = rig.intrinsics().to(device)[None], rig.extrinsics().to(device)[None]
        config = Config()
        config.max_range = MAX_RANGE
        self.frontend = OpenFrontend(YoloWorld(CARLA_WEIGHTS, CARLA_PROMPTS, device=device), make_depth(depth, device), config)
        self.rays = [G.pixel_rays()@G.EGO_FROM_OPTICAL[c][:3, :3].T for c in range(len(G.CAMERAS))]      # ego directions
        self.origins = [G.EGO_FROM_OPTICAL[c][:3, 3] for c in range(len(G.CAMERAS))]

    def reset(self):
        self.frontend.reset()

    def __call__(self, images, target, stamp, pose):
        """images [6, 300, 400, 3] RGB uint8 (rig.CAMERAS order), target (x m, y m, yaw deg) in the ego frame,
        stamp s, pose (x, y, yaw deg). Returns evidence uint8 [C, ROWS, COLS], velocity float16 [2, ROWS, COLS],
        the objects and a dict of timings."""
        torch, G = self.torch, self.G
        started = time.perf_counter()
        with torch.no_grad():
            pixels = torch.stack([self.rig.prepare_image(image) for image in images])[None].to(self.device)
            out = self.ddp(pixels, self.intrinsics, self.extrinsics, torch.tensor([target], dtype=torch.float32).to(self.device))
            ddp = torch.softmax(out['segmentation'][0].float(), 0).cpu().numpy()          # [3, 200, 200], row = 100 + 10 x
        ddp_seconds = time.perf_counter()-started
        front = self.frontend(images, target, stamp=stamp, pose=pose)
        evidence = np.zeros((len(CHANNELS), T.ROWS, T.COLS), np.uint8)
        velocity = np.zeros((2, T.ROWS, T.COLS), np.float16)
        counts = np.zeros((3, T.ROWS*T.COLS), np.int32)
        for camera, dmap in enumerate(front['depth']):
            points = self.rays[camera]*dmap[..., None]+self.origins[camera]
            x, y, z = points[..., 0].ravel(), points[..., 1].ravel(), points[..., 2].ravel()
            row, col, inside = cells(x, y)
            inside &= np.isfinite(z)
            smooth = self.frontend.smooth[camera].ravel()
            index = row*T.COLS+col
            for band, keep in enumerate((np.abs(z-G.GROUND_Z) <= GROUND_BAND,
                                         (z > G.GROUND_Z+LOW[0]) & (z <= G.GROUND_Z+LOW[1]) & smooth,
                                         (z > G.GROUND_Z+HIGH[0]) & (z <= G.GROUND_Z+HIGH[1]) & smooth)):
                counts[band] += np.bincount(index[inside & keep], minlength=T.ROWS*T.COLS)
        evidence[:3] = np.clip(counts, 0, 255).reshape(3, T.ROWS, T.COLS)
        objects = []
        for obj in front['objects']:
            category = int(obj['category'])
            mask = fill(np.asarray(obj['corners'], float), int(round(255*float(obj['score']))), evidence[3+category])
            moving = bool(obj.get('dynamic')) and float(np.linalg.norm(obj['velocity'])) > 0
            if moving:
                velocity[0][mask], velocity[1][mask] = obj['velocity'][0], obj['velocity'][1]
            objects.append(dict(category=category, score=round(float(obj['score']), 3),
                                centre=np.round(obj['centre'], 3).tolist(), size=np.round(obj['size'], 3).tolist(),
                                yaw=round(float(obj['yaw']), 4), velocity=np.round(obj['velocity'], 3).tolist(),
                                dynamic=bool(obj.get('dynamic')), id=int(obj.get('id', -1))))
        # the DDP grid (row = 100 + 10 x, the central 20 m) turned to row 0 ahead
        first = int(round((T.X_HALF-G.BEV_HALF)/T.CELL))
        evidence[6, first:first+G.BEV_SIZE] = np.round(255*ddp[1][::-1]).astype(np.uint8)
        evidence[7, first:first+G.BEV_SIZE] = np.round(255*ddp[2][::-1]).astype(np.uint8)
        timing = dict(front['timing'], ddp=ddp_seconds, all=time.perf_counter()-started)
        return evidence, velocity, objects, timing, front['scales']


def load_images(folder, index, cameras):
    from PIL import Image
    return np.stack([np.asarray(Image.open(os.path.join(folder, name, '{:04d}.png'.format(index))).convert('RGB'))
                     for name in cameras])


def target_of(scene, index):
    x, y, yaw = scene.pose(index)
    gx, gy, gyaw = scene.goal
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    dx, dy = gx-x, gy-y
    return [dx*c+dy*s, -dx*s+dy*c, (math.degrees(gyaw)-yaw+180.) % 360.-180.]


def episode(model, folder):
    scene = T.Scene(folder)
    frames = list(range(0, len(scene), STEP))
    evidence, velocity, objects, scales, seconds = [], [], [], [], []
    model.reset()
    for index in frames:
        images = load_images(folder, index, model.G.CAMERAS)
        e, v, o, timing, scale = model(images, target_of(scene, index), scene.frames[index]['t'], scene.pose(index))
        evidence.append(e)
        velocity.append(v)
        objects.append(o)
        scales.append(scale if scale is not None else [1.]*len(model.G.CAMERAS))
        seconds.append(timing['all'])
    np.savez_compressed(os.path.join(folder, 'bev_evidence.npz'), frames=np.asarray(frames, np.int32),
                        evidence=np.stack(evidence), velocity=np.stack(velocity), scales=np.asarray(scales, np.float32),
                        seconds=np.asarray(seconds, np.float32), channels=np.asarray(CHANNELS), depth=DEPTH)
    with open(os.path.join(folder, 'bev_objects.json'), 'w') as stream:
        json.dump(dict(frames=frames, objects=objects), stream)
    return len(frames), float(np.mean(seconds[1:] or seconds))


def picture(evidence, velocity, truth):
    """Check sheet: the truth, the object channels, the depth-point bands, the DDP probabilities, side by side."""
    def rgb(r, g, b):
        return np.stack([255-np.clip(v, 0, 255) for v in (r, g, b)], -1).astype(np.uint8)
    e = evidence.astype(np.float32)
    objects = rgb(e[4]+e[5], e[3]+e[5], e[3]+e[4])                    # vehicle red, person green, obstacle blue
    moving = np.hypot(velocity[0].astype(np.float32), velocity[1].astype(np.float32)) > .2
    objects[moving] = (255, 0, 255)
    points = rgb(8*e[2], 8*e[1]+8*e[2], 8*e[1]+2*e[0])                # ground faint yellow, low red, high cyan
    ddp = rgb(e[7], e[6], e[6]+e[7])                                  # vehicle red, slot green
    gap = np.full((T.ROWS, 4, 3), 0, np.uint8)
    return np.concatenate([T.picture(truth), gap, objects, gap, points, gap, ddp], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--check', help='PNG of one frame (root is one episode folder then); nothing is saved')
    parser.add_argument('--frame', type=int, default=-1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--redo', action='store_true')
    args = parser.parse_args()
    model = Evidence(device=args.device)
    if args.check:
        from PIL import Image
        scene = T.Scene(args.root)
        moving = [i for i in range(0, len(scene), STEP) if scene.walkers(i)]
        last = args.frame if args.frame >= 0 else (moving[len(moving)//2] if moving else len(scene)//2)
        model.reset()
        for index in range(max(0, last-10*STEP), last+1, STEP):       # 2 s of frames before it for the tracker
            images = load_images(args.root, index, model.G.CAMERAS)
            e, v, o, timing, scale = model(images, target_of(scene, index), scene.frames[index]['t'], scene.pose(index))
        Image.fromarray(np.repeat(np.repeat(picture(e, v, scene.render(last)), 2, 0), 2, 1)).save(args.check)
        print('frame', last, 'objects', [(x['category'], x['centre'], x['velocity'], x['dynamic']) for x in o])
        print('truth walkers (ego)', [np.round(T_world_to_ego(scene.pose(last), w), 2).tolist() for w in scene.walkers(last)])
        print('timing', {k: round(1000*float(x)) for k, x in timing.items()}, 'scales', np.round(scale, 2).tolist() if scale else None)
        return
    folders = sorted(str(p) for p in Path(args.root).iterdir() if (p/'meta.json').exists() and (p/'frames.json').exists()
                     and (p/'camera_front').is_dir())
    for folder in folders[args.worker::args.workers]:
        if os.path.exists(os.path.join(folder, 'bev_evidence.npz')) and not args.redo:
            continue
        began = time.perf_counter()
        count, mean = episode(model, folder)
        print(os.path.basename(folder), count, 'frames', round(1000*mean), 'ms/frame', round(time.perf_counter()-began, 1), 's', flush=True)
    print('evidence done', flush=True)


def T_world_to_ego(pose, walker):
    c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
    dx, dy = walker[0]-pose[0], walker[1]-pose[1]
    return [dx*c+dy*s, -dx*s+dy*c, walker[2]*c+walker[3]*s, -walker[2]*s+walker[3]*c]


if __name__ == '__main__':
    main()
