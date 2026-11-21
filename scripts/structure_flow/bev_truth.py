"""Ground-truth BEV of a collected lot episode (drive.py folders): the label of the BEV refiner and the clean
input of the avoidance Flow.

Grid (ego frame of the car's bounding-box centre, x forward, y right as CARLA): ROWS x COLS cells of CELL metres,
row = (X_HALF - x) / CELL (row 0 is X_HALF ahead), column = (y + Y_HALF) / CELL. 32 x 20 m: the scenes set a walker
down 12-16 m ahead or behind and the teacher reacts from 13 m, the DDP grid's 10 m would hide what it reacts to.
The central 200 rows are the Flow worker's static mask (supervisor.FlowDriver.static_mask: row r, x = 10-(r+.5)/10).

Channels (float32 unless noted):
  static    parked vehicles, props, poles (1) - what stands still
  border    outside the lot (supervisor.LOT_BOUND): no camera sees it as an object, kept apart from `static`
  dynamic   walkers (discs of their radius) at the pose the cameras show
  vx, vy    velocity of the dynamic cells in the ego frame, m/s (ground speed, not relative to the car)
  target    the goal slot (rectangle SLOT_LENGTH x SLOT_WIDTH)

Walkers: drawn at position + velocity * WALKER_LAG. 0 here: the people the open front end finds lie at CARLA's
transform of the walker.
usage: python structure_flow/bev_truth.py FOLDER [--png OUT.png --frame N]      (a check picture of one frame)
"""
import json
import math
import os

import numpy as np

CELL = .1
X_HALF, Y_HALF = 16., 10.
ROWS, COLS = int(round(2*X_HALF/CELL)), int(round(2*Y_HALF/CELL))
CHANNELS = ('static', 'border', 'dynamic', 'vx', 'vy', 'target')
LOT_BOUND = (264., 304., -241., -178.)          # supervisor.LOT_BOUND
POLES = ((276.3, -234.3), (294.8, -186.4), (294.7, -210.7), (294.7, -234.3), (276.2, -210.7), (276.3, -186.4))
POLE_SIZE = .4
SLOT_LENGTH, SLOT_WIDTH = 5.0, 2.6
WALKER_RADIUS = .3
WALKER_LAG = 0.
HIDDEN_X = 250.             # waiting / removed walkers are parked off the lot at x 232 / 240


def cell_centres():
    """Ego-frame x, y of every cell centre, [ROWS, COLS] each."""
    x = X_HALF-(np.arange(ROWS)+.5)*CELL
    y = -Y_HALF+(np.arange(COLS)+.5)*CELL
    return np.meshgrid(x, y, indexing='ij')


_X, _Y = cell_centres()


def paint(mask, pose, x, y, yaw, length, width):
    """Mark the cells whose centre lies in the world rectangle (yaw in radians), as the Flow worker's mask does.
    Only the cells of its bounding window are tested (cv2.fillPoly also takes the cells its outline touches, about
    8 % more area)."""
    c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
    dx, dy = x-pose[0], y-pose[1]
    ex, ey, eyaw = c*dx+s*dy, -s*dx+c*dy, yaw-math.radians(pose[2])
    reach = math.hypot(length, width)/2
    r0, r1 = max(0, int((X_HALF-ex-reach)/CELL)), min(ROWS, int((X_HALF-ex+reach)/CELL)+2)
    c0, c1 = max(0, int((ey-reach+Y_HALF)/CELL)), min(COLS, int((ey+reach+Y_HALF)/CELL)+2)
    if r0 >= r1 or c0 >= c1:
        return
    a, b = math.cos(eyaw), math.sin(eyaw)
    px, py = _X[r0:r1, c0:c1]-ex, _Y[r0:r1, c0:c1]-ey
    mask[r0:r1, c0:c1] |= (np.abs(a*px+b*py) <= length/2) & (np.abs(-b*px+a*py) <= width/2)


class Scene:
    """The static things of one episode (meta.json) and its frames (frames.json)."""

    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, 'meta.json'), encoding='utf-8') as stream:
            self.meta = json.load(stream)
        with open(os.path.join(folder, 'frames.json'), encoding='utf-8') as stream:
            self.frames = json.load(stream)
        boxes = [(p['x'], p['y'], math.radians(p['yaw']), p['length'], p['width']) for p in self.meta['parked']]
        boxes += [(p['x'], p['y'], math.radians(p['yaw']), p['length'], p['width']) for p in self.meta.get('props') or []]
        boxes += [(x, y, 0., POLE_SIZE, POLE_SIZE) for x, y in POLES]
        self.boxes = np.asarray(boxes, float)
        self.goal = self.meta['goal']            # x, y, yaw rad: the car parked in the slot

    def __len__(self):
        return len(self.frames)

    def pose(self, index):
        frame = self.frames[index]
        return frame['x'], frame['y'], frame['yaw']

    def walkers(self, index, lag=WALKER_LAG):
        """[x, y, vx, vy] (world) of the walkers on the lot at the frame, the position advanced by the render lag."""
        out = []
        for x, y, vx, vy in self.frames[index]['walkers']:
            if x > HIDDEN_X:
                out.append((x+vx*lag, y+vy*lag, vx, vy))
        return out

    def render(self, index, lag=WALKER_LAG):
        """[len(CHANNELS), ROWS, COLS] float32 for the frame."""
        pose = self.pose(index)
        out = np.zeros((len(CHANNELS), ROWS, COLS), np.float32)
        static = np.zeros((ROWS, COLS), bool)
        near = self.boxes[np.hypot(self.boxes[:, 0]-pose[0], self.boxes[:, 1]-pose[1]) < X_HALF+Y_HALF+4.]
        for x, y, yaw, length, width in near:
            paint(static, pose, x, y, yaw, length, width)
        out[0] = static
        lot = np.zeros((ROWS, COLS), bool)
        paint(lot, pose, (LOT_BOUND[0]+LOT_BOUND[1])/2, (LOT_BOUND[2]+LOT_BOUND[3])/2, 0.,
              LOT_BOUND[1]-LOT_BOUND[0], LOT_BOUND[3]-LOT_BOUND[2])
        out[1] = ~lot
        c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
        for x, y, vx, vy in self.walkers(index, lag):
            ex, ey = c*(x-pose[0])+s*(y-pose[1]), -s*(x-pose[0])+c*(y-pose[1])
            disc = np.hypot(_X-ex, _Y-ey) <= WALKER_RADIUS
            out[2][disc] = 1.
            out[3][disc] = c*vx+s*vy
            out[4][disc] = -s*vx+c*vy
        target = np.zeros((ROWS, COLS), bool)
        paint(target, pose, self.goal[0], self.goal[1], self.goal[2], SLOT_LENGTH, SLOT_WIDTH)
        out[5] = target
        return out


def picture(bev):
    """RGB uint8 [ROWS, COLS, 3] of a truth BEV: static grey, border dark, dynamic red, target green, the car blue."""
    image = np.full((ROWS, COLS, 3), 255, np.uint8)
    image[bev[1] > .5] = (70, 70, 70)
    image[bev[5] > .5] = (170, 230, 170)
    image[bev[0] > .5] = (150, 150, 150)
    image[bev[2] > .5] = (220, 40, 40)
    car = (np.abs(_X) <= 2.4) & (np.abs(_Y) <= 1.08)
    image[car] = (60, 110, 230)
    return image


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('folder')
    parser.add_argument('--png')
    parser.add_argument('--frame', type=int, default=-1)
    args = parser.parse_args()
    scene = Scene(args.folder)
    moving = [i for i in range(len(scene)) if scene.walkers(i)]
    index = args.frame if args.frame >= 0 else (moving[len(moving)//2] if moving else len(scene)//2)
    bev = scene.render(index)
    print('frames', len(scene), 'with walkers', len(moving), 'frame', index, 'pose', np.round(scene.pose(index), 2).tolist(),
          {name: round(float(bev[i].sum()), 1) if name not in ('vx', 'vy') else round(float(np.abs(bev[i]).max()), 2)
           for i, name in enumerate(CHANNELS)})
    if args.png:
        from PIL import Image
        Image.fromarray(np.repeat(np.repeat(picture(bev), 2, 0), 2, 1)).save(args.png)
        print('saved', args.png)


if __name__ == '__main__':
    main()
