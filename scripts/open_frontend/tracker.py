"""Object tracks in the world frame: velocity of each detection and its predicted sweep.

The ego pose (x, y, yaw deg, CARLA world) moves the ego-frame detections to the world, so a parked car
keeps zero velocity while the ego drives past it. Without a pose the ego is taken as standing still.

Velocity is the least-squares slope of the track over the last WINDOW seconds. Frame-to-frame differences of
monocular positions jitter by tens of centimetres, which a slope over ten frames averages out.
"""
import numpy as np

from .models import MOVABLE, PERSON, VEHICLE

GATE = {0: 2.0, 1: 1.2, 2: 1.0}         # association radius per category, metres
WINDOW = 1.0                             # seconds of history for the velocity
MIN_SAMPLES = 3
PERSON_SPEED = .3                        # m/s: slower people are taken as standing for the sweep
VEHICLE_SPEED, VEHICLE_TRAVEL = 1.0, .8  # a vehicle moves if its slope says so and it really travelled
VEHICLE_SAMPLES, VEHICLE_STEADY = 8, .75 # over 8+ frames, 75 % of the steps along the slope (fit jumps are not)
MAX_AGE = 1.0                            # seconds a track survives without a detection


def ego_to_world(xy, pose):
    x, y, yaw = pose
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    xy = np.asarray(xy, float)
    return np.stack((x+xy[..., 0]*c-xy[..., 1]*s, y+xy[..., 0]*s+xy[..., 1]*c), -1)


def world_to_ego(xy, pose):
    x, y, yaw = pose
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    dx, dy = np.asarray(xy, float)[..., 0]-x, np.asarray(xy, float)[..., 1]-y
    return np.stack((dx*c+dy*s, -dx*s+dy*c), -1)


def slope(history):
    times = np.array([h[0] for h in history])
    points = np.array([h[1] for h in history])
    times = times-times.mean()
    spread = (times**2).sum()
    if spread < 1e-6:
        return np.zeros(2)
    return (times[:, None]*(points-points.mean(0))).sum(0)/spread


class Tracker:
    def __init__(self):
        self.tracks, self.next_id = [], 0

    def reset(self):
        self.tracks, self.next_id = [], 0

    def update(self, objects, stamp, pose=None):
        """objects: dicts with 'category', 'centre' (ego x, y). Adds 'id', 'velocity' (ego frame m/s), 'dynamic'."""
        pose = (0., 0., 0.) if pose is None else pose
        alive = [t for t in self.tracks if stamp-t['stamp'] <= MAX_AGE]
        used = set()
        for obj in sorted(objects, key=lambda o: -o['score']):
            world = ego_to_world(obj['centre'], pose)
            best, best_distance = None, GATE[obj['category']]
            for index, track in enumerate(alive):
                if index in used or track['category'] != obj['category']:
                    continue
                predicted = track['world']+track['velocity']*(stamp-track['stamp'])
                distance = np.linalg.norm(world-predicted)
                if distance < best_distance:
                    best, best_distance = index, distance
            if best is None:
                track = dict(id=self.next_id, category=obj['category'], world=world, velocity=np.zeros(2),
                             stamp=stamp, history=[])
                self.next_id += 1
                alive.append(track)
                used.add(len(alive)-1)
            else:
                track = alive[best]
                used.add(best)
            track['history'] = [h for h in track['history'] if stamp-h[0] <= WINDOW]+[(stamp, world)]
            track.update(world=world, stamp=stamp)
            history = track['history']
            track['velocity'] = slope(history) if len(history) >= MIN_SAMPLES else np.zeros(2)
            speed = float(np.linalg.norm(track['velocity']))
            travel = float(np.linalg.norm(history[-1][1]-history[0][1]))
            if obj['category'] == PERSON:
                moving = speed > PERSON_SPEED
            elif obj['category'] == VEHICLE:
                steps = np.diff(np.array([h[1] for h in history]), axis=0)
                along = steps@track['velocity'] > 0 if len(steps) else np.zeros(0, bool)
                moving = (len(history) >= VEHICLE_SAMPLES and speed > VEHICLE_SPEED and travel > VEHICLE_TRAVEL
                          and along.mean() >= VEHICLE_STEADY)
            else:
                moving = False
            if not moving:
                track['velocity'] = np.zeros(2)
            c, s = np.cos(np.radians(pose[2])), np.sin(np.radians(pose[2]))
            vx, vy = track['velocity']
            obj['id'] = track['id']
            obj['velocity'] = np.array([vx*c+vy*s, -vx*s+vy*c])          # world velocity in ego axes
            obj['dynamic'] = obj['category'] in MOVABLE and (obj['category'] == PERSON or moving)
        self.tracks = alive
        return objects
