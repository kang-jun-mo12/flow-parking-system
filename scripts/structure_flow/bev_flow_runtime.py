"""The BEV avoidance Flow at run time (bev_flow.py), one decision at a time: the inputs are built exactly as
bev_flow_data.py builds the training rows, the candidates are the chunks of the two likeliest intents (fallback: then
the mean chunk of every other intent, likeliest first - the server takes the first safe one, so these are driven only
when none of the first eight is safe; without them the car stands, also where a walker walks on into it).

Used by system_server.py --student CHECKPOINT (closed-loop tests and DAgger collection): the server screens the
candidates with the worker's exact safety check and executes the first safe one.
"""
import math
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bev_truth as T                                   # noqa: E402
import bev_flow_data as D                               # noqa: E402
import bev_flow as F                                    # noqa: E402

SAMPLES = 4                 # chunks per intent: the chunk of the mean and SAMPLES-1 samples
STAGE, WAIT = 0, 1


class Runtime:
    def __init__(self, checkpoint, device='cuda', seed=0, fallback=False):
        self.model, saved = F.load(str(checkpoint), device)
        self.mode, self.device, self.fallback = saved['bev'], device, fallback
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.new_segment()

    def new_segment(self):
        """The Flow takes over anew: no previous plan, no previous intent."""
        self.previous_intent, self.elapsed, self.tail = None, 0, np.zeros(18, np.float32)
        self.last_vector = None

    def inputs(self, world, info, tracks, perceived=None):
        """perceived: dict(static [ROWS, COLS] bool, walkers [WALKERS, 5]) of the perception (bev_perceived.py) - read
        in place of the simulator's boxes and walkers."""
        pose, goal = world['pose'], world['goal']
        static = np.zeros((T.ROWS, T.COLS), bool)
        for x, y, yaw, length, width in world['statics']:
            if math.hypot(x-pose[0], y-pose[1]) < T.X_HALF+T.Y_HALF+4.:
                T.paint(static, pose, x, y, yaw, length, width)
        c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
        people = []
        for track in tracks:
            if track.get('class_name') != 'pedestrian':
                continue
            if pose[0]+c*track['x']-s*track['y'] <= T.HIDDEN_X:        # waiting off the lot
                continue
            people.append((math.hypot(track['x'], track['y']), track))
        walkers = np.zeros((D.WALKERS, 5), np.float32)
        for k, (_, track) in enumerate(sorted(people, key=lambda p: p[0])[:D.WALKERS]):
            walkers[k] = (track['x'], track['y'], track['vx'], track['vy'], 1.)
        if perceived is not None:
            static, walkers = np.asarray(perceived['static'], bool), np.asarray(perceived['walkers'], np.float32)
        condition = np.zeros(28, np.float32)
        condition[4] = float(info.get('base_steer_normalized', 0.))
        condition[10:28] = self.tail
        past = list(world.get('history') or [])
        history = [past[-k] if len(past) >= k else (past[0] if past else pose) for k in range(1, D.HISTORY+1)]
        vector = D.vector(float(world['speed']), float(world['steer']), bool(world['reverse']), pose, goal, condition,
                          history, self.previous_intent, self.elapsed)
        return pose, goal, static, walkers, vector

    @torch.no_grad()
    def candidates(self, world, info, tracks, perceived=None):
        """[dict(actions [4, 2], gears [4], intent, probability)]: the likeliest intent's mean chunk first, its
        samples, then the same of the second intent."""
        pose, goal, static, walkers, vector = self.inputs(world, info, tracks, perceived)
        self.last_vector = vector
        device = self.device
        t = lambda a, dtype=torch.float32: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)[None]      # noqa: E731
        w, v = t(walkers), t(vector)
        bev = F.bev_truth(t(pose), t(goal), t(static, torch.bool), w) if self.mode == 'truth' else None
        context, _ = self.model.encode(bev, v, w)
        probability = torch.softmax(self.model.intent(context).float(), 1)[0]
        out = []
        order = probability.argsort(descending=True).tolist()
        for rank, intent in enumerate(order if self.fallback else order[:2]):
            cond = self.model.condition(context, torch.tensor([intent], device=device))
            gears = self.model.gear(cond).view(F.STEPS, 3).argmax(-1).cpu().numpy().astype(np.int64)
            mean = self.model.sample(cond, 1, noise=0.)[0]
            more = self.model.sample(cond, SAMPLES-1)[0] if rank < 2 else mean[:0]
            for chunk in torch.cat((mean, more)).cpu().numpy().astype(np.float32):
                chunk = chunk.copy()
                chunk[gears == 1, 1] = 0.
                out.append(dict(actions=chunk, gears=gears.copy(), intent=int(intent), probability=float(probability[intent])))
        return out

    def executed(self, actions, gears, intent):
        """The chunk that was driven (the student's or the teacher's): the next decision is conditioned on it."""
        actions, gears = np.asarray(actions, np.float32), np.asarray(gears, np.int64)
        one_hot = np.zeros((3, 3), np.float32)
        one_hot[np.arange(3), gears[1:4]] = 1.
        self.tail = np.concatenate((actions[1:4].reshape(-1), one_hot.reshape(-1), np.ones(3, np.float32)))
        self.previous_intent, self.elapsed = int(intent), self.elapsed+1
