"""Model process of the action parking model (action_model.py) for the closed-loop benchmark (benchmark.py --actions).

Like plan_server.py: the CARLA 0.9.11 client only exists for Python 3.7, so the models live here. A request is one
JSON header {target [x m, y m, yaw deg] in the vehicle frame, speed (signed m/s), steer, reverse} and the 6 camera
images as raw RGB uint8 in rig.CAMERAS order. The DDP front end turns the images into the Fused BEV, the model samples
8 control sequences of 64 frames, and the reply is the medoid of their first second, as the car is to be given it:
{controls [64, 4] throttle, steer, brake, reverse; seconds}.
usage (Python 3.11, from scripts/): python structure_flow/action_server.py WEIGHTS [--port 6100]
"""
import argparse
import json
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ddp_frontend import rig                                        # noqa: E402
from ddp_frontend.model import DdpFrontend                          # noqa: E402
from structure_flow import action_model                             # noqa: E402
from sync_flow.windows import target_features                       # noqa: E402

AUTHKEY = b'structure-flow-benchmark'
SHAPE = (len(rig.CAMERAS), rig.HEIGHT, rig.WIDTH, 3)
FRONTEND = str(Path(__file__).resolve().parents[2]/'weights'/'ddp_frontend.pt')      # the Release asset (README)


class Driver:
    def __init__(self, weights, frontend, samples=8, seed=0):
        self.frontend = DdpFrontend.load(frontend).cuda().eval()
        self.model = action_model.load(weights)
        self.cfg, self.samples = self.model.cfg, samples
        self.intrinsics = rig.intrinsics().cuda()[None]
        self.extrinsics = rig.extrinsics().cuda()[None]
        self.generator = torch.Generator('cuda').manual_seed(seed)

    @torch.no_grad()
    def controls(self, images, state):
        cfg = self.cfg
        pixels = torch.stack([rig.prepare_image(image) for image in images])[None].cuda()
        out = self.frontend(pixels, self.intrinsics, self.extrinsics,
                            torch.tensor(state['target'], dtype=torch.float32)[None].cuda())
        bev = out['fused_bev'].half().float()           # the training cache holds fp16
        x, y, yaw = state['target']
        target = torch.as_tensor(target_features((x, y, np.radians(yaw)), cfg))[None].cuda()
        ego = torch.tensor([[state['speed']/cfg.max_speed_mps, state['steer'], float(state['reverse'])]],
                           dtype=torch.float32).cuda()
        samples = self.model.sample(bev, target, ego, self.samples, generator=self.generator).cpu().numpy()
        return action_model.executed(samples[action_model.medoid(samples)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weights', type=Path)
    parser.add_argument('--frontend', default=FRONTEND)
    parser.add_argument('--port', type=int, default=6100)
    parser.add_argument('--samples', type=int, default=8)
    args = parser.parse_args()
    driver = Driver(args.weights, args.frontend, args.samples)
    with Listener(('127.0.0.1', args.port), authkey=AUTHKEY) as listener:
        print('action model ready', args.weights, 'port', args.port, flush=True)
        while True:
            try:
                connection = listener.accept()
            except (EOFError, OSError, AuthenticationError):
                continue
            with connection:
                while True:
                    try:
                        state = json.loads(connection.recv_bytes().decode())
                        images = np.frombuffer(connection.recv_bytes(), np.uint8).reshape(SHAPE)
                    except (EOFError, ConnectionError, OSError):
                        break
                    started = time.perf_counter()
                    controls = driver.controls(images, state)
                    torch.cuda.synchronize()
                    reply = dict(controls=np.round(controls, 4).tolist(), seconds=time.perf_counter()-started)
                    connection.send_bytes(json.dumps(reply).encode())


if __name__ == '__main__':
    main()
