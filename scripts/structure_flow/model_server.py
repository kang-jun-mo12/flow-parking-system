"""Shared model process for the collection teachers (scene_server.py --model-port): one copy of the DDP
front end and the StructureFlow planner on the GPU, any number of teacher connections (a thread each, one
inference at a time).

Request: a JSON header {target [x m, y m, yaw deg] in the vehicle frame, speed (signed m/s), steer, reverse},
then the 6 camera images as raw RGB uint8 in rig.CAMERAS order. Reply: every sampled latent with its
structure, and the structure probabilities, so the teacher can choose among them with the true scene.
usage (Python 3.11, from scripts/): python structure_flow/model_server.py CHECKPOINT [--port 6400]
"""
import argparse
import json
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path
import sys
import threading
import time
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from structure_flow.plan_server import Planner, SHAPE
from sync_flow.windows import target_features

AUTHKEY = b'lot-model'
FRONTEND = str(Path(__file__).resolve().parents[2]/'weights'/'ddp_frontend.pt')      # the Release asset (README)


def serve_one(connection, planner, lock):
    with connection:
        while True:
            try:
                state = json.loads(connection.recv_bytes().decode())
                images = np.frombuffer(connection.recv_bytes(), np.uint8).reshape(SHAPE)
            except (EOFError, ConnectionError, OSError):
                return
            started = time.perf_counter()
            with lock, torch.no_grad():
                cfg = planner.cfg
                bev, seg = planner.perceive(images, state)
                x, y, yaw = state['target']
                target = target_features((x, y, np.radians(yaw)), cfg)
                ego = np.array([state['speed']/cfg.max_speed_mps, state['steer'], float(state['reverse'])], np.float32)
                scene = torch.as_tensor(seg)[None].cuda() if cfg.condition == 'seg' else bev
                latents, structures, probabilities = planner.model.sample(
                    scene, torch.as_tensor(target)[None].cuda(), torch.as_tensor(ego)[None].cuda())
                reply = dict(latents=latents.cpu().numpy().round(5).tolist(), h=structures.cpu().numpy().tolist(),
                             probabilities=probabilities.cpu().numpy().round(5).tolist(),
                             seconds=round(time.perf_counter()-started, 3))
            connection.send_bytes(json.dumps(reply).encode())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--frontend', default=FRONTEND)
    parser.add_argument('--port', type=int, default=6400)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    planner = Planner(args.checkpoint, args.frontend, 'ddp')
    lock = threading.Lock()
    with Listener(('127.0.0.1', args.port), authkey=AUTHKEY) as listener:
        print('model ready', args.checkpoint, 'port', args.port, flush=True)
        while True:
            try:
                connection = listener.accept()
            except (EOFError, OSError, AuthenticationError):
                continue
            threading.Thread(target=serve_one, args=(connection, planner, lock), daemon=True).start()


if __name__ == '__main__':
    main()
