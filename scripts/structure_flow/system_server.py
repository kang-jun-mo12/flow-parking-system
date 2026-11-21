"""The system's models on the GPU, for any number of cars (drive.py through supervisor.FlowDriver; a thread and its
own tracker, refiner memory and Flow state per connection, one inference at a time):

  perceive  the 6 cameras of a decision step (every 0.2 s) -> the world state: the frozen perception (bev_evidence.py:
            DDP front end, YOLO-World, UniDepth) gives the evidence BEV, the refiner and the people tracker
            (bev_perceived.py) the static mask and the people. With the car's forecast along the parking plan the
            arbiter (bev_arbiter.py) answers P(a walker in the plan's way within 3 s), P(a prop in it).
  decide    while the supervisor lets the avoidance Flow drive: its candidates from the world state
            (bev_flow_runtime.py: the two likeliest intents, 4 chunks each, then the mean chunk of every other
            intent), the first that passes the safety check (safety.safe) is driven, none: the car stands.
            The reply is the chunk's first step [steer, speed gate, direction request] and `hand_back`: the Flow has
            brought the car to where the reverse parking starts and means to stay there.

--screen: what the safety check reads - `truth`, the simulator's objects (the static mask and the walkers' way the
car's side sends), or `perceived`, the world state of the cameras (people going on at their velocity).

Messages (multiprocessing.connection, dicts): ping; perceive {t, pose, goal, first, forecast, planned, speed, reverse}
followed by the images as bytes (RGB uint8 [6, 300, 400, 3]); reposition_train_reset (the Flow takes over) and
reposition_train_transition (its next step, or terminated: it lets go) {observation, info, world, futures,
goal_local, t}; close.
usage (Python 3.11, from scripts/): python structure_flow/system_server.py --refiner W/bev_refiner.pt
    --flow W/avoidance_flow.pt --arbiter W/arbiter.pt [--screen truth|perceived] [--port 6510]
"""
import argparse
import math
from multiprocessing import AuthenticationError
from multiprocessing.connection import Listener
from pathlib import Path
import sys
import threading
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import safety                                       # noqa: E402
import bev_flow_data                                # noqa: E402

SHARED = {}                  # the frozen perception, the refiner and the arbiter: one copy for every car


class Car:
    """One connection: the perception's memory of this car's episode and its Flow state."""

    def __init__(self, args, lock):
        import bev_flow_runtime
        import bev_perceived
        from open_frontend.tracker import Tracker
        self.lock, self.screen = lock, args.screen
        self.flow = bev_flow_runtime.Runtime(args.flow, seed=args.seed, fallback=True)
        self.eyes = (bev_perceived.Perception(model=SHARED['refiner']), Tracker())
        self.seen = None            # the world state of the latest decision step
        self.staged = False

    def perceive(self, message, payload):
        """The 6 cameras of a decision step -> the static mask and the people of the perceived BEV, kept for the
        decision of the same step; the arbiter's verdict when the car's forecast along the plan comes with them."""
        began = time.perf_counter()
        perception, tracker = self.eyes
        if message.get('first'):
            tracker.reset()
            perception.reset()
        images = np.frombuffer(payload, np.uint8).reshape(6, 300, 400, 3)
        pose, goal = [float(v) for v in message['pose']], message['goal']
        c, s = math.cos(math.radians(pose[2])), math.sin(math.radians(pose[2]))
        dx, dy = goal[0]-pose[0], goal[1]-pose[1]
        target = [dx*c+dy*s, -dx*s+dy*c, (math.degrees(goal[2])-pose[2]+180.) % 360.-180.]
        evidence = SHARED['evidence']
        with self.lock:
            evidence.frontend.tracker = tracker
            e, v, _, _, _ = evidence(images, target, float(message['t']), pose)
            static, walkers = perception.step(e, v, pose)
        self.seen = dict(t=float(message['t']), static=static, walkers=walkers, seconds=time.perf_counter()-began)
        reply = dict(ok=True, people=int(walkers[:, 4].sum()), seconds=self.seen['seconds'])
        if message.get('forecast') is not None:
            forecast = np.asarray(message['forecast'], np.float32)
            planned, speed = bool(message.get('planned', True)), float(message.get('speed', 0.))
            with self.lock:
                verdict = SHARED['arbiter'](pose, [float(g) for g in goal], static, walkers, forecast, planned, speed,
                                            bool(message.get('reverse', False)))
            reply['arbiter'] = [float(verdict[0]), float(verdict[1])]
        return reply

    def decide(self, message):
        """One step of the avoidance Flow: the first of its candidates that is safe, else stand."""
        observation, info = message['observation'], message.get('info') or {}
        if message.get('futures') is None or message.get('world') is None:
            raise RuntimeError('a decision needs the car and the scene (world, futures, goal_local)')
        tracks = list((info.get('dynamic_state') or {}).get('tracks') or [])
        seen = self.seen
        if seen is None or abs(seen['t']-float(message.get('t', 0.))) > 1e-3:
            raise RuntimeError('no perceived BEV for the decision at t = {}'.format(message.get('t')))
        with self.lock:
            proposed = self.flow.candidates(message['world'], info, tracks, seen)
        if self.screen == 'perceived':
            mask200, moving = safety.perceived_scene(seen, message['world']['pose'])
        else:
            mask200 = np.asarray(observation['flow_static_physical_mask'], bool)
            moving = [f for f in message['futures'] if not f.get('static')]         # the props are in the mask
        chosen = next((c for c in proposed if safety.safe(c['actions'], c['gears'], info, mask200, moving)), None)
        if chosen is None:
            steer = float(observation['vector'][4])
            chosen = dict(actions=np.array([[steer, 0.]]*4, np.float32), gears=np.ones(4, np.int64),
                          intent=bev_flow_data.intent_of('stop', 1), probability=0.)
        self.flow.executed(chosen['actions'], chosen['gears'], chosen['intent'])
        # the Flow is done where the reverse parking starts and it means to stay there
        self.staged = chosen['intent'] == 0 and bool(safety.staged(message['goal_local']))
        return safety.first_action(chosen['actions'], chosen['gears'])


def serve_one(connection, args, lock):
    car = Car(args, lock)
    with connection:
        while True:
            try:
                message = connection.recv()
                cmd = message.get('cmd')
                payload = connection.recv_bytes() if cmd == 'perceive' else None
            except (EOFError, ConnectionError, OSError):
                return
            try:
                if cmd == 'ping':
                    reply = dict(ok=True, perceive=True, arbiter=True, collector=False, screen=args.screen)
                elif cmd == 'perceive':
                    reply = car.perceive(message, payload)
                elif cmd == 'reposition_train_reset':
                    car.flow.new_segment()
                    reply = dict(ok=True, action=car.decide(message).tolist(), hand_back=car.staged)
                elif cmd == 'reposition_train_transition':
                    if message.get('terminated') or message.get('truncated'):
                        reply = dict(ok=True, action=None, episode_done=True)
                    else:
                        reply = dict(ok=True, action=car.decide(message).tolist(), hand_back=car.staged)
                elif cmd == 'close':
                    connection.send(dict(ok=True))
                    return
                else:
                    reply = dict(ok=False, error='unknown command {}'.format(cmd))
            except Exception as error:      # keep the other cars running; the client raises on ok=False
                import traceback
                traceback.print_exc()
                reply = dict(ok=False, error=repr(error))
            connection.send(reply)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refiner', type=Path, required=True, help='bev_refiner.py checkpoint')
    parser.add_argument('--flow', type=Path, required=True, help='bev_flow.py checkpoint: the avoidance Flow')
    parser.add_argument('--arbiter', type=Path, required=True, help='bev_arbiter.py checkpoint')
    parser.add_argument('--screen', choices=('truth', 'perceived'), default='truth',
                        help='what the safety check reads: the simulator, or the perceived world state')
    parser.add_argument('--port', type=int, default=6510)
    parser.add_argument('--authkey', default='lot-flow')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    import bev_arbiter
    import bev_evidence
    import bev_refiner
    SHARED.update(evidence=bev_evidence.Evidence(), refiner=bev_refiner.load(str(args.refiner)),
                  arbiter=bev_arbiter.Runtime(str(args.arbiter)))
    lock = threading.Lock()
    with Listener(('127.0.0.1', args.port), authkey=args.authkey.encode()) as listener:
        print('[SYSTEM] listening {} safety check on {}'.format(args.port, args.screen), flush=True)
        while True:
            try:
                connection = listener.accept()
            except (EOFError, OSError, AuthenticationError):
                continue
            threading.Thread(target=serve_one, args=(connection, args, lock), daemon=True).start()


if __name__ == '__main__':
    main()
