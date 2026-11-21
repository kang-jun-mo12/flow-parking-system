#!/bin/bash
# The whole system on one CARLA 0.9.11 server that is already running: four processes on this machine.
#   system_server.py   perception (DDP front end, YOLO-World, UniDepth -> evidence BEV -> refiner -> people),
#                      arbiter, avoidance Flow, safety check                                       (Python 3.11, GPU)
#   model_server.py    parking model                                                               (Python 3.11, GPU)
#   scene_server.py    the scene of an episode (slot, props, walkers), the parking plan's check    (Python 3.11)
#   drive.py           CARLA client: cameras, supervisor, path tracker, control                    (Python 3.7)
# usage: CARLA_EGG=path/to/carla-0.9.11-py3.7-*.egg [PY311=python] [PY37=python3.7] [SCREEN=perceived] [WATCH=1]
#            ./run_system.sh OUT_FOLDER [CARLA_PORT] [FIRST_EPISODE] [EPISODES]
# SCREEN=perceived: the safety check reads the perceived world state (default: the simulator's objects).
# WATCH=1: the server window follows the car from above, in real time.
OUT=${1:?usage: run_system.sh OUT_FOLDER [CARLA_PORT] [FIRST_EPISODE] [EPISODES]}
PORT=${2:-2000}; FIRST=${3:-0}; COUNT=${4:-1}
PY311=${PY311:-python}; PY37=${PY37:-python3.7}
: "${CARLA_EGG:?CARLA_EGG: the carla 0.9.11 egg of Python 3.7}"
here() { pwd -W 2>/dev/null || pwd; }                # a Windows path under Git Bash: the Pythons are native
ROOT=$(cd "$(dirname "$0")" && here)
mkdir -p "$OUT" && OUT=$(cd "$OUT" && here)
export CARLA_EGG BENCH_DIR="$ROOT/scripts/structure_flow"
W="$ROOT/weights"
[ -f "$W/ddp_frontend.pt" ] || { echo "weights/ddp_frontend.pt is missing: download it from the Release (README)"; exit 1; }
cd "$ROOT/scripts" || exit 1

$PY311 -u structure_flow/system_server.py --port 6510 --refiner "$W/bev_refiner.pt" --flow "$W/avoidance_flow.pt" \
  --arbiter "$W/arbiter.pt" ${SCREEN:+--screen "$SCREEN"} > "$OUT/system.log" 2>&1 < /dev/null &
SYSTEM=$!
$PY311 -u structure_flow/model_server.py "$W/parking_model.pt" --port 6410 > "$OUT/model.log" 2>&1 < /dev/null &
MODEL=$!
stop() { kill $SYSTEM $MODEL ${SCENE:-} 2>/dev/null; }
trap stop EXIT
ready() {       # ready LOG WORDS PID: the process says it listens, or it has ended
  until grep -q "$2" "$1" 2>/dev/null; do
    kill -0 "$3" 2>/dev/null || { echo "failed: $1"; tail -5 "$1"; exit 1; }
    sleep 1
  done
}
ready "$OUT/model.log" "model ready" $MODEL
ready "$OUT/system.log" "listening" $SYSTEM
$PY311 -u structure_flow/scene_server.py "$OUT" --port 6315 --model-port 6410 > "$OUT/scene.log" 2>&1 < /dev/null &
SCENE=$!
ready "$OUT/scene.log" "listening" $SCENE

$PY37 -u structure_flow/drive.py "$OUT" --port "$PORT" --server-port 6315 --first "$FIRST" --count "$COUNT" \
  --renderer dx --depth 0 --images 0 ${WATCH:+--watch} --walkers cases --flow-port 6510 2>&1 \
  | grep -v "^WARNING" | tee -a "$OUT/client.log"
