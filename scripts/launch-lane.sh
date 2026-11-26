#!/usr/bin/env bash
# Launch a single H3 Studio worker lane in Docker.
# Usage: LANE=a GPU=0 PORT=8003 PEER_URL=http://h3b:8003 bash scripts/launch-lane.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
fi

LANE="${LANE:-a}"
GPU="${GPU:-0}"
PORT="${PORT:-8003}"
CONTAINER_NAME="${CONTAINER_NAME:-$([ "$LANE" = "a" ] && echo "h3" || echo "h3-$LANE")}"
PEER_URL="${PEER_URL:-}"
MODEL_DIR="${MODEL_DIR:-/data/stage/minimax-h3}"
OUT_DIR="${OUT_DIR:-/data/stage/h3-out}"
HFCACHE_DIR="${HFCACHE_DIR:-$ROOT_DIR/hfcache}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
IMAGE="${IMAGE:-h3-diffusers}"

mkdir -p "$OUT_DIR" "$HFCACHE_DIR"

# Removing a container holding ~130 GB pinned host RAM can take >10s; retry cleanly.
for _ in $(seq 60); do
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  docker inspect "$CONTAINER_NAME" >/dev/null 2>&1 || break
  sleep 2
done

docker network create h3net >/dev/null 2>&1 || true

exec docker run -d --name "$CONTAINER_NAME" --network h3net --restart unless-stopped \
  --health-cmd "python /app/healthcheck.py" --health-interval 30s --health-timeout 10s --health-start-period 600s --health-retries 3 \
  --gpus "device=$GPU" --ipc=host \
  --add-host=host.docker.internal:host-gateway \
  --env-file "$ENV_FILE" \
  -e HF_HUB_OFFLINE=1 -e HF_HUB_CACHE=/hf \
  -e H3_LANE="$LANE" -e H3_GPU="$GPU" -e H3_PEERS="$PEER_URL" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -p "${BIND_HOST}:${PORT}:8003" \
  -v "$HFCACHE_DIR:/hf" \
  -v "$MODEL_DIR:/hf/models--MiniMaxAI--MiniMax-H3/snapshots/0000000000000000000000000000000000000000:ro" \
  -v "$OUT_DIR:/out" \
  -v "$ROOT_DIR/server.py:/app/server.py:ro" \
  -v "$ROOT_DIR/index.html:/app/index.html:ro" \
  "$IMAGE"
