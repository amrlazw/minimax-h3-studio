#!/usr/bin/env bash
# Launch the optional H3 Studio auto-titler sidecar (polls /api/jobs and writes short titles via LLM).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
fi

: "${H3_API_KEY:?Set H3_API_KEY in .env or environment}"
IMAGE="${IMAGE:-h3-diffusers}"
H3_API="${H3_API:-http://127.0.0.1:8003}"
LLM_API="${LLM_API:-${H3_LLM_URL:-http://127.0.0.1:8001}}"
LLM_MODEL="${LLM_MODEL:-${H3_LLM_MODEL:-qwen3.8-neo-coder}}"
LLM_KEY="${LLM_KEY:-${H3_LLM_KEY:-}}"

for _ in $(seq 30); do
  docker rm -f h3-titler >/dev/null 2>&1 || true
  docker inspect h3-titler >/dev/null 2>&1 || break
  sleep 2
done

exec docker run -d --name h3-titler --restart unless-stopped --network host \
  -e H3_API_KEY="$H3_API_KEY" -e LLM_KEY="$LLM_KEY" \
  -e H3_API="$H3_API" -e LLM_API="$LLM_API" -e LLM_MODEL="$LLM_MODEL" \
  -v "$ROOT_DIR/titler.py:/app/titler.py:ro" \
  --entrypoint python \
  "$IMAGE" /app/titler.py
