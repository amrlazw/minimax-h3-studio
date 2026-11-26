#!/usr/bin/env bash
# Recreate BOTH H3 lanes (GPU 0 & GPU 1) from the same image in order, plus the optional titler.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT_DIR/.env}"
if [ -f "$ENV_FILE" ]; then
  set -a; source "$ENV_FILE"; set +a
fi

: "${H3_API_KEY:?Set H3_API_KEY in .env or environment}"

waitready() {
  local port=$1 name=$2
  for _ in $(seq 60); do
    local h
    h=$(curl -s -H "x-api-key: $H3_API_KEY" --max-time 5 "http://127.0.0.1:$port/api/health" 2>/dev/null || true)
    if echo "$h" | grep -q '"model":"ready"'; then
      echo "  $name ready$(echo "$h" | grep -o '"loaded_in":[0-9.]*' | sed 's/"loaded_in":/ in /;s/$/ s/')"
      return 0
    fi
    if echo "$h" | grep -q 'load failed'; then
      echo "  $name FAILED: $h" >&2
      return 1
    fi
    sleep 10
  done
  echo "  $name did not become ready in 10 min" >&2
  return 1
}

echo "== lane A (GPU 0)"
LANE=a GPU=0 PORT=8003 CONTAINER_NAME=h3 PEER_URL=http://h3b:8003 bash "$ROOT_DIR/scripts/launch-lane.sh" >/dev/null
waitready 8003 A

echo "== lane B (GPU 1)"
LANE=b GPU=1 PORT=8004 CONTAINER_NAME=h3b PEER_URL=http://h3:8003 bash "$ROOT_DIR/scripts/launch-lane.sh" >/dev/null
waitready 8004 B

if [ -n "${H3_LLM_URL:-}" ]; then
  echo "== titler"
  bash "$ROOT_DIR/scripts/launch-titler.sh" >/dev/null && echo "  titler up"
fi

echo "== summary"
curl -s -H "x-api-key: $H3_API_KEY" --max-time 8 http://127.0.0.1:8003/api/lanes | python3 -c '
import json, sys
for l in json.load(sys.stdin):
    g = l.get("gpu") or {}
    print("  %s  %-9s v%s  %s GB  running=%s  queued=%s"
          % (l.get("lane"), l.get("model"), l.get("version"), g.get("used_gb"),
             bool(l.get("running")), l.get("queued")))
'
