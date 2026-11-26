#!/usr/bin/env python3
"""Container healthcheck: the lane is healthy only when its worker reports the model ready.

A wedged render or a failed load otherwise still shows as "Up" in docker ps and in monitoring.
The start period in the launcher covers the ~290 s load.
"""
import json
import os
import sys
import urllib.request

try:
    req = urllib.request.Request("http://127.0.0.1:8003/api/health",
                                 headers={"x-api-key": os.environ["H3_API_KEY"]})
    with urllib.request.urlopen(req, timeout=5) as r:
        state = json.loads(r.read()).get("model")
except Exception as e:
    print(f"healthcheck: {type(e).__name__}: {e}")
    sys.exit(1)

if state != "ready":
    print(f"healthcheck: model is {state}")
    sys.exit(1)
sys.exit(0)
