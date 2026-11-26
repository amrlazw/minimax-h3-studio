#!/usr/bin/env python3
"""Give every job a short title, using the neo-coder already running on this box.

Why this runs on the host network: containers on this machine cannot reach host ports at all
(INPUT policy drops them — 172.17.0.1, 172.18.0.1 and the LAN address all time out), while the
host itself reaches both the H3 API on :8003 and llama.cpp on :8001. So the titler sits outside
the containers and bridges the two.

It is deliberately dumb and idempotent: every pass titles any job that has no title yet, oldest
first, so a restart backfills history. A title it cannot produce is simply skipped and retried
on the next pass; nothing here can block a render.
"""
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

H3 = os.environ.get("H3_API", "http://127.0.0.1:8003").rstrip("/")
H3_FALLBACK = os.environ.get("H3_API_FALLBACK", "http://127.0.0.1:8004").rstrip("/")
LLM = os.environ.get("LLM_API", "http://127.0.0.1:8001").rstrip("/")
H3_KEY = os.environ["H3_API_KEY"]
LLM_KEY = os.environ.get("LLM_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "qwen3.8-neo-coder")
EVERY = int(os.environ.get("TITLE_INTERVAL", "20"))

SYSTEM = ("You title short generated videos. Reply with a title of at most six words, lowercase, "
          "no quotes, no trailing full stop, no explanation. Describe what is on screen.")


def h3_get(path):
    req = urllib.request.Request(H3 + path, headers={"x-api-key": H3_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _h3_get_via_fallback(path):
    """Lane A on :8003, else lane B on :8004 — both serve the same job store."""
    req = urllib.request.Request(H3_FALLBACK + path, headers={"x-api-key": H3_KEY})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _hash(prompt):
    return hashlib.sha1(prompt.encode("utf-8", "ignore")).hexdigest()


def h3_post(path, fields):
    data = urllib.parse.urlencode(fields).encode()
    req = urllib.request.Request(H3 + path, data=data,
                                 headers={"x-api-key": H3_KEY,
                                          "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _chat(body):
    headers = {"Content-Type": "application/json"}
    if LLM_KEY:
        headers["Authorization"] = "Bearer " + LLM_KEY
    req = urllib.request.Request(LLM + "/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read())
    return (d["choices"][0]["message"].get("content") or "").strip()


def ask_model(prompt):
    """The coder is a thinking model: without enable_thinking=false it spends its whole budget on
    reasoning and returns an empty answer (measured: 24 tokens -> '', 231 tokens -> a title).
    With it, a title costs ~10 tokens and 1.1 s."""
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt[:4000]}],
            "max_tokens": 48, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}}
    title = _chat(body)
    if not title:
        # template ignored the switch: give it room to finish reasoning, then read the answer
        body.pop("chat_template_kwargs", None)
        body["max_tokens"] = 300
        title = _chat(body)
    title = title.splitlines()[0] if title else ""
    title = re.sub(r'^["\'`*\s]+|["\'`*.\s]+$', "", title)
    title = re.sub(r"^(title|summary)\s*[:\-]\s*", "", title, flags=re.I)
    title = " ".join(title.split())
    if len(title) > 70:
        title = title[:70].rsplit(" ", 1)[0]
    return title


def main():
    if not LLM_KEY:
        print("titler: no LLM_KEY, will try unauthenticated", flush=True)
    print(f"titler: {H3} -> {LLM} ({MODEL}), every {EVERY}s", flush=True)
    while True:
        try:
            jobs = h3_get("/api/jobs")
        except Exception as e:
            # lane A may be restarting; lane B serves the same job store
            try:
                jobs = _h3_get_via_fallback("/api/jobs")
            except Exception:
                print(f"titler: no H3 lane answering: {type(e).__name__}: {e}", flush=True)
                time.sleep(EVERY)
                continue
        # Identical prompts get the same title: three queued re-renders of one brief should not
        # read as three different shots.
        known = {}
        for j in jobs:
            if j.get("title"):
                known.setdefault(_hash(j["prompt"]), j["title"])
        todo = [j for j in jobs if not j.get("title")]
        todo.sort(key=lambda j: j.get("created") or 0)
        for j in todo[:6]:
            h = _hash(j["prompt"])
            if h in known:
                t = known[h]
            else:
                try:
                    t = ask_model(j["prompt"])
                except Exception as e:
                    print(f"titler: model failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
                    time.sleep(5)
                    break
            if not t:
                continue
            try:
                h3_post(f"/api/jobs/{j['id']}/title", {"title": t})
                known[h] = t
            except Exception as e:
                print(f"titler: store failed: {type(e).__name__}: {str(e)[:80]}", flush=True)
            time.sleep(0.5)
        time.sleep(EVERY)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
