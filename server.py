# H3 Studio v2: MiniMax H3 (t2va / fl2va / ref2va) worker + web page.
# One GPU, one job at a time. int8 weights + group offload to host RAM (diffusers "consumer card" path).
# Weights come from the local folder exposed as an HF cache snapshot (HF_HUB_OFFLINE=1, see ~/launch-h3.sh).
#
# v2 adds: live step progress from forward hooks, ETA from measured history, step presets,
# cancel/retry/delete, cookie sessions, reference duration checks, poster frames, stage timings.
import base64
from contextlib import contextmanager
import fcntl
from fractions import Fraction
import gc
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import re
import secrets
import shutil
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import uuid
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

VERSION = "2.4"
KEY = os.environ["H3_API_KEY"].encode()
JOBS = Path("/out/jobs"); JOBS.mkdir(parents=True, exist_ok=True)
SERIES = Path("/out/series"); SERIES.mkdir(parents=True, exist_ok=True)
STITCHES = Path("/out/stitches"); STITCHES.mkdir(parents=True, exist_ok=True)
CASTS = Path("/out/casts"); CASTS.mkdir(parents=True, exist_ok=True)
GHOST_REFS = Path("/out/ghost-refs"); GHOST_REFS.mkdir(parents=True, exist_ok=True)
DEMOS = Path("/out/demos"); DEMOS.mkdir(parents=True, exist_ok=True)
# One history file per lane. Two lanes finishing at the same moment would otherwise rewrite the
# same file: read-modify-write loses an entry, and a reader catching a partial write sees [] and
# falls back to the default cost model. Per-lane files remove the lock entirely.
PERF = Path(f"/out/perf-{os.environ.get('H3_LANE', 'a')}.json")
R = "MiniMaxAI/MiniMax-H3"
# Sharded checkpoints load from the path: diffusers asks the Hub API for the shard list even when offline.
LOCAL = "/hf/models--MiniMaxAI--MiniMax-H3/snapshots/" + "0" * 40
PARTITION = os.environ.get("H3_PARTITION", "both").lower()  # both | base | ref
LLM_URL = os.environ.get("H3_LLM_URL", "").rstrip("/")      # optional prompt polish backend
LLM_KEY = os.environ.get("H3_LLM_KEY", "")
LLM_MODEL = os.environ.get("H3_LLM_MODEL", "qwen3.8-neo-coder")
VLM_URL = os.environ.get("H3_VLM_URL", "").rstrip("/")      # optional vision model for series last-frame continuity
VLM_KEY = os.environ.get("H3_VLM_KEY", "")
VLM_MODEL = os.environ.get("H3_VLM_MODEL", "qwen3.8-27b")
FFN_CHUNK = int(os.environ.get("H3_FFN_CHUNK", "16384"))    # sequence chunk size for FFN & row-wise norm/AdaLN
ATTN_CHUNK = int(os.environ.get("H3_ATTN_CHUNK", "32768"))  # query chunk size for self-attention memory control
_CHUNK_CFG = {"ffn": FFN_CHUNK, "attn": ATTN_CHUNK}
# Canvas presets (height, width), multiples of 32 under the 768p short-edge / 768*1344 pixel budget.
ASPECTS = {
    "auto": None,
    "16:9": (768, 1344),
    "9:16": (1344, 768),
    "4:3": (768, 1024),
    "3:4": (1024, 768),
    "1:1": (1024, 1024),
    "21:9": (672, 1568),
}

# Two lanes render in parallel: same image, same job store, one GPU each. The claim file in a job
# directory is the lock, so either lane can take the oldest queued job regardless of who queued it.
LANE = os.environ.get("H3_LANE", "a")
GPU = int(os.environ.get("H3_GPU", "0"))

# Persistent rotating log file in /out/logs so worker stdout/stderr survives container rebuilds.
LOG_DIR = Path("/out/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"h3-{LANE}.log"


class _TeeStream:
    """Mirror stdout/stderr writes to both the original stream and a rotating file handler."""
    def __init__(self, orig, handler):
        self._orig = orig
        self._handler = handler
        self._buf = ""

    def write(self, data):
        self._orig.write(data)
        self._buf += data
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line:
                rec = logging.LogRecord("h3", logging.INFO, "", 0, line, (), None)
                self._handler.emit(rec)
        return len(data)

    def flush(self):
        self._orig.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self._orig.fileno()


if not isinstance(sys.stdout, _TeeStream):
    _log_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    _log_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%Y-%m-%d %H:%M:%S"))
    sys.stdout = _TeeStream(sys.stdout, _log_handler)
    sys.stderr = _TeeStream(sys.stderr, _log_handler)
PEERS = [u.strip().rstrip("/") for u in os.environ.get("H3_PEERS", "").split(",") if u.strip()]
VRAM_FRACTION = float(os.environ.get("H3_VRAM_FRACTION", "0")) or None

FPS = 24
# The video VAE requires frames on the 17n+5 grid. The pipeline enforces 5.0-15.0 s natively;
# we monkey-patch max_duration after load() to lift that cap to H3_MAX_DURATION (default 30 s).
# 17*7+5=124 is the shortest legal clip (5.17 s); floor 107 is on the grid but rejected at denoise.
_MAX_DURATION = float(os.environ.get("H3_MAX_DURATION", "30.0"))  # seconds; env-tunable
_MAX_N = int((_MAX_DURATION * FPS - 5) // 17)                     # largest n satisfying 17n+5 ≤ limit
FRAMES = [17 * n + 5 for n in range(7, _MAX_N + 1)]              # 124..(_MAX_N*17+5) frames
assert FRAMES, f"H3_MAX_DURATION={_MAX_DURATION} too small — no valid frame counts (need ≥ 5 s)"
assert FRAMES[0] >= 120, f"shortest frame count {FRAMES[0]} is below the pipeline's 5-s floor"
STEPS = {"draft": 8, "standard": 12, "fine": 25, "ultra": 36, "reference": 50}
MAX_UPLOAD = 300 * 2**20
MAX_STEPS = 50
FRAME_EXP = 1.5                                             # cost per step grows as frames^1.5
DEFAULT_K, DEFAULT_C = 0.015112, 0.0                        # s per step-frame^1.5, one-off overhead
MIN_FREE_GB = 20
LIMITS = {"image": 9, "video": 3, "audio": 3}
MAX_REFS_TOTAL = 12
MAX_PROMPT_CHARS = 20000
NATIVE_MAX_FRAMES = 345                                     # 17*20+5 = 14.38 s native open-weights cap
REF_VIDEO_MIN_S = 1.9
REF_VIDEO_MAX_S = _MAX_DURATION + 0.5                       # 0.5 s rounding headroom above max output
REF_AUDIO_MIN_S = 0.2
REF_EDGES = (2048, 1536, 1024)
ROLES = ("person", "place", "product", "style", "motion", "voice")
UI_ROLES = {
    "image": [
        "subject_person", "subject_outfit", "subject_place", "subject_product",
        "subject_style", "keyframe_anchor", "storyboard",
    ],
    "video": ["video_edit", "video_continue", "video_motion", "subject_person"],
    "audio": ["audio_voice", "audio_bgm", "audio_sfx"],
}
UI_ROLE_TO_LEGACY = {
    "subject_person": "person",
    "subject_outfit": "style",
    "subject_place": "place",
    "subject_product": "product",
    "subject_style": "style",
    "keyframe_anchor": "style",
    "storyboard": "style",
    "video_edit": "motion",
    "video_continue": "motion",
    "video_motion": "motion",
    "audio_voice": "voice",
    "audio_bgm": "voice",
    "audio_sfx": "voice",
}
DEFAULT_UI_ROLE = {"image": "subject_person", "video": "video_motion", "audio": "audio_voice"}
DEFAULT_LEGACY_ROLE = {"image": "person", "video": "motion", "audio": "voice"}
LEGACY_TO_UI_ROLE = {
    ("image", "person"): "subject_person",
    ("image", "place"): "subject_place",
    ("image", "product"): "subject_product",
    ("image", "style"): "subject_style",
    ("video", "person"): "subject_person",
    ("video", "motion"): "video_motion",
    ("audio", "voice"): "audio_voice",
}
EXT = {".jpg": "image", ".jpeg": "image", ".png": "image", ".webp": "image", ".gif": "image", ".bmp": "image",
       ".mp4": "video", ".mov": "video", ".webm": "video", ".mkv": "video", ".avi": "video",
       ".mp3": "audio", ".wav": "audio", ".m4a": "audio", ".aac": "audio", ".flac": "audio", ".ogg": "audio",
       ".opus": "audio"}
STAGE = {"queued": "Queued", "encode": "Encoding prompt", "denoise": "Denoising", "decode": "Decoding",
         "mux": "Writing MP4", "done": "Done", "error": "Failed", "cancelled": "Cancelled"}

state = {"model": "loading", "loaded_in": None, "running": None, "step": 0, "total": 0, "ticks": []}
lock = threading.Lock()
pipe = None


def queued_ids():
    out = []
    for p in JOBS.glob("*/job.json"):
        try:
            if json.loads(p.read_text()).get("status") == "queued":
                out.append((p.stat().st_mtime, p.parent.name))
        except Exception:
            continue
    return [jid for _, jid in sorted(out)]


def next_job():
    """Atomically claim the oldest queued job this lane can run.

    `O_CREAT|O_EXCL` on <job>/claim is the lock: only one lane creates it. A claim left behind by a
    worker that died between claiming and starting goes stale and is cleared after 90 s, so a crash
    cannot strand a job forever.
    """
    for jid in queued_ids():
        d = JOBS / jid
        try:
            job = json.loads((d / "job.json").read_text())
        except Exception:
            continue
        if job.get("status") != "queued":
            continue
        if job.get("cancel"):
            job.update(status="cancelled", stage="cancelled", finished=time.time())
            write(job)
            continue
        if PARTITION == "base" and job.get("workflow") == "ref2va":
            continue
        if PARTITION == "ref" and job.get("workflow") != "ref2va":
            continue
        if job.get("pin") and job["pin"] != LANE:
            continue        # pinned to the other lane; that lane will take it
        if job.get("frames") not in FRAMES:
            # legacy job from before the 5 s floor was known: raise it here so it cannot die again
            job["frames"] = legal_frames(job.get("frames"))
            job["frames_note"] = "raised to the shortest legal length (5 s floor)"
            write(job)
            print(f"claim: {jid[:8]} frames raised to {job['frames']}", flush=True)
        claim = d / "claim"
        try:
            fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - claim.stat().st_mtime > 90:
                    claim.unlink()
            except Exception:
                pass
            continue
        with os.fdopen(fd, "w") as f:
            f.write(LANE)
        job.update(status="running", stage="encode", step=0, total_steps=job["steps"],
                   started=time.time(), lane=LANE, error=None)
        write(job)
        return job
    return None


class Cancelled(Exception):
    """Raised from the step hook so a cancel lands at the next step boundary, not after the render."""


_peer_cache = {"at": 0.0, "data": []}
app = FastAPI(title="H3 Studio", version=VERSION)


# ---------------------------------------------------------------- auth

def _key_ok(request: Request) -> bool:
    cand = (request.headers.get("x-api-key")
            or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            or request.cookies.get("h3key")
            or request.query_params.get("key") or "")
    return bool(cand) and secrets.compare_digest(cand.encode(), KEY)


@app.middleware("http")
async def auth(request: Request, call_next):
    if _key_ok(request) or request.url.path in ("/", "/index.html", "/favicon.ico", "/api/session"):
        return await call_next(request)
    return JSONResponse({"detail": "bad or missing API key"}, 401)


@app.post("/api/session")
async def session(request: Request, key: str = Form(...)):
    if not secrets.compare_digest(key.strip().encode(), KEY):
        raise HTTPException(401, "bad API key")
    r = JSONResponse({"ok": True})
    r.set_cookie("h3key", key.strip(), max_age=30 * 86400, httponly=True, samesite="lax", path="/")
    return r


@app.delete("/api/session")
def logout():
    r = JSONResponse({"ok": True})
    r.delete_cookie("h3key", path="/")
    return r


# ---------------------------------------------------------------- job store

def read(jid):
    if not re.fullmatch(r"[0-9a-f]{32}", jid):
        raise HTTPException(404, "no such job")
    p = JOBS / jid / "job.json"
    if not p.exists():
        raise HTTPException(404, "no such job")
    return json.loads(p.read_text())


def write(job):
    p = JOBS / job["id"] / "job.json"
    t = p.with_suffix(".tmp")
    t.write_text(json.dumps(job))
    os.replace(t, p)


def set_state(**kw):
    """Update the live job record while the pipeline runs. Called from hooks."""
    with lock:
        jid = state.get("running")
        state.update({k: v for k, v in kw.items() if k in ("step", "total", "stage")})
    if not jid:
        return
    try:
        job = read(jid)
        if job.get("status") != "running":
            return
        if "step" in kw:
            job["step"] = kw["step"]
        if "total" in kw:
            job["total_steps"] = kw["total"]
        if "stage" in kw:
            job["stage"] = kw["stage"]
        write(job)
    except Exception:
        pass


def next_number():
    """Sequential, human-referenceable job number.

    Both lanes create jobs, so the counter is a file locked with flock: the lock is in the kernel,
    not the process, which makes it safe across containers sharing /out.
    """
    seq = JOBS / ".seq"
    fd = os.open(seq, os.O_RDWR | os.O_CREAT, 0o644)
    with os.fdopen(fd, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            raw = f.read().strip()
            n = int(raw) + 1 if raw.isdigit() else 1
            f.seek(0)
            f.truncate()
            f.write(str(n))
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return n


def backfill_numbers():
    missing = []
    for p in JOBS.glob("*/job.json"):
        try:
            j = json.loads(p.read_text())
        except Exception:
            continue
        if j.get("num") is None:
            missing.append((j.get("created") or 0, p))
    missing.sort()
    for _, p in missing:
        try:
            j = json.loads(p.read_text())
            if j.get("num") is None:
                j["num"] = next_number()
                write(j)
        except Exception:
            continue
    return len(missing)


# ---------------------------------------------------------------- media helpers

def kind_of(u: UploadFile):
    k = (u.content_type or "").split("/")[0]
    if k in LIMITS:
        return k
    return EXT.get(Path(u.filename or "").suffix.lower())


def media_duration(path):
    try:
        import av
        with av.open(str(path)) as c:
            if c.duration:
                return c.duration / 1e6
            for s in c.streams:
                if s.duration and s.time_base:
                    return float(s.duration * s.time_base)
    except Exception:
        pass
    return None


def probe_video(path):
    try:
        import av
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            w, h = s.width, s.height
            rot = 0.0
            for fr in c.decode(s):
                rot = float(getattr(fr, "rotation", 0.0) or 0.0)
                break
            if round(rot / 90.0) % 2:
                w, h = h, w
            fps = float(s.average_rate or s.guessed_rate or 0) or None
            dur = float(c.duration) / 1e6 if c.duration else None
            if dur is None and s.duration and s.time_base:
                dur = float(s.duration * s.time_base)
            audio_sr = int(c.streams.audio[0].codec_context.sample_rate) if c.streams.audio and c.streams.audio[0].codec_context.sample_rate else None
            return {
                "width": w,
                "height": h,
                "fps": round(fps, 3) if fps else None,
                "duration": round(dur, 3) if dur else 0.0,
                "has_audio": bool(c.streams.audio),
                "audio_sample_rate": audio_sr,
            }
    except Exception:
        return {}


def probe_audio(path):
    """Probe an audio file for duration, stream presence, and sample rate."""
    dur = media_duration(path)
    has_audio = False
    audio_sr = None
    try:
        import av
        with av.open(str(path)) as c:
            if c.streams.audio:
                s = c.streams.audio[0]
                has_audio = True
                if s.codec_context.sample_rate:
                    audio_sr = int(s.codec_context.sample_rate)
                if dur is None and s.duration and s.time_base:
                    dur = float(s.duration * s.time_base)
    except Exception:
        pass
    return {
        "duration": round(float(dur), 3) if dur else None,
        "has_audio": has_audio,
        "audio_sample_rate": audio_sr,
    }


def probe_image(path):
    try:
        from PIL import Image, ImageOps
        with Image.open(str(path)) as im:
            im = ImageOps.exif_transpose(im)
            return {"width": im.width, "height": im.height}
    except Exception:
        return {}


def _snap_canvas(w: float, h: float, short_edge: int = 768, max_pixels: int = 768 * 1344, mult: int = 32):
    ratio = max(0.25, min(4.0, float(w) / max(1.0, float(h))))
    cw, ch = (short_edge * ratio, float(short_edge)) if ratio >= 1.0 else (float(short_edge), short_edge / ratio)
    area = cw * ch
    if area > max_pixels:
        s = (max_pixels / area) ** 0.5
        cw *= s
        ch *= s
    return max(mult, round(ch / mult) * mult), max(mult, round(cw / mult) * mult)


def _resolve_canvas(workflow: str, aspect: str, ref_meta: list[dict]):
    if aspect != "auto":
        return ASPECTS[aspect]
    if workflow == "ref2va":
        for r in ref_meta:
            if (r.get("effective_kind") or r.get("kind")) == "video" and r.get("width") and r.get("height"):
                return _snap_canvas(r["width"], r["height"])
        return ASPECTS["16:9"]
    if workflow == "fl2va":
        return None, None
    return ASPECTS["16:9"]


def poster(path, out):
    """First frame at ~1 s, as a small JPEG, so the bin does not decode video for thumbnails."""
    try:
        import av
        from PIL import Image  # noqa: F401  (frame.to_image needs Pillow)
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            want = max(1, int((s.average_rate or FPS) * 1.0))
            frame, seen = None, 0
            for fr in c.decode(s):
                frame = fr
                seen += 1
                if seen >= want:
                    break
            if frame is None:
                return False
            img = frame.to_image()
            img.thumbnail((640, 640))
            img.save(str(out), quality=82, optimize=True)
            return True
    except Exception:
        return False


class Upload:
    async def save(self, u: UploadFile, d: Path, stem: str):
        suffix = re.sub(r"[^.a-z0-9]", "", Path(u.filename or "").suffix.lower())[:8]
        name = stem + suffix
        size = 0
        with open(d / name, "wb") as f:
            while chunk := await u.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD:
                    raise HTTPException(413, f"{u.filename}: larger than {MAX_UPLOAD >> 20} MB")
                f.write(chunk)
        return name, size


# ---------------------------------------------------------------- routes

@app.get("/")
def index():
    return FileResponse("/app/index.html")


@app.get("/api/engine")
def engine():
    k, c = fit_cost()
    return {"version": VERSION, "model": R, "partition": PARTITION, "fps": FPS,
            "frames": FRAMES, "native_max_frames": NATIVE_MAX_FRAMES,
            "max_duration_s": round(FRAMES[-1] / FPS, 2),
            "steps": STEPS, "max_steps": MAX_STEPS, "limits": LIMITS,
            "max_refs_total": MAX_REFS_TOTAL, "max_prompt_chars": MAX_PROMPT_CHARS,
            "ref_video_min_s": REF_VIDEO_MIN_S, "ref_video_max_s": REF_VIDEO_MAX_S,
            "ref_audio_min_s": REF_AUDIO_MIN_S,
            "aspects": list(ASPECTS), "aspect_dims": ASPECTS,
            "ref_edges": list(REF_EDGES), "ui_roles": UI_ROLES,
            "stages": STAGE, "polish": bool(LLM_URL),
            "series": True, "vlm": bool(VLM_URL or LLM_URL), "vlm_vision": bool(VLM_URL),
            "lane": LANE, "gpu_index": GPU,
            "ffn_chunk": FFN_CHUNK, "attn_chunk": ATTN_CHUNK,
            "est": {"k": round(k, 6), "p": FRAME_EXP, "c": round(c, 1), "n": len(_hist())}}


@app.post("/api/probe")
async def probe(file: UploadFile = File(...)):
    """Probe an uploaded image, video, or audio file without creating a job."""
    k = kind_of(file)
    if k is None:
        raise HTTPException(400, "unsupported file type; upload image, video, or audio")
    up = Upload()
    with tempfile.TemporaryDirectory() as td:
        name, size = await up.save(file, Path(td), "probe")
        path = Path(td) / name
        meta = {}
        dur = None
        if k == "image":
            meta = probe_image(path)
        elif k == "video":
            meta = probe_video(path)
            dur = meta.get("duration") or media_duration(path)
        elif k == "audio":
            meta = probe_audio(path)
            dur = meta.get("duration") or media_duration(path)
    return {
        "kind": k,
        "name": (file.filename or "")[:200],
        "bytes": size,
        "width": meta.get("width"),
        "height": meta.get("height"),
        "duration": round(float(dur), 3) if dur else None,
        "fps": meta.get("fps"),
        "has_audio": bool(meta.get("has_audio", False)),
        "audio_sample_rate": meta.get("audio_sample_rate"),
    }


@app.get("/api/health")
def health():
    gpu = None
    if pipe is not None:
        try:
            free, total = torch.cuda.mem_get_info(GPU)
            gpu = {"name": torch.cuda.get_device_name(GPU), "used_gb": round((total - free) / 2**30, 1),
                   "total_gb": round(total / 2**30, 1)}
        except Exception:
            pass
    running = None
    with lock:
        jid, step, total = state.get("running"), state.get("step", 0), state.get("total", 0)
        ticks = list(state.get("ticks") or [])
    if jid:
        try:
            j = read(jid)
            now = time.time()
            n = j.get("step", step)
            n_total = j.get("total_steps") or total
            per = (ticks[-1] - ticks[0]) / (len(ticks) - 1) if len(ticks) >= 2 else None
            if per and n_total:
                eta = round(per * max(0, n_total - n))
            else:
                eta = max(0, round((j.get("eta_s") or 0) - (now - (j.get("started") or now)), 0))
            running = {"id": jid, "stage": j.get("stage") or "denoise", "step": n, "steps": n_total,
                       "elapsed": round(now - (j.get("started") or now), 1), "eta": eta,
                       "sec_per_step": round(per, 1) if per else None,
                       "frames": j.get("frames"), "job_steps": j.get("steps")}
        except Exception:
            running = None
    queued = queued_ids()
    return {"model": state["model"], "loaded_in": state.get("loaded_in"), "queued": len(queued),
            "queue": queued, "running": running, "gpu": gpu, "version": VERSION,
            "lane": LANE, "gpu_index": GPU, "partition": PARTITION}


@app.get("/api/lanes")
def lanes():
    """This lane plus its peers, so one page can show both GPUs.

    Containers are on a bridge network, so a peer is reached through host.docker.internal — the
    published port, not the container IP. Peer answers are cached for 3 s: the page polls every
    3 s, and an unreachable peer must not stall every poll.
    """
    out = [health()]
    now = time.time()
    if now - _peer_cache["at"] < 3:
        return out + _peer_cache["data"]
    peers = []
    for url in PEERS:
        try:
            req = urllib.request.Request(url + "/api/health", headers={"x-api-key": KEY.decode()})
            with urllib.request.urlopen(req, timeout=1.5) as r:
                peers.append(json.loads(r.read()))
        except Exception:
            peers.append({"lane": url.rsplit(":", 1)[-1], "model": "unreachable", "gpu": None,
                          "running": None, "queued": None, "peer": url})
    _peer_cache.update(at=now, data=peers)
    return out + peers


_TAG_RE = re.compile(r"<(Picture|Video|Audio)\s+(\d+)>")
_BARE_TAG_RE = re.compile(r"\b(Picture|Video|Audio)\s+(\d+)\b")
_DIALOGUE_TAG_RE = re.compile(r"<(/?)d>")


def _parse_bool(val, default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes", "on"):
            return True
        if s in ("false", "0", "no", "off", ""):
            return False
    return bool(val)


def _resolve_role(effective_kind: str, raw_legacy: str = "", raw_ui_role: str = "") -> tuple[str, str]:
    """Resolve (legacy_role, ui_role) cleanly for an effective reference kind."""
    ui_cand = (raw_ui_role or "").strip().lower()
    leg_cand = (raw_legacy or "").strip().lower()
    allowed_ui = UI_ROLES.get(effective_kind, ())
    if ui_cand in allowed_ui:
        return UI_ROLE_TO_LEGACY[ui_cand], ui_cand
    if ui_cand in ROLES and not leg_cand:
        leg_cand = ui_cand
    if effective_kind == "audio" and leg_cand == "motion":
        leg_cand = "voice"
    role = leg_cand if leg_cand in ROLES else DEFAULT_LEGACY_ROLE.get(effective_kind, "person")
    ui_role = LEGACY_TO_UI_ROLE.get((effective_kind, role), DEFAULT_UI_ROLE.get(effective_kind, "subject_person"))
    return role, ui_role


def resolve_ref_tags(entries: list[dict]) -> list[dict]:
    """Assign canonical <Picture N>/<Video N>/<Audio N> tags mirroring MiniMaxH3Ref2VATextEncoderStep._build_presentation."""
    pi = vi = ai = 0
    out = []
    for e in entries:
        effective_kind = "audio" if e.get("mode") == "video_audio_only" else e["kind"]
        has_audio = True if effective_kind == "audio" else (
            effective_kind == "video" and bool(e.get("has_audio")) and bool(e.get("keep_video_audio", True))
        )
        atag = None
        if has_audio:
            ai += 1
            atag = f"<Audio {ai}>"
        audio_tag = None
        if effective_kind == "image":
            pi += 1
            primary_tag = f"<Picture {pi}>"
        elif effective_kind == "video":
            vi += 1
            primary_tag = f"<Video {vi}>"
            audio_tag = atag
        elif effective_kind == "audio":
            primary_tag = atag
        else:
            primary_tag = None
        out.append({
            **e,
            "effective_kind": effective_kind,
            "has_audio": has_audio,
            "primary_tag": primary_tag,
            "audio_tag": audio_tag,
        })
    return out


def parse_prompt_tags(
    prompt: str,
    workflow: str,
    resolved_refs: list[dict] | None = None,
    has_first: bool = False,
    has_last: bool = False,
) -> list[dict]:
    """Return human-readable tag and dialogue diagnostics for a MiniMax H3 prompt."""
    prompt = prompt or ""
    wf = (workflow or "t2va").strip().lower()
    refs = resolved_refs or []
    warnings: list[dict] = []

    if wf == "ref2va":
        valid_tags = {t for r in refs for t in (r.get("primary_tag"), r.get("audio_tag")) if t}
    elif wf == "fl2va":
        kf_count = int(bool(has_first)) + int(bool(has_last))
        valid_tags = {f"<Picture {n}>" for n in range(1, kf_count + 1)}
    else:
        valid_tags = set()

    mentioned: list[str] = []
    seen_mentioned: set[str] = set()
    for m in _TAG_RE.finditer(prompt):
        tag = f"<{m.group(1)} {int(m.group(2))}>"
        if tag not in seen_mentioned:
            seen_mentioned.add(tag)
            mentioned.append(tag)

    for tag in mentioned:
        if tag not in valid_tags:
            warnings.append({
                "code": "missing_ref",
                "tag": tag,
                "message": f"{tag} is referenced in the prompt, but no matching reference is attached ({wf}).",
            })

    if wf == "ref2va":
        for r in refs:
            ptag = r.get("primary_tag")
            atag = r.get("audio_tag")
            label = r.get("name") or r.get("effective_kind") or r.get("kind") or "reference"
            if ptag and ptag not in seen_mentioned:
                warnings.append({
                    "code": "unreferenced_ref",
                    "tag": ptag,
                    "message": f"{ptag} ({label}) is attached but not mentioned directly or in subject_definitions.",
                })
            if atag and atag not in seen_mentioned:
                warnings.append({
                    "code": "unreferenced_audio",
                    "tag": atag,
                    "message": f"{atag} (soundtrack of {ptag or label}) is enabled but not mentioned in the prompt.",
                })

        seen_bare: set[str] = set()
        for m in _BARE_TAG_RE.finditer(prompt):
            before = prompt[m.start() - 1] if m.start() > 0 else ""
            after = prompt[m.end()] if m.end() < len(prompt) else ""
            if before == "<" and after == ">":
                continue
            bare = f"{m.group(1)} {int(m.group(2))}"
            if bare not in seen_bare:
                seen_bare.add(bare)
                warnings.append({
                    "code": "bracketless_tag",
                    "tag": bare,
                    "message": f"Use angle brackets <{bare}> instead of '{bare}' in ref2va prompts.",
                })

    depth = 0
    nested = False
    orphan = False
    for m in _DIALOGUE_TAG_RE.finditer(prompt):
        if m.group(1):
            if depth == 0:
                orphan = True
            else:
                depth -= 1
        else:
            if depth > 0:
                nested = True
            depth += 1
    if depth > 0 or nested:
        warnings.append({
            "code": "unclosed_dialogue",
            "tag": "<d>",
            "message": "Unclosed <d> dialogue tag; close every spoken line with </d>.",
        })
    if orphan:
        warnings.append({
            "code": "unclosed_dialogue",
            "tag": "</d>",
            "message": "Unexpected </d> closing tag with no matching <d>.",
        })

    return warnings


def _parse_ref_configs(
    refs: list[UploadFile],
    kinds: list[str],
    meta_list: list,
    roles: list[str],
    keep_video_audio: bool,
    video_start: float,
) -> list[dict]:
    """Validate and normalize per-reference metadata before writing uploads to disk."""
    configs = []
    for i, (u, k) in enumerate(zip(refs, kinds)):
        m = meta_list[i] if i < len(meta_list) and meta_list[i] is not None else {}
        if not isinstance(m, dict):
            raise HTTPException(400, f"ref_meta[{i}] must be a JSON object")
        mode = str(m.get("mode") or k).strip().lower()
        valid_modes = ("video", "video_audio_only") if k == "video" else (k,)
        if mode not in valid_modes:
            raise HTTPException(400, f"{u.filename}: invalid mode '{mode}' for {k}")
        effective_kind = "audio" if k == "video" and mode == "video_audio_only" else k

        raw_legacy = str(m.get("role") or (roles[i] if i < len(roles) else "") or "")
        role, ui_role = _resolve_role(effective_kind, raw_legacy, str(m.get("ui_role") or ""))

        raw_sg = m.get("subject_group")
        subject_group = " ".join(str(raw_sg).split())[:60] or None if raw_sg is not None else None

        if "keep_video_audio" in m and m["keep_video_audio"] is not None:
            keep_va = _parse_bool(m["keep_video_audio"], bool(keep_video_audio))
        elif "keep_va" in m and m["keep_va"] is not None:
            keep_va = _parse_bool(m["keep_va"], bool(keep_video_audio))
        else:
            keep_va = bool(keep_video_audio)
        if effective_kind == "audio":
            keep_va = True
        elif effective_kind == "image":
            keep_va = False

        raw_start = m.get("start_sec")
        if effective_kind == "image":
            start_sec = 0.0
            end_sec = None
        else:
            if raw_start is None or raw_start == "":
                start_sec = float(video_start) if effective_kind == "video" else 0.0
            else:
                try:
                    start_sec = float(raw_start)
                except (TypeError, ValueError):
                    raise HTTPException(400, f"{u.filename}: invalid start_sec")
                if not math.isfinite(start_sec) or start_sec < 0:
                    raise HTTPException(400, f"{u.filename}: start_sec must be >= 0")

            raw_end = m.get("end_sec")
            if raw_end is None or raw_end == "":
                end_sec = None
            else:
                try:
                    end_sec = float(raw_end)
                except (TypeError, ValueError):
                    raise HTTPException(400, f"{u.filename}: invalid end_sec")
                if not math.isfinite(end_sec) or end_sec <= start_sec:
                    raise HTTPException(400, f"{u.filename}: end_sec must be > start_sec")

        raw_fps = m.get("fps_override")
        if raw_fps is None or raw_fps == "" or effective_kind != "video":
            fps_override = None
        else:
            try:
                fps_override = float(raw_fps)
            except (TypeError, ValueError):
                raise HTTPException(400, f"{u.filename}: invalid fps_override")
            if not math.isfinite(fps_override) or not (1.0 <= fps_override <= 120.0):
                raise HTTPException(400, f"{u.filename}: fps_override must be in 1.0..120.0")

        raw_sr = m.get("sample_rate_override")
        if raw_sr is None or raw_sr == "" or effective_kind == "image":
            sample_rate_override = None
        else:
            if isinstance(raw_sr, bool):
                raise HTTPException(400, f"{u.filename}: invalid sample_rate_override")
            try:
                sample_rate_override = int(raw_sr)
            except (TypeError, ValueError):
                raise HTTPException(400, f"{u.filename}: invalid sample_rate_override")
            if not (8000 <= sample_rate_override <= 96000):
                raise HTTPException(400, f"{u.filename}: sample_rate_override must be in 8000..96000")

        configs.append({
            "kind": k,
            "mode": mode,
            "effective_kind": effective_kind,
            "role": role,
            "ui_role": ui_role,
            "subject_group": subject_group,
            "keep_video_audio": keep_va,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3) if end_sec is not None else None,
            "fps_override": fps_override,
            "sample_rate_override": sample_rate_override,
        })
    return configs


def _validate_ref_duration(filename: str, effective_kind: str, dur: float | None, start_sec: float, end_sec: float | None):
    """Validate trimmed clip length for video (>= 1.9 s, <= max output duration) and audio (> 0.2 s).

    The reference video upper bound is tied to _MAX_DURATION so that lifting the output cap
    automatically lifts the reference cap too — a 29 s reference alongside a 30 s output is valid.
    """
    if effective_kind == "image":
        return
    if dur is not None:
        if start_sec >= dur:
            raise HTTPException(400, f"{filename}: start_sec {start_sec:.1f} s exceeds duration {dur:.1f} s")
        eff_end = min(dur, end_sec) if end_sec is not None else dur
        avail = eff_end - start_sec
    elif end_sec is not None:
        avail = end_sec - start_sec
    else:
        return

    if effective_kind == "video":
        if avail < REF_VIDEO_MIN_S:
            raise HTTPException(400, f"{filename}: selected segment ({avail:.1f} s) leaves < {REF_VIDEO_MIN_S:.1f} s of video")
        if avail > REF_VIDEO_MAX_S:
            raise HTTPException(400, f"{filename}: reference video is {avail:.1f} s; maximum is {_MAX_DURATION:.0f} s — trim with start_sec/end_sec")
    elif effective_kind == "audio" and avail <= REF_AUDIO_MIN_S:
        raise HTTPException(400, f"{filename}: selected audio segment ({avail:.2f} s) must be > {REF_AUDIO_MIN_S:.1f} s")


@app.post("/api/jobs")
async def create(prompt: str = Form(...), frames: int = Form(124), seed: int = Form(-1),
                 steps: int = Form(8), audio_mode: str = Form("raw"), pin: str = Form(""),
                 aspect: str = Form("auto"), keep_video_audio: bool = Form(True),
                 video_start: float = Form(0.0),
                 roles: list[str] = Form([]),
                 ref_meta: str = Form(""),
                 ref_short_edge: int = Form(2048),
                 image: UploadFile | None = File(None),
                 last_image: UploadFile | None = File(None), references: list[UploadFile] = File([])):
    prompt = prompt.strip()
    if audio_mode not in ("raw", "peak"):
        raise HTTPException(400, "audio_mode must be raw or peak")
    aspect = (aspect or "auto").strip().lower()
    if aspect not in ASPECTS:
        raise HTTPException(400, f"aspect must be one of {list(ASPECTS)}")
    if ref_short_edge not in REF_EDGES:
        raise HTTPException(400, f"ref_short_edge must be one of {list(REF_EDGES)}")
    if video_start < 0:
        raise HTTPException(400, "video_start must be >= 0")
    pin = (pin or "").strip().lower()
    if pin not in ("", "a", "b"):
        raise HTTPException(400, "pin must be a, b or blank")
    keys = {k: u for k, u in (("image", image), ("last_image", last_image)) if u and u.filename}
    refs = [u for u in references if u.filename]
    if not prompt:
        raise HTTPException(400, "write a prompt first")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(400, f"prompt is limited to {MAX_PROMPT_CHARS} characters")
    if frames not in FRAMES:
        raise HTTPException(400, f"frames must be {FRAMES[0]}-{FRAMES[-1]} on the 17n+5 grid "
                                 f"(5.0-{_MAX_DURATION:.0f}.0 s at {FPS} fps); got {frames}")
    if not 4 <= steps <= MAX_STEPS:
        raise HTTPException(400, f"steps must be 4-{MAX_STEPS}")
    if keys and refs:
        raise HTTPException(400, "use first/last frame or references, not both")
    if any(kind_of(u) != "image" for u in keys.values()):
        raise HTTPException(400, "first and last frame must be images")
    kinds = [kind_of(u) for u in refs]
    if any(k is None for k in kinds):
        raise HTTPException(400, "references must be images, video or audio")
    meta_list = []
    if ref_meta and ref_meta.strip():
        try:
            meta_list = json.loads(ref_meta)
        except Exception:
            raise HTTPException(400, "ref_meta must be valid JSON")
        if not isinstance(meta_list, list):
            raise HTTPException(400, "ref_meta must be a JSON list")

    ref_cfgs = _parse_ref_configs(refs, kinds, meta_list, roles, keep_video_audio, video_start)
    effective_kinds = [c["effective_kind"] for c in ref_cfgs]
    for k, n in LIMITS.items():
        if effective_kinds.count(k) > n:
            raise HTTPException(400, f"at most {n} {k} references")
    if len(refs) > 12:
        raise HTTPException(400, "at most 12 references")
    if effective_kinds and set(effective_kinds) == {"audio"}:
        raise HTTPException(400, "audio references need an image or video reference too")
    if PARTITION == "base" and refs:
        raise HTTPException(503, "this worker is running the base partition only; references need ref2va")
    if PARTITION == "ref" and keys:
        raise HTTPException(503, "this worker is running the ref partition only; keyframes need fl2va")
    free_gb = shutil.disk_usage(JOBS).free / 2**30
    if free_gb < MIN_FREE_GB:
        raise HTTPException(507, f"only {free_gb:.0f} GB free on /out; clear old jobs first")

    jid = uuid.uuid4().hex
    d = JOBS / jid
    d.mkdir()
    up = Upload()
    inputs = {}
    raw_entries = []
    try:
        for k, u in keys.items():
            inputs[k], _ = await up.save(u, d, k)
        for i, (u, cfg) in enumerate(zip(refs, ref_cfgs)):
            k = cfg["kind"]
            eff_k = cfg["effective_kind"]
            name, size = await up.save(u, d, f"ref{i}")
            meta = {}
            dur = None
            if k == "video":
                meta = probe_video(d / name)
                dur = meta.get("duration") or media_duration(d / name)
            elif k == "audio":
                meta = probe_audio(d / name)
                dur = meta.get("duration") or media_duration(d / name)
            elif k == "image":
                meta = probe_image(d / name)

            if cfg["mode"] == "video_audio_only" and not meta.get("has_audio"):
                raise HTTPException(400, f"{u.filename}: video has no audio track to extract")
            _validate_ref_duration(u.filename or name, eff_k, dur, cfg["start_sec"], cfg["end_sec"])

            entry = {
                "kind": k,
                "name": (u.filename or "")[:200],
                "file": name,
                "bytes": size,
                "duration": round(dur, 2) if dur else None,
                "role": cfg["role"],
                "ui_role": cfg["ui_role"],
                "mode": cfg["mode"],
                "subject_group": cfg["subject_group"],
                "keep_video_audio": cfg["keep_video_audio"],
                "start_sec": cfg["start_sec"],
                "end_sec": cfg["end_sec"],
                "fps_override": cfg["fps_override"],
                "sample_rate_override": cfg["sample_rate_override"],
                "has_audio": bool(meta.get("has_audio")) if k in ("video", "audio") else False,
            }
            if meta.get("width") and meta.get("height"):
                entry["width"] = meta["width"]
                entry["height"] = meta["height"]
            if k == "video":
                entry["source_has_audio"] = bool(meta.get("has_audio"))
                if meta.get("fps"):
                    entry["fps"] = meta["fps"]
            if meta.get("audio_sample_rate"):
                entry["audio_sample_rate"] = meta["audio_sample_rate"]
            raw_entries.append(entry)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise

    inputs["references"] = resolve_ref_tags(raw_entries)
    seed = int(seed)
    if seed < 0:
        seed = secrets.randbelow(2**31)
    workflow = "ref2va" if refs else "fl2va" if keys else "t2va"
    tag_warnings = parse_prompt_tags(
        prompt, workflow, inputs["references"], bool(keys.get("image")), bool(keys.get("last_image"))
    )
    target_h, target_w = _resolve_canvas(workflow, aspect, inputs["references"])
    est = estimate(frames, steps)
    job = {"id": jid, "num": next_number(), "prompt": prompt, "frames": frames, "seed": seed, "steps": steps, "inputs": inputs,
           "workflow": workflow, "aspect": aspect, "target_height": target_h, "target_width": target_w,
           "ref_short_edge": int(ref_short_edge), "tag_warnings": tag_warnings,
           "keep_video_audio": bool(keep_video_audio), "video_start": round(float(video_start), 2),
           "status": "queued", "stage": "queued", "step": 0, "total_steps": steps,
           "created": time.time(), "started": None, "finished": None, "error": None, "output": None,
           "poster": None, "eta_s": est, "timings": {}, "vram_peak_gb": None, "media": {},
           "audio_mode": audio_mode, "audio": None, "title": None, "name": None, "folder": None,
           "pin": pin or None}
    write(job)
    return job


@app.get("/api/jobs")
def jobs():
    out = []
    for p in JOBS.iterdir():
        if (p / "job.json").exists():
            try:
                out.append(json.loads((p / "job.json").read_text()))
            except Exception:
                pass
    return sorted(out, key=lambda j: -j.get("created", 0))


@app.get("/api/jobs/{jid}")
def job(jid: str):
    return read(jid)


@app.post("/api/jobs/{jid}/title")
async def set_title(jid: str, title: str = Form(...)):
    """Store a short title for a job. Written by the titler service, never by the browser."""
    j = read(jid)
    t = " ".join(title.split())[:80]
    if not t:
        raise HTTPException(400, "empty title")
    j["title"] = t
    write(j)
    return {"id": jid, "title": t}


@app.post("/api/jobs/{jid}/pause")
def pause(jid: str):
    """Hold the render at the next step boundary. The GPU goes idle; VRAM stays held."""
    j = read(jid)
    if j["status"] != "running":
        raise HTTPException(409, f"job is {j['status']}")
    j["pause"] = True
    write(j)
    return {"id": jid, "pause": True}


@app.post("/api/jobs/{jid}/resume")
def resume(jid: str):
    j = read(jid)
    j["pause"] = False
    write(j)
    return {"id": jid, "pause": False}


@app.post("/api/jobs/{jid}/prompt")
async def set_prompt(jid: str, prompt: str = Form(...)):
    """Edit the prompt of a job that is not running, so a mis-bound reference is fixed in place
    rather than re-uploaded as a duplicate clip."""
    j = read(jid)
    if j["status"] == "running":
        raise HTTPException(409, "cancel the running job first")
    p = prompt.strip()
    if not p:
        raise HTTPException(400, "empty prompt")
    if len(p) > 20000:
        raise HTTPException(400, "prompt is limited to 20000 characters")
    j["prompt"] = p
    j["title"] = None            # re-title it from the new prompt
    inp = j.get("inputs") or {}
    j["tag_warnings"] = parse_prompt_tags(
        p, j.get("workflow") or "t2va", inp.get("references") or [],
        bool(inp.get("image")), bool(inp.get("last_image")),
    )
    write(j)
    return {"id": jid, "prompt": p}


@app.post("/api/jobs/{jid}/rename")
async def rename(jid: str, name: str = Form("")):
    j = read(jid)
    j["name"] = " ".join(name.split())[:80] or None
    write(j)
    return {"id": jid, "name": j["name"]}


@app.post("/api/jobs/{jid}/folder")
async def set_folder(jid: str, folder: str = Form("")):
    j = read(jid)
    j["folder"] = " ".join(folder.split())[:40] or None
    write(j)
    return {"id": jid, "folder": j["folder"]}


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    j = read(jid)
    if j["status"] == "queued":
        j.update(status="cancelled", stage="cancelled", finished=time.time())
        write(j)
        return j
    if j["status"] == "running":
        j["cancel"] = True
        write(j)
        return j
    raise HTTPException(409, f"job is {j['status']}")


@app.post("/api/jobs/{jid}/requeue")
def requeue(jid: str):
    """Run this clip again **in place**.

    Same job id, so the bin does not fill with copies of one brief. The previous render is kept as
    take-<timestamp>.mp4 in the same directory and listed on the slate, so a re-take never destroys
    a good one.
    """
    j = read(jid)
    if j["status"] == "running":
        raise HTTPException(409, "cancel the running job first")
    if j["status"] == "queued":
        return j
    d = JOBS / jid
    if j.get("output") and (d / j["output"]).exists():
        name = "take-%s.mp4" % time.strftime("%Y%m%d-%H%M%S")
        try:
            (d / j["output"]).rename(d / name)
            j["takes"] = (list(j.get("takes") or []) + [name])[-12:]
        except OSError:
            pass
    try:
        (d / "claim").unlink()
    except OSError:
        pass
    old_frames = j["frames"]
    frames = legal_frames(old_frames)
    j.update(status="queued", stage="queued", step=0, total_steps=j["steps"], started=None,
             finished=None, error=None, output=None, poster=None, timings={}, vram_peak_gb=None,
             media={}, audio=None, cancel=False, pause=False, lane=None, created=time.time(),
             frames=frames, eta_s=estimate(frames, j["steps"]))
    if frames != old_frames:
        j["frames_note"] = f"raised from {old_frames} frames (below the 5 s floor)"
    write(j)
    return j


@app.post("/api/jobs/{jid}/retry")
def retry(jid: str):
    """Alias for requeue: an old client calling retry must not spawn a duplicate clip."""
    return requeue(jid)


@app.delete("/api/jobs/{jid}")
def delete(jid: str):
    j = read(jid)
    if j["status"] == "running":
        raise HTTPException(409, "cancel the running job first")
    d = JOBS / jid
    if d.exists():
        shutil.rmtree(d)
    if d.exists():
        raise HTTPException(500, "failed to remove job directory")
    return {"deleted": jid}


@app.get("/api/logs")
def get_logs(lane: str = LANE, lines: int = 200):
    """Return the last N lines from the persistent worker log for the requested lane."""
    safe_lane = re.sub(r"[^a-zA-Z0-9_-]", "", lane or LANE) or LANE
    p = LOG_DIR / f"h3-{safe_lane}.log"
    if not p.exists():
        return {"lane": safe_lane, "path": str(p), "lines": []}
    n = max(10, min(1000, int(lines)))
    text = p.read_text(encoding="utf-8", errors="replace").splitlines()
    return {"lane": safe_lane, "path": str(p), "lines": text[-n:]}


def _upscale_mp4_2k(src: Path, dst: Path, target_short: int = 1440):
    """Upscale an MP4 video stream to 2K short edge with LANCZOS while preserving audio."""
    import av
    from PIL import Image

    with av.open(str(src)) as inp:
        v_in = inp.streams.video[0]
        a_in = inp.streams.audio[0] if inp.streams.audio else None
        w0, h0 = v_in.width, v_in.height
        scale = target_short / min(w0, h0)
        if scale <= 1.05:
            shutil.copy2(src, dst)
            return w0, h0
        w1 = max(16, (round(w0 * scale) // 16) * 16)
        h1 = max(16, (round(h0 * scale) // 16) * 16)
        fps = v_in.average_rate or FPS

        with av.open(str(dst), mode="w") as out:
            v_out = out.add_stream("libx264", rate=fps)
            v_out.width = w1
            v_out.height = h1
            v_out.pix_fmt = "yuv420p"
            v_out.options = {"crf": "16", "preset": "medium"}

            a_out = None
            if a_in is not None:
                a_out = out.add_stream("aac", rate=a_in.sample_rate or 44100)
                a_out.layout = a_in.layout.name if a_in.layout else "stereo"

            for pkt in inp.demux([s for s in (v_in, a_in) if s is not None]):
                if pkt.stream.type == "video":
                    for frame in pkt.decode():
                        img = frame.to_image().resize((w1, h1), Image.Resampling.LANCZOS)
                        nf = av.VideoFrame.from_image(img)
                        nf.pts = frame.pts
                        nf.time_base = frame.time_base
                        for op in v_out.encode(nf):
                            out.mux(op)
                elif pkt.stream.type == "audio" and a_out is not None:
                    for frame in pkt.decode():
                        frame.pts = None
                        for op in a_out.encode(frame):
                            out.mux(op)

            for op in v_out.encode():
                out.mux(op)
            if a_out is not None:
                for op in a_out.encode():
                    out.mux(op)
    return w1, h1


@app.post("/api/jobs/{jid}/upscale")
def upscale_job(jid: str):
    """Upscale a completed job's output MP4 to 2K (1440p short edge)."""
    j = read(jid)
    if j.get("status") != "done" or not j.get("output"):
        raise HTTPException(400, "job is not completed yet")
    d = JOBS / jid
    src = d / j["output"]
    if not src.exists():
        raise HTTPException(404, "output video missing")
    out_name = "out-2k.mp4"
    dst = d / out_name
    tmp = d / f".tmp-{out_name}"
    try:
        w1, h1 = _upscale_mp4_2k(src, tmp)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    j["upscaled"] = out_name
    j["upscaled_size"] = f"{w1}x{h1}"
    write(j)
    return {"id": jid, "upscaled": out_name, "width": w1, "height": h1}


@app.get("/files/{jid}/{name}")
def files(jid: str, name: str, dl: int = 0):
    p = JOBS / jid / name
    if not re.fullmatch(r"[0-9a-f]{32}", jid) or not re.fullmatch(r"[\w.-]+", name) \
            or name.startswith("job.") or not p.is_file():
        raise HTTPException(404)
    if dl:
        try:
            j = read(jid)
            raw_title = (j.get("name") or j.get("title") or "").strip()
            safe = re.sub(r'[\\/:*?"<>|]+', " ", raw_title)
            safe = re.sub(r"\s+", " ", safe).strip()[:80]
            if safe:
                ext = p.suffix or ".mp4"
                suffix = " (2K)" if name == j.get("upscaled") else ""
                return FileResponse(p, filename=f"{safe}{suffix}{ext}")
        except Exception:
            pass
    return FileResponse(p)


# ── Series Automation, VLM Continuity & Selected-Video Stitching ───────────────────────────────

def _read_series(sid: str) -> dict:
    p = SERIES / f"{sid}.json"
    if not p.is_file():
        raise HTTPException(404, "series not found")
    return json.loads(p.read_text())


def _write_series(series: dict):
    series["updated"] = time.time()
    p = SERIES / f"{series['id']}.json"
    tmp = SERIES / f".{series['id']}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(series))
    tmp.replace(p)


@contextmanager
def _series_lock(sid: str, timeout_s: float = 10.0):
    """Atomic per-series lock across dual GPU lanes using O_CREAT|O_EXCL with 90s stale cleanup."""
    lk = SERIES / f"{sid}.lock"
    deadline = time.time() + timeout_s
    while True:
        try:
            fd = os.open(lk, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(LANE)
            break
        except FileExistsError:
            try:
                if time.time() - lk.stat().st_mtime > 90:
                    lk.unlink()
                    continue
            except Exception:
                pass
            if time.time() >= deadline:
                raise HTTPException(409, "series is busy updating on another lane; retry")
            time.sleep(0.15)
    try:
        yield
    finally:
        try:
            lk.unlink()
        except Exception:
            pass


def _extract_last_frame(mp4_path: Path, dst_png: Path) -> bool:
    """Decode the exact final video frame of an MP4 into a lossless PNG using PyAV."""
    import av
    last_img = None
    with av.open(str(mp4_path)) as c:
        vs = next((s for s in c.streams if s.type == "video"), None)
        if vs is None:
            return False
        if vs.duration and vs.time_base:
            dur_s = float(vs.duration * vs.time_base)
            if dur_s > 2.0:
                try:
                    c.seek(int(max(0.0, dur_s - 1.5) / float(vs.time_base)), stream=vs, backward=True)
                    for frame in c.decode(vs):
                        last_img = frame.to_image()
                except Exception:
                    last_img = None
        if last_img is None:
            c.seek(0)
            for frame in c.decode(vs):
                last_img = frame.to_image()
    if last_img is None:
        return False
    last_img.save(str(dst_png), format="PNG")
    return True


def _stitch_mp4s(job_ids: list[str], dst_mp4: Path) -> dict:
    """Concatenate ordered finished job output.mp4 files into one continuous MP4 via PyAV."""
    import av
    import numpy as np
    from PIL import Image

    sources: list[Path] = []
    for jid in job_ids:
        j = read(jid)
        if j.get("status") != "done":
            raise HTTPException(400, f"job #{j.get('num') or jid[:8]} is not finished yet")
        mp4 = JOBS / jid / (j.get("output") or "output.mp4")
        if not mp4.is_file():
            raise HTTPException(404, f"output video missing for job #{j.get('num') or jid[:8]}")
        sources.append(mp4)
    if len(sources) < 2:
        raise HTTPException(400, "select at least 2 finished clips to stitch")

    first_meta = probe_video(sources[0])
    target_w = int(first_meta.get("width") or 1344)
    target_h = int(first_meta.get("height") or 768)
    target_w -= target_w % 2
    target_h -= target_h % 2
    target_fps = FPS
    target_sr = 32000

    out_tmp = dst_mp4.with_suffix(".tmp.mp4")
    v_pts = 0
    try:
        with av.open(str(out_tmp), mode="w") as out_c:
            ov = out_c.add_stream("libx264", rate=target_fps)
            ov.width = target_w
            ov.height = target_h
            ov.pix_fmt = "yuv420p"
            ov.options = {"crf": "16", "preset": "medium"}

            oa = out_c.add_stream("aac", rate=target_sr)
            oa.layout = "stereo"

            for src in sources:
                clip_v_frames = 0
                clip_a_samples = 0
                with av.open(str(src)) as in_c:
                    ivs = next((s for s in in_c.streams if s.type == "video"), None)
                    ias = next((s for s in in_c.streams if s.type == "audio"), None)
                    if ivs is None:
                        continue
                    for frame in in_c.decode(ivs):
                        if frame.width != target_w or frame.height != target_h:
                            img = frame.to_image()
                            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                            vf = av.VideoFrame.from_image(img).reformat(format="yuv420p")
                        else:
                            vf = frame.reformat(format="yuv420p")
                        vf.pts = v_pts
                        vf.time_base = Fraction(1, target_fps)
                        v_pts += 1
                        clip_v_frames += 1
                        for pkt in ov.encode(vf):
                            out_c.mux(pkt)

                    expected_samples = int(round((clip_v_frames / target_fps) * target_sr))
                    if ias is not None:
                        in_c.seek(0)
                        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=target_sr)
                        for aframe in in_c.decode(ias):
                            for rf in resampler.resample(aframe):
                                if rf.samples <= 0:
                                    continue
                                rf.pts = None
                                clip_a_samples += rf.samples
                                for pkt in oa.encode(rf):
                                    out_c.mux(pkt)
                        for rf in resampler.resample(None):
                            if rf.samples <= 0:
                                continue
                            rf.pts = None
                            clip_a_samples += rf.samples
                            for pkt in oa.encode(rf):
                                out_c.mux(pkt)

                    pad = expected_samples - clip_a_samples
                    if pad > 0:
                        chunk_sz = 1024
                        while pad > 0:
                            n = min(pad, chunk_sz)
                            sil = av.AudioFrame.from_ndarray(
                                np.zeros((2, n), dtype=np.float32), format="fltp", layout="stereo"
                            )
                            sil.sample_rate = target_sr
                            sil.pts = None
                            pad -= n
                            for pkt in oa.encode(sil):
                                out_c.mux(pkt)

            for pkt in ov.encode(None):
                out_c.mux(pkt)
            for pkt in oa.encode(None):
                out_c.mux(pkt)
        out_tmp.replace(dst_mp4)
    except Exception:
        out_tmp.unlink(missing_ok=True)
        raise
    return probe_video(dst_mp4)


def _llm_chat(messages: list[dict], *, use_vlm: bool = False, temperature: float = 0.35, max_tokens: int = 2000) -> str:
    """Call H3_VLM_URL (when use_vlm=True and configured) or H3_LLM_URL, returning stripped text."""
    url = VLM_URL if (use_vlm and VLM_URL) else LLM_URL or VLM_URL
    key = VLM_KEY if (use_vlm and VLM_URL) else (LLM_KEY if LLM_URL else VLM_KEY)
    model = VLM_MODEL if (use_vlm and VLM_URL) else (LLM_MODEL if LLM_URL else VLM_MODEL)
    if not url:
        raise HTTPException(501, "No LLM or VLM backend configured (set H3_LLM_URL / H3_VLM_URL)")
    payload_obj = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if not (use_vlm and VLM_URL):
        payload_obj["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(payload_obj).encode(),
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read())
    text = (data["choices"][0]["message"].get("content") or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()
    text = re.sub(r"^```[a-zA-Z]*\n([\s\S]*?)\n```$", r"\1", text).strip()
    return text


def _adapt_next_episode_prompt(series: dict, prev_ep: dict, next_ep: dict, last_frame_path: Path | None) -> str:
    """Use VLM (with last_frame.png) or LLM fallback to write a seamless fl2va continuation prompt."""
    if not (VLM_URL or LLM_URL):
        return next_ep.get("prompt") or next_ep.get("beat") or ""
    frames = int(series["settings"].get("frames") or 362)
    dur_s = frames / FPS
    aspect = series["settings"].get("aspect") or "9:16"
    bible = series.get("bible") or {}
    director_note = (series.get("director_note") or "").strip()
    sys_msg = (
        "You are an expert MiniMax-H3 video director writing a seamless fl2va continuation episode.\n"
        "You MUST start the output with exact line:\n"
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n"
        "Then output the exact 3 sections:\n"
        "integrated_multimodal_description: [Shot 1] ... [Shot 2] At 00:XX.000, ...\n"
        "overall_soundscape: ...\n"
        "non_diegetic_music: ...\n\n"
        f"Target duration is {dur_s:.2f}s ({frames} frames at {FPS}fps), aspect {aspect}. "
        "Keep every [Shot N] timestamp strictly inside 00:00.000 to the target duration. "
        "Look closely at <Picture 1> (the final frame of the previous clip) and ensure [Shot 1] continues "
        "naturally from that exact lighting, framing, character/creature pose, and environment without jarring cuts."
    )
    text_ctx = (
        f"SERIES TITLE: {series.get('title')}\n"
        f"SERIES BIBLE:\n"
        f"- Style & Camera: {bible.get('style', '')}\n"
        f"- Characters / Entities: {bible.get('characters', '')}\n"
        f"- Environment & Lighting: {bible.get('environment', '')}\n"
        f"- Audio & Music Continuity: {bible.get('audio_continuity', '')}\n\n"
        f"PREVIOUS EPISODE PROMPT (Ep {prev_ep['index'] + 1}):\n{prev_ep.get('prompt', '')[:2500]}\n\n"
        f"NEXT EPISODE BEAT (Ep {next_ep['index'] + 1}):\n{next_ep.get('beat', '')}\n\n"
        f"NEXT EPISODE DRAFT PROMPT:\n{next_ep.get('prompt', '')[:2500]}\n"
        + (f"\nDIRECTOR NOTE:\n{director_note}\n" if director_note else "")
        + "\nRewrite the NEXT EPISODE prompt so it seamlessly continues from the attached final frame (<Picture 1>)."
    )
    use_vlm = bool(VLM_URL and last_frame_path and last_frame_path.is_file())
    if use_vlm:
        b64 = base64.b64encode(last_frame_path.read_bytes()).decode("ascii")
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": text_ctx},
        ]
    else:
        user_content = text_ctx
    ref_imgs = [Path(p) for p in (next_ep.get("ref_images") or []) if Path(p).is_file()]
    if ref_imgs:
        pic_list = ", ".join(f"<Picture {i + 2}>" for i in range(len(ref_imgs)))
        text_ctx += (
            f"\n\nNOTE: This episode uses ref2va with <Picture 1> as the previous scene continuity frame "
            f"and {pic_list} as exact character/ghost appearance references. Reference both <Picture 1> "
            f"(for setting/lighting continuity) and {pic_list} (for the ghost's exact appearance) in [Shot 1]/[Shot 2]."
        )
    try:
        raw = _llm_chat([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_content}],
                        use_vlm=use_vlm, temperature=0.35, max_tokens=1800)
    except Exception as e:
        if use_vlm and LLM_URL:
            print(f"series: VLM failed ({type(e).__name__}: {e}), falling back to LLM_URL", flush=True)
            raw = _llm_chat([{"role": "system", "content": sys_msg}, {"role": "user", "content": text_ctx}],
                            use_vlm=False, temperature=0.35, max_tokens=1800)
        else:
            raise
    if "<Picture 1>" not in raw:
        raw = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n" + raw
    return _repair_polished_prompt(raw, "ref2va" if ref_imgs else "fl2va")


def _enqueue_series_episode(series: dict, ep_idx: int, prev_job_id: str | None = None) -> dict:
    """Create and queue a job on disk for episode ep_idx using the series' locked settings."""
    ep = series["episodes"][ep_idx]
    st = series["settings"]
    frames = int(st.get("frames") or 362)
    if frames not in FRAMES:
        frames = legal_frames(frames)
    steps = max(4, min(MAX_STEPS, int(st.get("steps") or 36)))
    aspect = st.get("aspect") or "9:16"
    audio_mode = st.get("audio_mode") or "peak"
    ref_short_edge = int(st.get("ref_short_edge") or 2048)
    pin = (st.get("pin") or "").strip().lower() or None
    seed_mode = st.get("seed_mode") or "increment"
    base_seed = int(st.get("base_seed") if st.get("base_seed") is not None else -1)
    if base_seed < 0:
        seed = secrets.randbelow(2**31)
    elif seed_mode == "fixed":
        seed = base_seed
    elif seed_mode == "random":
        seed = secrets.randbelow(2**31)
    else:
        seed = base_seed + ep_idx

    prompt = (ep.get("prompt") or ep.get("beat") or "").strip()
    if not prompt:
        raise HTTPException(400, f"episode {ep_idx + 1} has no prompt")

    jid = uuid.uuid4().hex
    d = JOBS / jid
    d.mkdir()
    inputs = {"references": []}
    workflow = "t2va"
    ref_imgs = [Path(p) for p in (ep.get("ref_images") or []) if Path(p).is_file()]
    if ref_imgs:
        raw_entries = []
        idx_r = 0
        if prev_job_id:
            src_lf = JOBS / prev_job_id / "last_frame.png"
            if not src_lf.is_file():
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(500, f"missing last_frame.png from previous job {prev_job_id[:8]}")
            shutil.copyfile(src_lf, d / f"ref{idx_r}.png")
            meta = probe_image(d / f"ref{idx_r}.png")
            raw_entries.append({
                "kind": "image", "name": "last_frame.png", "file": f"ref{idx_r}.png",
                "bytes": (d / f"ref{idx_r}.png").stat().st_size, "role": "style", "ui_role": "keyframe_anchor",
                "mode": "image", "subject_group": 0, "keep_video_audio": True, "start_sec": 0.0,
                "end_sec": None, "fps_override": None, "sample_rate_override": None, "has_audio": False,
                **({k: meta[k] for k in ("width", "height") if meta.get(k)}),
            })
            idx_r += 1
        for rp in ref_imgs:
            ext = rp.suffix.lower() or ".jpg"
            fname = f"ref{idx_r}{ext}"
            shutil.copyfile(rp, d / fname)
            meta = probe_image(d / fname)
            raw_entries.append({
                "kind": "image", "name": rp.name, "file": fname,
                "bytes": (d / fname).stat().st_size, "role": "person", "ui_role": "subject_person",
                "mode": "image", "subject_group": 1, "keep_video_audio": True, "start_sec": 0.0,
                "end_sec": None, "fps_override": None, "sample_rate_override": None, "has_audio": False,
                **({k: meta[k] for k in ("width", "height") if meta.get(k)}),
            })
            idx_r += 1
        inputs["references"] = resolve_ref_tags(raw_entries)
        workflow = "ref2va"
    elif prev_job_id:
        src_lf = JOBS / prev_job_id / "last_frame.png"
        if not src_lf.is_file():
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(500, f"missing last_frame.png from previous job {prev_job_id[:8]}")
        shutil.copyfile(src_lf, d / "image.png")
        inputs["image"] = "image.png"
        workflow = "fl2va"

    target_h, target_w = _resolve_canvas(workflow, aspect, inputs["references"])
    tag_warnings = parse_prompt_tags(prompt, workflow, inputs["references"], bool(inputs.get("image")), False)
    job = {
        "id": jid, "num": next_number(), "prompt": prompt, "frames": frames, "seed": seed, "steps": steps,
        "inputs": inputs, "workflow": workflow, "aspect": aspect, "target_height": target_h, "target_width": target_w,
        "ref_short_edge": ref_short_edge, "tag_warnings": tag_warnings, "keep_video_audio": True, "video_start": 0.0,
        "status": "queued", "stage": "queued", "step": 0, "total_steps": steps,
        "created": time.time(), "started": None, "finished": None, "error": None, "output": None,
        "poster": None, "eta_s": estimate(frames, steps), "timings": {}, "vram_peak_gb": None, "media": {},
        "audio_mode": audio_mode, "audio": None, "title": f"{series['title']} Ep {ep_idx + 1}",
        "name": f"{series['title']} Ep {ep_idx + 1}", "folder": f"Series: {series['title']}",
        "pin": pin, "series_id": series["id"], "episode_index": ep_idx,
    }
    write(job)
    ep["job_id"] = jid
    ep["status"] = "queued"
    series["current_index"] = ep_idx
    return job


def _advance_series_for_job(job: dict):
    """Post-completion hook called by workers after a series job finishes, errors, or cancels."""
    sid = job.get("series_id")
    ep_idx = job.get("episode_index")
    if not sid or ep_idx is None:
        return
    try:
        with _series_lock(sid, timeout_s=30.0):
            series = _read_series(sid)
            if ep_idx < 0 or ep_idx >= len(series["episodes"]):
                return
            ep = series["episodes"][ep_idx]
            if job.get("status") != "done":
                ep["status"] = job.get("status") or "error"
                series["status"] = job.get("status") or "error"
                series["error"] = job.get("error")
                _write_series(series)
                return

            mp4 = JOBS / job["id"] / (job.get("output") or "output.mp4")
            lf_path = JOBS / job["id"] / "last_frame.png"
            if mp4.is_file() and _extract_last_frame(mp4, lf_path):
                ep["last_frame"] = "last_frame.png"
            ep["status"] = "done"

            nxt_idx = ep_idx + 1
            if nxt_idx >= len(series["episodes"]):
                series["status"] = "done"
                _write_series(series)
                return

            nxt = series["episodes"][nxt_idx]
            if series.get("auto_adapt_prompt", True) and not nxt.get("prompt_locked"):
                try:
                    nxt["prompt"] = _adapt_next_episode_prompt(series, ep, nxt, lf_path)
                    nxt["adapted_with"] = "vlm" if (VLM_URL and lf_path.is_file()) else "llm"
                except Exception as e:
                    nxt["adapt_error"] = f"{type(e).__name__}: {e}"
                    print(f"series {sid[:8]} adapt warning: {e}", flush=True)

            if series.get("mode") == "review" or series.get("status") == "paused":
                nxt["status"] = "waiting_review"
                series["status"] = "waiting_review"
                series["current_index"] = nxt_idx
            else:
                _enqueue_series_episode(series, nxt_idx, prev_job_id=job["id"])
                series["status"] = "running"
            _write_series(series)
    except Exception as e:
        print(f"series {sid[:8]} advance failed: {type(e).__name__}: {e}", flush=True)


@app.get("/api/series")
def list_series():
    out = []
    for p in SERIES.glob("*.json"):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    out.sort(key=lambda s: s.get("updated") or s.get("created") or 0, reverse=True)
    return out


@app.get("/api/series/{sid}")
def get_series(sid: str):
    s = _read_series(sid)
    for ep in s.get("episodes", []):
        jid = ep.get("job_id")
        if jid and (JOBS / jid / "job.json").is_file():
            try:
                j = read(jid)
                ep["job_status"] = j.get("status")
                ep["job_num"] = j.get("num")
                ep["poster"] = j.get("poster")
                ep["output"] = j.get("output")
                ep["timings"] = j.get("timings")
            except Exception:
                pass
    return s


@app.post("/api/series")
async def create_series(request: Request):
    b = await request.json()
    sid = uuid.uuid4().hex[:12]
    frames = int(b.get("frames") or 362)
    if frames not in FRAMES:
        frames = legal_frames(frames)
    steps = max(4, min(MAX_STEPS, int(b.get("steps") or 36)))
    aspect = (b.get("aspect") or "9:16").strip().lower()
    if aspect not in ASPECTS:
        aspect = "9:16"
    raw_eps = b.get("episodes") or []
    if not raw_eps:
        raise HTTPException(400, "series needs at least 1 episode")
    episodes = []
    for idx, e in enumerate(raw_eps):
        episodes.append({
            "index": idx,
            "beat": (e.get("beat") or "").strip(),
            "prompt": (e.get("prompt") or "").strip(),
            "prompt_locked": bool(e.get("prompt_locked", False)),
            "job_id": None,
            "status": "pending",
            "last_frame": None,
        })
    series = {
        "id": sid,
        "title": (b.get("title") or "Untitled Series").strip()[:100],
        "concept": (b.get("concept") or "").strip(),
        "status": "draft",
        "mode": b.get("mode") if b.get("mode") in ("autopilot", "review") else "autopilot",
        "auto_adapt_prompt": bool(b.get("auto_adapt_prompt", True)),
        "director_note": (b.get("director_note") or "").strip(),
        "bible": b.get("bible") or {"style": "", "characters": "", "environment": "", "audio_continuity": ""},
        "settings": {
            "frames": frames,
            "steps": steps,
            "aspect": aspect,
            "audio_mode": b.get("audio_mode") if b.get("audio_mode") in ("raw", "peak") else "peak",
            "ref_short_edge": int(b.get("ref_short_edge") or 2048),
            "seed_mode": b.get("seed_mode") if b.get("seed_mode") in ("increment", "fixed", "random") else "increment",
            "base_seed": int(b.get("base_seed") if b.get("base_seed") is not None else -1),
            "pin": (b.get("pin") or "").strip().lower() or None,
        },
        "current_index": 0,
        "episodes": episodes,
        "stitched_id": None,
        "created": time.time(),
        "updated": time.time(),
        "error": None,
    }
    _write_series(series)
    return series


@app.patch("/api/series/{sid}")
async def update_series(sid: str, request: Request):
    b = await request.json()
    with _series_lock(sid):
        s = _read_series(sid)
        for k in ("title", "concept", "mode", "auto_adapt_prompt", "director_note", "bible"):
            if k in b:
                s[k] = b[k]
        if "settings" in b and isinstance(b["settings"], dict):
            s["settings"].update(b["settings"])
        if "episodes" in b and isinstance(b["episodes"], list):
            for idx, patch_ep in enumerate(b["episodes"]):
                if idx < len(s["episodes"]):
                    cur = s["episodes"][idx]
                    if cur.get("status") in ("pending", "waiting_review", "draft"):
                        for ek in ("beat", "prompt", "prompt_locked"):
                            if ek in patch_ep:
                                cur[ek] = patch_ep[ek]
        _write_series(s)
        return s


@app.delete("/api/series/{sid}")
def delete_series(sid: str):
    with _series_lock(sid):
        p = SERIES / f"{sid}.json"
        if not p.is_file():
            raise HTTPException(404, "series not found")
        p.unlink()
    return {"deleted": sid}


@app.post("/api/series/plan")
async def plan_series(request: Request):
    """Stage 1 LLM Series Architect: turn a concept into a Series Bible + per-episode prompts."""
    b = await request.json()
    concept = (b.get("concept") or "").strip()
    if not concept:
        raise HTTPException(400, "enter a series concept or hook first")
    n_eps = max(2, min(12, int(b.get("num_episodes") or 3)))
    frames = int(b.get("frames") or 362)
    dur_s = frames / FPS
    aspect = (b.get("aspect") or "9:16").strip()
    style_hint = (b.get("style_hint") or "").strip()
    sys_msg = (
        "You are a viral short-form video series architect for MiniMax-H3.\n"
        "Return ONLY valid JSON (no markdown fences) with this exact schema:\n"
        "{\n"
        '  "title": "short catchy series title (max 6 words)",\n'
        '  "bible": {\n'
        '    "style": "camera rig, lens, frame rate, lighting, color grade",\n'
        '    "characters": "exact appearance, wardrobe, creature details",\n'
        '    "environment": "location details, props, atmosphere",\n'
        '    "audio_continuity": "recurring ambient soundscape + non-diegetic music style"\n'
        "  },\n"
        '  "episodes": [\n'
        "    {\n"
        '      "beat": "concise 1-2 sentence plot beat + ending handoff pose",\n'
        '      "prompt": "full MiniMax-H3 prompt"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Rules:\n"
        f"- Generate exactly {n_eps} episodes, each {dur_s:.1f}s ({frames} frames at {FPS}fps), aspect {aspect}.\n"
        "- Episode 1 prompt starts with: integrated_multimodal_description: [Shot 1] ...\n"
        "- Episodes 2..N are fl2va continuations from the previous clip's final frame, so each MUST start with:\n"
        "  For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n"
        "- Every prompt must include integrated_multimodal_description:, overall_soundscape:, and non_diegetic_music:.\n"
        "- End each episode with a clear visual/camera state so the next episode continues smoothly."
    )
    user_msg = f"CONCEPT: {concept}\n" + (f"STYLE HINT: {style_hint}\n" if style_hint else "")
    raw = _llm_chat([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                    use_vlm=False, temperature=0.45, max_tokens=3600)
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise HTTPException(502, "Series Architect did not return valid JSON")
    try:
        plan = json.loads(m.group(0))
    except Exception as e:
        raise HTTPException(502, f"Series Architect JSON parse failed: {e}")
    eps = plan.get("episodes") or []
    for idx, ep in enumerate(eps):
        mode = "t2va" if idx == 0 else "fl2va"
        p = (ep.get("prompt") or "").strip()
        if mode == "fl2va" and "<Picture 1>" not in p:
            p = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n" + p
        ep["prompt"] = _repair_polished_prompt(p, mode) if p else ""
    return plan


@app.post("/api/series/{sid}/replan")
async def replan_series(sid: str, request: Request):
    """Stage 3 Mid-Series Live Pivot: rewrite pending/waiting episodes with a new Director Note."""
    b = await request.json()
    note = (b.get("director_note") or "").strip()
    if not note:
        raise HTTPException(400, "enter a director note to re-plan remaining episodes")
    with _series_lock(sid):
        s = _read_series(sid)
        s["director_note"] = note
        frames = int(s["settings"].get("frames") or 362)
        dur_s = frames / FPS
        done_ctx = "\n".join(
            f"- Ep {e['index'] + 1} ({e['status']}): {e.get('beat', '')}"
            for e in s["episodes"] if e.get("status") == "done"
        )
        pending_eps = [e for e in s["episodes"] if e.get("status") in ("pending", "waiting_review") and not e.get("prompt_locked")]
        if not pending_eps:
            raise HTTPException(400, "no unlocked pending episodes left to re-plan")
        sys_msg = (
            "You are pivoting an in-progress MiniMax-H3 video series. Return ONLY valid JSON: "
            '{"episodes": [{"index": 1, "beat": "...", "prompt": "..."}]}\n'
            f"Target duration {dur_s:.1f}s ({frames} frames). Preserve the Series Bible and fl2va <Picture 1> syntax for episodes > 1."
        )
        user_msg = (
            f"BIBLE: {json.dumps(s.get('bible') or {})}\n"
            f"COMPLETED:\n{done_ctx or 'None yet'}\n"
            f"NEW DIRECTOR NOTE: {note}\n"
            f"RE-PLAN EPISODE INDICES: {[e['index'] for e in pending_eps]}"
        )
        raw = _llm_chat([{"role": "system", "content": sys_msg}, {"role": "user", "content": user_msg}],
                        use_vlm=False, temperature=0.4, max_tokens=3000)
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            raise HTTPException(502, "Re-plan LLM did not return valid JSON")
        upd = json.loads(m.group(0)).get("episodes") or []
        by_idx = {int(u["index"]): u for u in upd if "index" in u}
        for e in pending_eps:
            u = by_idx.get(e["index"])
            if not u:
                continue
            e["beat"] = (u.get("beat") or e.get("beat") or "").strip()
            mode = "t2va" if e["index"] == 0 else "fl2va"
            p = (u.get("prompt") or e.get("prompt") or "").strip()
            if mode == "fl2va" and "<Picture 1>" not in p:
                p = "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced.\n\n" + p
            e["prompt"] = _repair_polished_prompt(p, mode)
        _write_series(s)
        return s


@app.post("/api/series/{sid}/start")
def start_series(sid: str):
    with _series_lock(sid):
        s = _read_series(sid)
        if s.get("status") == "running":
            return s
        nxt = next((e for e in s["episodes"] if e.get("status") in ("pending", "waiting_review")), None)
        if not nxt:
            raise HTTPException(400, "no pending episodes to start")
        idx = nxt["index"]
        prev_jid = s["episodes"][idx - 1].get("job_id") if idx > 0 else None
        _enqueue_series_episode(s, idx, prev_job_id=prev_jid)
        s["status"] = "running"
        s["error"] = None
        _write_series(s)
        return s


@app.post("/api/series/{sid}/approve")
async def approve_series(sid: str, request: Request):
    b = {}
    try:
        b = await request.json()
    except Exception:
        pass
    with _series_lock(sid):
        s = _read_series(sid)
        idx = int(b.get("episode_index") if b.get("episode_index") is not None else s.get("current_index", 0))
        if idx < 0 or idx >= len(s["episodes"]):
            raise HTTPException(400, "invalid episode index")
        ep = s["episodes"][idx]
        if ep.get("status") not in ("waiting_review", "pending"):
            raise HTTPException(400, f"episode {idx + 1} is {ep.get('status')}, not waiting review")
        if b.get("prompt"):
            ep["prompt"] = str(b["prompt"]).strip()
        if "prompt_locked" in b:
            ep["prompt_locked"] = bool(b["prompt_locked"])
        prev_jid = s["episodes"][idx - 1].get("job_id") if idx > 0 else None
        _enqueue_series_episode(s, idx, prev_job_id=prev_jid)
        s["status"] = "running"
        _write_series(s)
        return s


@app.post("/api/series/{sid}/pause")
def pause_series(sid: str):
    with _series_lock(sid):
        s = _read_series(sid)
        s["status"] = "paused"
        _write_series(s)
        return s


@app.post("/api/series/{sid}/resume")
def resume_series(sid: str):
    with _series_lock(sid):
        s = _read_series(sid)
        waiting = next((e for e in s["episodes"] if e.get("status") == "waiting_review"), None)
        if waiting:
            idx = waiting["index"]
            prev_jid = s["episodes"][idx - 1].get("job_id") if idx > 0 else None
            _enqueue_series_episode(s, idx, prev_job_id=prev_jid)
        s["status"] = "running"
        _write_series(s)
        return s


@app.post("/api/stitch")
async def stitch_jobs(request: Request):
    """Concatenate any selected finished clips (from a series or the clip bin) into one MP4."""
    b = await request.json()
    job_ids = [str(x).strip() for x in (b.get("job_ids") or []) if str(x).strip()]
    title = (b.get("title") or "H3 Stitched Series").strip()[:100]
    series_id = (b.get("series_id") or "").strip() or None
    stid = uuid.uuid4().hex[:12]
    dst = STITCHES / f"{stid}.mp4"
    media = _stitch_mp4s(job_ids, dst)
    meta = {
        "id": stid,
        "title": title,
        "job_ids": job_ids,
        "series_id": series_id,
        "media": media,
        "created": time.time(),
    }
    (STITCHES / f"{stid}.json").write_text(json.dumps(meta))
    if series_id and (SERIES / f"{series_id}.json").is_file():
        with _series_lock(series_id):
            s = _read_series(series_id)
            s["stitched_id"] = stid
            _write_series(s)
    return meta


@app.get("/api/stitches/{stid}/file")
def stitch_file(stid: str, dl: int = 0):
    mp4 = STITCHES / f"{stid}.mp4"
    meta_p = STITCHES / f"{stid}.json"
    if not mp4.is_file():
        raise HTTPException(404, "stitched video not found")
    headers = {}
    if dl:
        raw_title = f"h3-stitch-{stid}"
        if meta_p.is_file():
            try:
                raw_title = json.loads(meta_p.read_text()).get("title") or raw_title
            except Exception:
                pass
        safe = re.sub(r'[\\/:*?"<>|\r\n]+', "", raw_title).strip().strip(".") or f"h3-stitch-{stid}"
        headers["Content-Disposition"] = f'attachment; filename="{safe}.mp4"'
    return FileResponse(mp4, media_type="video/mp4", headers=headers)


@app.get("/api/casts")
def list_casts():
    """Return shared server-side Cast presets stored in /out/casts/*.json."""
    out = []
    for p in sorted(CASTS.glob("*.json")):
        try:
            data = json.loads(p.read_text())
            if isinstance(data, list):
                out.extend(data)
            elif isinstance(data, dict):
                out.append(data)
        except Exception:
            continue
    return out


@app.get("/api/casts/file/{name}")
def cast_file(name: str):
    """Serve a shared Cast reference image from /out/ghost-refs/."""
    if not re.fullmatch(r"[\w.-]+", name):
        raise HTTPException(404)
    p = GHOST_REFS / name
    if not p.is_file():
        raise HTTPException(404, "cast reference file not found")
    return FileResponse(p)


@app.get("/api/demos/{name}")
def demo_file(name: str):
    """Serve a locally cached official MiniMax-H3 use-case demo video from /out/demos/."""
    if not re.fullmatch(r"[\w.-]+", name):
        raise HTTPException(404)
    p = DEMOS / name
    if not p.is_file():
        raise HTTPException(404, "demo video not found")
    return FileResponse(p, media_type="video/mp4")


@app.post("/api/prompt/validate")
async def validate_prompt(request: Request):
    """Validate prompt reference tags and dialogue blocks against attached references/keyframes."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "invalid JSON body")
    if not isinstance(body, dict):
        raise HTTPException(400, "request body must be a JSON object")
    prompt = str(body.get("prompt") or "")
    workflow = str(body.get("workflow") or "t2va").strip().lower()
    raw_refs = body.get("refs") or []
    if not isinstance(raw_refs, list):
        raise HTTPException(400, "refs must be a list")
    resolved = resolve_ref_tags([r for r in raw_refs if isinstance(r, dict)])
    return {
        "warnings": parse_prompt_tags(
            prompt, workflow, resolved, bool(body.get("has_first")), bool(body.get("has_last"))
        )
    }


def _format_polish_refs(raw_refs) -> str:
    """Format either a legacy ref summary string or structured ref list into prompt-polish context."""
    if isinstance(raw_refs, str):
        return raw_refs.strip()
    if not isinstance(raw_refs, list):
        return ""
    resolved = resolve_ref_tags([r for r in raw_refs if isinstance(r, dict)])
    lines = []
    for r in resolved:
        tags = " + ".join(t for t in (r.get("primary_tag"), r.get("audio_tag")) if t) or r.get("effective_kind", "ref")
        parts = []
        if r.get("mode") == "video_audio_only":
            parts.append("audio extracted from video")
        if r.get("ui_role") or r.get("role"):
            parts.append(f"role={r.get('ui_role') or r.get('role')}")
        if r.get("subject_group"):
            parts.append(f"group={r['subject_group']}")
        if r.get("name"):
            parts.append(r["name"])
        if r.get("width") and r.get("height"):
            parts.append(f"{r['width']}x{r['height']}")
        if r.get("duration"):
            parts.append(f"{r['duration']}s")
        start, end = r.get("start_sec"), r.get("end_sec")
        if (start and float(start) > 0) or end is not None:
            parts.append(f"trim={float(start or 0):.1f}s..{f'{float(end):.1f}s' if end is not None else 'end'}")
        lines.append(f"{tags} ({', '.join(parts)})" if parts else tags)
    return "\n".join(lines)


@app.post("/api/prompt/polish")
async def polish(request: Request):
    """Optional: rewrite a draft prompt with the local LLM lane. Off unless H3_LLM_URL is set."""
    if not LLM_URL:
        raise HTTPException(501, "prompt polish is not configured on this worker")
    body = await request.json()
    draft = (body.get("draft") or "").strip()
    mode = body.get("mode") or "t2va"
    refs_ctx = _format_polish_refs(body.get("refs"))
    aspect = str(body.get("aspect") or "").strip()
    intent = str(body.get("intent") or "").strip()
    raw_frames = body.get("frames")
    if not draft:
        raise HTTPException(400, "nothing to polish")
    guide = _guide(mode)
    ctx_parts = []
    if raw_frames is not None and str(raw_frames).strip():
        try:
            nf = int(raw_frames)
            if nf > 0:
                ctx_parts.append(f"Target duration: {nf / FPS:.2f}s ({nf} frames at {FPS} fps)")
        except (TypeError, ValueError):
            pass
    if aspect:
        ctx_parts.append(f"Target aspect ratio: {aspect}")
    if intent:
        ctx_parts.append(f"Target intent: {intent}")
    if refs_ctx:
        ctx_parts.append(f"Attached references:\n{refs_ctx}")
    user_msg = "\n\n".join(ctx_parts + [f"Request:\n{draft}"]) if ctx_parts else draft
    payload = json.dumps({
        "model": body.get("model") or LLM_MODEL,
        "temperature": 0.35,
        "max_tokens": 1600,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": guide}, {"role": "user", "content": user_msg}],
    }).encode()
    req = urllib.request.Request(LLM_URL + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {LLM_KEY}"} if LLM_KEY else {})})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            data = json.loads(r.read())
        text = (data["choices"][0]["message"].get("content") or "").strip()
        text = re.sub(r"<think>[\s\S]*?</think>\s*", "", text).strip()
        text = re.sub(r"^```[a-zA-Z]*\n([\s\S]*?)\n```$", r"\1", text).strip()
        text = _repair_polished_prompt(text, mode)
    except Exception as e:
        raise HTTPException(502, f"polish backend failed: {type(e).__name__}: {e}")
    return {"prompt": text}


def _repair_polished_prompt(text: str, mode: str) -> str:
    """Clean up common LLM formatting slips in polished MiniMax-H3 prompts."""
    if not text:
        return text
    # Fix bare ref tags like Picture 1 / Video 1 / Audio 1 / Subject 1 missing angle brackets
    text = re.sub(r"(?<![<\w])(Picture|Video|Audio|Subject)\s+(\d+)(?!\s*>)", r"<\1 \2>", text, flags=re.I)
    # Close unclosed <d> dialogue tags at end of line/text
    if text.count("<d>") > text.count("</d>"):
        text = re.sub(r"(<d>[^\n<]*?)(?=\n|$)(?![\s\S]*</d>)", r"\1</d>", text)
        if text.count("<d>") > text.count("</d>"):
            text += "</d>"
    # Ensure required trailing audio sections exist so H3 separates foley from BGM
    if "overall_soundscape:" not in text:
        text += "\n\noverall_soundscape: Natural ambient room tone and physical action sounds."
    if "non_diegetic_music:" not in text:
        text += "\n\nnon_diegetic_music: N/A"
    return text.strip()


def _guide(mode):
    p = Path("/app/guides")
    name = "ref_en.md" if mode == "ref2va" else "base_en.md"
    rules = (
        "Rewrite the user's request into a production-ready MiniMax H3 prompt in English "
        "(keep spoken dialogue inside <d>[Language] ...</d> and on-screen signage inside \"...\").\n"
        "CRITICAL PROMPTING RULES:\n"
        "1. Replace vague buzzwords ('cinematic', 'photorealistic', 'hyper-detailed') with concrete lighting, lens/framing, materials, and physical motion.\n"
        "2. Match shot count to Target duration: 1-2 shots for <=8s, 2-3 shots for 9-16s, 3-5 shots for >16s; [Shot 1] has NO timestamp, later shots use strictly increasing [Shot N] At mm:ss.mmm.\n"
        "3. Describe action continuity smoothly ('begins by..., then..., continuing as...') so subjects do not morph.\n"
        "4. Write camera motion naturally inside each shot as Motion Type + Amplitude + Speed (e.g. 'pushes in with small amplitude at slow speed').\n"
        "5. Wrap ALL spoken lines/lyrics in <d>[Language] ...</d> with global speaker IDs (S1), (S2); wrap ALL visible on-screen text in double quotes.\n"
        "6. Keep overall_soundscape strictly for ambience/foley (no dialogue or score) and non_diegetic_music strictly for BGM (or N/A).\n"
        "7. Return ONLY the final structured prompt — no markdown fences, no commentary.\n\n"
    )
    try:
        text = (p / name).read_text()
    except Exception:
        return rules + "Structure with integrated_multimodal_description, overall_soundscape, and non_diegetic_music."
    return rules + text[:24000]


# ---------------------------------------------------------------- model

def _patch_offload_copies():
    """Let int8 weights travel through the block offloader.

    torchao 0.18 routes `Tensor.to()` through `aten.to.dtype_layout`, whose handler accepts only
    dtype/layout/device kwargs. The offloader always passes `non_blocking=`, so any quantised
    parameter raises AssertionError as soon as its group is onloaded (i.e. on the second block of
    the first denoise step). Retry that one call without the async hint: same copy, just
    synchronous — roughly 30 ms per block, invisible next to a 17 s step.

    The method lives on `ModuleGroup` here, but it has moved between diffusers releases, so bind to
    whichever class in the module defines it and fail loudly if none does.
    """
    import inspect

    from diffusers.hooks import group_offloading as go

    owners = [c for _, c in vars(go).items()
              if inspect.isclass(c) and c.__module__ == go.__name__
              and "_transfer_tensor_to_device" in vars(c)]
    if not owners:
        print("offload shim: no owner class found (diffusers changed); not patched", flush=True)
        return False

    def make(orig):
        def transfer(self, tensor, source_tensor, default_stream):
            try:
                return orig(self, tensor, source_tensor, default_stream)
            except AssertionError:
                moved = source_tensor.to(self.onload_device)
                if go._is_torchao_tensor(tensor):
                    go._swap_torchao_tensor(tensor, moved)
                    if self.record_stream:
                        go._record_stream_torchao_tensor(tensor, default_stream)
                else:
                    tensor.data = moved
                    if self.record_stream:
                        tensor.data.record_stream(default_stream)
        transfer._h3_sync_retry = True
        return transfer

    for cls in owners:
        orig = vars(cls)["_transfer_tensor_to_device"]
        if getattr(orig, "_h3_sync_retry", False):
            continue
        setattr(cls, "_transfer_tensor_to_device", make(orig))
    return True


def _rope_slice(x: torch.Tensor, cos_c: torch.Tensor, sin_c: torch.Tensor) -> torch.Tensor:
    rot_dim = cos_c.shape[-1]
    x_rot = x[..., :rot_dim]
    x_pass = x[..., rot_dim:]
    c = cos_c[None, :, None, :]
    s = sin_c[None, :, None, :]
    x1, x2 = x_rot.chunk(2, dim=-1)
    x_rot = x_rot * c + torch.cat((-x2, x1), dim=-1) * s
    return torch.cat((x_rot, x_pass), dim=-1)


def _patch_transformer_memory(model):
    """Chunk row-wise block ops, query attention, and output heads along sequence dim.

    In Ref2VA with long outputs (345 frames) + reference videos, packed sequence lengths reach ~230k tokens.
    Full-sequence AdaLN modulation, Q/K norms, RoPE, and Q/K/V/attn_out peak >28 GB in default bf16 and
    OOM 32 GB cards. Chunking row-wise ops + Q-sliced SDPA preserves exact math while cutting peak VRAM >14 GB.
    """
    from diffusers.models.attention_dispatch import dispatch_attention_fn
    from diffusers.models.modeling_utils import get_parameter_dtype
    from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TransformerOutput
    from diffusers.utils import apply_lora_scale

    patched_blocks = 0
    for blk in getattr(model, "transformer_blocks", ()):
        if getattr(blk.forward, "_h3_mem_patched", False):
            continue
        orig_blk = blk.forward

        def make_block(b, orig):
            def block_fwd(hidden_states, temb, adaln_indices, rotary_emb, attention_mask=None):
                attn = b.attn
                if (
                    torch.is_grad_enabled()
                    or attention_mask is not None
                    or rotary_emb is None
                    or getattr(attn, "fused_projections", False)
                    or type(attn.processor).__name__ != "MiniMaxH3AttnProcessor"
                    or hidden_states.ndim != 3
                ):
                    return orig(hidden_states, temb, adaln_indices, rotary_emb, attention_mask)

                row_chunk = max(1024, int(_CHUNK_CFG["ffn"]))
                q_chunk = max(row_chunk, int(_CHUNK_CFG["attn"]))
                bsz, seq_len, _ = hidden_states.shape

                cos_all, sin_all = rotary_emb
                cos_bf = cos_all if cos_all.dtype == hidden_states.dtype else cos_all.to(hidden_states.dtype)
                sin_bf = sin_all if sin_all.dtype == hidden_states.dtype else sin_all.to(hidden_states.dtype)

                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = b.adaln_proj(temb)
                n_heads, hd = attn.heads, attn.head_dim
                backend = getattr(attn.processor, "_attention_backend", None)
                pconf = getattr(attn.processor, "_parallel_config", None)

                key = torch.empty((bsz, seq_len, n_heads, hd), device=hidden_states.device, dtype=hidden_states.dtype)
                value = torch.empty((bsz, seq_len, n_heads, hd), device=hidden_states.device, dtype=hidden_states.dtype)
                for s in range(0, seq_len, row_chunk):
                    e = min(s + row_chunk, seq_len)
                    idx = adaln_indices[s:e]
                    x = b.norm1(hidden_states[:, s:e])
                    x = x * (1.0 + scale_msa.index_select(0, idx)) + shift_msa.index_select(0, idx)
                    kc = attn.norm_k(attn.to_k(x).unflatten(-1, (n_heads, hd)))
                    key[:, s:e] = _rope_slice(kc, cos_bf[s:e], sin_bf[s:e])
                    value[:, s:e] = attn.to_v(x).unflatten(-1, (n_heads, hd))
                    del x, kc, idx

                out = torch.empty_like(hidden_states)
                for s in range(0, seq_len, q_chunk):
                    e = min(s + q_chunk, seq_len)
                    idx = adaln_indices[s:e]
                    x = b.norm1(hidden_states[:, s:e])
                    x = x * (1.0 + scale_msa.index_select(0, idx)) + shift_msa.index_select(0, idx)
                    qc = attn.norm_q(attn.to_q(x).unflatten(-1, (n_heads, hd)))
                    qc = _rope_slice(qc, cos_bf[s:e], sin_bf[s:e])
                    del x
                    ao = dispatch_attention_fn(
                        qc, key, value, attn_mask=None, dropout_p=0.0, is_causal=False,
                        backend=backend, parallel_config=pconf,
                    )
                    del qc
                    ao = ao.flatten(2, 3).type_as(hidden_states)
                    ao = attn.to_out[1](attn.to_out[0](ao))
                    out[:, s:e] = hidden_states[:, s:e] + gate_msa.index_select(0, idx) * ao
                    del ao, idx
                del key, value

                mlp_out = torch.empty_like(out)
                for s in range(0, seq_len, row_chunk):
                    e = min(s + row_chunk, seq_len)
                    idx = adaln_indices[s:e]
                    r = out[:, s:e]
                    y = b.norm2(r)
                    y = y * (1.0 + scale_mlp.index_select(0, idx)) + shift_mlp.index_select(0, idx)
                    y = b.ff(y)
                    mlp_out[:, s:e] = r + gate_mlp.index_select(0, idx) * y
                    del r, y, idx
                return mlp_out

            block_fwd._h3_mem_patched = True
            return block_fwd

        blk.forward = make_block(blk, orig_blk)
        patched_blocks += 1

    if not getattr(model.forward, "_h3_mem_patched", False):
        import types

        orig_model_fwd = model.forward

        @apply_lora_scale("attention_kwargs")
        def model_fwd(
            self,
            hidden_states,
            audio_hidden_states,
            encoder_hidden_states,
            timestep,
            timestep_indices,
            token_tags,
            position_ids,
            video_indices,
            audio_indices,
            text_indices,
            attention_kwargs=None,
            return_dict=True,
        ):
            if torch.is_grad_enabled() or self.gradient_checkpointing:
                return orig_model_fwd(
                    hidden_states, audio_hidden_states, encoder_hidden_states,
                    timestep, timestep_indices, token_tags, position_ids,
                    video_indices, audio_indices, text_indices,
                    attention_kwargs=attention_kwargs, return_dict=return_dict,
                )

            if position_ids.ndim != 2 or position_ids.shape[-1] != 3:
                raise ValueError(f"`position_ids` must be a `(seq_len, 3)` tensor, got {list(position_ids.shape)}.")
            sequence_length = position_ids.shape[0]
            if token_tags.shape != (sequence_length,) or timestep_indices.shape != (sequence_length,):
                raise ValueError(
                    "`token_tags` and `timestep_indices` must both be `(seq_len,)` tensors matching `position_ids`."
                )

            cos, sin = self.rope(position_ids)

            video_embeds = self.proj_in(hidden_states.to(get_parameter_dtype(self.proj_in)))
            audio_embeds = self.audio_proj_in(audio_hidden_states.to(get_parameter_dtype(self.audio_proj_in)))
            text_embeds = self.context_embedder(encoder_hidden_states.to(get_parameter_dtype(self.context_embedder)))
            text_embeds = self.token_refiner(text_embeds)

            seq_dtype = text_embeds.dtype
            cos = cos.to(seq_dtype)
            sin = sin.to(seq_dtype)
            rotary_emb = (cos, sin)

            hidden_states = text_embeds.new_zeros((text_embeds.shape[0], sequence_length, text_embeds.shape[-1]))
            hidden_states = hidden_states.index_copy(1, text_indices, text_embeds)
            hidden_states = hidden_states.index_copy(1, video_indices, video_embeds.to(seq_dtype))
            hidden_states = hidden_states.index_copy(1, audio_indices, audio_embeds.to(seq_dtype))
            del video_embeds, audio_embeds, text_embeds

            temb = self.time_proj(timestep)
            temb = self.time_embedder(temb.to(get_parameter_dtype(self.time_embedder)))
            adaln_indices = timestep_indices * 3 + token_tags

            for block in self.transformer_blocks:
                hidden_states = block(hidden_states, temb, adaln_indices, rotary_emb)
            del rotary_emb, cos, sin, adaln_indices

            # Row-select output indices BEFORE float32 cast & head projection to avoid allocating 230k float32 rows.
            proj_dtype = get_parameter_dtype(self.proj_out)
            audio_proj_dtype = get_parameter_dtype(self.audio_proj_out)
            shift, scale = self.norm_out.linear(
                torch.nn.functional.silu(temb).to(get_parameter_dtype(self.norm_out.linear))
            ).chunk(2, dim=-1)

            vh = hidden_states.index_select(1, video_indices)
            ah = hidden_states.index_select(1, audio_indices)
            del hidden_states

            v_t = timestep_indices.index_select(0, video_indices)
            nv = video_indices.shape[0]
            out_chunk = max(1024, int(_CHUNK_CFG["ffn"]))
            if nv <= out_chunk:
                vh = self.norm_out.norm(vh)
                vh = vh * (1.0 + scale.index_select(0, v_t)) + shift.index_select(0, v_t)
                video_output = self.proj_out(vh.to(proj_dtype))
            else:
                out_dim = getattr(self.proj_out, "out_features", None) or self.proj_out.weight.shape[0]
                video_output = torch.empty((vh.shape[0], nv, out_dim), device=vh.device, dtype=proj_dtype)
                for s in range(0, nv, out_chunk):
                    e = min(s + out_chunk, nv)
                    tc = v_t[s:e]
                    hc = self.norm_out.norm(vh[:, s:e])
                    hc = hc * (1.0 + scale.index_select(0, tc)) + shift.index_select(0, tc)
                    video_output[:, s:e] = self.proj_out(hc.to(proj_dtype))
            del vh, v_t

            a_t = timestep_indices.index_select(0, audio_indices)
            ah = self.norm_out.norm(ah)
            ah = ah * (1.0 + scale.index_select(0, a_t)) + shift.index_select(0, a_t)
            audio_output = self.audio_proj_out(ah.to(audio_proj_dtype))
            del ah, a_t

            if not return_dict:
                return (video_output, audio_output)
            return MiniMaxH3TransformerOutput(sample=video_output, audio_sample=audio_output)

        model_fwd._h3_mem_patched = True
        model.forward = types.MethodType(model_fwd, model)

    return patched_blocks


def _release_pipe_vram(p):
    """Force every offloaded hook group and VAE back to CPU so failed/cancelled runs never strand VRAM."""
    if p is None:
        return
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    for attr in ("transformer", "transformer_ref", "text_encoder"):
        mod = getattr(p, attr, None)
        if mod is None:
            continue
        for sub in mod.modules():
            reg = getattr(sub, "_diffusers_hook", None)
            if reg is None:
                continue
            for hook in getattr(reg, "hooks", {}).values():
                grp = getattr(hook, "group", None)
                if grp is not None:
                    try:
                        grp.offload_()
                    except Exception:
                        pass
    for attr in ("vae", "audio_vae"):
        mod = getattr(p, attr, None)
        if mod is not None:
            try:
                mod.to("cpu")
            except Exception:
                pass


def load():
    global pipe
    from diffusers import AutoencoderKLMiniMaxH3, MiniMaxH3Transformer3DModel, ModularPipeline, TorchAoConfig
    from diffusers.hooks import apply_group_offloading
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from transformers import TorchAoConfig as TransformersTorchAoConfig
    from torchao.quantization import Int8WeightOnlyConfig

    def dit(sub):
        return MiniMaxH3Transformer3DModel.from_pretrained(
            LOCAL, subfolder=sub, dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
                "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
                "token_refiner", "norm_out", "proj_out", "audio_proj_out"]))

    t0 = time.time()
    # Each container sees exactly one GPU (--gpus device=N), so cuda:0 is that card. The fraction
    # only matters on the shared card: lane B coexists with Whisper and must not starve it.
    if VRAM_FRACTION:
        torch.cuda.set_per_process_memory_fraction(VRAM_FRACTION, 0)
        print(f"vram cap: {VRAM_FRACTION:.0%} of GPU", flush=True)
    p = ModularPipeline.from_pretrained(R, workflow="ref2va") if PARTITION == "ref" else ModularPipeline.from_pretrained(R)
    comp = {
        "transformer": dit("transformer") if PARTITION != "ref" else None,
        "transformer_ref": dit("transformer_ref") if PARTITION != "base" else None,
        "vae": AutoencoderKLMiniMaxH3.from_pretrained(LOCAL, subfolder="vae", dtype=torch.bfloat16),
        "processor": Qwen3VLProcessor.from_pretrained(LOCAL, subfolder="processor"),
        "text_encoder": Qwen3VLForConditionalGeneration.from_pretrained(
            LOCAL, subfolder="text_encoder", dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(Int8WeightOnlyConfig(version=2), modules_to_not_convert=[
                "model.visual", "model.language_model.embed_tokens", "model.language_model.norm", "lm_head"])),
    }
    p.update_components(**{k: v for k, v in comp.items() if v is not None})
    p.load_components(dtype=torch.bfloat16)  # no workflow= on both/base: keeps t2va and fl2va on one pipeline
    print("offload shim:", _patch_offload_copies(), flush=True)
    offload = dict(onload_device=torch.device("cuda"), offload_device=torch.device("cpu"), use_stream=True)
    for m in (getattr(p, "transformer", None), getattr(p, "transformer_ref", None)):
        if m is None:
            continue
        m.requires_grad_(False)
        n_blk = _patch_transformer_memory(m)
        print(f"memory patch: chunked {n_blk} transformer blocks (ffn={FFN_CHUNK}, attn={ATTN_CHUNK})", flush=True)
        m.enable_group_offload(offload_type="block_level", num_blocks_per_group=1, **offload)
        m.register_forward_pre_hook(_tick)          # one call per denoise step
    p.text_encoder.requires_grad_(False)
    apply_group_offloading(p.text_encoder.model, offload_type="leaf_level", **offload)
    p.text_encoder.model.register_forward_hook(lambda *_: torch.cuda.empty_cache())
    # The VAEs are only needed at decode. Keeping them on the host during denoise frees ~5 GB at
    # exactly the point where the peak is. The move has to wrap decode() itself: the pipeline calls
    # components.vae.decode(), which never goes through __call__, so a forward pre-hook never fires.
    p.vae.to("cpu"); p.audio_vae.to("cpu")

    def _offload_weights(module):
        """Move a VAE to the GPU only while it is actually computing.

        The VAEs are used at more than one point: ref2va VAE-encodes the reference images *before*
        denoise, and every job decodes at the end. Wrapping decode() alone left the encode path
        holding host weights and produced "Input type (torch.cuda.FloatTensor) and weight type
        (torch.FloatTensor) should be the same". Re-entrancy is counted so a wrapper that calls
        another wrapped method does not park the weights mid-flight.
        """
        depth = {"n": 0}

        def wrap(fn, label):
            def wrapper(*args, **kwargs):
                depth["n"] += 1
                if depth["n"] == 1:
                    module.to("cuda")
                    set_state(stage=label)
                try:
                    return fn(*args, **kwargs)
                finally:
                    depth["n"] -= 1
                    if depth["n"] == 0:
                        module.to("cpu")
                        torch.cuda.empty_cache()
            return wrapper

        for name, label in (("encode", "encode"), ("decode", "decode"), ("forward", "decode")):
            fn = getattr(module, name, None)
            if callable(fn):
                setattr(module, name, wrap(fn, label))
        return module

    _offload_weights(p.vae)
    _offload_weights(p.audio_vae)

    # Lift the pipeline's native 15-s duration cap to match our extended FRAMES array.
    # MiniMaxH3ModularPipeline.max_duration is a plain property that returns 15.0;
    # patching the class (not the instance) is safe because there is only one pipeline
    # object per worker process and both lanes share nothing at runtime.
    _pipe_max_dur = FRAMES[-1] / FPS
    type(p).max_duration = property(lambda self: _pipe_max_dur)
    print(f"pipeline max_duration patched to {_pipe_max_dur:.2f} s ({FRAMES[-1]} frames)", flush=True)

    pipe = p
    state["loaded_in"] = round(time.time() - t0, 1)
    print(f"model loaded in {state['loaded_in']} s (partition={PARTITION})", flush=True)


def _tick(module, args):
    with lock:
        state["step"] = state.get("step", 0) + 1
        n = state["step"]
        state["ticks"].append(time.time())
        jid = state.get("running")
    set_state(stage="denoise", step=n)
    if not jid:
        return
    # One file read per step (~20-70 s apart): the only place a long render can be interrupted or
    # held. A pause blocks here, between steps, so the GPU goes idle and VRAM stays reserved.
    # The stage is written once on entering pause, not every poll, so a resume cannot be overwritten
    # by this loop's own read-modify-write.
    marked = False
    while True:
        try:
            job = read(jid)
        except HTTPException:
            raise Cancelled("cancelled: the job was removed")
        if job.get("cancel"):
            raise Cancelled(f"cancelled at step {n}")
        if not job.get("pause"):
            return
        if not marked:
            set_state(stage="paused")
            marked = True
        time.sleep(2)


def analyse_audio(a):
    """Peak and RMS in dBFS. Peak is what clips; RMS is what people actually hear as 'quiet'."""
    x = a.float()
    peak, rms = float(x.abs().max()), float(x.pow(2).mean().sqrt())
    db = lambda v: round(20 * math.log10(v), 2) if v > 0 else -120.0
    return db(peak), db(rms)


def level_audio(a, mode="raw", ceiling_db=-1.0, max_boost_db=24.0):
    """Return (audio, gain_db).

    Attenuation is always applied when a mix would clip, because a clipped master is a defect.
    Boosting only happens when asked, and never by more than max_boost_db, so a near-silent take
    cannot become amplified noise. Measured on this box: the hawker prompt peaked at -0.4 dBFS and
    the rain-street prompt at -25.0, so the two need opposite treatment.
    """
    peak_db, _ = analyse_audio(a)
    if peak_db <= -119:
        return a, 0.0
    gain_db = min(0.0, ceiling_db - peak_db)
    if mode == "peak" and gain_db == 0.0:
        gain_db = min(ceiling_db - peak_db, max_boost_db)
    if gain_db == 0.0:
        return a, 0.0
    return a * (10 ** (gain_db / 20)), round(gain_db, 2)


def _slice_audio(audio, sample_rate: int | None, start_sec: float, end_sec: float | None):
    """Slice a (channels, num_samples) waveform by [start_sec, end_sec] in seconds."""
    if audio is None or not sample_rate:
        return audio
    total = audio.shape[1]
    a0 = max(0, int(round(start_sec * sample_rate))) if start_sec > 0 else 0
    a1 = min(total, int(round(end_sec * sample_rate))) if end_sec is not None else total
    if a0 == 0 and a1 == total:
        return audio
    return audio[:, a0:a1] if a0 < a1 else None


def run(job):
    from diffusers.modular_pipelines.minimax_h3 import (MiniMaxH3AudioReference, MiniMaxH3ImageReference,
                                                        MiniMaxH3VideoReference)
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video
    d, i = JOBS / job["id"], job["inputs"]
    ref = {"image": MiniMaxH3ImageReference, "video": MiniMaxH3VideoReference, "audio": MiniMaxH3AudioReference}
    kw = {k: load_image(str(d / i[k])) for k in ("image", "last_image") if i.get(k)}
    if job.get("target_height") and job.get("target_width"):
        kw["height"] = int(job["target_height"])
        kw["width"] = int(job["target_width"])
    if i.get("references"):
        loaded_refs = []
        vstart = float(job.get("video_start") or 0.0)
        keep_va = job.get("keep_video_audio", True)
        for r in i["references"]:
            kind_to_load = r.get("effective_kind") or r["kind"]
            obj = ref[kind_to_load].from_file(str(d / r["file"]))
            start_sec = float(r["start_sec"] if r.get("start_sec") is not None else (vstart if kind_to_load == "video" else 0.0))
            end_sec = float(r["end_sec"]) if r.get("end_sec") is not None else None
            if kind_to_load == "video":
                # Slice frames using original container fps before applying any fps_override.
                src_fps = float(obj.fps or FPS)
                f0 = min(max(0, int(round(start_sec * src_fps))), max(0, len(obj.frames) - 1)) if start_sec > 0 else 0
                f1 = min(len(obj.frames), max(f0 + 1, int(round(end_sec * src_fps)))) if end_sec is not None else len(obj.frames)
                if f0 > 0 or f1 < len(obj.frames):
                    obj.frames = obj.frames[f0:f1]
                if not r.get("keep_video_audio", keep_va):
                    obj.audio = None
                    obj.sample_rate = None
                else:
                    obj.audio = _slice_audio(obj.audio, obj.sample_rate, start_sec, end_sec)
                    if obj.audio is None:
                        obj.sample_rate = None
                    elif r.get("sample_rate_override"):
                        obj.sample_rate = int(r["sample_rate_override"])
                if r.get("fps_override"):
                    obj.fps = float(r["fps_override"])
            elif kind_to_load == "audio":
                sliced = _slice_audio(obj.audio, obj.sample_rate, start_sec, end_sec)
                if sliced is None:
                    raise ValueError(f"{r.get('name', 'audio')}: start_sec/end_sec leaves empty audio")
                obj.audio = sliced
                if r.get("sample_rate_override"):
                    obj.sample_rate = int(r["sample_rate_override"])
            loaded_refs.append(obj)
        kw["references"] = loaded_refs
    t = {}
    job.update(status="running", stage="encode", started=time.time(), step=0)
    write(job)
    with lock:
        state.update(running=job["id"], step=0, total=job["steps"], ticks=[])
    torch.cuda.reset_peak_memory_stats(0)

    def attempt():
        # A fresh generator per attempt, so an OOM retry reproduces the same take.
        return pipe(prompt=job["prompt"], num_frames=job["frames"], num_inference_steps=job["steps"],
                    generator=torch.Generator().manual_seed(job["seed"]),
                    output=["videos", "audio", "sampling_rate"], **kw)

    ref_edge = int(job.get("ref_short_edge") or 2048)
    with torch.inference_mode():
        pipe.register_to_config(reference_image_short_edge=ref_edge)
        try:
            oom = False
            r = None
            try:
                r = attempt()
            except torch.OutOfMemoryError:
                oom = True
            if oom:
                # Retry outside `except` so dead frame locals/tracebacks do not pin GPU tensors.
                _release_pipe_vram(pipe)
                gc.collect()
                torch.cuda.empty_cache()
                job["oom_retry"] = True
                write(job)
                print("out of memory: released offloaded weights and retrying with smaller chunks", flush=True)
                _CHUNK_CFG["ffn"] = min(FFN_CHUNK, 4096)
                _CHUNK_CFG["attn"] = min(ATTN_CHUNK, 8192)
                try:
                    r = attempt()
                except torch.OutOfMemoryError as e:
                    raise RuntimeError(
                        f"out of GPU memory at {job['frames']} frames / {job['steps']} steps "
                        f"({job['frames'] / FPS:.1f} s). Shorten the clip or lower the steps; "
                        f"long clips with references are the heaviest case. ({e})") from e
        finally:
            _CHUNK_CFG["ffn"] = FFN_CHUNK
            _CHUNK_CFG["attn"] = ATTN_CHUNK
            pipe.register_to_config(reference_image_short_edge=2048)
        set_state(stage="mux")
        a = r["audio"][0]
        peak_db, rms_db = analyse_audio(a)
        a, gain_db = level_audio(a, job.get("audio_mode", "raw"))
        job["audio"] = {"peak_db": peak_db, "rms_db": rms_db, "gain_db": gain_db}
        encode_video(r["videos"][0], fps=FPS, output_path=str(d / "output.mp4"),
                     audio=a, audio_sample_rate=r["sampling_rate"])
    now = time.time()
    t["total"] = round(now - job["started"], 1)
    job["vram_peak_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    if poster(d / "output.mp4", d / "poster.jpg"):
        job["poster"] = "poster.jpg"
    job["media"] = probe_video(d / "output.mp4")
    job["timings"] = t
    job["stage"] = "done"
    write(job)  # main() re-reads the job from disk, so anything kept only in memory is lost
    record(job)


def record(job):
    """Keep the raw measurement; the cost model is fitted from these, not assumed."""
    try:
        hist = json.loads(PERF.read_text()) if PERF.exists() else []
    except Exception:
        hist = []
    total, steps, frames = job.get("timings", {}).get("total"), job.get("steps"), job.get("frames")
    if total and steps and frames:
        entry = {"frames": frames, "steps": steps, "total": total, "at": time.time()}
        hist = [h for h in hist if not (h.get("frames") == frames and h.get("steps") == steps)]
        hist.append(entry)
        tmp = PERF.with_suffix(".tmp")
        tmp.write_text(json.dumps(sorted(hist, key=lambda h: -h["at"])[:40]))
        os.replace(tmp, PERF)


def _hist():
    """Every lane's measurements. The old shared perf.json is still read so history is not lost."""
    out = []
    for p in sorted(PERF.parent.glob("perf*.json")):
        try:
            out.extend(json.loads(p.read_text()))
        except Exception:
            continue
    return out


def _cost_x(frames, steps):
    """Denoise work for one job. Measured on this box: cost per step grows as frames^1.5,
    not linearly — 124 frames costs 0.165 s per step-frame, 243 frames costs 0.235 s."""
    return max(1, steps) * (max(1, frames) ** FRAME_EXP)


def fit_cost():
    """Least squares of total seconds against steps x frames^1.5.

    Four measured runs — 124 f / 8 steps 163 s, 124 f / 50 steps 1027 s and 1056 s,
    243 f / 25 steps 1426 s — fit this to within 1.5%. The linear model I started with was
    39% low on the 243-frame job, which is the error that matters when quoting a client.
    """
    pts = [(_cost_x(h["frames"], h["steps"]), h["total"]) for h in _hist()
           if h.get("total") and h.get("steps") and h.get("frames")]
    if len(pts) < 2:
        return DEFAULT_K, DEFAULT_C
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    var = sum((x - mx) ** 2 for x, _ in pts)
    if var <= 0:
        return DEFAULT_K, DEFAULT_C
    k = sum((x - mx) * (y - my) for x, y in pts) / var
    if k <= 0:
        return DEFAULT_K, DEFAULT_C
    return k, max(0.0, my - k * mx)


def estimate(frames, steps):
    k, c = fit_cost()
    return round(k * _cost_x(frames, steps) + c)


def legal_frames(n):
    """Nearest legal frame count at or above n.

    The pipeline needs 5.0-15.0 s at 24 fps after alignment to the 17n+5 grid, so anything below
    124 is refused at denoise time. Legacy jobs and retries can still carry 107.
    """
    n = int(n or 0)
    for f in FRAMES:
        if f >= n:
            return f
    return FRAMES[-1]


# ---------------------------------------------------------------- worker

def main():
    try:
        load(); state["model"] = "ready"
    except Exception as e:
        traceback.print_exc()
        state["model"] = f"load failed: {type(e).__name__}: {e}"[:500]
        return
    while True:
        job = next_job()
        if job is None:
            time.sleep(2)
            continue
        jid = job["id"]
        with lock:
            state.update(running=jid, step=0, total=job["steps"], ticks=[])
        try:
            run(job)
            job = read(jid)
            if job.get("cancel"):
                job.update(status="cancelled", stage="cancelled", finished=time.time(),
                           error="cancelled after the render finished")
            else:
                job.update(status="done", stage="done", finished=time.time(), output="output.mp4",
                           step=job.get("total_steps") or job.get("steps"))
        except Cancelled as e:
            job = read(jid)
            job.update(status="cancelled", stage="cancelled", finished=time.time(), error=str(e))
        except Exception as e:
            tb = traceback.format_exc()
            traceback.print_exc()
            job = read(jid)
            head = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
            tail = "\n".join(l.rstrip() for l in tb.strip().splitlines()[-8:])
            job.update(status="error", stage="error", error=(head + "\n" + tail)[:2000],
                       finished=time.time())
        write(job)
        _advance_series_for_job(job)
        _release_pipe_vram(pipe)
        try:
            (JOBS / jid / "claim").unlink()
        except Exception:
            pass
        with lock:
            state.update(running=None, step=0, total=0, ticks=[])
        gc.collect()
        torch.cuda.empty_cache()


# Restart recovery, per lane: only jobs this lane was running are failed, and only its own claims
# are cleared. A job another lane is rendering right now must not be touched.
try:
    n = backfill_numbers()
    if n:
        print(f"numbered {n} existing job(s)", flush=True)
except Exception as e:
    print(f"numbering failed: {type(e).__name__}: {e}", flush=True)
for p in JOBS.glob("*/job.json"):
    try:
        j = json.loads(p.read_text())
    except Exception:
        continue
    if j.get("status") == "running" and j.get("lane") == LANE:
        j.update(status="error", stage="error", error="worker restarted during this job",
                 finished=time.time())
        write(j)
        try:
            (p.parent / "claim").unlink()
        except Exception:
            pass
    elif j.get("status") == "running" and j.get("lane") in (None, ""):
        # claimed by a build that predates lanes: it cannot be running anywhere now
        j.update(status="error", stage="error", error="worker restarted during this job",
                 finished=time.time())
        write(j)
        try:
            (p.parent / "claim").unlink()
        except Exception:
            pass
threading.Thread(target=main, daemon=True).start()
