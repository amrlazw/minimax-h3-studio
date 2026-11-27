# How H3 Studio Actually Works

This document explains the real internal architecture, execution flow, memory patches, and operational behavior of **H3 Studio** (`server.py`, `index.html`, `titler.py`, and `scripts/`).

---

## 1. System Topology

H3 Studio is a self-hosted, zero-build web studio for **MiniMaxAI/MiniMax-H3** (`t2va`, `fl2va`, `ref2va`) designed to run one worker container per GPU while sharing a single persistent job directory (`/out`).

```mermaid
flowchart LR
    Browser["Browser SPA (index.html)"] -->|HTTP + x-api-key / cookie| LaneA["Container h3 (Lane A, GPU 0, :8003)"]
    Browser -->|Optional direct / peer proxy| LaneB["Container h3b (Lane B, GPU 1, :8004)"]

    LaneA <-->|h3net /api/health, /api/logs| LaneB

    LaneA -->|Atomic O_CREAT|O_EXCL claim| Out[("Shared /out/jobs + /out/logs")]
    LaneB -->|Atomic O_CREAT|O_EXCL claim| Out

    Titler["h3-titler (--network host)"] -->|Poll /api/jobs every 20s| LaneA
    Titler -->|Chat completions :8001| LLM["Local LLM (llama.cpp)"]
    LaneA -->|POST /api/prompt/polish| LLM
```

### Key Components
- **`server.py`** — FastAPI web server + background daemon worker thread (`uvicorn server:app --host 0.0.0.0 --port 8003`). Serves `index.html` at `/`, exposes `/api/*` and `/files/*`, and runs the single-GPU inference loop in a background thread.
- **`index.html`** — Single-file vanilla HTML/CSS/JS studio UI with no build step. Reads all engine limits, frame grids, step presets, and aspect rules dynamically from `GET /api/engine`.
- **`titler.py`** — Optional host-network sidecar (`h3-titler`) that polls `/api/jobs` every 20 seconds and asks a local OpenAI-compatible LLM (`H3_LLM_URL`) to generate concise `<= 6`-word lowercase titles for untitled jobs. It runs with `--network host` because container firewall rules on the host block bridge containers from reaching host-bound ports like `:8001`.
- **`healthcheck.py`** — Docker `HEALTHCHECK` probe that checks `GET /api/health` and returns healthy only once `model == "ready"`.

---

## 2. Authentication & API Access

All `/api/*` and `/files/*` endpoints (except `GET /` and `GET /api/health` / `GET /api/engine`) are gated by `_key_ok()` in `server.py`:
- Accepts either the `x-api-key` request header, a `?key=` query parameter (useful for `<video>`/`<img>` tags), or an `h3_session` cookie issued by `POST /api/session`.
- `POST /api/lanes` / `GET /api/lanes` aggregates the local worker's `/api/health` with peer URLs listed in `H3_PEERS` (e.g., `http://h3b:8003`).

---

## 3. Model Loading & 32 GB VRAM Memory Patches

MiniMax-H3 is a dual-transformer pipeline (`transformer` for `t2va`/`fl2va` and `transformer_ref` for `ref2va`, plus `text_encoder`, `vae`, and `audio_vae`). On 32 GB GPUs (RTX 5090), loading and running 15–30 second clips requires several runtime patches applied inside `load()` in `server.py`:

1. **Lifting the 15 s Duration Cap**
   - Diffusers' `MiniMaxH3Pipeline` exposes `max_duration` as a read-only property returning `15.0`.
   - `server.py` monkey-patches `type(pipe).max_duration` to return `MAX_DURATION` (`H3_MAX_DURATION`, default `30.0` seconds), extending the legal `17n + 5` frame grid from `124..345` up to `124..719` frames (`5.2 s` to `30.0 s` at `24 fps`).
2. **CPU Block Offloading & Offload-Copy Patch**
   - `pipe.enable_model_cpu_offload()` keeps inactive modules in host RAM (~130 GB across two lanes) and streams blocks onto the GPU as needed.
   - `_patch_offload_copies()` wraps `NonBlockingGroupOffloadingHook` / `cpu_offload` to avoid unnecessary host-to-device allocation spikes.
3. **Chunked Attention, FFN, and AdaLN (`_patch_transformer_memory`)**
   - Patches `transformer` and `transformer_ref` blocks so:
     - **Self-attention** slices queries along the sequence dimension using `H3_ATTN_CHUNK` (default `32768`), keeping peak attention memory bounded on long clips.
     - **Feed-forward (FFN), RoPE, and final `norm_out` / `proj_out`** slice along the sequence dimension using `H3_FFN_CHUNK` (default `16384`).
4. **VAE CPU Offloading**
   - Video and audio VAEs are kept offloaded to CPU between decode/encode steps.
5. **Automatic OOM Retry**
   - Inside `run()`, if `torch.OutOfMemoryError` occurs during `pipe(...)`:
     - `_release_pipe_vram(pipe)` forces all hook groups and VAEs back to CPU, runs `gc.collect()` and `torch.cuda.empty_cache()`.
     - Chunk sizes are temporarily tightened to `ffn = min(FFN_CHUNK, 4096)` and `attn = min(ATTN_CHUNK, 8192)`.
     - The exact same `torch.Generator().manual_seed(job["seed"])` is recreated and `pipe(...)` is retried once.

---

## 4. Job Submission & Workflow Routing

`POST /api/jobs` accepts `multipart/form-data` and infers the workflow automatically from the uploaded files:

| Workflow | Trigger Condition | Pipeline Weights |
|---|---|---|
| `t2va` | Prompt only (no keyframe images or references) | `transformer` |
| `fl2va` | `image` (first frame) and/or `last_image` (last frame) | `transformer` |
| `ref2va` | `references[]` (images, videos, audio; up to 9 images, 3 videos, 3 audio, 12 total; audio cannot be the sole reference) | `transformer_ref` |

> **Important constraint:** `fl2va` and `ref2va` use separate transformer weights (`transformer` vs `transformer_ref`) and cannot be combined in a single job. To anchor a character/scene from a previous clip's last frame while using other references, attach that frame inside `references[]` with the `Keyframe / Anchor` role (`<Picture N>`).

### Frame Grid & Duration
- MiniMax-H3's 3D VAE requires frames on the `17 * n + 5` grid (minimum `124` frames = `5.17 s` at `24 fps`).
- `server.py` builds `FRAMES` dynamically from `124` up to `MAX_DURATION * 24` (`719` frames = `29.96 s` at `30.0 s` max).
- Legacy or out-of-grid frame counts are snapped by `legal_frames()`.

### Reference Trimming & Tagging
- Uploaded reference metadata is parsed by `_parse_ref_configs()` (supporting per-reference `start_sec`, `end_sec`, `keep_video_audio`, `fps_override`, `sample_rate_override`, and UI role).
- `parse_prompt_tags()` validates `<Picture N>`, `<Video N>`, `<Audio N>`, `<Subject N>`, and `<d>...</d>` dialogue tags against the attached references and records non-fatal `tag_warnings` if counts or types mismatch.
- `_validate_ref_duration()` rejects reference videos/audio longer than `MAX_DURATION + 0.5 s` (unless trimmed via `start_sec`/`end_sec`).

---

## 5. Multi-Lane Queue & Atomic Claiming

Both Lane A (`h3`) and Lane B (`h3b`) watch the same `/out/jobs/` directory:

1. Each job lives in `/out/jobs/<job_id>/job.json` along with its uploaded inputs and outputs (`output.mp4`, `poster.jpg`, optional `output_2k.mp4`).
2. `next_job()` scans `status == "queued"` jobs sorted oldest-first by `job.json` mtime:
   - Checks `PARTITION` (`both`, `base` for `t2va`/`fl2va` only, `ref` for `ref2va` only) and `job["pin"]` (optional lane affinity).
   - Attempts to create `/out/jobs/<job_id>/claim` using `os.open(..., os.O_CREAT | os.O_EXCL | os.O_WRONLY)`. Only one container can win this atomic lock.
   - Stale `claim` files older than `90 s` (from a crashed worker) are automatically cleared.
3. On worker startup, `server.py` scans `/out/jobs/*/job.json` and marks any job left in `status == "running"` **for its own `LANE`** as `error` (`"worker restarted during this job"`) without touching jobs actively running on the peer lane.

---

## 6. Inference, Audio Leveling & ETA Model

Inside `run(job)`:
1. **Inputs & Canvas** — Loads `image`/`last_image` and/or `references[]` (slicing video frames at native container fps before applying `fps_override`, and slicing audio via `_slice_audio()`). `_resolve_canvas()` computes `target_width` / `target_height` on a `768p` short-edge grid (snapped to multiples of 32, capped at `768 * 1344` pixels).
2. **Denoising Loop** — Diffusers invokes `_tick(module, args)` on every transformer step, updating `state` (`step`, `total`, `ticks`) and checking `job["cancel"]` or `job["pause"]` flags so jobs can be paused or cancelled mid-render.
3. **Audio Analysis & Muxing** — Raw generated stereo `32 kHz` audio is analyzed (`peak_db`, `rms_db`) and leveled according to `audio_mode` (`raw`, `peak` to `-1 dBFS`, or `loud` up to `+24 dB` boost), then muxed with the `24 fps` video via `encode_video()` to `output.mp4`.
4. **Metadata & Cost Model** — `poster()` extracts `poster.jpg`, `probe_video()` records media dimensions/duration/audio sample rate, and `record(job)` appends timing data to `/out/history.jsonl`. `fit_cost()` fits a linear cost model over recent completed jobs to power `estimate(frames, steps)` ETAs in `/api/engine` and the UI.

---

## 7. Post-Processing, Downloads & Logs

- **2K Upscaling (`POST /api/jobs/{jid}/upscale`)**
  - Uses PyAV (`av`) to decode `output.mp4` frame-by-frame, resizes each frame with Pillow `Image.Resampling.LANCZOS` so the short edge is `1440 px` (even dimensions), re-encodes H.264 (`libx264`, `yuv420p`, `crf=16`) while copying the original audio stream into `output_2k.mp4`.
  - Sets `job["upscaled_2k"] = "output_2k.mp4"` and `job["media_2k"]` in `job.json`.
  - Note: `ffmpeg` CLI is not installed in the container image; all video probing, poster extraction, and upscaling use PyAV + Pillow directly in Python.
- **Title-Based Downloads**
  - In `index.html`, download links sanitize `job.title` (falling back to `job.name` or `h3-job-<num>`) for the `download="<title>.mp4"` attribute and pass `?dl=1`.
  - In `GET /files/{jid}/{name}?dl=1`, `server.py` also sets `Content-Disposition: attachment; filename="..."` using the sanitized job title so direct/cross-origin downloads save with the video's title instead of `output.mp4`.
- **Persistent Worker Logs (`GET /api/logs`)**
  - `_TeeStream` tees stdout/stderr to `/out/logs/h3-{lane}.log` (rotating `5 MB` files, `3` backups). The UI's **Worker logs** modal can inspect either lane's live tail.

---

## 8. Frontend Studio (`index.html`)

- **Dynamic Engine Sync** — Fetches `/api/engine` on boot so duration sliders, frame counts, step presets (`Draft 8`, `Std 12`, `Fine 25`, `Ultra 36`, `Max 50`), aspect ratios, and reference rules always match the backend.
- **Dark-Cinema Glassmorphism UI** — Translucent layered surfaces (`backdrop-filter: blur`), subtle specular borders, and mobile-friendly responsive layout.
- **Prompt Builder & Presets** — Includes 3-part Camera Grammar (`shot size + angle + movement`), Multi-Speaker Dialogue `(S1, S2...)` `<d>...</d>` tagging, a **Repair** helper (`repairPromptText`), and **21 Use-Case Starters (`3.1`–`3.3.4.4`)** served locally from `/out/demos/` via `GET /api/demos/{name}`.
- **Multi-Cast Presets & `<CastName>` Shorthand** — Stores reusable character/voice reference sets in browser `IndexedDB` and shared server JSON (`/out/casts/*.json` via `GET /api/casts`), auto-assigns distinct `<Subject N>` groups per Cast, expands `<CastName>` shorthand tags on submit/save, injects background-free facial/body `retention_analysis`, and runs a pre-flight GPU VRAM fit estimator (`estimateSingleJobVram()`) before queuing single jobs.
- **Series Studio & Dedicated Queue Tab (`?tab=series`)** — Automates multi-episode series stored in `/out/series/` with atomic dual-lane locks (`<sid>.lock`), PyAV `last_frame.png` extraction (`_extract_last_frame`), `qwen3.8-27b` (`H3_VLM_URL`) vision-grounded JIT continuity prompts, per-episode `ref_images` (`ref2va`) or `fl2va` chaining, **Autopilot / Review Gate**, **Mid-Series Live Pivot**, and **Selected-Videos Stitching** (`POST /api/stitch` via `_stitch_mp4s()`).

---

## 9. Deployment & Live Updates

Containers are launched by `scripts/launch-lane.sh`, `scripts/launch-both.sh`, and `scripts/launch-titler.sh` with `server.py` and `index.html` bind-mounted read-only (`:ro`) into `/app/`:
- **`index.html` edits** take effect immediately on browser refresh.
- **`server.py` edits** require recreating the container (`bash scripts/launch-both.sh`). Avoid `docker restart` because tearing down ~130 GB of pinned host RAM exceeds Docker's default stop timeout; `launch-lane.sh` retries `docker rm -f` cleanly until the container is gone.
