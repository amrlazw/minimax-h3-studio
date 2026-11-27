# H3 Studio

Self-hosted, single- or dual-GPU web studio for **MiniMax-H3** video + stereo audio generation (`t2va`, `fl2va`, `ref2va`), built with FastAPI and a zero-build vanilla HTML/JS frontend.

## Features

- **Three workflows** inferred automatically from inputs:
  - `t2va` — text to video + audio
  - `fl2va` — first and/or last keyframe image to video + audio
  - `ref2va` — multi-reference generation (up to 9 images, 3 videos, 3 audio clips, 12 total) with `<Picture N>`, `<Video N>`, `<Audio N>`, `<Subject N>`, and `<d>...</d>` dialogue linking
- **Dual-GPU lane architecture** — run one worker per GPU sharing a single `/out` job directory via atomic file-lock claiming
- **Extended clip lengths** — supports the `17n + 5` VAE frame grid up to `H3_MAX_DURATION` (default `30.0 s` / `719` frames at `24 fps`) by lifting the pipeline's native `15 s` property cap
- **32 GB VRAM memory patch** — query-chunked self-attention (`H3_ATTN_CHUNK`), chunked FFN/AdaLN (`H3_FFN_CHUNK`), VAE CPU offloading, and automatic OOM weight release + smaller-chunk retry
- **Dynamic UI configuration** — `index.html` reads limits, frame grid, steps, aspects, and reference rules live from `/api/engine`
- **Optional LLM & VLM sidecars** — prompt polish (`H3_LLM_URL`), automatic short job titles (`titler.py`), and vision-grounded series continuity (`H3_VLM_URL`, e.g., `qwen3.8-27b`)
- **Series Studio & Queue Tab (`?tab=series`)** — automate multi-episode viral shorts with locked settings, PyAV `last_frame.png` `fl2va` chaining (or per-episode `ref_images` `ref2va`), Autopilot/Review Gate, Mid-Series Pivot, and Commercial/Use-Case Starters with local `/api/demos/` video previews
- **Selected-Video Stitching** — concatenate any finished series episodes or multi-selected clips from the bin via PyAV (`POST /api/stitch`)
- **Cast Presets & Pre-Flight VRAM Guard** — reusable local & shared Casts (`GET /api/casts`) with `<CastName>` shorthand expansion, background-free facial/body `retention_analysis`, and dynamic pre-flight GPU VRAM fit checks before single-job submission

---

## Requirements

- Linux host with **Docker** and **NVIDIA Container Toolkit**
- NVIDIA GPU(s) with ≥ 32 GB VRAM (tested on RTX 5090) and sufficient host RAM for CPU block offloading (~130 GB for dual-lane)
- Local copy of [`MiniMaxAI/MiniMax-H3`](https://huggingface.co/MiniMaxAI/MiniMax-H3) weights

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/amrlazw/minimax-h3-studio.git
cd minimax-h3-studio
cp .env.example .env
# Edit .env and set H3_API_KEY (and paths to your model weights & output dir)
```

### 2. Build the Docker image

```bash
docker build -t h3-diffusers .
```

### 3. Launch

**Single GPU (Lane A on GPU 0, port 8003):**

```bash
LANE=a GPU=0 PORT=8003 bash scripts/launch-lane.sh
```

**Dual GPU (Lane A on GPU 0 `:8003`, Lane B on GPU 1 `:8004` + Titler):**

```bash
bash scripts/launch-both.sh
```

Open `http://<server-ip>:8003/` in your browser and paste your `H3_API_KEY` once to connect.

---

## Configuration (Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `H3_API_KEY` | *(required)* | Shared API key required for `/api/*` and `/files/*` |
| `H3_LANE` | `a` | Worker lane identifier (`a`, `b`, ...) |
| `H3_GPU` | `0` | GPU index reported in `/api/health` and UI |
| `H3_PEERS` | `""` | Comma-separated peer lane URLs (e.g. `http://h3b:8003`) |
| `H3_PARTITION` | `both` | Model partition to load: `both`, `base`, or `ref` |
| `H3_MAX_DURATION` | `30.0` | Maximum clip & reference video duration in seconds |
| `H3_FFN_CHUNK` | `16384` | Sequence chunk size for FFN & row-wise norm/AdaLN |
| `H3_ATTN_CHUNK` | `32768` | Query chunk size for self-attention |
| `H3_VRAM_FRACTION` | `0` | Optional `torch.cuda.set_per_process_memory_fraction` (`0` = disabled) |
| `H3_LLM_URL` | `""` | Optional OpenAI-compatible endpoint for prompt polish |
| `H3_LLM_MODEL` | `qwen3.8-neo-coder` | Model name for prompt polish |
| `H3_LLM_KEY` | `""` | API key for the polish LLM endpoint |
| `H3_VLM_URL` | `""` | Optional Vision-Language Model endpoint for Series `last_frame.png` continuity |
| `H3_VLM_MODEL` | `qwen3.8-27b` | Vision model name for Series JIT continuity |
| `H3_VLM_KEY` | `""` | API key for `H3_VLM_URL` |

### Launch script overrides

`scripts/launch-lane.sh` and `scripts/launch-both.sh` also accept:
- `MODEL_DIR` — path to local `MiniMax-H3` weights (default: `/data/stage/minimax-h3`)
- `OUT_DIR` — shared persistent output/jobs directory (default: `/data/stage/h3-out`)
- `HFCACHE_DIR` — Hugging Face cache directory (default: `./hfcache`)
- `BIND_HOST` — host IP to bind ports on (default: `0.0.0.0`)

---

## API Overview

All `/api/*` endpoints (except `/api/health` and `/api/engine`) require either the `x-api-key` header or an authenticated session cookie (`POST /api/session`).

- `GET /api/health` — worker model readiness (`loading` / `ready`), queue count, GPU usage
- `GET /api/engine` — dynamic limits, frame grid, steps, aspects, ref edges, ETA cost model
- `GET /api/lanes` — aggregated status across this worker and `H3_PEERS`
- `GET /api/jobs` / `GET /api/jobs/{id}` — list or inspect jobs
- `POST /api/jobs` — queue a new generation job (`multipart/form-data`)
- `POST /api/jobs/{id}/cancel` / `retry` / `DELETE /api/jobs/{id}` — manage jobs
- `POST /api/probe` — inspect uploaded image/video/audio metadata before submitting
- `POST /api/prompt/polish` — polish prompt via configured `H3_LLM_URL`
- `GET/POST/PATCH/DELETE /api/series*` & `POST /api/series/plan` / `replan` / `start` / `approve` / `upload_ref` — manage automated multi-episode series
- `POST /api/stitch` & `GET /api/stitches/{id}/file` — concatenate selected finished clips into one MP4
- `GET /api/casts` / `GET /api/casts/file/{name}` & `GET /api/demos/{name}` — shared Cast presets and cached official demo videos

---

## Project Layout

```
├── server.py          # FastAPI app, worker loop, memory patches, job store
├── index.html         # Single-file studio UI (vanilla HTML/CSS/JS)
├── titler.py          # Optional sidecar that auto-titles completed/queued jobs
├── healthcheck.py     # Docker HEALTHCHECK script
├── Dockerfile         # Container build definition
├── guides/            # Prompting guides (base_en.md, ref_en.md)
├── scripts/           # Portable Docker launch scripts
├── PRODUCT.md         # Product brief & design constraints
└── .env.example       # Environment variable template
```
