# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

One person for now: the server owner, testing MiniMax H3 on the Inwin GPU server. They judge output quality and speed before deciding what the model is for. They use it on a desktop and check results on a phone.

## Product Purpose

H3 Studio submits video-with-audio generation jobs to MiniMax H3 and shows the results. Submitting a good job and reviewing results are equal halves of the job. Success: a prompt plus optional inputs goes in with little friction, and every finished clip can be found, played, compared with the prompt and inputs that made it, and downloaded.

## Positioning

It runs on our own server (Inwin, 4x RTX 5090, model on local disk), not a cloud API. The page must make that visible.

## Operating Context

- Reached on the LAN (http://192.168.103.31:8003/) or Tailscale; behind an API key the user pastes once.
- One GPU, one job at a time. A clip takes minutes; jobs wait in a queue. The model takes about 5 minutes to load after a restart, and the page must show that state.
- Three workflows, picked from the inputs: t2va (text only), fl2va (first and/or last frame image), ref2va (up to 9 images, 3 videos, 3 audio clips, 12 total; audio cannot be the only reference; not combined with first/last frame).
- Output: 24 fps MP4 with stereo audio, 768p short edge, 124 to 345 frames (5.2 to 14.4 s).

## Capabilities and Constraints

- Single file `index.html`, vanilla JS, no build step, served by the FastAPI app in `server.py` (container `h3` on Inwin, source in `~/h3/`).
- API contract (must not change): `GET /api/health` -> {model, queued}; `GET /api/jobs` -> list newest first; `GET /api/jobs/{id}`; `POST /api/jobs` multipart (prompt, frames, seed, steps, image, last_image, references[]); files at `/files/{id}/{name}?key=`. Auth header `x-api-key`.
- Job fields: id, prompt, frames, seed, steps, inputs {image, last_image, references[{kind, name, file}]}, workflow, status (queued | running | done | error), created, started, finished, error, output.
- No progress percentage from the server today; only status and timestamps.
- No external network assumptions beyond the LAN; any font or library must load from an allowed CDN or be omitted.

## Brand Commitments

Name: H3 Studio. Model: MiniMax H3 (licence restricts use outside the listed regions; Malaysia is fine). No VSTECS branding required on this page.

## Evidence on Hand

Real generated clips only (under /data/stage/h3-out/jobs). No benchmarks, sample galleries or claims beyond what jobs record (duration, seed, run time).

## Product Principles

1. Results are the hero once they exist; the form never hides them.
2. Show machine truth: model state, queue, run time, the local GPU.
3. Every clip stays tied to the exact prompt and inputs that made it.
4. Works one-handed on a phone for review.
