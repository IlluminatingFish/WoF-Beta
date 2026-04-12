# WoF-Beta

Beta version — speech-driven 3D head animation (ARTalk + GAGAvatar stack).

## Layout

| Entry | Role |
|--------|------|
| `src/inference.py` | Core engine (`ARTAvatarInferEngine`), CLI batch inference, launcher for UIs |
| `src/gradio_app.py` | **Gradio** demo (optional; heavier I/O stack) |
| `src/flask_app.py` | **Flask**: **Offline** = chunked MP4 after generate; **Online** = **GPU producer + JPEG consumer** queue, MJPEG (`/session` + `/online_mjpeg/<token>`), same final MP4 |
| `src/stream_utils.py` | Preview tensor → uint8 / JPEG + **streaming timing logs** |
| `src/stream_render.py` | **`iter_render_frames`**; used by Gradio/Flask; `inference.rendering()` pipes frames into **`IncrementalVideoWriter`** (incremental H.264, no full `stack`) |

## Setup

From `src/` (after conda env + `build_resources.sh` per original project):

```bash
pip install flask
```

(Gradio and other deps remain in `environment.yml`.)

## Run — CLI (unchanged)

```bash
cd src
python inference.py --audio_path ./demo/your.wav --shape_id mesh --style_id natural_0
```

## Run — Gradio

```bash
cd src
python gradio_app.py                    # or: python inference.py --run_app
python gradio_app.py --online           # default render mode → online
python inference.py --run_app --online
```

Gradio listens on **8960** (see `gradio_app.py`).

## Run — Flask (recommended for demos / custom I/O)

```bash
cd src
python flask_app.py                     # or: python inference.py --run_app --flask
python inference.py --run_app --flask --flask_port 8961
```

Open **http://127.0.0.1:8961/** (or host:port shown in the terminal).

- **`POST /api/offline`** — Same pipeline as online; **body** = **chunked** **MP4** (`transfer_chunk_kb`, no sleep between chunks). JSON metadata (`timing`, URLs, `transfer` info) in **`X-WoF-Meta`**. Log: **`[offline_mp4_stream]`**.
- **`POST /api/session`** then **`GET /api/online_mjpeg/<token>`** — After motion, a **producer thread** runs **`iter_render_frames`** (GPU) and pushes frames into a bounded queue; the response generator **JPEG-encodes and yields** MJPEG on the main thread so **JPEG/network can overlap the next frame’s GPU work** (`timing.transfer.mjpeg_overlap_estimate_s`, `mjpeg_parallel_phase_wall_s`). Then stack / `write_video` / save `.pt` as offline. Demo **drains** the stream with `fetch` (no preview). **`GET /api/session/<token>/status`** returns **`timing`** when done. Log: **`[online_mjpeg]`**.

CLI batch (`python inference.py --audio_path ...`) prints the same compute breakdown under **`=== timing (seconds) ===`**.

## Gradio streaming logs (optional)

If you use **Gradio** with **Render mode → online**, the server prints `[gradio_online]` per-frame lines from `stream_utils.StreamTimingLog` (separate from Flask offline/online comparison).

## Docker note

The bundled `Dockerfile` still runs `python inference.py --run_app` (Gradio). To use Flask inside the container, override the command, e.g.:

```bash
python inference.py --run_app --flask
```
