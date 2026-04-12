# Beta Release Notes

We describe some changes in the codebase and improvement techniques below
---

## Overview

In the original demo the user had to wait for the entire pipeline
(motion generation → mesh rendering → avatar rendering) to complete
before any video appeared.  On a 30-second clip this could mean
60–90 seconds of a blank screen.

The beta version streams both the FLAME mesh and the photorealistic
avatar frame-by-frame using the browser's **Media Source Extensions
(MSE)** API, so playback begins within a second of inference starting.

---

## 1. Streaming Architecture

### Fragmented MP4 (fMP4) in Memory

Frames are encoded into a **fragmented MP4** stream held entirely in a
`io.BytesIO` buffer (class `FMP4StreamWriter`).

Key encoder flags:

| FFmpeg flag | Effect |
|---|---|
| `frag_keyframe` | Every frame starts a new fragment |
| `empty_moov` | Write an empty `moov` box upfront (no back-seeking needed) |
| `default_base_moof` | Compatible `moof` layout for incremental append |
| `g=1` | Every frame is a keyframe → every frame is independently decodable |

The `get_new_b64()` method returns only the bytes written since the
previous call, so each yield to the browser carries exactly the new
data accumulated since the last poll.

### Data Channel: Python → JavaScript

Gradio does not provide a built-in push channel for binary data.
Two hidden `gr.Textbox` components (`elem_id="mesh_seg"` and
`elem_id="avatar_seg"`) serve as the data channel:

1. Python base64-encodes the new fMP4 bytes and yields them as the
   textbox value.
2. A JavaScript `setInterval` loop polls the textboxes every **150 ms**.
3. When a new value is detected, JS decodes it and appends the bytes to
   the MSE `SourceBuffer`.

> **Gradio 5.x note:** `visible=False` uses Svelte's `{#if}` gate and
> removes the element from the DOM entirely, making `getElementById`
> return `null`.  The textboxes are therefore kept `visible=True` and
> hidden with `display:none` via an injected `<style>` tag.

### MSE Player (JavaScript)

Each `<video>` element has a dedicated MSE player created by
`createMSEPlayer()`:

- **Lazy initialization** — `MediaSource` is created only when the
  first fMP4 data arrives, never on page load.  This prevents Chrome's
  idle-MediaSource timeout (~30–60 s) from killing the avatar player
  before avatar frames are ready.
- **Automatic stream detection** — a new stream is detected by checking
  bytes 4–7 of the incoming data for the `ftyp` box magic bytes
  (`0x66 0x74 0x79 0x70`).  No explicit `RESET` signal is needed, and
  there is no risk of JS missing a reset if the poll fires late.
- **Append queue + drain loop** — incoming segments are queued and
  drained one at a time via the `updateend` event, preventing
  `QuotaExceededError` from concurrent appends.
- **Codec fallback list** — four AVC descriptor strings are tried in
  order (`avc1.42C01F` → `avc1.42E01E` → `avc1.640028` →
  `avc1.4D401F`) for maximum browser compatibility.

---

## 2. Mesh Rendering Latency Improvements

### Batched GPU Producer + Consumer Thread

The original code rendered one frame at a time.  The beta runs a
dedicated GPU producer thread that calls PyTorch3D in batches of
`MESH_RENDER_BATCH = 64` frames per call.  Results are pushed into a
bounded `queue.Queue(maxsize=30)`.  The main thread acts as a consumer:
it pulls frames, writes them to both the final MP4 and the fMP4
preview stream, and yields to Gradio.

This overlap of GPU rendering and CPU encoding eliminates the GPU
idle time that occurred when encoding blocked rendering in the
original single-threaded loop.

### Buffer Phase + Rate-Limited Yields

| Constant | Value | Purpose |
|---|---|---|
| `MESH_BUFFER_FRAMES` | 25 | Frames to accumulate silently before first yield |
| `MESH_YIELD_INTERVAL_S` | 0.20 s | Minimum time between successive yields |

**Phase 1 (buffering):** The consumer accumulates 25 frames (~1 second
of video at 25 fps) before sending anything to the browser.  This
ensures the MSE buffer is pre-filled so playback starts without
stalling on the first decode.

**Phase 2 (streaming):** After the buffer phase, yields are
rate-limited to one per 200 ms.  Because the JS poll runs at 150 ms,
every Python yield is guaranteed to be observed by the poller.  Each
yield carries all fMP4 bytes written since the previous yield — never
just one frame's worth — so chunk sizes grow proportionally with
render speed.

---

## 3. Avatar Rendering Latency Improvements

### fps Declaration Matches Render Speed

The avatar fMP4 stream is declared at `AVATAR_FMP4_FPS = 5 fps` instead
of the original 20 fps.

With fps=20 declared but the avatar rendering at ~5 fps, each frame
has a declared duration of 50 ms.  After the initial buffer plays out
(~1.25 s), each subsequent yield contains only ~1 frame = 50 ms of
video.  The browser exhausts that in 50 ms, then stalls for the next
150 ms yield — producing visible freezes.

With fps=5, each frame has a declared duration of 200 ms, so a
10-frame chunk represents 2 seconds of video.  The browser always has
significantly more buffered video than the time until the next yield.

### Frame-Count-Based Yields

| Constant | Value | Purpose |
|---|---|---|
| `AVATAR_FMP4_FPS` | 5 | Declared fps of the avatar preview stream |
| `AVATAR_YIELD_EVERY` | 10 | Frames between successive avatar yields |

Rather than time-based polling, the avatar loop yields every
`AVATAR_YIELD_EVERY = 10` frames.  At a declared fps of 5 this
represents 2 seconds of video per chunk, which comfortably exceeds the
200 ms delivery interval regardless of actual GPU render speed
fluctuations.

---

## 4. Memory Efficiency

### Incremental Video Writer

The `IncrementalVideoWriter` class encodes frames one at a time directly
to disk via PyAV, replacing the pattern of collecting all rendered
frames into a large tensor in RAM and then calling `write_video()`.

For a 30-second clip at 512×512 this reduces peak RAM usage from
~4 GB (all frames as float32) to a constant small footprint regardless
of clip length.

---

## 5. Warmup

A `warmup()` function runs `N` (default 3) full inference passes before
the Gradio server starts accepting requests.  This triggers PyTorch's
JIT compilation, CUDA kernel caching, and cuDNN algorithm selection
upfront, so the first user request has the same latency as subsequent
ones.

---

## 6. Running the Beta Demo

```bash
cd src/
python gradio_demo_streaming.py
```

The interface launches on `http://0.0.0.0:8961` (port 8961, one above
the original demo).

Optional flags:

```
--clip_length  -l   Maximum frames to render (default 750 = 30 s)
--warmup_steps -w   Warmup iterations before launch (default 3, 0 to skip)
```

## 7. Improvements
1.  W/o improve mesh rendering: 3.5 FPS (whole pipeline) with I/O to web demo
2. W/ improve mesh rendering: 9.5 FPS (whole pipeline) with I/O to web demo
2. W/ improve mesh rendering + I/O: ~11.3 FPS (whole pipeline)

We benchmark on A100 40GB, but we found that because of the problem of CPU and share resources with other project's resources, the result will be faster if we can use single one. We will update the results