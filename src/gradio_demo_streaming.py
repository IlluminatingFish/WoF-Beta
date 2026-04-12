#!/usr/bin/env python
"""
Streaming version of the Gradio demo with live frame preview.

Architecture:
  - Mesh rendering uses a batched GPU producer thread + CPU consumer.
  - Avatar rendering uses a simple sequential loop (no threading).
  - Preview uses Media Source Extensions (MSE): frames are encoded into
    a fragmented-MP4 stream in memory, and base64 segments are pushed
    to the browser where JavaScript appends them to a SourceBuffer.
    The <video> element never changes its source, so there is ZERO
    black-frame flash.
"""

import os
import sys
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import av
import io
import time
import queue
import base64
import torch
import argparse
import threading
import torchaudio
import numpy as np
import gradio as gr
from gtts import gTTS

from gradio_demo import ARTAvatarInferEngine

# How many mesh frames to render in one batched PyTorch3D call.
MESH_RENDER_BATCH = 64
# How many frames the GPU producer can render ahead of the consumer.
QUEUE_MAXSIZE = 30

# ── Mesh streaming ───────────────────────────────────────────
# Frames to buffer silently before the first mesh preview yield.
# Mesh renders at 50+ fps so 25 frames ≈ 0.5 s of head-start.
MESH_BUFFER_FRAMES = 25
# Minimum seconds between successive mesh yields (JS polls at 150 ms).
MESH_YIELD_INTERVAL_S = 0.20

# ── Avatar streaming ─────────────────────────────────────────
# Avatar typically renders at 3–10 fps.  We declare the preview stream
# at 5 fps so each decoded frame has 200 ms duration.  When render
# rate ≥ 5 fps the video plays with zero stalls; slower renders stall
# proportionally.  (Mesh preview stays at 20 fps because it renders fast.)
AVATAR_FMP4_FPS = 5
# Yield a new fMP4 chunk every N avatar frames.  At fps=5 this produces
# N/5 seconds of video per yield, matching the N/fps_render render time.
AVATAR_YIELD_EVERY = 10


# ── Helpers ──────────────────────────────────────────────────


class FMP4StreamWriter:
    """Writes H264 frames into a fragmented-MP4 stream in memory.

    Each frame is its own fragment (GOP=1), so ``get_new_bytes()``
    returns complete, appendable fMP4 data after every write.  The
    first call returns the init segment (ftyp+moov) plus the first
    media segment; subsequent calls return only new media segments.
    """

    def __init__(self, width=512, height=512, fps=25):
        self._buf = io.BytesIO()
        self._container = av.open(
            self._buf, mode="w", format="mp4",
            options={
                "movflags": "frag_keyframe+empty_moov+default_base_moof",
            },
        )
        self._stream = self._container.add_stream("h264", rate=int(fps))
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "crf": "23",
            "g": "1",           # every frame is a keyframe → own fragment
            "profile": "baseline",
            "level": "3.1",
        }
        self._last_pos = 0
        self.n_frames = 0

    def write_frame(self, frame_chw_uint8):
        """Encode one (C,H,W) uint8 numpy frame."""
        hwc = frame_chw_uint8.transpose(1, 2, 0)
        vf = av.VideoFrame.from_ndarray(hwc, format="rgb24")
        for pkt in self._stream.encode(vf):
            self._container.mux(pkt)
        self.n_frames += 1

    def get_new_bytes(self):
        """Return all bytes written since the last call."""
        end = self._buf.seek(0, 2)
        if end <= self._last_pos:
            return b""
        self._buf.seek(self._last_pos)
        data = self._buf.read(end - self._last_pos)
        self._last_pos = end
        return data

    def get_new_b64(self):
        """Return new bytes as a base64 string (for sending to JS)."""
        raw = self.get_new_bytes()
        if not raw:
            return ""
        return base64.b64encode(raw).decode("ascii")

    def close(self):
        for pkt in self._stream.encode():
            self._container.mux(pkt)
        self._container.close()


class IncrementalVideoWriter:
    """Encodes video frames one-by-one via PyAV — no frame list in RAM."""

    def __init__(self, output_path, fps, sample_rate=None):
        self.output_path = output_path
        self.fps = fps
        self.sample_rate = sample_rate
        self._container = None
        self._video_stream = None
        self._audio_stream = None
        self.n_frames = 0

    def _init_container(self, height, width):
        self._container = av.open(self.output_path, mode="w")
        self._video_stream = self._container.add_stream(
            "h264", rate=int(self.fps)
        )
        self._video_stream.width = width
        self._video_stream.height = height
        self._video_stream.pix_fmt = "yuv420p"
        self._video_stream.options = {"crf": "18", "preset": "ultrafast"}
        if self.sample_rate is not None:
            self._audio_stream = self._container.add_stream(
                "aac", rate=self.sample_rate
            )
            self._audio_stream.format = "fltp"

    def write_frame(self, frame_chw_uint8):
        if self._container is None:
            _, h, w = frame_chw_uint8.shape
            self._init_container(h, w)
        frame_hwc = frame_chw_uint8.transpose(1, 2, 0)
        vf = av.VideoFrame.from_ndarray(frame_hwc, format="rgb24")
        for packet in self._video_stream.encode(vf):
            self._container.mux(packet)
        self.n_frames += 1

    def finalize(self, audio_samples=None):
        if self._container is None:
            return
        for packet in self._video_stream.encode():
            self._container.mux(packet)
        if audio_samples is not None and self._audio_stream is not None:
            if isinstance(audio_samples, torch.Tensor):
                audio_samples = audio_samples.cpu().numpy()
            n_per_frame = int(self.sample_rate // self.fps)
            for i in range(0, audio_samples.shape[0], n_per_frame):
                chunk = audio_samples[i : i + n_per_frame]
                if chunk.shape[0] < n_per_frame:
                    chunk = np.pad(
                        chunk, (0, n_per_frame - chunk.shape[0]),
                        mode="constant",
                    )
                af = av.AudioFrame.from_ndarray(
                    chunk[None], format="fltp", layout="mono"
                )
                af.rate = self.sample_rate
                for packet in self._audio_stream.encode(af):
                    self._container.mux(packet)
            for packet in self._audio_stream.encode():
                self._container.mux(packet)
        self._container.close()


# ── Warmup ───────────────────────────────────────────────────

WARMUP_AUDIO = "demo/eng1.wav"
WARMUP_AVATAR = "12.jpg"
WARMUP_STYLE = "natural_0"
WARMUP_CLIP_LENGTH = 25


def warmup(engine, steps=3):
    import torchaudio as _ta
    print(f"[warmup] Running {steps} warmup step(s) ...")
    audio, sr = _ta.load(WARMUP_AUDIO)
    audio = _ta.transforms.Resample(sr, 16000)(audio).mean(dim=0)
    engine.set_style_motion(WARMUP_STYLE)
    for step in range(1, steps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_motions, _ = engine.inference(audio, clip_length=WARMUP_CLIP_LENGTH)
        shape_code = torch.zeros(1, 300, device=engine.device).expand(
            pred_motions.shape[0], -1)
        verts = engine.generator.basic_vae.get_flame_verts(
            engine.flame_model, shape_code, pred_motions, with_global=True)
        rgb_mesh = engine.mesh_renderer(verts)[0]
        del rgb_mesh, verts, shape_code
        if hasattr(engine, "GAGAvatar"):
            avatar_ids = sorted(engine.GAGAvatar.all_gagavatar_id.keys())
            aid = WARMUP_AVATAR if WARMUP_AVATAR in avatar_ids else avatar_ids[0]
            engine.GAGAvatar.set_avatar_id(aid)
            for motion in pred_motions[:5]:
                batch = engine.GAGAvatar.build_forward_batch(
                    motion[None], engine.GAGAvatar_flame)
                rgb = engine.GAGAvatar.forward_expression(batch)
                del rgb
        torch.cuda.synchronize()
        print(f"[warmup] Step {step}/{steps} done in "
              f"{time.perf_counter() - t0:.2f}s")
    torch.cuda.empty_cache()
    print("[warmup] Complete.\n")


# ── MSE JavaScript ───────────────────────────────────────────

MSE_HTML = """
<div style="display:flex; gap:1em;">
  <div style="flex:1;">
    <div style="text-align:center; font-weight:bold; margin-bottom:4px;">FLAME Mesh</div>
    <video id="mse_mesh_video" autoplay muted playsinline
           style="width:100%; height:384px; background:#111; object-fit:contain;"></video>
  </div>
  <div style="flex:1;">
    <div style="text-align:center; font-weight:bold; margin-bottom:4px;">Photorealistic Avatar</div>
    <video id="mse_avatar_video" autoplay muted playsinline
           style="width:100%; height:384px; background:#111; object-fit:contain;"></video>
  </div>
</div>
"""

# All MSE logic lives here and is injected via demo.load(js=...).
# gr.HTML does NOT reliably execute <script> tags in all Gradio versions,
# but demo.load(js=...) is guaranteed to run.
#
# Key design decisions:
#   - No "RESET" string signal.  Instead, JavaScript detects a new fMP4
#     stream automatically by checking for the ftyp box magic bytes at the
#     start of the first data chunk.  This is robust to the JS poll
#     missing the RESET yield entirely.
#   - Python yields are rate-limited to YIELD_INTERVAL_S (> 150 ms poll),
#     so every yield is caught.  Each yield carries ALL fMP4 bytes since
#     the previous yield, never just one frame worth.
MSE_BOOT_JS = """
() => {
  console.log('[MSE] boot JS running');

  /* ── Utilities ──────────────────────────────────────────── */
  function b64ToBytes(b64) {
    var raw = atob(b64);
    var buf = new Uint8Array(raw.length);
    for (var i = 0; i < raw.length; i++) buf[i] = raw.charCodeAt(i);
    return buf;
  }

  /* fMP4 init segment starts with an 'ftyp' box:
     bytes 4-7 = 0x66 0x74 0x79 0x70 ('ftyp') */
  function isInitSegment(buf) {
    return buf.length >= 8 &&
           buf[4] === 0x66 && buf[5] === 0x74 &&
           buf[6] === 0x79 && buf[7] === 0x70;
  }

  function createMSEPlayer(videoId) {
    var video = document.getElementById(videoId);
    if (!video) { console.warn('[MSE] video not found:', videoId); return null; }

    var mediaSource = null, sourceBuffer = null, queue = [], ready = false;

    function isAlive() {
      return mediaSource && mediaSource.readyState === 'open'
             && sourceBuffer && !sourceBuffer.updating;
    }

    function init() {
      /* LAZY: called only when real data arrives, never on page load.
         This prevents Chrome from timing out an idle MediaSource
         (which would leave the video element in an unrecoverable
         error state before any avatar frames are ready). */
      ready = false; sourceBuffer = null;
      mediaSource = new MediaSource();
      /* Fully reset the video element before assigning a new source. */
      video.removeAttribute('src');
      video.load();
      video.src = URL.createObjectURL(mediaSource);

      mediaSource.addEventListener('sourceopen', function() {
        if (mediaSource.readyState !== 'open') return;
        var codecs = [
          'video/mp4; codecs="avc1.42C01F"',
          'video/mp4; codecs="avc1.42E01E"',
          'video/mp4; codecs="avc1.640028"',
          'video/mp4; codecs="avc1.4D401F"'
        ];
        for (var i = 0; i < codecs.length; i++) {
          if (MediaSource.isTypeSupported(codecs[i])) {
            sourceBuffer = mediaSource.addSourceBuffer(codecs[i]);
            console.log('[MSE] ' + videoId + ' ready, codec=' + codecs[i]);
            break;
          }
        }
        if (!sourceBuffer) { console.error('[MSE] no supported codec for ' + videoId); return; }
        sourceBuffer.addEventListener('updateend', drain);
        ready = true;
        drain();
      });
    }

    function drain() {
      if (!isAlive() || queue.length === 0) return;
      try { sourceBuffer.appendBuffer(queue.shift()); }
      catch(e) {
        console.warn('[MSE] ' + videoId + ' appendBuffer error:', e.message);
        /* Keep remaining queue items, reinit. */
        ready = false; sourceBuffer = null; mediaSource = null;
        init();
      }
    }

    function append(b64) {
      if (!b64 || b64.length === 0) return;
      var buf = b64ToBytes(b64);

      if (isInitSegment(buf)) {
        /* New fMP4 stream (ftyp box detected).
           Tear down everything, queue the init segment, reinit. */
        console.log('[MSE] ' + videoId + ' new stream (' + buf.length + ' B), reinitializing');
        queue  = [buf];
        ready  = false; sourceBuffer = null; mediaSource = null;
        init();
      } else {
        /* Continuation media segment. */
        queue.push(buf);
        if (ready && isAlive()) drain();
        /* If mediaSource is null (not yet initialized), we wait —
           an init segment must arrive before we can play. */
      }
    }

    /* NOTE: init() is NOT called here.  The player initializes itself
       the first time an init segment arrives in append(). */
    return { append: append, _video: video };
  }

  /* ── Poll hidden textboxes for segment data ─────────────────
     Python rate-limits yields to YIELD_INTERVAL_S (200 ms) which
     is > the 150 ms poll interval, so every Python yield is
     guaranteed to be seen by at least one poll cycle.            */
  var meshP = null, avatarP = null;
  var lastM = '', lastA = '';

  function findInput(eid) {
    var el = document.getElementById(eid);
    if (!el) return null;
    /* Gradio 5.x may place elem_id directly on the textarea when
       container=False; handle both that case and the wrapper-div case. */
    if (el.tagName === 'TEXTAREA' || el.tagName === 'INPUT') return el;
    return el.querySelector('textarea') || el.querySelector('input');
  }

  function poll() {
    /* (Re)create players if video elements just appeared/changed. */
    var mv = document.getElementById('mse_mesh_video');
    var av = document.getElementById('mse_avatar_video');
    if (mv && (!meshP  || meshP._video  !== mv)) { meshP   = createMSEPlayer('mse_mesh_video');   }
    if (av && (!avatarP || avatarP._video !== av)) { avatarP = createMSEPlayer('mse_avatar_video'); }

    var mIn = findInput('mesh_seg');
    var aIn = findInput('avatar_seg');
    if (!mIn) console.warn('[MSE] mesh_seg textarea not found in DOM');
    if (!aIn) console.warn('[MSE] avatar_seg textarea not found in DOM');

    if (mIn && mIn.value !== lastM) {
      var mVal = mIn.value; lastM = mVal;
      if (mVal.length > 0 && meshP) {
        console.log('[MSE] mesh segment received, len=' + mVal.length);
        meshP.append(mVal);
      }
    }
    if (aIn && aIn.value !== lastA) {
      var aVal = aIn.value; lastA = aVal;
      if (aVal.length > 0 && avatarP) {
        console.log('[MSE] avatar segment received, len=' + aVal.length);
        avatarP.append(aVal);
      }
    }
  }

  setInterval(poll, 150);
}
"""


# ── Gradio App ───────────────────────────────────────────────


def run_streaming_gradio_app(engine):
    has_gaga = hasattr(engine, "GAGAvatar")
    all_gagavatar_id = (
        sorted(engine.GAGAvatar.all_gagavatar_id.keys()) if has_gaga else []
    )
    all_style_id = sorted(
        s.split(".")[0]
        for s in os.listdir("assets/style_motion")
        if s.endswith(".pt")
    )
    default_avatar = all_gagavatar_id[0] if all_gagavatar_id else None
    default_preview = (
        engine.get_avatar_preview(default_avatar) if default_avatar else None
    )

    def on_preset_select(avatar_id):
        return engine.get_avatar_preview(avatar_id), avatar_id

    def on_custom_upload(image_path):
        if image_path is None:
            return None, None, gr.update()
        try:
            new_key = engine.track_custom_avatar(image_path)
            preview = engine.get_avatar_preview(new_key)
            gr.Info(f"Successfully tracked custom avatar: {new_key}")
            return (
                preview, new_key,
                gr.update(
                    choices=sorted(engine.GAGAvatar.all_gagavatar_id.keys()),
                    value=new_key),
            )
        except Exception as e:
            gr.Warning(str(e))
            return None, gr.update(), gr.update()

    # ── Generator core ──

    def process_audio_streaming(
        input_type, audio_input, text_input, text_language, avatar_id,
        style_id,
    ):
        if input_type == "Audio" and audio_input is None:
            gr.Warning("Please upload an audio file"); return
        if input_type == "Text" and (
            not text_input or len(text_input.strip()) == 0
        ):
            gr.Warning("Please input text content"); return
        if avatar_id is None:
            gr.Warning("Please select or upload an avatar"); return

        if input_type == "Text":
            gtts_lang = {
                "English": "en", "中文": "zh", "日本語": "ja",
                "Deutsch": "de", "Français": "fr", "Español": "es",
            }
            tts = gTTS(text=text_input, lang=gtts_lang[text_language])
            tts.save("./render_results/tts_output.wav")
            audio_input = "./render_results/tts_output.wav"

        pipeline_start = time.perf_counter()

        audio, sr = torchaudio.load(audio_input)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
        audio_duration = audio.shape[0] / 16000.0

        if style_id == "default":
            engine.style_motion = None
        else:
            engine.set_style_motion(style_id)

        pred_motions, motion_stats = engine.inference(audio)
        n_frames = motion_stats["motion_frames"]

        base_name = audio_input.split("/")[-1].split(".")[0]
        style_tag = style_id.replace(".", "_")
        avatar_tag = avatar_id.replace(".", "_") if avatar_id else "mesh"

        # ────────────────────────────────────────────────────
        # Mesh rendering — batched GPU producer + CPU consumer
        # ────────────────────────────────────────────────────
        mesh_save_name = f"{base_name}_{style_tag}_mesh"
        mesh_path = os.path.join(engine.output_dir, f"{mesh_save_name}.mp4")
        mesh_writer = IncrementalVideoWriter(
            mesh_path, fps=25.0, sample_rate=16000
        )
        mesh_fmp4 = FMP4StreamWriter(width=512, height=512, fps=20)

        shape_code = torch.zeros(1, 300, device=engine.device).expand(
            pred_motions.shape[0], -1
        )
        verts = engine.generator.basic_vae.get_flame_verts(
            engine.flame_model, shape_code, pred_motions, with_global=True
        )
        n_mesh = len(verts)

        mesh_queue = queue.Queue(maxsize=QUEUE_MAXSIZE)
        mesh_error = [None]
        mesh_render_time = 0.0

        def _mesh_producer():
            try:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for cs in range(0, n_mesh, MESH_RENDER_BATCH):
                    chunk = verts[cs : cs + MESH_RENDER_BATCH]
                    rgb_batch = engine.mesh_renderer(chunk)[0]
                    rgb_cpu = rgb_batch.cpu()
                    del rgb_batch
                    for j in range(rgb_cpu.shape[0]):
                        f8 = rgb_cpu[j].clamp(0, 255).byte().numpy()
                        mesh_queue.put((cs + j, f8))
                    del rgb_cpu
                torch.cuda.synchronize()
                mesh_error.append(time.perf_counter() - t0)
            except Exception as exc:
                mesh_error[0] = exc
            finally:
                mesh_queue.put(None)

        mesh_thread = threading.Thread(target=_mesh_producer, daemon=True)
        mesh_thread.start()

        # Consumer — encode final video + push MSE segments.
        # Phase 1: buffer MESH_BUFFER_FRAMES silently (build 1-second head-start).
        # Phase 2: rate-limited yields every MESH_YIELD_INTERVAL_S so the JS
        #          poll (150 ms) never misses a yield.  Each yield sends ALL
        #          fMP4 bytes accumulated since the previous yield.
        mesh_frames_written = 0
        mesh_buffering = True
        last_mesh_yield_t = 0.0  # set to 0 so first post-buffer yield fires

        while True:
            item = mesh_queue.get()
            if item is None:
                break
            if mesh_error[0] is not None:
                mesh_thread.join()
                raise mesh_error[0]
            i, f8 = item
            mesh_writer.write_frame(f8)
            mesh_fmp4.write_frame(f8)
            mesh_frames_written += 1

            is_last = i == n_mesh - 1
            now = time.perf_counter()

            if mesh_buffering:
                # Phase 1: accumulate silently until buffer is full.
                if mesh_frames_written >= MESH_BUFFER_FRAMES or is_last:
                    mesh_buffering = False
                    b64 = mesh_fmp4.get_new_b64()
                    if b64:
                        yield (
                            b64, "",
                            None, None,
                            f"Rendering mesh: frame {i + 1} / {n_mesh}",
                        )
                        last_mesh_yield_t = now
            else:
                # Phase 2: yield at most once per MESH_YIELD_INTERVAL_S.
                if (now - last_mesh_yield_t >= MESH_YIELD_INTERVAL_S) or is_last:
                    b64 = mesh_fmp4.get_new_b64()
                    if b64:
                        yield (
                            b64, "",
                            None, None,
                            f"Rendering mesh: frame {i + 1} / {n_mesh}",
                        )
                        last_mesh_yield_t = now

        mesh_thread.join()
        if mesh_error[0] is not None:
            raise mesh_error[0]
        mesh_render_time = mesh_error[1] if len(mesh_error) > 1 else 0.0
        mesh_fmp4.close()

        audio_clip = audio[: int(mesh_writer.n_frames / 25.0 * 16000)]
        mesh_writer.finalize(audio_clip)

        # ────────────────────────────────────────────────────
        # Avatar rendering — sequential loop
        # ────────────────────────────────────────────────────
        avatar_path = None
        avatar_render_time = 0.0
        n_avatar_frames = 0

        if has_gaga and avatar_id in engine.GAGAvatar.all_gagavatar_id:
            avatar_save_name = f"{base_name}_{style_tag}_{avatar_tag}"
            avatar_path = os.path.join(
                engine.output_dir, f"{avatar_save_name}.mp4"
            )
            avatar_writer = IncrementalVideoWriter(
                avatar_path, fps=25.0, sample_rate=16000
            )
            avatar_fmp4 = FMP4StreamWriter(width=512, height=512, fps=AVATAR_FMP4_FPS)
            engine.GAGAvatar.set_avatar_id(avatar_id)
            n_avatar = len(pred_motions)

            torch.cuda.synchronize()
            avatar_t0 = time.perf_counter()
            avatar_frames_written = 0
            avatar_frames_since_yield = 0

            for i, motion in enumerate(pred_motions):
                batch = engine.GAGAvatar.build_forward_batch(
                    motion[None], engine.GAGAvatar_flame
                )
                rgb = engine.GAGAvatar.forward_expression(batch)
                f8 = rgb.cpu()[0].clamp(0, 1).mul(255).byte().numpy()
                del rgb

                avatar_writer.write_frame(f8)
                avatar_fmp4.write_frame(f8)
                avatar_frames_written += 1
                avatar_frames_since_yield += 1

                is_last = i == n_avatar - 1

                if avatar_frames_since_yield >= AVATAR_YIELD_EVERY or is_last:
                    b64 = avatar_fmp4.get_new_b64()
                    if b64:
                        yield (
                            "", b64,
                            None, None,
                            f"Rendering avatar: frame {i + 1} / {n_avatar}",
                        )
                        avatar_frames_since_yield = 0

            torch.cuda.synchronize()
            avatar_render_time = time.perf_counter() - avatar_t0
            n_avatar_frames = avatar_writer.n_frames
            avatar_fmp4.close()

            audio_clip2 = audio[
                : int(avatar_writer.n_frames / 25.0 * 16000)
            ]
            avatar_writer.finalize(audio_clip2)

        # ────────────────────────────────────────────────────
        # Benchmark
        # ────────────────────────────────────────────────────
        pipeline_end = time.perf_counter()
        total_time = pipeline_end - pipeline_start
        e2e_fps = n_frames / total_time

        bench_rows = (
            f"| Motion Generation | {n_frames} "
            f"| {motion_stats['motion_time']:.2f} "
            f"| {motion_stats['motion_fps']:.1f} |\n"
            f"| Mesh Rendering | {mesh_writer.n_frames} "
            f"| {mesh_render_time:.2f} "
            f"| {mesh_writer.n_frames / max(mesh_render_time, 1e-6):.1f} |\n"
        )
        if n_avatar_frames > 0:
            bench_rows += (
                f"| Avatar Rendering | {n_avatar_frames} "
                f"| {avatar_render_time:.2f} "
                f"| {n_avatar_frames / max(avatar_render_time, 1e-6):.1f}"
                f" |\n"
            )
        bench = (
            f"### Benchmark Results\n"
            f"| Stage | Frames | Time (s) | FPS |\n"
            f"|---|---|---|---|\n"
            f"{bench_rows}"
            f"| **End-to-End** | **{n_frames}** | **{total_time:.2f}** "
            f"| **{e2e_fps:.1f}** |\n\n"
            f"Audio duration: {audio_duration:.1f}s &nbsp; | &nbsp; "
            f"Real-time factor: {total_time / audio_duration:.2f}x"
        )

        # Final yield — empty segments, final videos + benchmark.
        yield "", "", mesh_path, avatar_path, bench

    # ── Build Gradio UI ──

    with gr.Blocks() as demo:
        active_avatar_id = gr.State(value=default_avatar)

        gr.Markdown("""
## Speech-Driven 3D Head Animation Demo (Streaming)

**Workflow:** Provide speech (audio or text) &rarr; autoregressive motion
generation &rarr; live-streamed mesh & avatar rendering.

Preview streams in real time via MSE. Final videos with audio appear
once encoding completes.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Speech Input")
                input_type = gr.Radio(
                    choices=["Audio", "Text"], value="Audio",
                    label="Input Mode",
                )
                audio_group = gr.Group()
                with audio_group:
                    audio_input = gr.Audio(
                        type="filepath", label="Upload or Record Audio"
                    )
                text_group = gr.Group(visible=False)
                with text_group:
                    text_input = gr.Textbox(
                        label="Text Content",
                        placeholder="Enter text to synthesize...",
                        lines=3,
                    )
                    text_language = gr.Dropdown(
                        choices=[
                            "English", "中文", "日本語",
                            "Deutsch", "Français", "Español",
                        ],
                        value="English", label="Language",
                    )
                gr.Markdown("**Motion Style**")
                style = gr.Dropdown(
                    choices=["default"] + all_style_id,
                    value="natural_0", label="Select Style",
                    info="Choose the speaking style and mannerisms",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Step 2: Choose Avatar")
                avatar_source = gr.Radio(
                    choices=["Preset", "Upload Headshot"],
                    value="Preset", label="Avatar Source",
                )
                preset_group = gr.Group()
                with preset_group:
                    appearance = gr.Dropdown(
                        choices=all_gagavatar_id, value=default_avatar,
                        label="Select Preset Avatar",
                        info="Choose from pre-tracked avatar identities",
                    )
                upload_group = gr.Group(visible=False)
                with upload_group:
                    custom_upload = gr.Image(
                        type="filepath", label="Upload Headshot",
                        sources=["upload"],
                    )
                    gr.Markdown(
                        "*Upload a clear, front-facing headshot. "
                        "We will track the face geometry automatically.*"
                    )
                avatar_preview = gr.Image(
                    value=default_preview, label="Avatar Preview",
                    interactive=False, height=256,
                )
                btn = gr.Button(
                    "Generate Animation", variant="primary", size="lg"
                )

        # ── MSE Live Preview ──
        # Custom <video> elements + JS MediaSource API for seamless
        # streaming (no source reload, no black flash).
        # Data flows: Python → hidden gr.Textbox (base64) → JS polls
        # textbox value → decodes → appends to MSE SourceBuffer.
        gr.Markdown("---")
        gr.Markdown("### Live Preview")
        gr.HTML(value=MSE_HTML)

        # Data-channel textboxes (CSS-hidden, NOT visible=False).
        # In Gradio 5.x, visible=False removes the element from the DOM
        # entirely (Svelte {#if} gate), so JS getElementById returns null.
        # We keep them visible=True and hide via CSS so the DOM element
        # exists and JS can read .value from it.
        gr.HTML('<style>#mesh_seg,#avatar_seg{display:none!important;}</style>')
        mesh_segment = gr.Textbox(
            visible=True, elem_id="mesh_seg",
            show_label=False, container=False,
        )
        avatar_segment = gr.Textbox(
            visible=True, elem_id="avatar_seg",
            show_label=False, container=False,
        )

        # ── Final video section ──
        gr.Markdown("---")
        gr.Markdown("### Final Videos (with audio)")
        with gr.Row():
            with gr.Column(scale=1):
                mesh_output = gr.Video(
                    autoplay=True, label="Mesh Rendering"
                )
            with gr.Column(scale=1):
                avatar_output = gr.Video(
                    autoplay=True, label="Avatar Rendering"
                )

        benchmark_output = gr.Markdown(value="", label="Benchmark")

        # ── Event wiring ──
        inputs = [
            input_type, audio_input, text_input,
            text_language, active_avatar_id, style,
        ]
        outputs = [
            mesh_segment, avatar_segment,
            mesh_output, avatar_output,
            benchmark_output,
        ]
        btn.click(
            fn=process_audio_streaming, inputs=inputs, outputs=outputs
        )
        appearance.change(
            fn=on_preset_select, inputs=[appearance],
            outputs=[avatar_preview, active_avatar_id],
        )
        custom_upload.change(
            fn=on_custom_upload, inputs=[custom_upload],
            outputs=[avatar_preview, active_avatar_id, appearance],
        )

        def toggle_input(choice):
            if choice == "Audio":
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)

        input_type.change(
            fn=toggle_input, inputs=[input_type],
            outputs=[audio_group, text_group],
        )

        def toggle_avatar_source(choice):
            if choice == "Preset":
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)

        avatar_source.change(
            fn=toggle_avatar_source, inputs=[avatar_source],
            outputs=[preset_group, upload_group],
        )

        gr.Markdown("---")
        gr.Markdown("### Try These Examples")
        examples = [
            ["Audio", "demo/jp1.wav", None, None, "12.jpg", "curious_0"],
            ["Audio", "demo/jp2.wav", None, None, "12.jpg", "natural_3"],
            ["Audio", "demo/eng1.wav", None, None, "12.jpg", "natural_2"],
            ["Audio", "demo/eng2.wav", None, None, "12.jpg", "happy_1"],
            ["Audio", "demo/cn1.wav", None, None, "11.jpg", "natural_1"],
            ["Audio", "demo/cn2.wav", None, None, "12.jpg", "happy_2"],
            ["Text", None,
             "Hello, this is a demo! Let's create something fun together.",
             "English", "12.jpg", "happy_0"],
            ["Text", None,
             "让我们一起创造一些有趣的东西吧。",
             "中文", "12.jpg", "natural_0"],
        ]
        gr.Examples(
            examples=examples,
            inputs=[
                input_type, audio_input, text_input,
                text_language, appearance, style,
            ],
        )

        # Inject MSE JavaScript on page load (must be inside Blocks context).
        demo.load(fn=None, js=MSE_BOOT_JS)

    demo.launch(server_name="0.0.0.0", server_port=8961, share=True)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_length", "-l", default=750, type=int)
    parser.add_argument(
        "--warmup_steps", "-w", default=3, type=int,
        help="Warmup iterations before launching (0 to skip)",
    )
    args = parser.parse_args()

    engine = ARTAvatarInferEngine(
        load_gaga=True, fix_pose=False, clip_length=args.clip_length
    )
    if args.warmup_steps > 0:
        warmup(engine, steps=args.warmup_steps)
    run_streaming_gradio_app(engine)
