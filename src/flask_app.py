#!/usr/bin/env python
"""Flask: offline chunked MP4; online session + parallel mux then chunked MP4 stream."""

import json
import os
import queue
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torchaudio
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, stream_with_context

from app.utils_videos import IncrementalVideoWriter
from inference import ARTAvatarInferEngine
from stream_render import iter_render_frames

ONLINE_RENDER_QUEUE_MAXSIZE = 4
_ONLINE_QUEUE_SENTINEL = object()
ONLINE_MP4_CHUNK_BYTES = 24 * 1024

_sessions_lock = threading.Lock()
SESSIONS: Dict[str, Dict[str, Any]] = {}


def _parse_clip_length(form) -> int | None:
    raw = (form.get("clip_length") or "").strip()
    return int(raw) if raw else None


def _safe_name(name: str) -> str:
    base = os.path.basename(name or "upload")
    base = re.sub(r"[^a-zA-Z0-9._-]+", "_", base)[:120]
    return base or "upload"


def _session_get(token: str) -> Optional[Dict[str, Any]]:
    with _sessions_lock:
        return SESSIONS.get(token)


def _session_put(token: str, data: Dict[str, Any]) -> None:
    with _sessions_lock:
        SESSIONS[token] = data


def _session_update(token: str, **kwargs: Any) -> None:
    with _sessions_lock:
        if token in SESSIONS:
            SESSIONS[token].update(kwargs)


def _finalize_timing(
    t_wall0: float,
    load_resample_audio_s: float,
    motion_inference_s: float,
    rt: Dict[str, Any],
    save_motion_s: float,
) -> Dict[str, Any]:
    total_s = time.perf_counter() - t_wall0
    render_and_encode_s = (
        rt["render_frames_s"] + rt["stack_tensors_s"] + rt["encode_video_s"]
    )
    out = {
        "total_s": round(total_s, 4),
        "load_resample_audio_s": round(load_resample_audio_s, 4),
        "motion_inference_s": round(motion_inference_s, 4),
        "render_frames_s": rt["render_frames_s"],
        "stack_tensors_s": rt["stack_tensors_s"],
        "encode_video_s": rt["encode_video_s"],
        "render_and_encode_s": round(render_and_encode_s, 4),
        "save_motion_pt_s": round(save_motion_s, 4),
        "n_frames": rt["n_frames"],
    }
    if "h264_interleaved_wall_s" in rt:
        out["h264_interleaved_wall_s"] = rt["h264_interleaved_wall_s"]
    return out


def create_app(engine: ARTAvatarInferEngine) -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    gpu_lock = threading.Lock()
    upload_dir = Path(engine.output_dir) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    @app.route("/")
    def index():
        return render_template("flask_demo.html")

    @app.route("/files/<path:fname>")
    def serve_output(fname):
        return send_from_directory(engine.output_dir, fname, as_attachment=False)

    @app.route("/api/offline", methods=["POST"])
    def api_offline():
        """POST audio; response body is chunked MP4; timing in X-WoF-Meta."""
        if "audio" not in request.files:
            return jsonify({"ok": False, "error": "missing audio file"}), 400
        f = request.files["audio"]
        shape_id = request.form.get("shape_id", "mesh")
        style_id = request.form.get("style_id", "natural_0")
        clip_length = _parse_clip_length(request.form)
        chunk_kb = max(1, int(request.form.get("transfer_chunk_kb", 24)))
        chunk_bytes = chunk_kb * 1024

        raw_name = _safe_name(f.filename)
        token = uuid.uuid4().hex[:12]
        wav_path = upload_dir / f"{token}_{raw_name}"
        t_upload0 = time.perf_counter()
        f.save(wav_path)
        audio_upload_receive_s = time.perf_counter() - t_upload0

        with gpu_lock:
            t_wall0 = time.perf_counter()
            t_load0 = time.perf_counter()
            audio, sr = torchaudio.load(str(wav_path))
            audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
            load_resample_audio_s = time.perf_counter() - t_load0

            engine.clip_length = clip_length
            if style_id == "default":
                engine.style_motion = None
            else:
                engine.set_style_motion(style_id)

            t_gen0 = time.perf_counter()
            pred_motions = engine.inference(audio)
            motion_inference_s = time.perf_counter() - t_gen0

            save_name = f"{Path(raw_name).stem}_{style_id.replace('.', '_')}_{shape_id.replace('.', '_')}_{token}"
            rt = engine.rendering(
                audio, pred_motions, shape_id=shape_id, save_name=save_name, return_timings=True
            )
            video_generation_wall_s = time.perf_counter() - t_gen0

            motion_path = os.path.join(engine.output_dir, f"{save_name}_motions.pt")
            t_save0 = time.perf_counter()
            torch.save(pred_motions.float().cpu(), motion_path)
            save_motion_s = time.perf_counter() - t_save0

            timing = _finalize_timing(
                t_wall0,
                load_resample_audio_s,
                motion_inference_s,
                rt,
                save_motion_s,
            )
            timing["audio_upload_receive_s"] = round(audio_upload_receive_s, 4)
            timing["video_generation_wall_s"] = round(video_generation_wall_s, 4)

        video_rel = f"{save_name}.mp4"
        motion_rel = f"{save_name}_motions.pt"
        video_path = os.path.join(engine.output_dir, video_rel)
        file_size = os.path.getsize(video_path)
        n_chunks_est = (file_size + chunk_bytes - 1) // chunk_bytes if chunk_bytes else 1

        meta = {
            "ok": True,
            "mode": "offline",
            "timing": timing,
            "video_url": f"/files/{video_rel}",
            "motion_url": f"/files/{motion_rel}",
            "transfer": {
                "style": "whole_mp4_chunks",
                "chunk_bytes": chunk_bytes,
                "file_bytes": file_size,
                "n_chunks_estimate": int(n_chunks_est),
            },
            "infer_s": timing["motion_inference_s"],
            "render_and_encode_s": timing["render_and_encode_s"],
        }
        meta_json = json.dumps(meta, separators=(",", ":"))

        def chunked_mp4():
            t_stream0 = time.perf_counter()
            n_chunks = 0
            with open(video_path, "rb") as fp:
                data = fp.read(chunk_bytes)
                while data:
                    n_chunks += 1
                    yield data
                    data = fp.read(chunk_bytes)
            wall = time.perf_counter() - t_stream0
            print(
                f"[offline_mp4_stream] chunks={n_chunks} stream_wall_s={wall:.4f} "
                f"file_bytes={file_size}"
            )

        return Response(
            stream_with_context(chunked_mp4()),
            mimetype="video/mp4",
            headers={
                "X-WoF-Meta": meta_json,
                "Cache-Control": "no-store",
                "Content-Disposition": f'inline; filename="{video_rel}"',
            },
        )

    @app.route("/api/session", methods=["POST"])
    def api_session():
        """POST creates session; GET /api/online_mjpeg/<token> streams MP4 after encode."""
        if "audio" not in request.files:
            return jsonify({"ok": False, "error": "missing audio file"}), 400
        f = request.files["audio"]
        shape_id = request.form.get("shape_id", "mesh")
        style_id = request.form.get("style_id", "natural_0")
        clip_length = _parse_clip_length(request.form)

        raw_name = _safe_name(f.filename)
        token = uuid.uuid4().hex[:12]
        wav_path = upload_dir / f"{token}_{raw_name}"
        f.save(wav_path)

        _session_put(
            token,
            {
                "wav_path": str(wav_path),
                "raw_name": raw_name,
                "shape_id": shape_id,
                "style_id": style_id,
                "clip_length": clip_length,
                "status": "ready",
                "timing": None,
                "video_url": None,
                "motion_url": None,
                "error": None,
            },
        )
        return jsonify({"ok": True, "token": token})

    @app.route("/api/session/<token>/status", methods=["GET"])
    def api_session_status(token: str):
        s = _session_get(token)
        if not s:
            return jsonify({"ok": False, "error": "unknown token"}), 404
        out = {
            "ok": True,
            "status": s["status"],
            "timing": s.get("timing"),
            "video_url": s.get("video_url"),
            "motion_url": s.get("motion_url"),
            "error": s.get("error"),
        }
        return jsonify(out)

    @app.route("/api/online_mjpeg/<token>", methods=["GET"])
    def api_online_mjpeg(token: str):
        s = _session_get(token)
        if not s:
            return "not found", 404
        if s["status"] == "running":
            return "stream already in progress", 409
        if s["status"] not in ("ready",):
            if s["status"] == "done":
                return "session finished", 410
            return "bad state", 400

        def online_mp4_stream():
            try:
                with gpu_lock:
                    _session_update(token, status="running")
                    t_wall0 = time.perf_counter()
                    wav_path = s["wav_path"]
                    raw_name = s["raw_name"]
                    shape_id = s["shape_id"]
                    style_id = s["style_id"]
                    clip_length = s["clip_length"]

                    t_load0 = time.perf_counter()
                    audio, sr = torchaudio.load(wav_path)
                    audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
                    load_resample_audio_s = time.perf_counter() - t_load0

                    engine.clip_length = clip_length
                    if style_id == "default":
                        engine.style_motion = None
                    else:
                        engine.set_style_motion(style_id)

                    t_infer0 = time.perf_counter()
                    pred_motions = engine.inference(audio)
                    motion_inference_s = time.perf_counter() - t_infer0

                    save_name = f"{Path(raw_name).stem}_{style_id.replace('.', '_')}_{shape_id.replace('.', '_')}_{token}"
                    video_path = os.path.join(engine.output_dir, f"{save_name}.mp4")
                    motion_path = os.path.join(engine.output_dir, f"{save_name}_motions.pt")

                    h264_mux_acc = 0.0
                    writer = IncrementalVideoWriter(video_path, 25.0)
                    q: "queue.Queue[Any]" = queue.Queue(maxsize=ONLINE_RENDER_QUEUE_MAXSIZE)
                    prod_state: Dict[str, Any] = {"err": None}

                    def producer():
                        try:
                            it = iter(
                                iter_render_frames(
                                    engine, audio, pred_motions, shape_id=shape_id
                                )
                            )
                            while True:
                                t_f0 = time.perf_counter()
                                try:
                                    rgb = next(it)
                                except StopIteration:
                                    break
                                render_acc_local = time.perf_counter() - t_f0
                                prod_state["render_acc"] = (
                                    prod_state.get("render_acc", 0.0) + render_acc_local
                                )
                                q.put(rgb)
                        except Exception as e:
                            prod_state["err"] = e
                        finally:
                            q.put(_ONLINE_QUEUE_SENTINEL)

                    t_par0 = time.perf_counter()
                    th = threading.Thread(target=producer, daemon=True)
                    th.start()
                    n_frames = 0
                    while True:
                        rgb = q.get()
                        if rgb is _ONLINE_QUEUE_SENTINEL:
                            th.join()
                            if prod_state["err"] is not None:
                                raise prod_state["err"]
                            break
                        t_h0 = time.perf_counter()
                        writer.write_frame(rgb)
                        h264_mux_acc += time.perf_counter() - t_h0
                        n_frames += 1
                    parallel_wall_s = time.perf_counter() - t_par0
                    render_acc = float(prod_state.get("render_acc", 0.0))
                    serial_phase_s = render_acc + h264_mux_acc
                    overlap_est_s = max(0.0, serial_phase_s - parallel_wall_s)

                    stack_s = 0.0
                    audio_trim = audio[: int(writer.n_frames / 25.0 * 16000)]
                    t_fin0 = time.perf_counter()
                    writer.finalize(audio_trim, 16000, "aac")
                    finalize_s = time.perf_counter() - t_fin0
                    encode_video_s = h264_mux_acc + finalize_s
                    t_save0 = time.perf_counter()
                    torch.save(pred_motions.float().cpu(), motion_path)
                    save_motion_s = time.perf_counter() - t_save0

                    render_frames_s = round(render_acc, 4)
                    stack_tensors_s = round(stack_s, 4)
                    encode_video_s_r = round(encode_video_s, 4)
                    n_fr = writer.n_frames
                    render_and_encode_s = round(
                        render_acc + stack_s + encode_video_s, 4
                    )
                    total_wall_s = time.perf_counter() - t_wall0
                    total_compute_s = (
                        load_resample_audio_s
                        + motion_inference_s
                        + render_acc
                        + stack_s
                        + encode_video_s
                        + save_motion_s
                    )
                    timing = {
                        "total_s": round(total_wall_s, 4),
                        "total_serial_sum_s": round(total_compute_s, 4),
                        "load_resample_audio_s": round(load_resample_audio_s, 4),
                        "motion_inference_s": round(motion_inference_s, 4),
                        "render_frames_s": render_frames_s,
                        "stack_tensors_s": stack_tensors_s,
                        "encode_video_s": encode_video_s_r,
                        "render_and_encode_s": render_and_encode_s,
                        "save_motion_pt_s": round(save_motion_s, 4),
                        "n_frames": n_fr,
                        "transfer": {
                            "style": "parallel_render_mux_then_chunked_mp4",
                            "parallel_render_mux_wall_s": round(parallel_wall_s, 4),
                            "serial_render_plus_h264_mux_s": round(serial_phase_s, 4),
                            "h264_mux_accum_before_finalize_s": round(h264_mux_acc, 4),
                            "overlap_estimate_s": round(overlap_est_s, 4),
                            "queue_maxsize": ONLINE_RENDER_QUEUE_MAXSIZE,
                            "mp4_stream_chunk_bytes": ONLINE_MP4_CHUNK_BYTES,
                        },
                    }

                    file_size = os.path.getsize(video_path)
                    t_stream0 = time.perf_counter()
                    n_chunks = 0
                    with open(video_path, "rb") as fp:
                        data = fp.read(ONLINE_MP4_CHUNK_BYTES)
                        while data:
                            n_chunks += 1
                            yield data
                            data = fp.read(ONLINE_MP4_CHUNK_BYTES)
                    stream_wall = time.perf_counter() - t_stream0
                    _session_update(
                        token,
                        status="done",
                        timing=timing,
                        video_url=f"/files/{save_name}.mp4",
                        motion_url=f"/files/{save_name}_motions.pt",
                    )
                    print(
                        f"[online_mp4] frames={n_frames} parallel_wall_s={parallel_wall_s:.4f} "
                        f"overlap_est_s={overlap_est_s:.4f} stream_chunks={n_chunks} "
                        f"stream_wall_s={stream_wall:.4f} file_bytes={file_size}"
                    )
            except Exception as e:
                _session_update(token, status="error", error=str(e))
                raise

        return Response(
            stream_with_context(online_mp4_stream()),
            mimetype="video/mp4",
            headers={"Cache-Control": "no-cache, no-store", "X-Accel-Buffering": "no"},
        )

    return app


def run_flask_app(engine: ARTAvatarInferEngine, host: str = "0.0.0.0", port: int = 8961) -> None:
    app = create_app(engine)
    print(f"Flask demo: http://{host}:{port}/")
    print("  POST /api/offline — generate all, then chunked MP4 stream; meta in X-WoF-Meta")
    print(
        "  POST /api/session + GET /api/online_mjpeg/<token> — parallel mux, chunked MP4 stream"
    )
    app.run(host=host, port=port, threaded=True)


if __name__ == "__main__":
    import argparse

    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8961)
    parser.add_argument(
        "--clip_length",
        "-l",
        type=int,
        default=None,
        help="Max motion frames; default from audio at 25 fps.",
    )
    cli = parser.parse_args()
    eng = ARTAvatarInferEngine(load_gaga=True, fix_pose=False, clip_length=cli.clip_length)
    run_flask_app(eng, host=cli.host, port=cli.port)
