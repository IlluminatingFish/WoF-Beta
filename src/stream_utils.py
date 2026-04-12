#!/usr/bin/env python
"""Shared preview encoding + streaming timing (Gradio / Flask)."""

import base64
import io
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

try:
    from PIL import Image
except ImportError:
    Image = None


def chw_tensor_to_uint8_hwc(t: torch.Tensor) -> np.ndarray:
    """CHW float [0,1] or ~[0,255] -> uint8 HWC for preview / JPEG."""
    x = t.detach().float().cpu()
    if x.max() <= 1.0 + 1e-3:
        x = x * 255.0
    x = x.clamp(0, 255).byte().permute(1, 2, 0).numpy()
    return np.ascontiguousarray(x)


def uint8_hwc_to_jpeg_b64(arr: np.ndarray, quality: int = 82) -> str:
    raw = uint8_hwc_to_jpeg_bytes(arr, quality=quality, max_side=None)
    return base64.standard_b64encode(raw).decode("ascii")


def uint8_hwc_to_jpeg_bytes(arr: np.ndarray, quality: int = 82, max_side: Optional[int] = None) -> bytes:
    """Encode preview JPEG; optional max_side (long edge) to cut wire size for streaming."""
    if Image is None:
        raise RuntimeError("Pillow is required for JPEG preview streaming (pip install pillow).")
    pil = Image.fromarray(arr)
    if max_side is not None and max_side > 0:
        w, h = pil.size
        m = max(w, h)
        if m > max_side:
            s = max_side / float(m)
            pil = pil.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class StreamTimingLog:
    """Per-frame timings to stdout + final summary (bottleneck visibility)."""

    def __init__(self, label: str = "stream"):
        self.label = label
        self.t0: Optional[float] = None
        self.t_motion_end: Optional[float] = None
        self.t_first_frame_out: Optional[float] = None
        self.frame_rows: List[Dict[str, Any]] = []
        self._prev: Optional[float] = None

    def start_after_motion(self) -> None:
        self.t0 = time.perf_counter()
        self.t_motion_end = self.t0
        self._prev = self.t0

    def log_frame(self, frame_index: int, convert_s: float, encode_s: float) -> None:
        now = time.perf_counter()
        gap_s = now - self._prev if self._prev is not None else 0.0
        self._prev = now
        if self.t_first_frame_out is None:
            self.t_first_frame_out = now
            ttff = now - self.t_motion_end if self.t_motion_end else 0.0
            print(f"[{self.label}] time_to_first_frame_after_motion_s={ttff:.4f}")
        row = {
            "i": frame_index,
            "since_prev_s": gap_s,
            "convert_s": convert_s,
            "encode_s": encode_s,
        }
        self.frame_rows.append(row)
        print(
            f"[{self.label}] frame={frame_index} "
            f"since_prev_ms={gap_s*1000:.2f} convert_ms={convert_s*1000:.2f} encode_ms={encode_s*1000:.2f}"
        )

    def finalize(self, n_frames: int, encode_video_s: float) -> Dict[str, Any]:
        now = time.perf_counter()
        total_stream_s = (now - self.t0) if self.t0 else 0.0
        avg_frame_ms = (total_stream_s / n_frames * 1000.0) if n_frames else 0.0
        summary = {
            "n_frames": n_frames,
            "total_stream_phase_s": round(total_stream_s, 4),
            "avg_ms_per_frame_loop": round(avg_frame_ms, 3),
            "encode_video_s": round(encode_video_s, 4),
        }
        if self.frame_rows:
            enc = [r["encode_s"] for r in self.frame_rows]
            summary["jpeg_encode_total_s"] = round(float(np.sum(enc)), 4)
            summary["jpeg_encode_avg_ms"] = round(float(np.mean(enc)) * 1000.0, 3)
        print(f"[{self.label}] summary {summary}")
        return summary
