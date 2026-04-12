#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import av
import torch
import numpy as np

def write_video(video_frames, output_path, fps, audio_samples=None, sample_rate=None, acodec="aac"):
    assert video_frames.ndim == 4, "Input frames should be a 4D array."
    assert video_frames.shape[1] == 3, "Input frames should have 3 channels (RGB)."
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.cpu().numpy()
    if video_frames.dtype != np.uint8:
        video_frames = video_frames.astype(np.uint8)
    _, _, height, width = video_frames.shape
    container = av.open(output_path, mode="w")
    stream = container.add_stream("h264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18"}
    if audio_samples is not None:
        if acodec == "aac":
            audio_stream = container.add_stream("aac", rate=sample_rate)
            audio_stream.format = "fltp"
        elif acodec in ("vs_preview", "debug"):
            audio_stream = container.add_stream("mp3", rate=sample_rate)
            audio_stream.format = "fltp"
        else:
            raise ValueError("Unsupported audio codec.")

    for frame in video_frames:
        frame = frame.transpose(1, 2, 0)
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    if audio_samples is not None:
        if isinstance(audio_samples, torch.Tensor):
            audio_samples = audio_samples.cpu().numpy()
        assert audio_samples.ndim == 1, "Input audio samples should be a 1D array."
        num_samples_per_frame = int(sample_rate // fps)
        for i in range(0, audio_samples.shape[0], num_samples_per_frame):
            chunk = audio_samples[i : i + num_samples_per_frame]
            if chunk.shape[0] < num_samples_per_frame:
                chunk = np.pad(chunk, (0, num_samples_per_frame - chunk.shape[0]), mode="constant")
            audio_frame = av.AudioFrame.from_ndarray(chunk[None], format="fltp", layout="mono")
            audio_frame.rate = sample_rate
            for packet in audio_stream.encode(audio_frame):
                container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    if audio_samples is not None:
        for packet in audio_stream.encode():
            container.mux(packet)

    container.close()


def _tensor_chw_to_rgb24_hwc_numpy(frame):
    """Single frame CHW (torch, float 0–1 or uint8) -> contiguous uint8 HWC for PyAV."""
    if isinstance(frame, torch.Tensor):
        x = frame.detach().cpu()
        if x.dtype != torch.uint8:
            x = x.float()
            if x.max() <= 1.0 + 1e-3:
                x = x * 255.0
            x = x.clamp(0, 255).byte()
        x = x.permute(1, 2, 0).contiguous()
        return np.ascontiguousarray(x.numpy())
    raise TypeError("frame must be a torch.Tensor CHW")


class IncrementalVideoWriter:
    """Incremental H.264 (+ optional AAC opened before first video packet) into one MP4."""

    def __init__(
        self,
        output_path: str,
        fps: float,
        sample_rate: int | None = 16000,
        acodec: str = "aac",
    ):
        self.output_path = output_path
        self.fps = int(fps)
        self.sample_rate = sample_rate
        self.acodec = acodec
        self.container = None
        self.stream = None
        self.audio_stream = None
        self.n_frames = 0

    def _add_audio_stream(self, sample_rate: int) -> None:
        if self.acodec == "aac":
            self.audio_stream = self.container.add_stream("aac", rate=sample_rate)
            self.audio_stream.format = "fltp"
        elif self.acodec in ("vs_preview", "debug"):
            self.audio_stream = self.container.add_stream("mp3", rate=sample_rate)
            self.audio_stream.format = "fltp"
        else:
            raise ValueError("Unsupported audio codec.")

    def write_frame(self, frame_chw) -> None:
        arr = _tensor_chw_to_rgb24_hwc_numpy(frame_chw)
        height, width = arr.shape[0], arr.shape[1]
        if self.container is None:
            self.container = av.open(self.output_path, mode="w")
            self.stream = self.container.add_stream("h264", rate=self.fps)
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.options = {"crf": "18"}
            if self.sample_rate is not None:
                self._add_audio_stream(self.sample_rate)
        video_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)
        self.n_frames += 1

    def finalize(
        self,
        audio_samples=None,
        sample_rate=None,
        acodec="aac",
    ) -> None:
        if self.container is None or self.stream is None:
            raise RuntimeError("IncrementalVideoWriter: no frames written")
        for packet in self.stream.encode():
            self.container.mux(packet)

        if audio_samples is not None:
            if isinstance(audio_samples, torch.Tensor):
                audio_samples = audio_samples.cpu().numpy()
            assert audio_samples.ndim == 1, "audio_samples should be 1D"
            if self.audio_stream is None:
                sr = sample_rate if sample_rate is not None else self.sample_rate
                if sr is None:
                    raise RuntimeError(
                        "IncrementalVideoWriter: audio_samples given but no audio stream; "
                        "construct with sample_rate=... or call finalize(sample_rate=...)"
                    )
                self._add_audio_stream(int(sr))
            sr_eff = int(self.audio_stream.rate)
            num_samples_per_frame = int(sr_eff // self.fps)
            for i in range(0, audio_samples.shape[0], num_samples_per_frame):
                chunk = audio_samples[i : i + num_samples_per_frame]
                if chunk.shape[0] < num_samples_per_frame:
                    chunk = np.pad(
                        chunk,
                        (0, num_samples_per_frame - chunk.shape[0]),
                        mode="constant",
                    )
                audio_frame = av.AudioFrame.from_ndarray(
                    chunk[None], format="fltp", layout="mono"
                )
                audio_frame.rate = sr_eff
                for packet in self.audio_stream.encode(audio_frame):
                    self.container.mux(packet)
            for packet in self.audio_stream.encode():
                self.container.mux(packet)

        self.container.close()
        self.container = None
        self.stream = None
        self.audio_stream = None


def read_video_frames(video_path):
    container = av.open(video_path)
    for frame in container.decode(video=0):
        yield torch.tensor(frame.to_ndarray(format="rgb24")).permute(2, 0, 1)


def get_video_info(video_path):
    info_dict = {}
    container = av.open(video_path)
    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    if video_stream is None:
        info_dict["video"] = None
    else:
        info_dict["video"] = {
            "width": video_stream.width,
            "height": video_stream.height,
            "frame_rate": float(video_stream.average_rate),
            "num_frames": video_stream.frames,
        }
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if audio_stream is None:
        info_dict["audio"] = None
    else:
        info_dict["audio"] = {
            "channels": audio_stream.channels,
            "sample_rate": audio_stream.rate,
            "duration": audio_stream.duration,
        }
    return info_dict


def read_all_video_frames(video_path):
    container = av.open(video_path)
    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    if video_stream is None:
        print("No video stream found in the file.")
        return np.zeros((0), dtype=np.uint8), 0
    frames = []
    for frame in container.decode(video=0):  # Decode only video stream
        if frame.pts is None:  # Ignore invalid frames
            continue
        frames.append(frame.to_ndarray(format="rgb24"))
        # frame_id = int(frame.pts * video_stream.time_base * float(video_stream.average_rate))
    frames = torch.tensor(np.stack(frames, axis=0)).permute(0, 3, 1, 2)
    return frames, float(video_stream.average_rate)


def read_audio_samples(video_path, stero=False):
    container = av.open(video_path)
    audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
    if audio_stream is None:
        print("No audio stream found in the file.")
        return None, None
    audio_samples = []
    for frame in container.decode(audio=0):  # Decode all audio frames
        audio_samples.append(frame.to_ndarray())  # Convert to NumPy array
    # Concatenate all audio frames into a single array
    audio_data = np.concatenate(audio_samples, axis=-1)
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32768.0  # for PCM (WAV)
    elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / (2**31) # for FLAC
    if not stero:
        audio_data = audio_data.mean(axis=0)
    if audio_data.max() > 1.0 or audio_data.min() < -1.0:
        print("Warning: Audio samples are not normalized, max={}, min={}.".format(audio_data.max(), audio_data.min()))
    return audio_data, audio_stream.rate


if __name__ == "__main__":
    from tqdm import tqdm
    # Example Usage
    video_path = '../MultiTalk_dataset/multitalk_dataset/english/-OknSRRyFJE_0.mp4'
    vres, fps = read_all_video_frames(video_path)
    print(vres.shape, fps)
    
    ares, sample_rate = read_audio_samples(video_path)
    print(ares.shape, sample_rate)
    
    video_length = get_video_info(video_path)['video']['num_frames']
    print(get_video_info(video_path))

    for frame in tqdm(read_video_frames(video_path), total=video_length):
        pass

    # write_video(vres, "output_debug.mp4", fps)
