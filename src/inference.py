#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
# Default GPUs 
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import json
import queue
import threading
import time
import torch
import argparse
import torchaudio
from tqdm import tqdm

from app import BitwiseARModel
from app.flame_model import FLAMEModel, RenderMesh
from app.utils_videos import IncrementalVideoWriter

_RENDER_PIPE_QUEUE = 4
_RENDER_SENTINEL = object()


class ARTAvatarInferEngine:
    def __init__(self, load_gaga=False, fix_pose=False, clip_length=None, device='cuda'):
        self.device = device
        self.fix_pose = fix_pose
        self.clip_length = clip_length
        audio_encoder = 'wav2vec'
        ckpt = torch.load('./assets/ARTalk_{}.pt'.format(audio_encoder), map_location='cpu', weights_only=True)
        configs = json.load(open("./assets/config.json"))
        configs['AR_CONFIG']['AUDIO_ENCODER'] = audio_encoder
        self.ARTalk = BitwiseARModel(configs).eval().to(device)
        self.ARTalk.load_state_dict(ckpt, strict=True)
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=1.0, no_lmks=True).to(device)
        self.mesh_renderer = RenderMesh(image_size=512, faces=self.flame_model.get_faces(), scale=1.0)
        
        self.output_dir = 'render_results/ARTAvatar_{}'.format(audio_encoder)
        os.makedirs(self.output_dir, exist_ok=True)
        self.style_motion = None

        if load_gaga:
            from app.GAGAvatar import GAGAvatar
            self.GAGAvatar = GAGAvatar().to(device)
            self.GAGAvatar_flame = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True).to(device)

    def set_style_motion(self, style_motion):
        if isinstance(style_motion, str):
            style_motion = torch.load('assets/style_motion/{}.pt'.format(style_motion), map_location='cpu', weights_only=True)
        assert style_motion.shape == (50, 106), f'Invalid style_motion shape: {style_motion.shape}.'
        self.style_motion = style_motion[None].to(self.device)

    def inference(self, audio, clip_length=None):
        audio_batch = {'audio': audio[None].to(self.device), 'style_motion': self.style_motion}
        print('Inferring motion...')
        pred_motions = self.ARTalk.inference(audio_batch, with_gtmotion=False)[0]
        pred_motions = self.smooth_motion_savgol(pred_motions)
        clip_length = clip_length if clip_length is not None else self.clip_length
        if clip_length is not None:
            pred_motions = pred_motions[:clip_length]
        else:
            n_audio_frames = max(1, int(audio.numel() * 25 // 16000))
            pred_motions = pred_motions[: min(int(pred_motions.shape[0]), n_audio_frames)]
        if self.fix_pose:
            pred_motions[..., 100:103] *= 0.0
        print('Done!')
        pred_motions[..., 104:] *= 0.0
        return pred_motions

    def rendering(
        self,
        audio,
        pred_motions,
        shape_id="mesh",
        shape_code=None,
        save_name="ARTAvatar",
        return_timings=False,
    ):
        from stream_render import iter_render_frames

        print("Rendering (GPU pipe + incremental H.264)...")
        dump_path = os.path.join(self.output_dir, "{}.mp4".format(save_name))
        writer = IncrementalVideoWriter(dump_path, 25.0)
        q = queue.Queue(maxsize=_RENDER_PIPE_QUEUE)
        prod_state: dict = {"render_acc": 0.0, "err": None}

        def producer():
            try:
                it = iter(
                    iter_render_frames(self, audio, pred_motions, shape_id, shape_code)
                )
                while True:
                    t_g0 = time.perf_counter()
                    try:
                        rgb = next(it)
                    except StopIteration:
                        break
                    prod_state["render_acc"] += time.perf_counter() - t_g0
                    q.put(rgb)
            except Exception as e:
                prod_state["err"] = e
            finally:
                q.put(_RENDER_SENTINEL)

        t_par0 = time.perf_counter()
        th = threading.Thread(target=producer, daemon=True)
        th.start()
        h264_mux_acc = 0.0
        pbar = tqdm(total=int(pred_motions.shape[0]))
        while True:
            rgb = q.get()
            if rgb is _RENDER_SENTINEL:
                th.join()
                if prod_state["err"] is not None:
                    raise prod_state["err"]
                break
            t_h0 = time.perf_counter()
            writer.write_frame(rgb)
            h264_mux_acc += time.perf_counter() - t_h0
            pbar.update(1)
        pbar.close()
        h264_interleaved_wall_s = time.perf_counter() - t_par0

        n_frames = writer.n_frames
        audio_trim = audio[: int(n_frames / 25.0 * 16000)]
        print("Muxing audio + closing container...")
        t_fin0 = time.perf_counter()
        writer.finalize(audio_trim, 16000, "aac")
        finalize_s = time.perf_counter() - t_fin0
        encode_video_s = h264_mux_acc + finalize_s
        print("Done!")
        if return_timings:
            return {
                "n_frames": n_frames,
                "render_frames_s": round(prod_state["render_acc"], 4),
                "stack_tensors_s": 0.0,
                "encode_video_s": round(encode_video_s, 4),
                "h264_interleaved_wall_s": round(h264_interleaved_wall_s, 4),
            }
        return None

    @staticmethod
    def smooth_motion_savgol(motion_codes):
        from scipy.signal import savgol_filter
        motion_np = motion_codes.clone().detach().cpu().numpy()
        motion_np_smoothed = savgol_filter(motion_np, window_length=5, polyorder=2, axis=0)
        motion_np_smoothed[..., 100:103] = savgol_filter(motion_np[..., 100:103], window_length=9, polyorder=3, axis=0)
        return torch.tensor(motion_np_smoothed).type_as(motion_codes)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', '-a', default=None, type=str)
    parser.add_argument(
        '--clip_length',
        '-l',
        default=None,
        type=int,
        help='Cap motion frames; omit to use min(model length, audio length at 25 fps).',
    )
    parser.add_argument("--shape_id", '-i', default='mesh', type=str)
    parser.add_argument("--style_id", '-s', default='default', type=str)

    parser.add_argument("--run_app", action='store_true')
    parser.add_argument(
        "--flask",
        action="store_true",
        help="With --run_app: run Flask UI (port 8961) instead of Gradio (see gradio_app.py).",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="With --run_app and Gradio only: default 'Render mode' to online.",
    )
    parser.add_argument("--flask_port", type=int, default=8961, help="Port for Flask when using --flask.")
    args = parser.parse_args()

    engine = ARTAvatarInferEngine(load_gaga=True, fix_pose=False, clip_length=args.clip_length)
    if args.run_app:
        if args.flask:
            from flask_app import run_flask_app
            run_flask_app(engine, port=args.flask_port)
        else:
            from gradio_app import run_gradio_app
            run_gradio_app(engine, default_online=args.online)
    else:
        shape_id = 'mesh' if args.shape_id not in engine.GAGAvatar.all_gagavatar_id.keys() else args.shape_id
        t_wall0 = time.perf_counter()
        t_load0 = time.perf_counter()
        audio, sr = torchaudio.load(args.audio_path)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
        load_resample_audio_s = time.perf_counter() - t_load0

        base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
        save_name = f'{base_name}_{args.style_id.replace(".", "_")}_{args.shape_id.replace(".", "_")}'
        engine.set_style_motion(args.style_id)
        t_infer0 = time.perf_counter()
        pred_motions = engine.inference(audio)
        motion_inference_s = time.perf_counter() - t_infer0
        rt = engine.rendering(
            audio, pred_motions, shape_id=args.shape_id, save_name=save_name, return_timings=True
        )
        motion_path = os.path.join(engine.output_dir, f"{save_name}_motions.pt")
        t_save0 = time.perf_counter()
        torch.save(pred_motions.float().cpu(), motion_path)
        save_motion_s = time.perf_counter() - t_save0
        total_s = time.perf_counter() - t_wall0
        timing = {
            "total_s": round(total_s, 4),
            "load_resample_audio_s": round(load_resample_audio_s, 4),
            "motion_inference_s": round(motion_inference_s, 4),
            "save_motion_pt_s": round(save_motion_s, 4),
            **rt,
        }
        timing["render_and_encode_s"] = round(
            rt["render_frames_s"] + rt["stack_tensors_s"] + rt["encode_video_s"], 4
        )
        print("\n=== timing (seconds) ===")
        for k, v in timing.items():
            print(f"  {k}: {v}")
