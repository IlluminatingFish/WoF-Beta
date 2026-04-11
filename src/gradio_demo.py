#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import sys
import json
import time
import torch
import argparse
import torchaudio
import numpy as np
import gradio as gr
from gtts import gTTS
from tqdm import tqdm

from app import BitwiseARModel
from app.flame_model import FLAMEModel, RenderMesh
from app.utils_videos import write_video


class ARTAvatarInferEngine:
    def __init__(self, load_gaga=False, fix_pose=False, clip_length=750, device='cuda'):
        self.device = device
        self.fix_pose = fix_pose
        self.clip_length = clip_length
        audio_encoder = 'wav2vec'
        ckpt = torch.load('./assets/ARTalk_{}.pt'.format(audio_encoder), map_location='cpu', weights_only=True)
        configs = json.load(open("./assets/config.json"))
        configs['AR_CONFIG']['AUDIO_ENCODER'] = audio_encoder
        self.generator = BitwiseARModel(configs).eval().to(device)
        self.generator.load_state_dict(ckpt, strict=True)
        self.flame_model = FLAMEModel(n_shape=300, n_exp=100, scale=1.0, no_lmks=True).to(device)
        self.mesh_renderer = RenderMesh(image_size=512, faces=self.flame_model.get_faces(), scale=1.0)
        
        self.output_dir = 'render_results/ARTAvatar_{}'.format(audio_encoder)
        os.makedirs(self.output_dir, exist_ok=True)
        self.style_motion = None

        if load_gaga:
            from app.GAGAvatar import GAGAvatar
            self.GAGAvatar = GAGAvatar().to(device)
            self.GAGAvatar_flame = FLAMEModel(n_shape=300, n_exp=100, scale=5.0, no_lmks=True).to(device)

        tracker_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'GAGAvatar_track')
        if os.path.isdir(tracker_root):
            if tracker_root not in sys.path:
                sys.path.insert(0, tracker_root)
            from engines import CoreEngine
            self.tracker = CoreEngine(focal_length=12.0, device=device)
            print('GAGAvatar tracker loaded.')
        else:
            self.tracker = None
            print('GAGAvatar_track not found, custom avatar upload disabled.')

    def set_style_motion(self, style_motion):
        if isinstance(style_motion, str):
            style_motion = torch.load('assets/style_motion/{}.pt'.format(style_motion), map_location='cpu', weights_only=True)
        assert style_motion.shape == (50, 106), f'Invalid style_motion shape: {style_motion.shape}.'
        self.style_motion = style_motion[None].to(self.device)

    def inference(self, audio, clip_length=None):
        audio_batch = {'audio': audio[None].to(self.device), 'style_motion': self.style_motion}
        print('Inferring motion...')
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred_motions = self.generator.inference(audio_batch, with_gtmotion=False)[0]
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        clip_length = clip_length if clip_length is not None else self.clip_length
        pred_motions = self.smooth_motion_savgol(pred_motions)[:clip_length]
        if self.fix_pose:
            pred_motions[..., 100:103] *= 0.0
        pred_motions[..., 104:] *= 0.0
        n_frames = pred_motions.shape[0]
        motion_time = t1 - t0
        motion_fps = n_frames / motion_time
        print(f'Motion inference: {n_frames} frames in {motion_time:.2f}s ({motion_fps:.1f} FPS)')
        return pred_motions, {'motion_frames': n_frames, 'motion_time': motion_time, 'motion_fps': motion_fps}

    def rendering(self, audio, pred_motions, shape_id="mesh", shape_code=None, save_name='ARTAvatar.mp4'):
        print(f'Rendering ({shape_id})...')
        pred_images = []
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if shape_id == "mesh":
            if shape_code is None:
                shape_code = audio.new_zeros(1, 300).to(self.device).expand(pred_motions.shape[0], -1)
            else:
                assert shape_code.dim() == 2, f'Invalid shape_code dim: {shape_code.dim()}.'
                assert shape_code.shape[0] == 1, f'Invalid shape_code shape: {shape_code.shape}.'
                shape_code = shape_code.to(self.device).expand(pred_motions.shape[0], -1)
            verts = self.generator.basic_vae.get_flame_verts(self.flame_model, shape_code, pred_motions, with_global=True)
            for v in tqdm(verts):
                rgb = self.mesh_renderer(v[None])[0]
                pred_images.append(rgb.cpu()[0] / 255.0)
        else:
            self.GAGAvatar.set_avatar_id(shape_id)
            for motion in tqdm(pred_motions):
                batch = self.GAGAvatar.build_forward_batch(motion[None], self.GAGAvatar_flame)
                rgb = self.GAGAvatar.forward_expression(batch)
                pred_images.append(rgb.cpu()[0])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        n_frames = len(pred_images)
        render_time = t1 - t0
        render_fps = n_frames / render_time
        print(f'Render ({shape_id}): {n_frames} frames in {render_time:.2f}s ({render_fps:.1f} FPS)')
        print('Saving video...')
        pred_images = torch.stack(pred_images)
        dump_path = os.path.join(self.output_dir, '{}.mp4'.format(save_name))
        t2 = time.perf_counter()
        audio = audio[:int(pred_images.shape[0]/25.0*16000)]
        write_video(pred_images*255.0, dump_path, 25.0, audio, 16000, "aac")
        t3 = time.perf_counter()
        encode_time = t3 - t2
        print(f'Video encoding: {encode_time:.2f}s')
        return {'render_frames': n_frames, 'render_time': render_time, 'render_fps': render_fps, 'encode_time': encode_time}

    def get_avatar_preview(self, avatar_id):
        if not hasattr(self, 'GAGAvatar') or avatar_id not in self.GAGAvatar.all_gagavatar_id:
            return None
        image = self.GAGAvatar.all_gagavatar_id[avatar_id]['image']
        if isinstance(image, np.ndarray):
            image = (image * 255).clip(0, 255).astype(np.uint8)
            if image.shape[0] == 3:
                image = image.transpose(1, 2, 0)
        elif isinstance(image, torch.Tensor):
            image = image.clamp(0, 1).mul(255).byte()
            if image.dim() == 3 and image.shape[0] == 3:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()
        return image

    def track_custom_avatar(self, image_path):
        if self.tracker is None:
            raise RuntimeError("GAGAvatar_track not available. Custom avatar upload is disabled.")
        import torchvision as tv
        inp_image = tv.io.read_image(image_path, mode=tv.io.ImageReadMode.RGB).to(self.device).float()
        avatar_key = os.path.basename(image_path)
        results = self.tracker.track_image([inp_image], [avatar_key])
        if results is None or avatar_key not in results:
            raise RuntimeError("Face detection failed on the uploaded image. Please try a clearer headshot.")
        tracked_data = results[avatar_key]
        tracked_data.pop('vis_image', None)
        self.GAGAvatar.all_gagavatar_id[avatar_key] = tracked_data
        return avatar_key

    @staticmethod
    def smooth_motion_savgol(motion_codes):
        from scipy.signal import savgol_filter
        motion_np = motion_codes.clone().detach().cpu().numpy()
        motion_np_smoothed = savgol_filter(motion_np, window_length=5, polyorder=2, axis=0)
        motion_np_smoothed[..., 100:103] = savgol_filter(motion_np[..., 100:103], window_length=9, polyorder=3, axis=0)
        return torch.tensor(motion_np_smoothed).type_as(motion_codes)


def run_gradio_app(engine):
    if hasattr(engine, 'GAGAvatar'):
        all_gagavatar_id = sorted(engine.GAGAvatar.all_gagavatar_id.keys())
    else:
        all_gagavatar_id = []
    all_style_id = [os.path.basename(i) for i in os.listdir('assets/style_motion')]
    all_style_id = sorted([i.split('.')[0] for i in all_style_id if i.endswith('.pt')])
    default_avatar = all_gagavatar_id[0] if all_gagavatar_id else None
    default_preview = engine.get_avatar_preview(default_avatar) if default_avatar else None

    def on_preset_select(avatar_id):
        preview = engine.get_avatar_preview(avatar_id)
        return preview, avatar_id

    def on_custom_upload(image_path):
        if image_path is None:
            return None, None, gr.update()
        try:
            new_key = engine.track_custom_avatar(image_path)
            preview = engine.get_avatar_preview(new_key)
            gr.Info(f"Successfully tracked custom avatar: {new_key}")
            return preview, new_key, gr.update(choices=sorted(engine.GAGAvatar.all_gagavatar_id.keys()), value=new_key)
        except Exception as e:
            gr.Warning(str(e))
            return None, gr.update(), gr.update()

    def process_audio(input_type, audio_input, text_input, text_language, avatar_id, style_id):
        if input_type == "Audio" and audio_input is None:
            gr.Warning("Please upload an audio file")
            return None, None, ""
        if input_type == "Text" and (text_input is None or len(text_input.strip()) == 0):
            gr.Warning("Please input text content")
            return None, None, ""
        if avatar_id is None:
            gr.Warning("Please select or upload an avatar")
            return None, None, ""
        if input_type == "Text":
            gtts_lang = {"English": "en", "中文": "zh", "日本語": "ja", "Deutsch": "de", "Français": "fr", "Español": "es"}
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

        base_name = audio_input.split("/")[-1].split(".")[0]
        style_tag = style_id.replace(".", "_")
        avatar_tag = avatar_id.replace(".", "_")

        mesh_save_name = f'{base_name}_{style_tag}_mesh'
        mesh_stats = engine.rendering(audio, pred_motions, shape_id="mesh", save_name=mesh_save_name)
        mesh_video = os.path.join(engine.output_dir, '{}.mp4'.format(mesh_save_name))

        avatar_save_name = f'{base_name}_{style_tag}_{avatar_tag}'
        avatar_stats = engine.rendering(audio, pred_motions, shape_id=avatar_id, save_name=avatar_save_name)
        avatar_video = os.path.join(engine.output_dir, '{}.mp4'.format(avatar_save_name))

        pipeline_end = time.perf_counter()
        total_time = pipeline_end - pipeline_start
        n_frames = motion_stats['motion_frames']
        e2e_fps = n_frames / total_time

        bench = (
            f"### Benchmark Results\n"
            f"| Stage | Frames | Time (s) | FPS |\n"
            f"|---|---|---|---|\n"
            f"| Motion Generation | {n_frames} | {motion_stats['motion_time']:.2f} | {motion_stats['motion_fps']:.1f} |\n"
            f"| Mesh Rendering | {mesh_stats['render_frames']} | {mesh_stats['render_time']:.2f} | {mesh_stats['render_fps']:.1f} |\n"
            f"| Avatar Rendering | {avatar_stats['render_frames']} | {avatar_stats['render_time']:.2f} | {avatar_stats['render_fps']:.1f} |\n"
            f"| Mesh Video Encode | — | {mesh_stats['encode_time']:.2f} | — |\n"
            f"| Avatar Video Encode | — | {avatar_stats['encode_time']:.2f} | — |\n"
            f"| **End-to-End** | **{n_frames}** | **{total_time:.2f}** | **{e2e_fps:.1f}** |\n\n"
            f"Audio duration: {audio_duration:.1f}s &nbsp; | &nbsp; "
            f"Real-time factor: {total_time / audio_duration:.2f}x"
        )

        return mesh_video, avatar_video, bench

    with gr.Blocks() as demo:
        active_avatar_id = gr.State(value=default_avatar)
        gr.Markdown("""
            ## Speech-Driven 3D Head Animation Demo

            **Workflow Overview:**

            We generate realistic 3D talking head animations from audio or text input through a three-stage pipeline:

            1. **Input Stage** — Provide speech input either as an audio file (WAV/MP3) or as text (which we convert to speech via TTS).
            2. **Motion Generation** — We use an autoregressive model to analyze the audio and produce a sequence of facial motion parameters, including lip synchronization, natural facial animations, eye blinks, and head pose dynamics.
            3. **Rendering Stage** — We render the motion sequence into two videos simultaneously:
               - **FLAME Mesh** — a 3D wireframe mesh visualization of the facial motion.
               - **Avatar** — a photorealistic talking head rendered with Gaussian Splatting.

            **Customization**: Select different avatar appearances and motion styles to control the output characteristics.
        """)

        with gr.Row():
            # ── Column 1: Speech Input ──
            with gr.Column(scale=1):
                gr.Markdown("### Step 1: Speech Input")
                input_type = gr.Radio(
                    choices=["Audio", "Text"],
                    value="Audio",
                    label="Input Mode"
                )

                audio_group = gr.Group()
                with audio_group:
                    audio_input = gr.Audio(type="filepath", label="Upload or Record Audio")

                text_group = gr.Group(visible=False)
                with text_group:
                    text_input = gr.Textbox(
                        label="Text Content",
                        placeholder="Enter the text you want to synthesize...",
                        lines=3
                    )
                    text_language = gr.Dropdown(
                        choices=["English", "中文", "日本語", "Deutsch", "Français", "Español"],
                        value="English",
                        label="Language"
                    )

                gr.Markdown("**Motion Style**")
                style = gr.Dropdown(
                    choices=["default"] + all_style_id,
                    value="natural_0",
                    label="Select Style",
                    info="Choose the speaking style and mannerisms"
                )

            # ── Column 2: Avatar Selection + Preview ──
            with gr.Column(scale=1):
                gr.Markdown("### Step 2: Choose Avatar")

                avatar_source = gr.Radio(
                    choices=["Preset", "Upload Headshot"],
                    value="Preset",
                    label="Avatar Source"
                )

                preset_group = gr.Group()
                with preset_group:
                    appearance = gr.Dropdown(
                        choices=all_gagavatar_id,
                        value=default_avatar,
                        label="Select Preset Avatar",
                        info="Choose from pre-tracked avatar identities"
                    )

                upload_group = gr.Group(visible=False)
                with upload_group:
                    custom_upload = gr.Image(
                        type="filepath",
                        label="Upload Headshot",
                        sources=["upload"],
                    )
                    gr.Markdown("*Upload a clear, front-facing headshot. We will automatically track the face geometry for rendering.*")

                avatar_preview = gr.Image(
                    value=default_preview,
                    label="Avatar Preview",
                    interactive=False,
                    height=256,
                )

                btn = gr.Button("Generate Animation", variant="primary", size="lg")

        gr.Markdown("---")
        gr.Markdown("### Step 3: View Results")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("**FLAME Mesh**")
                mesh_output = gr.Video(autoplay=True, label="Mesh Rendering")

            with gr.Column(scale=1):
                gr.Markdown("**Photorealistic Avatar**")
                avatar_output = gr.Video(autoplay=True, label="Avatar Rendering")

        benchmark_output = gr.Markdown(value="", label="Benchmark")

        # ── Event wiring ──
        inputs = [input_type, audio_input, text_input, text_language, active_avatar_id, style]
        btn.click(fn=process_audio, inputs=inputs, outputs=[mesh_output, avatar_output, benchmark_output])

        appearance.change(
            fn=on_preset_select,
            inputs=[appearance],
            outputs=[avatar_preview, active_avatar_id],
        )

        custom_upload.change(
            fn=on_custom_upload,
            inputs=[custom_upload],
            outputs=[avatar_preview, active_avatar_id, appearance],
        )

        def toggle_input(choice):
            if choice == "Audio":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        input_type.change(
            fn=toggle_input, inputs=[input_type], outputs=[audio_group, text_group]
        )

        def toggle_avatar_source(choice):
            if choice == "Preset":
                return gr.update(visible=True), gr.update(visible=False)
            else:
                return gr.update(visible=False), gr.update(visible=True)

        avatar_source.change(
            fn=toggle_avatar_source,
            inputs=[avatar_source],
            outputs=[preset_group, upload_group],
        )

        # ── Examples ──
        gr.Markdown("---")
        gr.Markdown("### Try These Examples")
        examples = [
            ["Audio", "demo/jp1.wav", None, None, "12.jpg", "curious_0"],
            ["Audio", "demo/jp2.wav", None, None, "12.jpg", "natural_3"],
            ["Audio", "demo/eng1.wav", None, None, "12.jpg", "natural_2"],
            ["Audio", "demo/eng2.wav", None, None, "12.jpg", "happy_1"],
            ["Audio", "demo/cn1.wav", None, None, "11.jpg", "natural_1"],
            ["Audio", "demo/cn2.wav", None, None, "12.jpg", "happy_2"],
            ["Text", None, "Hello, this is a demo! Let's create something fun together.", "English", "12.jpg", "happy_0"],
            ["Text", None, "让我们一起创造一些有趣的东西吧。", "中文", "12.jpg", "natural_0"],
        ]
        example_inputs = [input_type, audio_input, text_input, text_language, appearance, style]
        gr.Examples(examples=examples, inputs=example_inputs, outputs=[mesh_output, avatar_output, benchmark_output])

    demo.launch(server_name="0.0.0.0", server_port=8961, share=True)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_path', '-a', default=None, type=str)
    parser.add_argument('--clip_length', '-l', default=750, type=int)
    parser.add_argument("--shape_id", '-i', default='mesh', type=str)
    parser.add_argument("--style_id", '-s', default='default', type=str)
    parser.add_argument("--run_app", action='store_true')
    args = parser.parse_args()

    engine = ARTAvatarInferEngine(load_gaga=True, fix_pose=False, clip_length=args.clip_length)
    if args.run_app:
        run_gradio_app(engine)
    else:
        shape_id = 'mesh' if args.shape_id not in engine.GAGAvatar.all_gagavatar_id.keys() else args.shape_id
        audio, sr = torchaudio.load(args.audio_path)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)

        base_name = os.path.splitext(os.path.basename(args.audio_path))[0]
        save_name = f'{base_name}_{args.style_id.replace(".", "_")}_{args.shape_id.replace(".", "_")}'
        engine.set_style_motion(args.style_id)
        pred_motions, motion_stats = engine.inference(audio)
        render_stats = engine.rendering(audio, pred_motions, shape_id=args.shape_id, save_name=save_name)
        total_fps = motion_stats['motion_frames'] / (motion_stats['motion_time'] + render_stats['render_time'] + render_stats['encode_time'])
        print(f'\n=== Benchmark Summary ===')
        print(f'Motion:    {motion_stats["motion_fps"]:.1f} FPS ({motion_stats["motion_time"]:.2f}s)')
        print(f'Render:    {render_stats["render_fps"]:.1f} FPS ({render_stats["render_time"]:.2f}s)')
        print(f'Encode:    {render_stats["encode_time"]:.2f}s')
        print(f'End-to-End: {total_fps:.1f} FPS')
