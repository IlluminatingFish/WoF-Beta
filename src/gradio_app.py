#!/usr/bin/env python
"""Gradio UI entry (optional). Prefer flask_app.py for a lighter, self-controlled stack."""

import os
import time

import gradio as gr
import torch
import torchaudio
from gtts import gTTS

from app.utils_videos import IncrementalVideoWriter
from inference import ARTAvatarInferEngine
from stream_render import iter_render_frames
from stream_utils import StreamTimingLog, chw_tensor_to_uint8_hwc


def run_gradio_app(engine: ARTAvatarInferEngine, default_online: bool = False) -> None:
    def process_audio(input_type, audio_input, text_input, text_language, shape_id, style_id, render_mode):
        def bail():
            yield gr.skip(), gr.skip(), gr.skip()

        if input_type == "Audio" and audio_input is None:
            gr.Warning("Please upload an audio file")
            yield from bail()
            return
        if input_type == "Text" and (text_input is None or len(text_input.strip()) == 0):
            gr.Warning("Please input text content")
            yield from bail()
            return
        if input_type == "Text":
            gtts_lang = {
                "English": "en",
                "Chinese": "zh",
                "Japanese": "ja",
                "German": "de",
                "French": "fr",
                "Spanish": "es",
            }
            tts = gTTS(text=text_input, lang=gtts_lang[text_language])
            tts.save("./render_results/tts_output.wav")
            audio_input = "./render_results/tts_output.wav"
        audio, sr = torchaudio.load(audio_input)
        audio = torchaudio.transforms.Resample(sr, 16000)(audio).mean(dim=0)
        if style_id == "default":
            engine.style_motion = None
        else:
            engine.set_style_motion(style_id)
        pred_motions = engine.inference(audio)
        save_name = f'{audio_input.split("/")[-1].split(".")[0]}_{style_id.replace(".", "_")}_{shape_id.replace(".", "_")}'
        motion_path = os.path.join(engine.output_dir, "{}_motions.pt".format(save_name))
        video_path = os.path.join(engine.output_dir, "{}.mp4".format(save_name))

        if render_mode == "online":
            print("Rendering (streaming frames + incremental H.264)...")
            timing = StreamTimingLog("gradio_online")
            timing.start_after_motion()
            writer = IncrementalVideoWriter(video_path, 25.0)
            for i, rgb in enumerate(iter_render_frames(engine, audio, pred_motions, shape_id=shape_id)):
                t0 = time.perf_counter()
                preview = chw_tensor_to_uint8_hwc(rgb)
                t1 = time.perf_counter()
                convert_s = t1 - t0
                writer.write_frame(rgb)
                encode_s = 0.0
                timing.log_frame(i, convert_s, encode_s)
                yield gr.skip(), gr.skip(), preview
            print("Saving video (audio mux)...")
            audio_trim = audio[: int(writer.n_frames / 25.0 * 16000)]
            t_enc0 = time.perf_counter()
            writer.finalize(audio_trim, 16000, "aac")
            encode_video_s = time.perf_counter() - t_enc0
            torch.save(pred_motions.float().cpu(), motion_path)
            timing.finalize(writer.n_frames, encode_video_s)
            yield video_path, motion_path, gr.update(value=None)
        else:
            engine.rendering(audio, pred_motions, shape_id=shape_id, save_name=save_name)
            torch.save(pred_motions.float().cpu(), motion_path)
            yield video_path, motion_path, gr.update(value=None)

    if hasattr(engine, "GAGAvatar"):
        all_gagavatar_id = sorted(list(engine.GAGAvatar.all_gagavatar_id.keys()))
    else:
        all_gagavatar_id = []
    all_style_id = [os.path.basename(i) for i in os.listdir("assets/style_motion")]
    all_style_id = sorted([i.split(".")[0] for i in all_style_id if i.endswith(".pt")])
    with gr.Blocks(title="ARTalk: Speech-Driven 3D Head Animation via Autoregressive Model") as demo:
        gr.Markdown(
            """
            <center>
            <h1>ARTalk: Speech-Driven 3D Head Animation via Autoregressive Model</h1>
            </center>

            **ARTalk generates realistic 3D head motions from given audio, including accurate lip sync, natural facial animations, eye blinks, and head poses.**
            Please refer to our [paper](https://arxiv.org/abs/2502.20323), [project page](https://xg-chu.site/project_artalk), and [github](https://github.com/xg-chu/ARTalk) for more details about ARTalk.
            The apperance is powered by [GAGAvatar](https://xg-chu.site/project_gagavatar).

            Usage: Upload an audio file or input text -> Select an appearance and style -> Click generate!
            """
        )
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Input Audio & Text")
                input_type = gr.Radio(choices=["Audio", "Text"], value="Audio", label="Choose input type")
                audio_group = gr.Group()
                with audio_group:
                    audio_input = gr.Audio(type="filepath", label="Input Audio")
                text_group = gr.Group(visible=False)
                with text_group:
                    text_input = gr.Textbox(label="Input Text")
                    text_language = gr.Dropdown(
                        choices=["English", "Chinese", "Japanese", "German", "French", "Spanish"],
                        value="English",
                        label="Choose the language of the input text",
                    )
            with gr.Column():
                gr.Markdown("### Avatar Control")
                appearance = gr.Dropdown(
                    choices=["mesh"] + all_gagavatar_id,
                    value="mesh",
                    label="Choose the apperance of the speaker",
                )
                style = gr.Dropdown(
                    choices=["default"] + all_style_id,
                    value="natural_0",
                    label="Choose the style of the speaker",
                )
                render_mode = gr.Radio(
                    choices=["offline", "online"],
                    value="online" if default_online else "offline",
                    label="Render mode",
                    info="online: stream each frame to Live preview, then save MP4. offline: original batch render (same as CLI).",
                )
            with gr.Column():
                gr.Markdown("### Generated Video")
                live_preview = gr.Image(label="Live preview (online mode)", type="numpy")
                video_output = gr.Video(autoplay=True)
                motion_output = gr.File(label="motion sequence", file_types=[".pt"])

        inputs = [input_type, audio_input, text_input, text_language, appearance, style, render_mode]
        btn = gr.Button("Generate")
        btn.click(fn=process_audio, inputs=inputs, outputs=[video_output, motion_output, live_preview])

        ex_off = "offline"
        if hasattr(engine, "GAGAvatar"):
            examples = [
                ["Audio", "demo/jp1.wav", None, None, "12.jpg", "curious_0", ex_off],
                ["Audio", "demo/jp2.wav", None, None, "12.jpg", "natural_3", ex_off],
                ["Audio", "demo/eng1.wav", None, None, "12.jpg", "natural_2", ex_off],
                ["Audio", "demo/eng2.wav", None, None, "12.jpg", "happy_1", ex_off],
                ["Audio", "demo/cn1.wav", None, None, "11.jpg", "natural_1", ex_off],
                ["Audio", "demo/cn2.wav", None, None, "12.jpg", "happy_2", ex_off],
                ["Text", None, "Hello, this is a demo of ARTalk! Let's create something fun together.", "English", "12.jpg", "happy_0", ex_off],
                ["Text", None, "Bonjour, voici une courte démo.", "French", "12.jpg", "natural_0", ex_off],
            ]
        else:
            examples = [
                ["Audio", "demo/jp1.wav", None, None, "mesh", "curious_0", ex_off],
                ["Audio", "demo/jp2.wav", None, None, "mesh", "natural_3", ex_off],
                ["Audio", "demo/eng1.wav", None, None, "mesh", "natural_2", ex_off],
                ["Audio", "demo/eng2.wav", None, None, "mesh", "happy_1", ex_off],
                ["Audio", "demo/cn1.wav", None, None, "mesh", "natural_1", ex_off],
                ["Audio", "demo/cn2.wav", None, None, "mesh", "happy_2", ex_off],
                ["Text", None, "Hello, this is a demo of ARTalk! Let's create something fun together.", "English", "mesh", "happy_0", ex_off],
                ["Text", None, "Bonjour, voici une courte démo.", "French", "mesh", "natural_0", ex_off],
            ]
        gr.Examples(examples=examples, inputs=inputs, outputs=video_output)

        def toggle_input(choice):
            if choice == "Audio":
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)

        input_type.change(fn=toggle_input, inputs=[input_type], outputs=[audio_group, text_group])

    demo.launch(server_name="0.0.0.0", server_port=8960, share=True)


if __name__ == "__main__":
    import argparse

    torch.set_float32_matmul_precision("high")
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true", help="Default render mode to online in the UI.")
    p.add_argument(
        "-l",
        "--clip_length",
        default=None,
        type=int,
        help="Max motion frames; default from audio at 25 fps.",
    )
    args = p.parse_args()
    eng = ARTAvatarInferEngine(load_gaga=True, fix_pose=False, clip_length=args.clip_length)
    run_gradio_app(eng, default_online=args.online)
