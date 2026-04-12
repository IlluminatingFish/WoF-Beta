1. add GPU setting

import os
Default GPUs 
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

2. stream_utils.chw_tensor_to_uint8_hwc()

rendered (tensor) -> preview uint8 HWC (Gradio / Flask)

3. iter_render_frames() in stream_render.py (not on the engine class)

Core change: yield frame by frame; Gradio/Flask import it; inference.rendering() uses the same function for one code path.

Using command "python inference.py --run_app --online" to run in frame-by-frame mode

Using command "python inference.py --run_app" to run in the original/default mode

4. Two paths: offline uses engine.rendering() (tqdm over iter_render_frames + write_video); online uses iter_render_frames directly in the UI, pushes each frame to Live preview, then write_video and saves the motion file.

5. When running in the online mode, there will be a live preview window on the website to show the updated frames (somehow)