1. add GPU setting

import os
# Default GPUs 
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

2. add function

_chw_float_to_uint8_hwc()

rendered (tensor) -> Gradio Image (uint8 HWC)

3. iter_render_frames() in ARTAvatarInferEngine

Core changes. Yeild frame by frame

Using command "python inference.py --run_app --online" to run in frame-by-frame mode

Using command "python inference.py --run_app" to run in the original/default mode

4. Two paths: offline still uses engine.rendering() (original tqdm loop + write_video); online uses iter_render_frames, pushes each frame to Live preview, then write_video and saves the motion file.

5. When running in the online mode, there will be a live preview window on the website to show the updated frames (somehow)