#!/usr/bin/env python
"""
Incremental mesh / GAGAvatar rendering for online UIs (Gradio, Flask).
Lives outside inference.py so the core engine file stays batch-oriented.
"""

from __future__ import annotations

import torch


def iter_render_frames(engine, audio: torch.Tensor, pred_motions: torch.Tensor, shape_id="mesh", shape_code=None):
    """Yield each frame as CHW float (mesh: [0,1]; GAGAvatar: model range)."""
    if shape_id == "mesh":
        if shape_code is None:
            shape_code = audio.new_zeros(1, 300).to(engine.device).expand(pred_motions.shape[0], -1)
        else:
            assert shape_code.dim() == 2, f"Invalid shape_code dim: {shape_code.dim()}."
            assert shape_code.shape[0] == 1, f"Invalid shape_code shape: {shape_code.shape}."
            shape_code = shape_code.to(engine.device).expand(pred_motions.shape[0], -1)
        verts = engine.ARTalk.basic_vae.get_flame_verts(engine.flame_model, shape_code, pred_motions, with_global=True)
        for v in verts:
            rgb = engine.mesh_renderer(v[None])[0]
            yield rgb.cpu()[0] / 255.0
    else:
        engine.GAGAvatar.set_avatar_id(shape_id)
        for motion in pred_motions:
            batch = engine.GAGAvatar.build_forward_batch(motion[None], engine.GAGAvatar_flame)
            rgb = engine.GAGAvatar.forward_expression(batch)
            yield rgb.cpu()[0]
