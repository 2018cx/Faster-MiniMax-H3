"""Compatibility shims for the two public MiniMax-H3 Diffusers layouts.

The fine-tuning baseline was authored against the initial H3 integration,
where packing helpers lived in ``minimax_h3.packing``.  The later public
integration moved the same operations into the modular pipeline steps.  Keep
one trainer source compatible with both without modifying either dependency.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

try:
    from diffusers.modular_pipelines.minimax_h3.packing import (  # type: ignore[attr-defined]
        MINIMAX_H3_PIXEL_MEAN,
        MINIMAX_H3_PIXEL_STD,
        align_num_frames,
        audio_latent_num_frames,
        build_packed_sequence,
        build_row_timesteps,
        patchify_video_latents,
    )
except ModuleNotFoundError:
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
        MiniMaxH3SetTimestepsStep,
        patchify_video_latents,
    )
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        align_num_frames as _align_num_frames,
        audio_latent_num_frames,
    )

    MINIMAX_H3_PIXEL_MEAN = (0.485, 0.456, 0.406)
    MINIMAX_H3_PIXEL_STD = (0.229, 0.224, 0.225)

    class PackedSequence(NamedTuple):
        position_ids: torch.Tensor
        token_tags: torch.Tensor
        video_indices: torch.Tensor
        audio_indices: torch.Tensor
        text_indices: torch.Tensor
        num_condition_video_rows: int
        num_condition_audio_rows: int

        @property
        def sequence_length(self) -> int:
            return int(self.token_tags.numel())

    def align_num_frames(num_frames: int) -> int:
        return _align_num_frames(num_frames, frames_per_chunk=17, latents_per_chunk=5)

    def build_packed_sequence(
        text_token_tags: torch.Tensor,
        num_latent_frames: int,
        latent_height: int,
        latent_width: int,
        num_audio_latents: int,
        patch_size: tuple[int, int, int],
    ) -> PackedSequence:
        values = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags,
            num_latent_frames,
            latent_height,
            latent_width,
            num_audio_latents,
            patch_size,
            audio_channels=2,
            audio_tag=2,
            video_tag=0,
        )
        return PackedSequence(*values)

    def build_row_timesteps(
        layout: PackedSequence,
        video_timestep: float,
        audio_timestep: float,
        condition_video_timestep: float,
        condition_audio_timestep: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return MiniMaxH3SetTimestepsStep.build_row_timesteps(
            layout.video_indices,
            layout.audio_indices,
            layout.num_condition_video_rows,
            layout.num_condition_audio_rows,
            int(layout.sequence_length - layout.video_indices.numel() - layout.audio_indices.numel()),
            video_timestep,
            audio_timestep,
            condition_video_timestep,
            condition_audio_timestep,
        )
