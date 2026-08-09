"""Pair cache: concatenate two consecutively-shot clips into one long sample.

Built for training on ~30 s story-continuous material assembled from pairs of
~15 s clips (clip A's ending flows into clip B's opening — same scene, same
story). Each manifest line describes one pair; the builder decodes both clips,
concatenates on the time axis, re-aligns the frame count to 17n+5 (cap 719 ≈
30 s @ 24 fps), and encodes video + audio + merged caption exactly like
prepare_cache.py. The cached item carries `seam_frame` (index of the first
frame that came from clip B) for any later seam-aware analysis.

Manifest (jsonl), one pair per line:
  {"id": ..., "video_a": path, "video_b": path, "caption": merged prompt,
   "width": W, "height": H, "est_frames": int (sort/shard key),
   "bucket": optional tag, "pair_id": optional}

Two-pass usage (so the 64 GB text encoder never shares a GPU with the VAEs):
  python prepare_cache_pairs.py --stage text  --manifest m.jsonl --output cache/ --model /path/MiniMax-H3
  python prepare_cache_pairs.py --stage video --manifest m.jsonl --output cache/ --model /path/MiniMax-H3 \
      --rank 0 --world 8   # video stage shards across ranks/GPUs
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio
from PIL import Image

from diffusers import AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio
from diffusers.modular_pipelines.minimax_h3.packing import (
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    audio_latent_num_frames,
    patchify_video_latents,
)
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration

MAX_FRAMES = 719  # 17*42+5 ≈ 30 s @ 24 fps


def align_17n5(n: int) -> int:
    return ((n - 5) // 17) * 17 + 5 if n >= 22 else 0


def decode_cover(path: str, target_w: int, target_h: int, max_frames: int):
    """Cover-fit decode (scale to cover + center crop, never stretch)."""
    container = av.open(path)
    frames = []
    for frame in container.decode(container.streams.video[0]):
        img = frame.to_image().convert("RGB")
        w, h = img.size
        scale = max(target_w / w, target_h / h)
        rw, rh = math.ceil(w * scale), math.ceil(h * scale)
        img = img.resize((rw, rh), Image.Resampling.BICUBIC)
        left, top = (rw - target_w) // 2, (rh - target_h) // 2
        img = img.crop((left, top, left + target_w, top + target_h))
        frames.append(np.asarray(img, dtype=np.float32) / 255.0)
        if len(frames) >= max_frames:
            break
    container.close()
    return frames


def extract_wav(video_path: str, tmpdir: str, tag: str) -> Path | None:
    # tag ('a'/'b') keeps the two clips' temp wavs distinct even when their
    # filenames collide (e.g. both named video.mp4 in sibling directories).
    out = Path(tmpdir) / f"{tag}.wav"
    r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                        "-vn", "-ac", "2", "-ar", "32000", str(out)],
                       capture_output=True, timeout=120)
    return out if (r.returncode == 0 and out.exists() and out.stat().st_size > 1000) else None


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--stage", choices=("text", "video"), required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    emb_path = args.output.parent / f"embeds_pairs_rank{args.rank}.pt"

    rows = [json.loads(l) for l in args.manifest.open() if l.strip()]
    rows.sort(key=lambda r: -r["est_frames"])  # longest first: OOM/capacity issues surface immediately
    rows = rows[args.rank :: args.world]
    if args.stage == "video":
        rows = [r for r in rows if not (args.output / f"{min(r['est_frames'], 9999):04d}_{r['id']}.pt").exists()]
    if not rows:
        print(f"[rank {args.rank}] nothing to do", flush=True)
        return
    print(f"[rank {args.rank}] {len(rows)} pairs", flush=True)

    if args.stage == "text":
        te = Qwen3VLForConditionalGeneration.from_pretrained(args.model / "text_encoder", torch_dtype=torch.bfloat16).to(device)
        tok = Qwen2TokenizerFast.from_pretrained(args.model / "tokenizer")
        embeds = {}
        for i, r in enumerate(rows):
            ids = tok(r["caption"], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            out = te.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, output_hidden_states=True)
            embeds[r["id"]] = out.hidden_states[50][0].to(torch.bfloat16).cpu()
            if i % 20 == 0:
                print(f"[rank {args.rank}] text {i}/{len(rows)}", flush=True)
        torch.save(embeds, emb_path)
        print(f"TEXT_STAGE_DONE rank={args.rank} n={len(embeds)}", flush=True)
        return

    # ---- video stage: two-clip concat + both VAEs ----
    embeds = {}
    for p in sorted(args.output.parent.glob("embeds_pairs_rank*.pt")):
        embeds.update(torch.load(p, map_location="cpu", weights_only=False))
    vae = AutoencoderKLMiniMaxH3.from_pretrained(args.model, subfolder="vae", torch_dtype=torch.float32).to(device)
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    avae = AutoencoderKLMiniMaxH3Audio.from_pretrained(args.model, subfolder="audio_vae", torch_dtype=torch.float32).to(device).eval()
    mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device)[:, None, None, None]
    std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device)[:, None, None, None]
    lat_mean = torch.as_tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
    lat_std = torch.as_tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
    a_mean = torch.as_tensor(avae.config.latents_mean).view(1, -1, 1).to(device)
    a_std = torch.as_tensor(avae.config.latents_std).view(1, -1, 1).to(device)

    done = skipped = 0
    for i, r in enumerate(rows):
        if r["id"] not in embeds:
            print(f"[rank {args.rank}] no embed for {r['id']}, skip", flush=True)
            skipped += 1
            continue
        try:
            fa = decode_cover(r["video_a"], r["width"], r["height"], MAX_FRAMES)
            fb = decode_cover(r["video_b"], r["width"], r["height"], max(0, MAX_FRAMES - len(fa)))
        except Exception as exc:
            print(f"[rank {args.rank}] decode fail {r['id']}: {exc}", flush=True)
            skipped += 1
            continue
        n_a = len(fa)
        frames = fa + fb
        keep = align_17n5(len(frames))
        if not keep:
            skipped += 1
            continue
        video = torch.from_numpy(np.stack(frames[:keep])).permute(3, 0, 1, 2).contiguous()
        del frames, fa, fb
        n_frames = video.shape[1]
        latent = vae.encode(((video.to(device) - mean) / std)[None]).latent_dist.mode()
        latent = (latent - lat_mean) / lat_std
        video_rows = patchify_video_latents(latent.float(), (1, 2, 2))
        n_audio = audio_latent_num_frames(n_frames)
        # Audio: extract both tracks, cut each to its clip's kept-frame length,
        # concatenate, then encode (channel-major rows, zero rows on failure).
        audio_rows = torch.zeros(n_audio * 2, 32)
        try:
            with tempfile.TemporaryDirectory() as td:
                wa, wb = extract_wav(r["video_a"], td, "a"), extract_wav(r["video_b"], td, "b")
                pieces = []
                for w, nf in ((wa, n_a), (wb, n_frames - n_a)):
                    if nf <= 0:
                        continue
                    need = int(round(nf / 24.0 * 32000))
                    if w is not None:
                        wav, sr = torchaudio.load(str(w))
                        if wav.shape[0] == 1:
                            wav = wav.repeat(2, 1)
                        if wav.shape[1] < need:
                            wav = torch.nn.functional.pad(wav, (0, need - wav.shape[1]))
                        pieces.append(wav[:, :need])
                    else:
                        pieces.append(torch.zeros(2, need))
                if pieces:
                    wave = torch.cat(pieces, dim=1)
                    need = n_audio * 800  # 800x downsample aligns to the latent grid
                    if wave.shape[1] < need:
                        wave = torch.nn.functional.pad(wave, (0, need - wave.shape[1]))
                    wave = wave[:, :need]
                    if wave.abs().sum() > 0:
                        post = avae.encode(wave.to(device)[:, None], return_dict=False)[0]
                        lat_a = ((post.mode().float() - a_mean) / a_std).transpose(1, 2)
                        if lat_a.shape[1] < n_audio:
                            lat_a = torch.nn.functional.pad(lat_a, (0, 0, 0, n_audio - lat_a.shape[1]))
                        audio_rows = lat_a[:, :n_audio].reshape(-1, lat_a.shape[-1]).cpu().contiguous()
        except Exception as exc:
            print(f"[rank {args.rank}] audio fail {r['id']}: {str(exc)[:80]} (keeping zero rows)", flush=True)
        item = {
            "video": video_rows.cpu().to(torch.float32),
            "audio": audio_rows,
            "prompt": embeds[r["id"]],
            "caption": r["caption"],
            "height": r["height"], "width": r["width"],
            "latent_frames": latent.shape[2], "audio_frames": n_audio,
            "bucket": r.get("bucket"), "num_frames": n_frames,
            "pair_id": r.get("pair_id"), "seam_frame": n_a,
        }
        torch.save(item, args.output / f"{min(r['est_frames'], 9999):04d}_{r['id']}.pt")
        done += 1
        print(f"[rank {args.rank}] video {i}/{len(rows)} f={n_frames} seam={n_a} rows={tuple(video_rows.shape)}", flush=True)
        del video, latent, video_rows
        torch.cuda.empty_cache()
    print(f"VIDEO_STAGE_DONE rank={args.rank} done={done} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
