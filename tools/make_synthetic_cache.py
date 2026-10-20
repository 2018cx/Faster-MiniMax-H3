"""Create deterministic public H3 cache fixtures for correctness and timing.

The tensors follow ``prepare_cache.py`` exactly, but are generated from a
fixed RNG instead of private media.  This keeps the optimization benchmark
self-contained and safe to redistribute.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def is_val_id(sample_id: str) -> bool:
    return int(hashlib.md5(sample_id.encode()).hexdigest()[:4], 16) % 20 == 0


def sample_ids(count: int, want_val: bool) -> list[str]:
    selected: list[str] = []
    candidate = 0
    while len(selected) < count:
        sample_id = f"{candidate:08d}"
        if is_val_id(sample_id) == want_val:
            selected.append(sample_id)
        candidate += 1
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--train-files-per-rank", type=int, default=1)
    parser.add_argument("--val-files-per-rank", type=int, default=0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=22)
    parser.add_argument("--text-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height % 32 or args.width % 32:
        raise SystemExit("height and width must be divisible by 32")
    if args.num_frames % 17 != 5:
        raise SystemExit("num_frames must satisfy num_frames % 17 == 5")
    if args.world_size < 1 or args.train_files_per_rank < 1 or args.val_files_per_rank < 0:
        raise SystemExit("invalid file-count argument")

    args.output.mkdir(parents=True, exist_ok=True)
    existing = sorted(args.output.glob("*.pt"))
    if existing:
        raise SystemExit(f"refusing to overwrite {len(existing)} existing fixture files under {args.output}")

    # H3's visual VAE maps 17*n+5 pixel frames to 5*n+2 latent frames.
    latent_frames = (args.num_frames - 5) // 17 * 5 + 2
    video_rows = latent_frames * (args.height // 32) * (args.width // 32)
    audio_frames = round(args.num_frames / 24 * 40)
    counts = {
        False: args.world_size * args.train_files_per_rank,
        True: args.world_size * args.val_files_per_rank,
    }
    all_ids = [(sid, split) for split, count in counts.items() for sid in sample_ids(count, split)]

    for index, (sample_id, is_val) in enumerate(all_ids):
        generator = torch.Generator(device="cpu").manual_seed(args.seed + index)
        item = {
            "video": torch.randn(video_rows, 96, generator=generator, dtype=torch.float32),
            "audio": torch.randn(audio_frames * 2, 32, generator=generator, dtype=torch.float32),
            "prompt": torch.randn(args.text_tokens, 5120, generator=generator, dtype=torch.bfloat16),
            "caption": f"deterministic synthetic {'validation' if is_val else 'training'} sample {sample_id}",
            "height": args.height,
            "width": args.width,
            "latent_frames": latent_frames,
            "audio_frames": audio_frames,
            "num_frames": args.num_frames,
        }
        path = args.output / f"fixture_{sample_id}.pt"
        torch.save(item, path)
        print(f"{path.name} split={'val' if is_val else 'train'} video={tuple(item['video'].shape)} "
              f"audio={tuple(item['audio'].shape)} text={tuple(item['prompt'].shape)}")

    packed_tokens = video_rows + audio_frames * 2 + args.text_tokens
    print(f"created={len(all_ids)} packed_tokens={packed_tokens} latent_frames={latent_frames} "
          f"audio_frames={audio_frames}")


if __name__ == "__main__":
    main()
