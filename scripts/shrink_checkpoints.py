"""Shrink the trained checkpoints so they fit in a normal GitHub repo (no Git LFS needed).

Floating-point weights are stored as float16 (about half the size). The app converts them back to
float32 when loading, so nothing else changes. Predictions normally differ by far less than 0.001.

Usage (run on your computer, in the repo root):
    pip install torch
    python scripts/shrink_checkpoints.py --src "path/to/pneumonia_project/models" --dst models \
        --config "path/to/pneumonia_project/app_config.json"
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch

NAMES = ("resnet50", "densenet121", "efficientnet_b0")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing the original .pt checkpoints")
    ap.add_argument("--dst", default="models", help="output folder inside the repo")
    ap.add_argument("--config", help="optional path to app_config.json to copy into --dst")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    for name in NAMES:
        f = src / f"{name}.pt"
        if not f.exists():
            print(f"[skip] {f} not found")
            continue
        state = torch.load(f, map_location="cpu", weights_only=True)
        small, worst = {}, 0.0
        for k, v in state.items():
            if v.is_floating_point():
                half = v.half()
                worst = max(worst, (half.float() - v).abs().max().item())
                small[k] = half
            else:
                small[k] = v
        out = dst / f"{name}.pt"
        torch.save(small, out)
        mb_in, mb_out = f.stat().st_size / 1e6, out.stat().st_size / 1e6
        flag = "  <-- over GitHub's 100 MB limit!" if mb_out > 100 else ("  (over 50 MB: GitHub warns)" if mb_out > 50 else "")
        print(f"[ok] {name}: {mb_in:.1f} MB -> {mb_out:.1f} MB | max weight error {worst:.2e}{flag}")

    if args.config:
        shutil.copy(args.config, dst / "app_config.json")
        print(f"[ok] copied {args.config} -> {dst / 'app_config.json'}")


if __name__ == "__main__":
    main()
