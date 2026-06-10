"""inference script for DeLocNet (PETCalibrator).

Loads a trained DeLocNet model, runs it on a folder of PET flood-map
``.npy`` files, and writes the predicted 2D peak coordinates plus a
visualization overlay.

This is the official inference entry point for the model introduced in:
    Li et al., "A Geometric Context Fusion De-bias Learning Framework for
    Resilient Positron Emission Tomography Detector Calibration",
    Information Fusion, 2026.
"""

from __future__ import annotations

import argparse
import os
from typing import Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.initialize import *  # noqa: F401,F403  (seeds + project-root path)
import models.DeLocNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeLocNet inference on PET flood maps and dump peak coordinates."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./dataset/test/img",
        help="Directory containing flood-map .npy files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./inference_results",
        help="Directory where predicted coordinates and visualisations are saved.",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default="./checkpoints/DeLocNet_best.pth",
        help="Path to the trained DeLocNet state-dict file.",
    )
    parser.add_argument(
        "--mean_model",
        type=str,
        default="./checkpoints/mean_model.pth",
        help="Path to the pre-computed mean model tensor (the MMFM prior).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for inference.",
    )
    parser.add_argument(
        "--no_visualisation",
        action="store_true",
        help="If set, skip writing PNG visualisations (only coordinates are saved).",
    )
    return parser.parse_args()


def load_flood_maps(input_dir: str) -> Tuple[list, list]:
    """Read every .npy file in `input_dir` and return (paths, arrays)."""
    paths = sorted(
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".npy")
    )
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")
    arrays = [np.load(p) for p in paths]
    return paths, arrays


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo, hi = float(x.min()), float(x.max())
    if hi > lo:
        return (x - lo) / (hi - lo)
    return x


def build_model(args) -> torch.nn.Module:
    print(f"[infer] loading DeLocNet from {args.mean_model}")
    model = models.DeLocNet.build_DeLocNet(args.mean_model)
    print(f"[infer] loading weights from {args.model_weights}")
    state_dict = torch.load(args.model_weights, map_location=args.device)
    model.load_state_dict(state_dict, strict=True)
    model.to(args.device)
    model.eval()
    return model


@torch.no_grad()
def predict_single(model: torch.nn.Module, flood: np.ndarray, device: str) -> np.ndarray:
    """Run DeLocNet on a single flood map and return the (256, 2) peak coordinates."""
    flood_norm = normalize_01(flood)
    x = torch.from_numpy(flood_norm)[None, None, :, :].to(torch.float32).to(device)
    out = model(x)
    coords = out.reshape(256, 2).cpu().numpy()
    return coords


def save_visualisation(flood: np.ndarray, coords: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(flood, cmap="viridis", alpha=1.0)
    ax.scatter(coords[:, 1], coords[:, 0], s=8, c="red", marker="x", label="DeLocNet")
    ax.set_title("Predicted Signal Peaks")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    coords_dir = os.path.join(args.output_dir, "coordinates")
    vis_dir = os.path.join(args.output_dir, "visualisations")
    os.makedirs(coords_dir, exist_ok=True)
    if not args.no_visualisation:
        os.makedirs(vis_dir, exist_ok=True)

    model = build_model(args)
    paths, floods = load_flood_maps(args.input_dir)
    print(f"[infer] running inference on {len(floods)} flood map(s)")

    for path, flood in zip(paths, floods):
        name = os.path.splitext(os.path.basename(path))[0]
        coords = predict_single(model, flood, args.device)

        coord_path = os.path.join(coords_dir, f"{name}_peaks.npy")
        np.save(coord_path, coords)

        if not args.no_visualisation:
            vis_path = os.path.join(vis_dir, f"{name}_peaks.png")
            save_visualisation(flood, coords, vis_path)

        print(f"[infer] {name}: saved {coord_path} ({coords.shape[0]} peaks)")

    print(f"[infer] done. results in {args.output_dir}")


if __name__ == "__main__":
    main()
