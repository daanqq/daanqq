#!/usr/bin/env python3
"""Validate the generated README animations."""

from pathlib import Path

import numpy as np
from PIL import Image


EXPECTED_SIZE = (840, 360)
EXPECTED_FRAMES = 72
ASSETS = ("mobius-dark.webp", "mobius-light.webp")


def check_asset(path: Path) -> None:
    image = Image.open(path)

    assert image.size == EXPECTED_SIZE, f"{path}: unexpected size {image.size}"
    assert image.width * 9 == image.height * 21, f"{path}: not 21:9"
    assert image.n_frames == EXPECTED_FRAMES, (
        f"{path}: expected {EXPECTED_FRAMES} frames, got {image.n_frames}"
    )
    assert image.info.get("loop") == 0, f"{path}: animation does not loop forever"

    center = np.array((image.width / 2, image.height / 2))
    max_center_offset = np.array((image.width, image.height)) * 0.05

    for frame_index in range(image.n_frames):
        image.seek(frame_index)
        alpha = np.asarray(image.convert("RGBA").getchannel("A"), dtype=float)
        total_alpha = alpha.sum()
        assert total_alpha > 0, f"{path}: frame {frame_index} is empty"
        assert np.any(alpha == 0), f"{path}: frame {frame_index} has no transparency"

        yy, xx = np.indices(alpha.shape)
        visible_center = np.array(
            ((xx * alpha).sum() / total_alpha, (yy * alpha).sum() / total_alpha)
        )
        assert np.all(np.abs(visible_center - center) < max_center_offset), (
            f"{path}: frame {frame_index} is not centered"
        )


def main() -> None:
    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    for asset in ASSETS:
        check_asset(assets_dir / asset)
    print("Möbius assets are valid: 840x360, 21:9, centered, transparent, 72 frames.")


if __name__ == "__main__":
    main()
