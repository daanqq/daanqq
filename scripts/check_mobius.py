#!/usr/bin/env python3
"""Validate the generated README animations."""

from pathlib import Path

import numpy as np
from PIL import Image

from render_mobius import HALF_WIDTH, mobius, transform


EXPECTED_SIZE = (1000, 360)
EXPECTED_FRAMES = 72
ASSETS = ("mobius-dark.webp", "mobius-light.webp")


def check_asset(path: Path) -> None:
    image = Image.open(path)

    assert image.size == EXPECTED_SIZE, f"{path}: unexpected size {image.size}"
    assert image.width * 9 == image.height * 25, f"{path}: not 25:9"
    assert image.n_frames == EXPECTED_FRAMES, (
        f"{path}: expected {EXPECTED_FRAMES} frames, got {image.n_frames}"
    )
    assert image.info.get("loop") == 0, f"{path}: animation does not loop forever"

    center = np.array((image.width / 2, image.height / 2))
    max_center_offset = np.array((image.width, image.height)) * 0.05
    vertical_boxes = []

    for frame_index in range(image.n_frames):
        image.seek(frame_index)
        alpha = np.asarray(image.convert("RGBA").getchannel("A"), dtype=float)
        total_alpha = alpha.sum()
        assert total_alpha > 0, f"{path}: frame {frame_index} is empty"
        assert np.any(alpha == 0), f"{path}: frame {frame_index} has no transparency"

        bbox = image.convert("RGBA").getchannel("A").getbbox()
        assert bbox is not None
        vertical_boxes.append(bbox)
        visible_center = np.array(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2))
        assert np.all(np.abs(visible_center - center) <= max_center_offset), (
            f"{path}: frame {frame_index} is not centered"
        )

    assert min(box[1] for box in vertical_boxes) <= 1, (
        f"{path}: animation does not reach the top edge"
    )
    assert max(box[3] for box in vertical_boxes) >= image.height - 1, (
        f"{path}: animation does not reach the bottom edge"
    )


def check_intrinsic_roll() -> None:
    u = np.linspace(0.0, np.pi * 2.0, 32)
    centerline = np.zeros_like(u)
    edge = np.full_like(u, HALF_WIDTH)

    center_start = transform(mobius(u, centerline, 0.0))
    center_quarter_turn = transform(mobius(u, centerline, np.pi / 2.0))
    edge_start = transform(mobius(u, edge, 0.0))
    edge_quarter_turn = transform(mobius(u, edge, np.pi / 2.0))

    assert np.allclose(center_start, center_quarter_turn), (
        "the strip centerline rotates instead of staying fixed"
    )
    assert not np.allclose(edge_start, edge_quarter_turn), (
        "the strip does not roll around its centerline"
    )


def main() -> None:
    check_intrinsic_roll()
    assets_dir = Path(__file__).resolve().parents[1] / "assets"
    for asset in ASSETS:
        check_asset(assets_dir / asset)
    print(
        "Möbius assets are valid: intrinsic roll, 1000x360, 25:9, "
        "centered, transparent, 72 frames."
    )


if __name__ == "__main__":
    main()
