#!/usr/bin/env python3
"""Render a centered, seamless Möbius-strip animation for a GitHub README."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


WIDTH = 1000
HEIGHT = 360
FPS = 24
DURATION_SECONDS = 3
FRAME_COUNT = 72
SUPERSAMPLING = 3

RADIUS = 1.0
HALF_WIDTH = 0.34
SURFACE_U_SEGMENTS = 112
SURFACE_V_SEGMENTS = 8
EDGE_SEGMENTS = 448
EDGE_MARGIN = 1.0 * SUPERSAMPLING
MAX_EDGE_WIDTH = 3.0 * SUPERSAMPLING


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))


def mobius(u: np.ndarray, v: np.ndarray, roll: float) -> np.ndarray:
    cross_section_angle = u / 2.0 + roll
    radial = RADIUS + v * np.cos(cross_section_angle)
    return np.stack(
        (
            radial * np.cos(u),
            radial * np.sin(u),
            v * np.sin(cross_section_angle),
        ),
        axis=-1,
    )


def transform(points: np.ndarray) -> np.ndarray:
    # Keep the ring itself fixed: animation comes from rolling the ribbon's
    # cross-section around its centerline, not from rotating a static picture.
    matrix = rotation_y(math.radians(-8.0)) @ rotation_x(math.radians(57.0))
    return points @ matrix.T


def projection_parameters(
    width: int = WIDTH, height: int = HEIGHT, padding: float = 0.0
) -> tuple[float, np.ndarray]:
    # Fit the union of all poses to the canvas, so the ribbon reaches the
    # limiting edges without drifting outside the image while it rolls.
    u = np.linspace(0.0, math.tau * 2.0, 1024)
    v = np.linspace(-HALF_WIDTH, HALF_WIDTH, 9)
    phases = np.linspace(0.0, math.pi, FRAME_COUNT * 4 + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    points = np.concatenate(
        [transform(mobius(uu, vv, phase))[..., :2].reshape(-1, 2) for phase in phases]
    )

    low, high = points.min(axis=0), points.max(axis=0)
    canvas = np.array((width, height)) * SUPERSAMPLING
    fit_margin = EDGE_MARGIN + MAX_EDGE_WIDTH / 2.0 + padding * SUPERSAMPLING
    scale = float(np.min((canvas - 2.0 * fit_margin) / (high - low)))
    middle = (high + low) / 2.0
    center = canvas / 2.0 - middle * np.array((scale, -scale))
    return scale, center


def project(points: np.ndarray, projection: tuple[float, np.ndarray]) -> np.ndarray:
    scale, center = projection
    return center + points[..., :2] * np.array((scale, -scale))


def surface_primitives(
    phase: float, projection: tuple[float, np.ndarray]
) -> list[tuple[float, str, np.ndarray]]:
    u = np.linspace(0.0, math.tau, SURFACE_U_SEGMENTS + 1)
    v = np.linspace(-HALF_WIDTH, HALF_WIDTH, SURFACE_V_SEGMENTS + 1)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    world = transform(mobius(uu, vv, phase))
    screen = project(world, projection)

    primitives: list[tuple[float, str, np.ndarray]] = []
    for ui in range(SURFACE_U_SEGMENTS):
        for vi in range(SURFACE_V_SEGMENTS):
            corners = (
                (ui, vi),
                (ui + 1, vi),
                (ui + 1, vi + 1),
                (ui, vi + 1),
            )
            for triangle in ((0, 1, 2), (0, 2, 3)):
                indexes = [corners[index] for index in triangle]
                points = np.array([screen[index] for index in indexes])
                depth = float(np.mean([world[index][2] for index in indexes]))
                primitives.append((depth, "surface", points))

    return primitives


def edge_primitives(
    phase: float, projection: tuple[float, np.ndarray]
) -> list[tuple[float, str, np.ndarray]]:
    # u=0..4π with fixed +width traces the strip's single continuous boundary.
    u = np.linspace(0.0, math.tau * 2.0, EDGE_SEGMENTS + 1)
    v = np.full_like(u, HALF_WIDTH)
    world = transform(mobius(u, v, phase))
    screen = project(world, projection)

    return [
        (
            float((world[index, 2] + world[index + 1, 2]) / 2.0),
            "edge",
            screen[index : index + 2],
        )
        for index in range(EDGE_SEGMENTS)
    ]


def render_frame(
    phase: float,
    color: tuple[int, int, int],
    width: int = WIDTH,
    height: int = HEIGHT,
    projection: tuple[float, np.ndarray] | None = None,
    background: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> Image.Image:
    projection = projection or projection_parameters(width, height)
    size = (width * SUPERSAMPLING, height * SUPERSAMPLING)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")

    primitives = surface_primitives(phase, projection) + edge_primitives(
        phase, projection
    )
    depths = [primitive[0] for primitive in primitives]
    min_depth, max_depth = min(depths), max(depths)
    depth_span = max(max_depth - min_depth, 1e-6)

    # Painter's algorithm: far geometry first, near geometry last.
    for depth, kind, points in sorted(primitives, key=lambda primitive: primitive[0]):
        proximity = (depth - min_depth) / depth_span
        coordinates = [tuple(point) for point in points]

        if kind == "surface":
            # A barely visible face makes the flat strip readable while retaining
            # the requested transparent, outline-led appearance.
            alpha = round(5 + proximity * 9)
            draw.polygon(coordinates, fill=(*color, alpha))
        else:
            alpha = round(120 + proximity * 125)
            line_width = round((2.2 + proximity * 0.8) * SUPERSAMPLING)
            draw.line(coordinates, fill=(*color, alpha), width=line_width)

    image = image.resize((width, height), Image.Resampling.LANCZOS)
    # ImageDraw replaces alpha instead of compositing, so draw on transparency
    # and lay the result over the background afterwards.
    return Image.alpha_composite(Image.new("RGBA", image.size, background), image)


def save_animation(
    output: Path,
    color: tuple[int, int, int],
    width: int = WIDTH,
    height: int = HEIGHT,
    padding: float = 0.0,
    background: tuple[int, int, int] | None = None,
) -> None:
    projection = projection_parameters(width, height, padding)
    fill = (*background, 255) if background else (0, 0, 0, 0)
    frames = [
        # A half-turn returns an untextured Möbius surface to the same geometry.
        render_frame(
            math.pi * frame / FRAME_COUNT, color, width, height, projection, fill
        )
        for frame in range(FRAME_COUNT)
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / FPS),
        loop=0,
        lossless=False,
        quality=78,
        method=6,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "assets",
    )
    return parser.parse_args()


def main() -> None:
    output_dir = parse_args().output_dir
    save_animation(output_dir / "mobius-dark.webp", (248, 250, 252))
    save_animation(output_dir / "mobius-light.webp", (36, 41, 47))
    save_animation(
        output_dir / "mobius-square-white.webp",
        (36, 41, 47),
        width=800,
        height=800,
        padding=48,
        background=(255, 255, 255),
    )


if __name__ == "__main__":
    main()
