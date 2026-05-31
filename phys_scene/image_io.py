from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import PhysSceneConfig
from .io_utils import ensure_dir
from .schemas import LoadedImage


class ImageLoader:
    """Load input images and normalize HEIC files into RGB PNG previews."""

    def __init__(self, config: PhysSceneConfig):
        self.config = config

    def load(self, image_paths: list[Path]) -> list[LoadedImage]:
        output_dir = ensure_dir(self.config.resolved_output_dir() / "preprocessed")
        frames: list[LoadedImage] = []
        for view_id, path in enumerate(image_paths):
            image = self._load_rgb_image(path, output_dir, view_id)
            image.thumbnail(
                (self.config.max_image_size, self.config.max_image_size),
                Image.Resampling.LANCZOS,
            )
            png_path = output_dir / f"view_{view_id:02d}.png"
            image.save(png_path)
            rgb = np.asarray(image, dtype=np.uint8)
            height, width = rgb.shape[:2]
            frames.append(
                LoadedImage(
                    view_id=view_id,
                    source_path=path.resolve(),
                    png_path=png_path,
                    width=width,
                    height=height,
                    rgb=rgb,
                )
            )
        return frames

    def _load_rgb_image(self, path: Path, output_dir: Path, view_id: int) -> Image.Image:
        load_error: Exception | None = None
        try:
            self._register_heif_if_available()
            return Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            load_error = exc

        if path.suffix.lower() not in {".heic", ".heif"}:
            raise RuntimeError(f"Cannot decode image {path}: {load_error}") from load_error

        converted = output_dir / f"_heic_decode_{view_id:02d}.png"
        if converted.exists():
            return Image.open(converted).convert("RGB")

        sips = shutil.which("sips")
        if sips is None:
            raise RuntimeError(
                f"Cannot decode {path.name}. Install pillow-heif or run on macOS with sips."
            )

        result = subprocess.run(
            [sips, "-s", "format", "png", str(path), "--out", str(converted)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"sips failed to convert {path}: {result.stderr.strip()}"
            )
        return Image.open(converted).convert("RGB")

    @staticmethod
    def _register_heif_if_available() -> None:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            return
