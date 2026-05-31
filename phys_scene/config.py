from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PhysSceneConfig:
    """Runtime settings for the PhysScene prototype.

    The default backend is deliberately lightweight so the full route can be
    tested without downloading VGGT, SAM2, a VLM, or a TDW build. Each stage is
    written behind a small interface so a production backend can replace the
    heuristic implementation module by module.
    """

    input_glob: str = "view*.HEIC"
    output_dir: Path = Path("phys_scene_output")
    max_image_size: int = 512
    point_sample_stride: int = 8
    virtual_camera_radius: float = 2.8
    scene_scale: float = 3.0
    segmentation_colors_per_view: int = 6
    slic_segments_per_view: int = 80
    max_objects: int = 8
    min_cluster_points: int = 120
    mesh_sample_points: int = 1200
    random_seed: int = 7
    min_box_extent: float = 0.08
    tdw_model_name: str = "iron_box"
    tdw_room_size: int = 12
    write_debug_artifacts: bool = True

    def resolved_output_dir(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()
