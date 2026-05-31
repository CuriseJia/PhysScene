from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PhysSceneConfig
from .io_utils import write_json
from .schemas import LoadedImage, PointCloud, SceneObject


class SceneManifestWriter:
    def __init__(self, config: PhysSceneConfig):
        self.config = config

    def write(
        self,
        frames: list[LoadedImage],
        cloud: PointCloud,
        objects: list[SceneObject],
        extra: dict[str, Any] | None = None,
    ) -> Path:
        path = self.config.resolved_output_dir() / "scene_manifest.json"
        payload = {
            "name": "PhysScene example generated from view1-view5",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pipeline": {
                "geometry": "HeuristicVGGTReconstructor",
                "segmentation": "HeuristicSAM2Segmenter",
                "attribute_estimation": "HeuristicVLMAttributeEstimator",
                "meshing": "BoundingBoxMeshBuilder",
                "simulation_export": "TDWCommandExporter",
            },
            "inputs": [frame.metadata() for frame in frames],
            "cameras": [camera.to_dict() for camera in cloud.cameras],
            "point_cloud": {
                "path": str(self.config.resolved_output_dir() / "geometry" / "point_cloud.npz"),
                "ply_path": str(self.config.resolved_output_dir() / "geometry" / "point_cloud.ply"),
                "num_points": int(len(cloud.points)),
            },
            "objects": [obj.to_dict() for obj in objects],
            "extra": extra or {},
        }
        write_json(path, payload)
        return path
