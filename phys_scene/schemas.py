from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def to_builtin(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(v) for v in value]
    return value


@dataclass
class LoadedImage:
    view_id: int
    source_path: Path
    png_path: Path
    width: int
    height: int
    rgb: np.ndarray = field(repr=False)

    def metadata(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "source_path": str(self.source_path),
            "png_path": str(self.png_path),
            "width": self.width,
            "height": self.height,
        }


@dataclass
class CameraPose:
    view_id: int
    position: np.ndarray
    rotation_c2w: np.ndarray
    intrinsics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return to_builtin(
            {
                "view_id": self.view_id,
                "position": self.position,
                "rotation_c2w": self.rotation_c2w,
                "intrinsics": self.intrinsics,
            }
        )


@dataclass
class PointCloud:
    points: np.ndarray
    colors: np.ndarray
    view_ids: np.ndarray
    pixels: np.ndarray
    depths: np.ndarray
    cameras: list[CameraPose]


@dataclass
class ObjectCluster:
    object_id: str
    point_indices: np.ndarray
    source_views: list[int]
    centroid: np.ndarray
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    color_mean: np.ndarray
    color_std: np.ndarray

    @property
    def size(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min


@dataclass
class MeshAsset:
    object_id: str
    obj_path: Path
    mtl_path: Path
    collision_path: Path
    center: np.ndarray
    size: np.ndarray
    color: np.ndarray
    vertices: np.ndarray
    faces: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return to_builtin(
            {
                "object_id": self.object_id,
                "obj_path": self.obj_path,
                "mtl_path": self.mtl_path,
                "collision_path": self.collision_path,
                "center": self.center,
                "size": self.size,
                "color": self.color,
                "num_vertices": int(len(self.vertices)),
                "num_faces": int(len(self.faces)),
            }
        )


@dataclass
class PhysicalProperties:
    semantic_label: str
    material: str
    mass: float
    dynamic_friction: float
    static_friction: float
    restitution: float
    density: float
    kinematic: bool
    pbr: dict[str, Any]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return to_builtin(self.__dict__)


@dataclass
class SceneObject:
    object_id: str
    mesh: MeshAsset
    physics: PhysicalProperties
    source_views: list[int]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return to_builtin(
            {
                "object_id": self.object_id,
                "semantic_label": self.physics.semantic_label,
                "material": self.physics.material,
                "transform": {
                    "position": self.mesh.center,
                    "scale": self.mesh.size,
                    "rotation_euler": {"x": 0.0, "y": 0.0, "z": 0.0},
                },
                "mesh": self.mesh.to_dict(),
                "physics": self.physics.to_dict(),
                "source_views": self.source_views,
                "diagnostics": self.diagnostics,
            }
        )
