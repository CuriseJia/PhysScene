from __future__ import annotations

import colorsys

import numpy as np

from .config import PhysSceneConfig
from .schemas import MeshAsset, ObjectCluster, PhysicalProperties


MATERIAL_TABLE = {
    "wood": {
        "density": 520.0,
        "dynamic_friction": 0.38,
        "static_friction": 0.52,
        "restitution": 0.28,
        "roughness": 0.62,
        "metallic": 0.0,
    },
    "metal": {
        "density": 2700.0,
        "dynamic_friction": 0.24,
        "static_friction": 0.38,
        "restitution": 0.42,
        "roughness": 0.28,
        "metallic": 1.0,
    },
    "plastic": {
        "density": 950.0,
        "dynamic_friction": 0.32,
        "static_friction": 0.46,
        "restitution": 0.45,
        "roughness": 0.48,
        "metallic": 0.0,
    },
    "fabric": {
        "density": 320.0,
        "dynamic_friction": 0.68,
        "static_friction": 0.82,
        "restitution": 0.12,
        "roughness": 0.9,
        "metallic": 0.0,
    },
    "ceramic": {
        "density": 1700.0,
        "dynamic_friction": 0.34,
        "static_friction": 0.55,
        "restitution": 0.38,
        "roughness": 0.34,
        "metallic": 0.0,
    },
    "rubber": {
        "density": 1100.0,
        "dynamic_friction": 0.85,
        "static_friction": 1.05,
        "restitution": 0.18,
        "roughness": 0.86,
        "metallic": 0.0,
    },
    "painted_wall": {
        "density": 900.0,
        "dynamic_friction": 0.48,
        "static_friction": 0.65,
        "restitution": 0.15,
        "roughness": 0.78,
        "metallic": 0.0,
    },
}


class HeuristicVLMAttributeEstimator:
    """VLM-shaped attribute estimator.

    The fallback maps visual statistics to material text and then to physical
    values. Swap this class for a Gemini/Qwen/OpenAI VLM wrapper when API keys
    and model checkpoints are available.
    """

    def __init__(self, config: PhysSceneConfig):
        self.config = config

    def estimate(
        self, clusters: list[ObjectCluster], meshes: list[MeshAsset]
    ) -> dict[str, PhysicalProperties]:
        mesh_by_id = {mesh.object_id: mesh for mesh in meshes}
        properties: dict[str, PhysicalProperties] = {}
        for cluster in clusters:
            mesh = mesh_by_id[cluster.object_id]
            semantic_label = self._semantic_label(mesh)
            material = self._material_from_visuals(cluster, mesh, semantic_label)
            base = MATERIAL_TABLE[material]
            volume = float(np.prod(mesh.size))
            effective_volume = volume * 0.025
            mass = float(np.clip(base["density"] * effective_volume, 0.05, 80.0))
            kinematic = semantic_label in {"floor_or_wall", "large_surface"} or volume > 3.5
            if kinematic:
                mass = max(mass, 50.0)
            properties[cluster.object_id] = PhysicalProperties(
                semantic_label=semantic_label,
                material=material,
                mass=mass,
                dynamic_friction=float(base["dynamic_friction"]),
                static_friction=float(base["static_friction"]),
                restitution=float(base["restitution"]),
                density=float(base["density"]),
                kinematic=bool(kinematic),
                pbr={
                    "albedo": np.clip(mesh.color, 0.0, 1.0),
                    "roughness": float(base["roughness"]),
                    "metallic": float(base["metallic"]),
                },
                rationale=self._rationale(cluster, mesh, material, semantic_label),
            )
        return properties

    @staticmethod
    def _semantic_label(mesh: MeshAsset) -> str:
        size = mesh.size
        max_extent = float(size.max())
        min_extent = float(size.min())
        flatness = min_extent / (max_extent + 1e-6)
        volume = float(np.prod(size))
        if max_extent > 2.2 and flatness < 0.18:
            return "floor_or_wall"
        if volume > 3.2:
            return "large_surface"
        if size[1] < 0.85 and max(size[0], size[2]) > 1.0:
            return "tabletop_or_device_cluster"
        if flatness < 0.22:
            return "thin_panel"
        if size[1] > 1.0 and size[1] > size[0] * 1.2:
            return "upright_object"
        return "box_like_object"

    @staticmethod
    def _material_from_visuals(
        cluster: ObjectCluster, mesh: MeshAsset, semantic_label: str
    ) -> str:
        rgb = np.clip(cluster.color_mean / 255.0, 0.0, 1.0)
        h, s, v = colorsys.rgb_to_hsv(float(rgb[0]), float(rgb[1]), float(rgb[2]))
        texture = float(np.linalg.norm(cluster.color_std) / 255.0)
        r, g, b = rgb
        brownish = r > g > b and r > 0.30 and g > 0.18

        if semantic_label in {"floor_or_wall", "large_surface"} and s < 0.22 and v > 0.45:
            return "painted_wall"
        if brownish and s > 0.16:
            return "wood"
        if v < 0.22:
            return "rubber"
        if s < 0.12 and v > 0.72 and mesh.size.max() > 1.1:
            return "painted_wall"
        if s < 0.16 and 0.52 < v < 0.90:
            return "ceramic"
        if s < 0.18 and texture > 0.14 and 0.25 < v < 0.72:
            return "metal"
        if texture > 0.20 and s < 0.45:
            return "fabric"
        if h < 0.07 or h > 0.92:
            return "wood" if brownish else "plastic"
        return "plastic"

    @staticmethod
    def _rationale(
        cluster: ObjectCluster,
        mesh: MeshAsset,
        material: str,
        semantic_label: str,
    ) -> str:
        mean = ", ".join(f"{v:.0f}" for v in cluster.color_mean)
        std = float(np.linalg.norm(cluster.color_std))
        size = ", ".join(f"{v:.2f}" for v in mesh.size)
        return (
            f"Heuristic VLM fallback inferred {semantic_label}/{material} "
            f"from mean RGB ({mean}), color variation {std:.1f}, and bbox size ({size})."
        )
