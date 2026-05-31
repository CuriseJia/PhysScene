from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PhysSceneConfig
from .io_utils import ensure_dir, write_json
from .schemas import MeshAsset, ObjectCluster, PointCloud


class BoundingBoxMeshBuilder:
    """Convert semantic point clusters into simulator-ready proxy meshes."""

    def __init__(self, config: PhysSceneConfig):
        self.config = config

    def build(self, cloud: PointCloud, clusters: list[ObjectCluster]) -> list[MeshAsset]:
        mesh_dir = ensure_dir(self.config.resolved_output_dir() / "meshes")
        collision_dir = ensure_dir(self.config.resolved_output_dir() / "collision")
        meshes: list[MeshAsset] = []
        for cluster in clusters:
            pts = cloud.points[cluster.point_indices]
            bbox_min = np.quantile(pts, 0.02, axis=0)
            bbox_max = np.quantile(pts, 0.98, axis=0)
            center = ((bbox_min + bbox_max) / 2.0).astype(np.float32)
            size = np.maximum(
                (bbox_max - bbox_min).astype(np.float32), self.config.min_box_extent
            )
            vertices, faces, mesh_type = self._cluster_geometry(pts, center, size)
            color = np.clip(cluster.color_mean / 255.0, 0.0, 1.0)
            obj_path = mesh_dir / f"{cluster.object_id}.obj"
            mtl_path = mesh_dir / f"{cluster.object_id}.mtl"
            collision_path = collision_dir / f"{cluster.object_id}_vhacd_proxy.json"
            self._write_obj(obj_path, mtl_path, vertices, faces, color)
            write_json(
                collision_path,
                {
                    "object_id": cluster.object_id,
                    "type": "vhacd_proxy",
                    "note": (
                        f"{mesh_type} convex proxy. Replace with VHACD for "
                        "multi-hull concave collision in production."
                    ),
                    "center": center,
                    "size": size,
                    "num_hulls": 1,
                },
            )
            meshes.append(
                MeshAsset(
                    object_id=cluster.object_id,
                    obj_path=obj_path,
                    mtl_path=mtl_path,
                    collision_path=collision_path,
                    center=center,
                    size=size,
                    color=color.astype(np.float32),
                    vertices=vertices,
                    faces=faces,
                )
            )
        return meshes

    def _cluster_geometry(
        self, pts: np.ndarray, center: np.ndarray, size: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, str]:
        hull = self._convex_hull_geometry(pts)
        if hull is not None:
            return hull[0], hull[1], "scipy_convex_hull"
        vertices, faces = self._box_geometry(center, size)
        return vertices, faces, "axis_aligned_box"

    def _convex_hull_geometry(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        if len(pts) < 8:
            return None
        try:
            from scipy.spatial import ConvexHull, QhullError
        except ImportError:
            return None

        try:
            if len(pts) > self.config.mesh_sample_points:
                rng = np.random.default_rng(self.config.random_seed)
                pts = pts[rng.choice(len(pts), size=self.config.mesh_sample_points, replace=False)]
            hull = ConvexHull(pts, qhull_options="QJ")
        except (QhullError, ValueError):
            return None
        vertices = pts.astype(np.float32)
        faces = (hull.simplices.astype(np.int32) + 1)
        return vertices, faces

    @staticmethod
    def _box_geometry(center: np.ndarray, size: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sx, sy, sz = size / 2.0
        cx, cy, cz = center
        vertices = np.array(
            [
                [cx - sx, cy - sy, cz - sz],
                [cx + sx, cy - sy, cz - sz],
                [cx + sx, cy + sy, cz - sz],
                [cx - sx, cy + sy, cz - sz],
                [cx - sx, cy - sy, cz + sz],
                [cx + sx, cy - sy, cz + sz],
                [cx + sx, cy + sy, cz + sz],
                [cx - sx, cy + sy, cz + sz],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [1, 2, 3],
                [1, 3, 4],
                [5, 8, 7],
                [5, 7, 6],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 8],
                [3, 8, 4],
                [4, 8, 5],
                [4, 5, 1],
            ],
            dtype=np.int32,
        )
        return vertices, faces

    @staticmethod
    def _write_obj(
        obj_path: Path,
        mtl_path: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: np.ndarray,
    ) -> None:
        if BoundingBoxMeshBuilder._write_with_trimesh(
            obj_path, mtl_path, vertices, faces, color
        ):
            return
        material_name = obj_path.stem + "_mat"
        obj_lines = [f"mtllib {mtl_path.name}", f"usemtl {material_name}"]
        for vertex in vertices:
            obj_lines.append(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")
        for face in faces:
            obj_lines.append(f"f {face[0]} {face[1]} {face[2]}")
        obj_path.write_text("\n".join(obj_lines) + "\n", encoding="utf-8")

        r, g, b = color
        mtl_lines = [
            f"newmtl {material_name}",
            f"Kd {r:.6f} {g:.6f} {b:.6f}",
            "Ks 0.120000 0.120000 0.120000",
            "Ns 24.000000",
            "d 1.0",
        ]
        mtl_path.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")

    @staticmethod
    def _write_with_trimesh(
        obj_path: Path,
        mtl_path: Path,
        vertices: np.ndarray,
        faces: np.ndarray,
        color: np.ndarray,
    ) -> bool:
        try:
            import trimesh
        except ImportError:
            return False

        face_zero = np.asarray(faces, dtype=np.int64) - 1
        rgba = np.asarray([*np.clip(color, 0.0, 1.0), 1.0]) * 255.0
        mesh = trimesh.Trimesh(vertices=vertices, faces=face_zero, process=False)
        mesh.visual.face_colors = np.tile(rgba.astype(np.uint8), (len(face_zero), 1))
        mesh.export(obj_path)
        if not mtl_path.exists():
            r, g, b = color
            mtl_path.write_text(
                "\n".join(
                    [
                        f"newmtl {obj_path.stem}_mat",
                        f"Kd {r:.6f} {g:.6f} {b:.6f}",
                        "Ks 0.120000 0.120000 0.120000",
                        "Ns 24.000000",
                        "d 1.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
        return True
