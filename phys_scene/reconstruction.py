from __future__ import annotations

from pathlib import Path

import numpy as np

from .config import PhysSceneConfig
from .io_utils import ensure_dir
from .schemas import CameraPose, LoadedImage, PointCloud


class HeuristicVGGTReconstructor:
    """VGGT-shaped reconstruction interface with a deterministic local backend.

    A production implementation can replace this class with a wrapper around
    VGGT. The return object is the same: dense-ish point cloud, per-view camera
    poses, colors, pixels, and estimated depths.
    """

    def __init__(self, config: PhysSceneConfig):
        self.config = config

    def reconstruct(self, frames: list[LoadedImage]) -> PointCloud:
        points: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        view_ids: list[np.ndarray] = []
        pixels: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        cameras: list[CameraPose] = []

        for frame in frames:
            pose = self._camera_pose(frame.view_id, len(frames), frame.width, frame.height)
            cameras.append(pose)
            depth = self._estimate_depth(frame.rgb)
            yy, xx = np.mgrid[
                0 : frame.height : self.config.point_sample_stride,
                0 : frame.width : self.config.point_sample_stride,
            ]
            sample_depth = depth[yy, xx]
            fx = pose.intrinsics["fx"]
            fy = pose.intrinsics["fy"]
            cx = pose.intrinsics["cx"]
            cy = pose.intrinsics["cy"]
            x_cam = (xx.astype(np.float32) - cx) / fx * sample_depth
            y_cam = -(yy.astype(np.float32) - cy) / fy * sample_depth
            z_cam = sample_depth
            cam_points = np.stack([x_cam, y_cam, z_cam], axis=-1).reshape(-1, 3)
            world_points = cam_points @ pose.rotation_c2w.T + pose.position

            sampled_colors = frame.rgb[yy, xx].reshape(-1, 3)
            sample_pixels = np.stack([xx, yy], axis=-1).reshape(-1, 2)
            points.append(world_points.astype(np.float32))
            colors.append(sampled_colors.astype(np.uint8))
            view_ids.append(np.full(len(world_points), frame.view_id, dtype=np.int32))
            pixels.append(sample_pixels.astype(np.int32))
            depths.append(sample_depth.reshape(-1).astype(np.float32))

        cloud = PointCloud(
            points=np.concatenate(points, axis=0),
            colors=np.concatenate(colors, axis=0),
            view_ids=np.concatenate(view_ids, axis=0),
            pixels=np.concatenate(pixels, axis=0),
            depths=np.concatenate(depths, axis=0),
            cameras=cameras,
        )
        self.write_debug_artifacts(cloud)
        return cloud

    def write_debug_artifacts(self, cloud: PointCloud) -> None:
        if not self.config.write_debug_artifacts:
            return
        out = ensure_dir(self.config.resolved_output_dir() / "geometry")
        np.savez_compressed(
            out / "point_cloud.npz",
            points=cloud.points,
            colors=cloud.colors,
            view_ids=cloud.view_ids,
            pixels=cloud.pixels,
            depths=cloud.depths,
        )
        self._write_ply(out / "point_cloud.ply", cloud.points, cloud.colors)

    def _camera_pose(
        self, view_id: int, num_views: int, width: int, height: int
    ) -> CameraPose:
        angle = 2.0 * np.pi * view_id / max(num_views, 1)
        eye = np.array(
            [
                self.config.virtual_camera_radius * np.sin(angle),
                1.25,
                self.config.virtual_camera_radius * np.cos(angle),
            ],
            dtype=np.float32,
        )
        rotation = self._look_at_rotation(eye, np.array([0.0, 0.55, 0.0], dtype=np.float32))
        focal = 0.85 * float(max(width, height))
        return CameraPose(
            view_id=view_id,
            position=eye,
            rotation_c2w=rotation,
            intrinsics={
                "fx": focal,
                "fy": focal,
                "cx": float(width) / 2.0,
                "cy": float(height) / 2.0,
            },
        )

    def _estimate_depth(self, rgb: np.ndarray) -> np.ndarray:
        gray = rgb.astype(np.float32).mean(axis=2) / 255.0
        gy, gx = np.gradient(gray)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = (edge - edge.min()) / (np.ptp(edge) + 1e-6)
        h, _ = gray.shape
        vertical = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
        depth = 0.9 + self.config.scene_scale * (
            0.55 * (1.0 - vertical) + 0.30 * (1.0 - gray) + 0.15 * (1.0 - edge)
        )
        return depth.astype(np.float32)

    @staticmethod
    def _look_at_rotation(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
        forward = target - eye
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, up_hint)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        return np.stack([right, up, forward], axis=1).astype(np.float32)

    @staticmethod
    def _write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(points)}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
        for p, c in zip(points, colors):
            lines.append(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
