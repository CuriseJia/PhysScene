from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from .config import PhysSceneConfig
from .io_utils import compact_list, ensure_dir
from .schemas import LoadedImage, ObjectCluster, PointCloud


class HeuristicSAM2Segmenter:
    """SAM2-shaped segmenter with color/spatial clustering fallback."""

    def __init__(self, config: PhysSceneConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)

    def segment(self, frames: list[LoadedImage], cloud: PointCloud) -> list[ObjectCluster]:
        label_maps = self._segment_images(frames)
        if self.config.write_debug_artifacts:
            self._write_label_previews(frames, label_maps)
        return self._cluster_points(cloud, label_maps)

    def _segment_images(self, frames: list[LoadedImage]) -> dict[int, np.ndarray]:
        label_maps: dict[int, np.ndarray] = {}
        for frame in frames:
            slic_labels = self._slic_segments(frame.rgb)
            if slic_labels is not None:
                label_maps[frame.view_id] = slic_labels
                continue

            rgb = frame.rgb.astype(np.float32) / 255.0
            h, w = rgb.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            features = np.concatenate(
                [
                    rgb.reshape(-1, 3),
                    (xx.reshape(-1, 1) / max(w - 1, 1)) * 0.35,
                    (yy.reshape(-1, 1) / max(h - 1, 1)) * 0.35,
                ],
                axis=1,
            )
            labels = self._kmeans(
                features,
                k=self.config.segmentation_colors_per_view,
                iterations=10,
                max_fit_samples=12000,
            )
            label_maps[frame.view_id] = labels.reshape(h, w)
        return label_maps

    def _cluster_points(
        self, cloud: PointCloud, label_maps: dict[int, np.ndarray]
    ) -> list[ObjectCluster]:
        point_labels = np.zeros(len(cloud.points), dtype=np.float32)
        for i, (view_id, pix) in enumerate(zip(cloud.view_ids, cloud.pixels)):
            labels = label_maps[int(view_id)]
            x = int(np.clip(pix[0], 0, labels.shape[1] - 1))
            y = int(np.clip(pix[1], 0, labels.shape[0] - 1))
            point_labels[i] = labels[y, x]
        point_label_scale = max(float(point_labels.max()), 1.0)

        xyz = cloud.points.astype(np.float32)
        rgb = cloud.colors.astype(np.float32) / 255.0
        xyz_centered = xyz - np.median(xyz, axis=0, keepdims=True)
        xyz_scale = np.percentile(np.abs(xyz_centered), 90, axis=0, keepdims=True) + 1e-6
        xyz_norm = xyz_centered / xyz_scale
        features = np.concatenate(
            [
                xyz_norm * 1.0,
                rgb * 0.65,
                (point_labels[:, None] / point_label_scale) * 0.35,
            ],
            axis=1,
        )
        k = min(self.config.max_objects, max(1, len(features) // self.config.min_cluster_points))
        labels = self._kmeans(features, k=k, iterations=16, max_fit_samples=20000)

        clusters: list[ObjectCluster] = []
        for raw_id in range(k):
            indices = np.flatnonzero(labels == raw_id)
            if len(indices) < self.config.min_cluster_points:
                continue
            pts = cloud.points[indices]
            cols = cloud.colors[indices]
            bbox_min = np.quantile(pts, 0.02, axis=0).astype(np.float32)
            bbox_max = np.quantile(pts, 0.98, axis=0).astype(np.float32)
            clusters.append(
                ObjectCluster(
                    object_id=f"object_{len(clusters):03d}",
                    point_indices=indices.astype(np.int32),
                    source_views=compact_list(cloud.view_ids[indices]),
                    centroid=pts.mean(axis=0).astype(np.float32),
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    color_mean=cols.mean(axis=0).astype(np.float32),
                    color_std=cols.std(axis=0).astype(np.float32),
                )
            )
        clusters.sort(key=lambda c: len(c.point_indices), reverse=True)
        for idx, cluster in enumerate(clusters):
            cluster.object_id = f"object_{idx:03d}"
        return clusters

    def _slic_segments(self, rgb: np.ndarray) -> np.ndarray | None:
        try:
            from skimage.segmentation import slic
        except ImportError:
            return None

        h, w = rgb.shape[:2]
        segments = slic(
            rgb,
            n_segments=self.config.slic_segments_per_view,
            compactness=14.0,
            start_label=0,
            channel_axis=-1,
            convert2lab=True,
        )
        yy, xx = np.mgrid[0:h, 0:w]
        raw_features = []
        raw_labels = np.unique(segments)
        for label in raw_labels:
            mask = segments == label
            mean_rgb = rgb[mask].astype(np.float32).mean(axis=0) / 255.0
            mean_x = float(xx[mask].mean() / max(w - 1, 1))
            mean_y = float(yy[mask].mean() / max(h - 1, 1))
            raw_features.append([*mean_rgb, mean_x * 0.25, mean_y * 0.25])
        coarse = self._kmeans(
            np.asarray(raw_features, dtype=np.float32),
            k=min(self.config.segmentation_colors_per_view, len(raw_labels)),
            iterations=12,
            max_fit_samples=10000,
        )
        label_map = np.zeros_like(segments, dtype=np.int32)
        for raw_label, coarse_label in zip(raw_labels, coarse):
            label_map[segments == raw_label] = int(coarse_label)
        return label_map

    def _kmeans(
        self,
        features: np.ndarray,
        k: int,
        iterations: int,
        max_fit_samples: int,
    ) -> np.ndarray:
        features = features.astype(np.float32, copy=False)
        if len(features) == 0:
            return np.zeros(0, dtype=np.int32)
        k = max(1, min(k, len(features)))
        sklearn_labels = self._sklearn_kmeans(features, k, iterations, max_fit_samples)
        if sklearn_labels is not None:
            return sklearn_labels
        fit_indices = (
            self.rng.choice(len(features), size=max_fit_samples, replace=False)
            if len(features) > max_fit_samples
            else np.arange(len(features))
        )
        fit = features[fit_indices]
        centers = fit[self.rng.choice(len(fit), size=k, replace=False)].copy()
        for _ in range(iterations):
            distances = ((fit[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            for label in range(k):
                mask = labels == label
                if np.any(mask):
                    centers[label] = fit[mask].mean(axis=0)
        full_distances = ((features[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        return full_distances.argmin(axis=1).astype(np.int32)

    def _sklearn_kmeans(
        self, features: np.ndarray, k: int, iterations: int, max_fit_samples: int
    ) -> np.ndarray | None:
        os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            return None

        batch_size = min(4096, max(256, len(features)))
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=self.config.random_seed,
            max_iter=iterations,
            batch_size=batch_size,
            n_init=3,
        )
        if len(features) > max_fit_samples:
            fit_indices = self.rng.choice(len(features), size=max_fit_samples, replace=False)
            model.fit(features[fit_indices])
            return model.predict(features).astype(np.int32)
        return model.fit_predict(features).astype(np.int32)

    def _write_label_previews(
        self, frames: list[LoadedImage], label_maps: dict[int, np.ndarray]
    ) -> None:
        out = ensure_dir(self.config.resolved_output_dir() / "segmentation")
        palette = np.array(
            [
                [229, 57, 53],
                [67, 160, 71],
                [30, 136, 229],
                [251, 192, 45],
                [142, 36, 170],
                [0, 172, 193],
                [244, 81, 30],
                [124, 179, 66],
            ],
            dtype=np.uint8,
        )
        for frame in frames:
            labels = label_maps[frame.view_id]
            preview = palette[labels % len(palette)]
            Image.fromarray(preview).save(out / f"view_{frame.view_id:02d}_segments.png")
