from __future__ import annotations

from pathlib import Path
from typing import Any

from .attributes import HeuristicVLMAttributeEstimator
from .config import PhysSceneConfig
from .image_io import ImageLoader
from .io_utils import discover_images
from .manifest import SceneManifestWriter
from .mesh_builder import BoundingBoxMeshBuilder
from .reconstruction import HeuristicVGGTReconstructor
from .schemas import SceneObject
from .segmentation import HeuristicSAM2Segmenter
from .tdw_exporter import TDWCommandExporter


class PhysScenePipeline:
    def __init__(self, config: PhysSceneConfig | None = None):
        self.config = config or PhysSceneConfig()
        self.image_loader = ImageLoader(self.config)
        self.reconstructor = HeuristicVGGTReconstructor(self.config)
        self.segmenter = HeuristicSAM2Segmenter(self.config)
        self.mesh_builder = BoundingBoxMeshBuilder(self.config)
        self.attribute_estimator = HeuristicVLMAttributeEstimator(self.config)
        self.tdw_exporter = TDWCommandExporter(self.config)
        self.manifest_writer = SceneManifestWriter(self.config)

    def run(
        self,
        root: Path | None = None,
        input_paths: list[Path] | None = None,
    ) -> dict[str, Any]:
        root = (root or Path.cwd()).expanduser().resolve()
        if input_paths is None:
            input_paths = discover_images(root, self.config.input_glob)
        input_paths = [Path(p).expanduser().resolve() for p in input_paths]
        if len(input_paths) < 1:
            raise FileNotFoundError(f"No input images matched {self.config.input_glob!r}")

        frames = self.image_loader.load(input_paths)
        cloud = self.reconstructor.reconstruct(frames)
        clusters = self.segmenter.segment(frames, cloud)
        meshes = self.mesh_builder.build(cloud, clusters)
        properties = self.attribute_estimator.estimate(clusters, meshes)
        mesh_by_id = {mesh.object_id: mesh for mesh in meshes}
        cluster_by_id = {cluster.object_id: cluster for cluster in clusters}
        objects = [
            SceneObject(
                object_id=object_id,
                mesh=mesh_by_id[object_id],
                physics=properties[object_id],
                source_views=cluster_by_id[object_id].source_views,
                diagnostics={
                    "num_points": int(len(cluster_by_id[object_id].point_indices)),
                    "bbox_min": cluster_by_id[object_id].bbox_min,
                    "bbox_max": cluster_by_id[object_id].bbox_max,
                    "color_mean_rgb": cluster_by_id[object_id].color_mean,
                    "color_std_rgb": cluster_by_id[object_id].color_std,
                },
            )
            for object_id in sorted(properties.keys())
        ]
        tdw_paths = self.tdw_exporter.export(objects)
        manifest_path = self.manifest_writer.write(
            frames,
            cloud,
            objects,
            extra={"tdw": {key: str(value) for key, value in tdw_paths.items()}},
        )
        return {
            "manifest": manifest_path,
            "tdw": tdw_paths,
            "num_images": len(frames),
            "num_points": int(len(cloud.points)),
            "num_objects": len(objects),
            "objects": objects,
        }
