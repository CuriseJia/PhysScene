from __future__ import annotations

import argparse
from pathlib import Path

from phys_scene import PhysSceneConfig, PhysScenePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PhysScene route on example images.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Folder containing view images.")
    parser.add_argument("--input-glob", default="view*.HEIC", help="Input image glob, e.g. view*.HEIC.")
    parser.add_argument("--output-dir", type=Path, default=Path("phys_scene_output"))
    parser.add_argument("--max-image-size", type=int, default=512)
    parser.add_argument("--point-sample-stride", type=int, default=8)
    parser.add_argument("--max-objects", type=int, default=8)
    parser.add_argument("--tdw-model-name", default="iron_box")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PhysSceneConfig(
        input_glob=args.input_glob,
        output_dir=args.output_dir,
        max_image_size=args.max_image_size,
        point_sample_stride=args.point_sample_stride,
        max_objects=args.max_objects,
        tdw_model_name=args.tdw_model_name,
    )
    result = PhysScenePipeline(config).run(root=args.root)
    print("PhysScene finished")
    print(f"  images:   {result['num_images']}")
    print(f"  points:   {result['num_points']}")
    print(f"  objects:  {result['num_objects']}")
    print(f"  manifest: {result['manifest']}")
    print(f"  tdw:      {result['tdw']['controller']}")


if __name__ == "__main__":
    main()
