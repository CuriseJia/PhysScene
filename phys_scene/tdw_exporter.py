from __future__ import annotations

from pathlib import Path

from .config import PhysSceneConfig
from .io_utils import write_json
from .schemas import SceneObject


class TDWCommandExporter:
    """Export a TDW controller script and command preview.

    TDW custom OBJ import requires Unity asset-bundle conversion. This exporter
    therefore uses TDW physics proxy objects for immediate simulation testing
    and keeps the reconstructed OBJ paths in the manifest for asset-bundle
    conversion when a production TDW build is available.
    """

    def __init__(self, config: PhysSceneConfig):
        self.config = config

    def export(self, objects: list[SceneObject]) -> dict[str, Path]:
        out_dir = self.config.resolved_output_dir() / "tdw"
        out_dir.mkdir(parents=True, exist_ok=True)
        commands_path = out_dir / "tdw_scene_commands.json"
        controller_path = out_dir / "run_tdw_scene.py"
        command_preview = self._command_preview(objects)
        write_json(commands_path, command_preview)
        controller_path.write_text(self._controller_source(objects), encoding="utf-8")
        return {"commands": commands_path, "controller": controller_path}

    def _command_preview(self, objects: list[SceneObject]) -> dict:
        return {
            "note": (
                "Preview of TDW commands. The generated controller uses "
                "Controller.get_add_physics_object() so the TDW Python package "
                "can resolve model libraries at runtime."
            ),
            "room": {
                "$type": "create_empty_room",
                "width": self.config.tdw_room_size,
                "height": self.config.tdw_room_size,
            },
            "objects": [
                {
                    "object_id": 1000 + i,
                    "model_name": self.config.tdw_model_name,
                    "position": {
                        "x": float(obj.mesh.center[0]),
                        "y": max(float(obj.mesh.center[1]), 0.05),
                        "z": float(obj.mesh.center[2]),
                    },
                    "scale_factor": {
                        "x": float(max(obj.mesh.size[0], self.config.min_box_extent)),
                        "y": float(max(obj.mesh.size[1], self.config.min_box_extent)),
                        "z": float(max(obj.mesh.size[2], self.config.min_box_extent)),
                    },
                    "mass": obj.physics.mass,
                    "dynamic_friction": obj.physics.dynamic_friction,
                    "static_friction": obj.physics.static_friction,
                    "bounciness": obj.physics.restitution,
                    "kinematic": obj.physics.kinematic,
                    "source_mesh": str(obj.mesh.obj_path),
                    "material": obj.physics.material,
                }
                for i, obj in enumerate(objects)
            ],
        }

    def _controller_source(self, objects: list[SceneObject]) -> str:
        object_literals = []
        for i, obj in enumerate(objects):
            object_literals.append(
                {
                    "id": 1000 + i,
                    "model_name": self.config.tdw_model_name,
                    "position": {
                        "x": float(obj.mesh.center[0]),
                        "y": max(float(obj.mesh.center[1]), 0.05),
                        "z": float(obj.mesh.center[2]),
                    },
                    "scale_factor": {
                        "x": float(max(obj.mesh.size[0], self.config.min_box_extent)),
                        "y": float(max(obj.mesh.size[1], self.config.min_box_extent)),
                        "z": float(max(obj.mesh.size[2], self.config.min_box_extent)),
                    },
                    "mass": float(obj.physics.mass),
                    "dynamic_friction": float(obj.physics.dynamic_friction),
                    "static_friction": float(obj.physics.static_friction),
                    "bounciness": float(obj.physics.restitution),
                    "kinematic": bool(obj.physics.kinematic),
                    "semantic_label": obj.physics.semantic_label,
                    "material": obj.physics.material,
                    "source_mesh": str(obj.mesh.obj_path),
                }
            )
        return f'''"""Run the PhysScene proxy scene in ThreeDWorld.

Install optional dependencies first:
    pip install tdw

This script intentionally uses TDW physics proxy objects. Convert the OBJ files
listed in scene_manifest.json to TDW asset bundles if you need exact meshes.
"""

from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils


OBJECTS = {object_literals!r}


def main() -> None:
    c = Controller()
    commands = [TDWUtils.create_empty_room({self.config.tdw_room_size}, {self.config.tdw_room_size})]
    for item in OBJECTS:
        commands.extend(
            Controller.get_add_physics_object(
                model_name=item["model_name"],
                object_id=item["id"],
                position=item["position"],
                rotation={{"x": 0, "y": 0, "z": 0}},
                scale_factor=item["scale_factor"],
                kinematic=item["kinematic"],
                gravity=not item["kinematic"],
                default_physics_values=False,
                mass=item["mass"],
                dynamic_friction=item["dynamic_friction"],
                static_friction=item["static_friction"],
                bounciness=item["bounciness"],
                scale_mass=False,
            )
        )
    c.communicate(commands)
    for _ in range(120):
        c.communicate([])
    c.communicate({{"$type": "terminate"}})


if __name__ == "__main__":
    main()
'''
