"""Scene description (JSON-serializable) and MuJoCo model construction via MjSpec."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PANDA_DIR = Path(
    os.environ.get("ER2_PANDA_DIR", REPO_ROOT / "third_party/mujoco_menagerie/franka_emika_panda")
)
PRESET_DIR = Path(__file__).resolve().parent / "presets"

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])
TCP_OFFSET = 0.1034  # hand frame → fingertip centre
GRIPPER_OPEN, GRIPPER_CLOSED = 255.0, 0.0

TABLE_TOP = 0.0
CAMERA_SIZE = (640, 480)  # width, height

COLORS: dict[str, tuple[float, float, float]] = {
    "red": (0.85, 0.15, 0.15),
    "green": (0.2, 0.7, 0.25),
    "blue": (0.15, 0.35, 0.9),
    "yellow": (0.95, 0.8, 0.1),
    "orange": (1.0, 0.5, 0.05),
    "purple": (0.55, 0.25, 0.75),
    "white": (0.92, 0.92, 0.92),
    "black": (0.12, 0.12, 0.12),
    "grey": (0.45, 0.45, 0.47),
}

# kind → default size. cube: edge; cuboid: (x, y, z) full extents; cylinder: (radius, height);
# sphere: radius; bowl: (radius, wall height); bin: (inner side, wall height).
DEFAULT_SIZES: dict[str, tuple[float, ...]] = {
    "cube": (0.045,),
    "cuboid": (0.06, 0.035, 0.035),
    "cylinder": (0.022, 0.06),
    "sphere": (0.024,),
    "bowl": (0.085, 0.04),
    "bin": (0.16, 0.05),
    "mat": (0.3, 0.22),  # flat, non-colliding zone marker: (x extent, y extent)
}
GRASPABLE = ("cube", "cuboid", "cylinder", "sphere")
CONTAINERS = ("bowl", "bin")


@dataclass
class ObjectSpec:
    name: str
    kind: str
    color: str
    x: float
    y: float
    yaw: float = 0.0
    size: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in DEFAULT_SIZES:
            raise ValueError(f"unknown object kind {self.kind!r}")
        if self.color not in COLORS:
            raise ValueError(f"unknown color {self.color!r}")
        if not self.size:
            self.size = list(DEFAULT_SIZES[self.kind])

    @property
    def graspable(self) -> bool:
        return self.kind in GRASPABLE

    @property
    def height(self) -> float:
        s = self.size
        if self.kind == "mat":
            return 0.002
        if self.kind == "cube":
            return s[0]
        if self.kind == "cuboid":
            return s[2]
        if self.kind == "sphere":
            return 2 * s[0]
        return s[1]  # cylinder, bowl, bin

    @property
    def grip_width(self) -> float:
        """Width across the fingers for the canonical grasp (along object's local y)."""
        s = self.size
        if self.kind == "cube":
            return s[0]
        if self.kind == "cuboid":
            return s[1]
        if self.kind in ("cylinder", "sphere"):
            return 2 * s[0]
        return 0.0

    @property
    def display_name(self) -> str:
        return self.name.replace("_", " ")


@dataclass
class SceneSpec:
    name: str
    prompt: str
    objects: list[ObjectSpec]
    max_steps: int = 12

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> SceneSpec:
        return cls(
            name=data["name"],
            prompt=data.get("prompt", ""),
            objects=[ObjectSpec(**o) for o in data["objects"]],
            max_steps=int(data.get("max_steps", 12)),
        )

    @classmethod
    def load(cls, path: Path) -> SceneSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_json())

    def get(self, name: str) -> ObjectSpec:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(name)

    def unique_name(self, base: str) -> str:
        names = {o.name for o in self.objects}
        if base not in names:
            return base
        i = 2
        while f"{base}_{i}" in names:
            i += 1
        return f"{base}_{i}"


def list_presets() -> dict[str, Path]:
    return {p.stem: p for p in sorted(PRESET_DIR.glob("*.json"))}


def look_at_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Quaternion for a MuJoCo camera (looks along -z, y up) at ``eye`` facing ``target``."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    mat = np.column_stack([right, up, -forward])
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return quat


CAMERAS = {
    # Front view facing the robot: the robot is at the back of the image, objects in front.
    "front": dict(eye=np.array([1.2, 0.0, 0.75]), target=np.array([0.45, 0.0, 0.0]), fovy=55.0),
    "side": dict(eye=np.array([0.5, -0.95, 0.6]), target=np.array([0.5, 0.0, 0.05]), fovy=55.0),
}


def _yaw_quat(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _add_object(spec: mujoco.MjSpec, obj: ObjectSpec) -> None:
    rgba = [*COLORS[obj.color], 1.0]
    z = TABLE_TOP + (obj.height / 2 if obj.graspable else 0.0) + 0.001
    if obj.kind == "mat":
        # Mocap body: draggable in the editor, invisible to physics.
        body = spec.worldbody.add_body(name=obj.name, pos=[obj.x, obj.y, 0.0], quat=_yaw_quat(obj.yaw), mocap=True)
        body.add_geom(name=f"{obj.name}_g", type=mujoco.mjtGeom.mjGEOM_BOX, size=[obj.size[0] / 2, obj.size[1] / 2, 0.001],
                      pos=[0, 0, 0.001], rgba=rgba, contype=0, conaffinity=0)
        return
    body = spec.worldbody.add_body(name=obj.name, pos=[obj.x, obj.y, z], quat=_yaw_quat(obj.yaw))
    body.add_freejoint(name=f"{obj.name}_free")
    common = dict(rgba=rgba, friction=[1.5, 0.02, 0.002], condim=4, solref=[0.01, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2])
    s = obj.size
    G = mujoco.mjtGeom
    if obj.kind == "cube":
        body.add_geom(name=f"{obj.name}_g", type=G.mjGEOM_BOX, size=[s[0] / 2] * 3, density=400, **common)
    elif obj.kind == "cuboid":
        body.add_geom(name=f"{obj.name}_g", type=G.mjGEOM_BOX, size=[v / 2 for v in s[:3]], density=400, **common)
    elif obj.kind == "cylinder":
        body.add_geom(name=f"{obj.name}_g", type=G.mjGEOM_CYLINDER, size=[s[0], s[1] / 2, 0], density=400, **common)
    elif obj.kind == "sphere":
        body.add_geom(name=f"{obj.name}_g", type=G.mjGEOM_SPHERE, size=[s[0], 0, 0], density=300, **common)
    elif obj.kind == "bowl":
        radius, wall_h = s[0], s[1]
        thick = 0.008
        body.add_geom(name=f"{obj.name}_base", type=G.mjGEOM_CYLINDER, size=[radius, thick / 2, 0],
                      pos=[0, 0, thick / 2], density=1500, **common)
        n = 20
        seg = 2 * radius * math.tan(math.pi / n) + 0.004
        for i in range(n):
            a = 2 * math.pi * i / n
            body.add_geom(name=f"{obj.name}_w{i}", type=G.mjGEOM_BOX, size=[thick / 2, seg / 2, wall_h / 2],
                          pos=[radius * math.cos(a), radius * math.sin(a), wall_h / 2], quat=_yaw_quat(a),
                          density=1500, **common)
    elif obj.kind == "bin":
        side, wall_h = s[0], s[1]
        thick = 0.008
        half = side / 2 + thick / 2
        body.add_geom(name=f"{obj.name}_base", type=G.mjGEOM_BOX, size=[half + thick / 2, half + thick / 2, thick / 2],
                      pos=[0, 0, thick / 2], density=1500, **common)
        for i, (px, py, sx, sy) in enumerate([(half, 0, thick / 2, half + thick / 2), (-half, 0, thick / 2, half + thick / 2),
                                              (0, half, half, thick / 2), (0, -half, half, thick / 2)]):
            body.add_geom(name=f"{obj.name}_w{i}", type=G.mjGEOM_BOX, size=[sx, sy, wall_h / 2],
                          pos=[px, py, wall_h / 2], density=1500, **common)


def panda_spec() -> mujoco.MjSpec:
    """The Menagerie Panda with a fingertip TCP site and grippier finger pads."""
    path = PANDA_DIR / "panda.xml"
    if not path.exists():
        raise FileNotFoundError(f"Panda MJCF not found at {path}; see third_party/README.md")
    spec = mujoco.MjSpec.from_file(str(path))
    hand = spec.body("hand")
    hand.add_site(name="tcp", pos=[0, 0, TCP_OFFSET], size=[0.005, 0, 0], rgba=[1, 0, 1, 0])
    for geom in spec.geoms:
        if geom.classname.name.startswith("fingertip_pad_collision"):
            geom.friction = [2.0, 0.05, 0.002]
    return spec


def build_model(scene: SceneSpec) -> tuple[mujoco.MjModel, dict[int, str]]:
    """Compile the full scene. Returns the model and a geom-id → object-name map."""
    spec = panda_spec()
    spec.option.timestep = 0.002
    spec.option.noslip_iterations = 3
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.stat.center = [0.45, 0.0, 0.2]
    spec.stat.extent = 1.0

    world = spec.worldbody
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.62, 0.52, 0.42], rgb2=[0.58, 0.48, 0.38], width=512, height=512)
    del tex
    spec.add_material(name="table_mat", textures=["", "grid"], texrepeat=[6, 6], reflectance=0.0)
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75],
                   rgba=[0.35, 0.37, 0.4, 1])
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.6, 0.7, 0.375], pos=[0.35, 0, -0.375],
                   material="table_mat", friction=[1.0, 0.02, 0.002])
    world.add_light(name="key", pos=[0.6, -0.4, 1.8], dir=[-0.2, 0.3, -1], diffuse=[0.6, 0.6, 0.6],
                    castshadow=True)
    world.add_light(name="fill", pos=[1.2, 0.6, 1.2], dir=[-0.6, -0.4, -0.7], diffuse=[0.3, 0.3, 0.3],
                    castshadow=False)
    for name, cam in CAMERAS.items():
        world.add_camera(name=name, pos=cam["eye"].tolist(), quat=look_at_quat(cam["eye"], cam["target"]).tolist(),
                         fovy=cam["fovy"])

    for obj in scene.objects:
        _add_object(spec, obj)
        if obj.graspable:
            eq = spec.add_equality(name=f"weld_{obj.name}", type=mujoco.mjtEq.mjEQ_WELD, name1="hand", name2=obj.name,
                                   objtype=mujoco.mjtObj.mjOBJ_BODY, active=False)
            eq.solref = [0.02, 1]

    # Start at home with the gripper open.
    spec.delete(spec.key("home"))
    model = spec.compile()
    geom_to_object: dict[int, str] = {}
    names = {o.name for o in scene.objects}
    for gid in range(model.ngeom):
        body_name = model.body(model.geom_bodyid[gid]).name
        if body_name in names:
            geom_to_object[gid] = body_name
    return model, geom_to_object


def robot_model() -> mujoco.MjModel:
    """Robot-only model used by the IK planner (independent of the live scene)."""
    spec = panda_spec()
    spec.delete(spec.key("home"))
    return spec.compile()


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for i, joint in enumerate(ARM_JOINTS):
        data.qpos[model.joint(joint).qposadr[0]] = HOME_Q[i]
        data.ctrl[model.actuator(f"actuator{i + 1}").id] = HOME_Q[i]
    for finger in ("finger_joint1", "finger_joint2"):
        data.qpos[model.joint(finger).qposadr[0]] = 0.04
    data.ctrl[model.actuator("actuator8").id] = GRIPPER_OPEN
    mujoco.mj_forward(model, data)
