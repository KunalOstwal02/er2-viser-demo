"""The real-time simulation thread. It is the ONLY code that touches MjModel/MjData.

Other threads talk to it through ``call`` (run a function on the sim thread and get a
Future back) and a few convenience wrappers built on it.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any

import mujoco
import numpy as np

from er2_demo.planner import DT, JointTrajectory, arm_q_from, yaw_of
from er2_demo.scene import CAMERA_SIZE, GRIPPER_OPEN, ObjectSpec, SceneSpec, build_model, reset_to_home

log = logging.getLogger(__name__)

SUBSTEPS = round(DT / 0.002)
VIEW_EVERY = 3  # push a frame to the viewer every 3 control ticks (~33 Hz)


@dataclass
class ObjectState:
    spec: ObjectSpec
    pos: np.ndarray
    yaw: float


@dataclass
class RobotState:
    q: np.ndarray
    grip: float
    tcp_pos: np.ndarray
    tcp_yaw: float
    held: str | None
    objects: dict[str, ObjectState]


@dataclass
class Capture:
    """Everything the agent needs from one observation, copied out of the sim."""

    images: dict[str, np.ndarray]  # camera → RGB uint8 (H, W, 3)
    depth: np.ndarray  # front camera planar depth (m)
    segmentation: np.ndarray  # front camera geom ids (-1 = none)
    cam_pos: np.ndarray
    cam_mat: np.ndarray
    fovy: float
    geom_to_object: dict[int, str]
    state: RobotState


@dataclass
class ExecResult:
    status: str  # "done" | "stopped" | "error"
    held: str | None = None
    grasped: bool | None = None
    assist_used: bool = False
    message: str = ""


@dataclass
class _Execution:
    traj: JointTrajectory
    future: Future
    index: int = 0
    grasp_target: str | None = None
    close_z: float = 0.0
    close_relpose: tuple[np.ndarray, np.ndarray] | None = None
    between_fingers: bool = False
    result: ExecResult = field(default_factory=lambda: ExecResult("done"))


class SimRunner:
    GRASP_MODES = ("physics+assist", "physics", "always weld")

    def __init__(self, scene: SceneSpec, *, realtime: bool = True) -> None:
        self.scene = scene
        self.realtime = realtime
        self.initial_scene = SceneSpec.from_dict(_scene_dict(scene))
        self.grasp_mode = "physics+assist"
        self.listeners: list[Callable[[mujoco.MjModel, mujoco.MjData], None]] = []
        self.model_listeners: list[Callable[[mujoco.MjModel], None]] = []
        self._commands: queue.Queue[tuple[Callable[[], Any], Future]] = queue.Queue()
        self._stop_thread = threading.Event()
        self._exec: _Execution | None = None
        self._held: str | None = None
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sim", daemon=True)

    # ---- lifecycle ------------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(60):
            raise RuntimeError("simulation failed to start")

    def close(self) -> None:
        self._stop_thread.set()
        self._thread.join(timeout=5)

    @property
    def busy(self) -> bool:
        return self._exec is not None

    # ---- thread-safe API ------------------------------------------------------------------

    def call(self, fn: Callable[[], Any]) -> Future:
        future: Future = Future()
        self._commands.put((fn, future))
        return future

    def execute(self, traj: JointTrajectory, grasp_target: str | None = None) -> Future:
        """Start playing a trajectory; the Future resolves with an ExecResult when it ends."""
        future: Future = Future()

        def start() -> None:
            if self._exec is not None:
                future.set_result(ExecResult("error", message="robot is already moving"))
                return
            self._exec = _Execution(traj, future, grasp_target=grasp_target)

        self.call(start)
        return future

    def stop(self) -> None:
        self.call(self._halt)

    def capture(self) -> Capture:
        return self.call(self._capture).result(timeout=30)

    def state(self) -> RobotState:
        return self.call(self._state).result(timeout=10)

    def set_object_pose(self, name: str, pos: np.ndarray, yaw: float | None = None) -> None:
        def apply() -> None:
            if self._exec is not None or name == self._held:
                return
            adr = self.model.joint(f"{name}_free").qposadr[0]
            vadr = self.model.joint(f"{name}_free").dofadr[0]
            self.data.qpos[adr:adr + 3] = pos
            if yaw is not None:
                self.data.qpos[adr + 3:adr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            self.data.qvel[vadr:vadr + 6] = 0
            mujoco.mj_forward(self.model, self.data)

        self.call(apply)

    def load_scene(self, scene: SceneSpec, *, as_initial: bool = True) -> Future:
        def apply() -> None:
            self._rebuild(scene, keep_robot=False)
            if as_initial:
                self.initial_scene = SceneSpec.from_dict(_scene_dict(scene))

        return self.call(apply)

    def reset(self) -> Future:
        return self.load_scene(SceneSpec.from_dict(_scene_dict(self.initial_scene)), as_initial=False)

    def spawn(self, obj: ObjectSpec) -> Future:
        def apply() -> None:
            scene = self._scene_from_live()
            obj.name = scene.unique_name(obj.name)
            scene.objects.append(obj)
            self._rebuild(scene, keep_robot=True)

        return self.call(apply)

    def delete(self, name: str) -> Future:
        def apply() -> None:
            scene = self._scene_from_live()
            scene.objects = [o for o in scene.objects if o.name != name]
            self._rebuild(scene, keep_robot=True)

        return self.call(apply)

    def current_scene(self) -> SceneSpec:
        return self.call(self._scene_from_live).result(timeout=10)

    # ---- sim-thread internals -------------------------------------------------------------

    def _run(self) -> None:
        try:
            self._rebuild(self.scene, keep_robot=False)
        except Exception:
            log.exception("initial scene build failed")
            raise
        finally:
            self._ready.set()
        tick = 0
        next_time = time.perf_counter()
        while not self._stop_thread.is_set():
            self._drain_commands()
            try:
                self._advance_execution()
                for _ in range(SUBSTEPS):
                    mujoco.mj_step(self.model, self.data)
            except Exception as error:  # keep the sim alive whatever happens
                log.exception("sim step failed")
                if self._exec is not None:
                    self._finish(ExecResult("error", message=f"simulation error: {error}"))
                mujoco.mj_resetData(self.model, self.data)
                reset_to_home(self.model, self.data)
            tick += 1
            if tick % VIEW_EVERY == 0:
                for listener in list(self.listeners):
                    try:
                        listener(self.model, self.data)
                    except Exception:
                        log.exception("viewer listener failed")
            if not self.realtime:
                continue
            next_time += DT
            delay = next_time - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:  # fell far behind (e.g. rebuild) – resync instead of racing
                next_time = time.perf_counter()
        if self._renderer is not None:
            self._renderer.close()

    def _drain_commands(self) -> None:
        while True:
            try:
                fn, future = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                future.set_result(fn())
            except Exception as error:
                log.exception("sim command failed")
                future.set_exception(error)

    def _rebuild(self, scene: SceneSpec, *, keep_robot: bool) -> None:
        if self._exec is not None:
            self._finish(ExecResult("stopped", message="scene changed"))
        old = (self.model, self.data)
        model, geom_to_object = build_model(scene)
        data = mujoco.MjData(model)
        reset_to_home(model, data)
        if keep_robot and old[0] is not None:
            om, od = old
            for i in range(1, 8):
                data.qpos[model.joint(f"joint{i}").qposadr[0]] = od.qpos[om.joint(f"joint{i}").qposadr[0]]
            data.ctrl[: om.nu] = od.ctrl[: om.nu]
            mujoco.mj_forward(model, data)
        self._held = None
        self.scene, self.model, self.data, self.geom_to_object = scene, model, data, geom_to_object
        if self._renderer is not None:
            self._renderer.close()
        self._renderer = mujoco.Renderer(model, CAMERA_SIZE[1], CAMERA_SIZE[0])
        # Let objects settle so the first observation is at rest.
        for _ in range(150):
            mujoco.mj_step(model, data)
        for listener in list(self.model_listeners):
            try:
                listener(model)
            except Exception:
                log.exception("model listener failed")

    def _scene_from_live(self) -> SceneSpec:
        objects = []
        for obj in self.scene.objects:
            body = self.data.body(obj.name)
            yaw = yaw_of(body.xmat.reshape(3, 3))
            objects.append(ObjectSpec(obj.name, obj.kind, obj.color, float(body.xpos[0]), float(body.xpos[1]),
                                      float(yaw), list(obj.size)))
        return SceneSpec(self.scene.name, self.scene.prompt, objects)

    def _state(self) -> RobotState:
        tcp = self.data.site("tcp")
        objects = {}
        for obj in self.scene.objects:
            body = self.data.body(obj.name)
            objects[obj.name] = ObjectState(obj, body.xpos.copy(), yaw_of(body.xmat.reshape(3, 3)))
        return RobotState(
            q=arm_q_from(self.model, self.data),
            grip=float(self.data.ctrl[self.model.actuator("actuator8").id]),
            tcp_pos=tcp.xpos.copy(),
            tcp_yaw=yaw_of(tcp.xmat.reshape(3, 3)),
            held=self._held,
            objects=objects,
        )

    def _capture(self) -> Capture:
        r = self._renderer
        images = {}
        for cam in ("front", "side"):
            r.update_scene(self.data, cam)
            images[cam] = r.render().copy()
        r.update_scene(self.data, "front")
        r.enable_depth_rendering()
        depth = r.render().copy()
        r.disable_depth_rendering()
        r.enable_segmentation_rendering()
        seg_raw = r.render().copy()
        r.disable_segmentation_rendering()
        seg = np.where(seg_raw[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg_raw[..., 0], -1)
        # Multisampling blends ids at object edges into unrelated ids: keep only pixels
        # whose label agrees with all four neighbours.
        padded = np.pad(seg, 1, mode="edge")
        stable = ((padded[:-2, 1:-1] == seg) & (padded[2:, 1:-1] == seg)
                  & (padded[1:-1, :-2] == seg) & (padded[1:-1, 2:] == seg))
        seg = np.where(stable, seg, -1)
        cam = self.data.cam("front")
        return Capture(images, depth, seg, cam.xpos.copy(), cam.xmat.reshape(3, 3).copy(),
                       float(self.model.cam("front").fovy[0]), dict(self.geom_to_object), self._state())

    # ---- trajectory playback and grasp handling -------------------------------------------

    def _halt(self) -> None:
        if self._exec is not None:
            self._finish(ExecResult("stopped", held=self._held, message="stopped by user"))

    def _finish(self, result: ExecResult) -> None:
        execution, self._exec = self._exec, None
        if execution is not None and not execution.future.done():
            result.held = self._held
            execution.future.set_result(result)

    def _advance_execution(self) -> None:
        ex = self._exec
        if ex is None:
            return
        i = ex.index
        for j in range(7):
            self.data.ctrl[j] = ex.traj.q[i, j]
        self.data.ctrl[7] = ex.traj.grip[i]
        event = ex.traj.events.get(i)
        if event:
            self._handle_event(ex, event)
        ex.index += 1
        if ex.index >= len(ex.traj.q):
            self._finish(ex.result)

    def _handle_event(self, ex: _Execution, event: str) -> None:
        target = ex.grasp_target
        if event == "closed" and target:
            obj = self.data.body(target)
            tcp = self.data.site("tcp")
            ex.close_z = float(obj.xpos[2])
            offset = obj.xpos - tcp.xpos
            ex.between_fingers = bool(np.linalg.norm(offset[:2]) < 0.035 and abs(offset[2]) < 0.04)
            ex.close_relpose = self._relpose(target)
            if self.grasp_mode == "always weld" and ex.between_fingers:
                self._weld(target, True, ex.close_relpose)
                ex.result.assist_used = True
        elif event == "lift_check" and target:
            lifted = float(self.data.body(target).xpos[2]) - ex.close_z > 0.02
            if not lifted and self.grasp_mode != "physics" and ex.between_fingers and ex.close_relpose:
                self._weld(target, True, ex.close_relpose)
                ex.result.assist_used = True
        elif event == "picked" and target:
            held = float(self.data.body(target).xpos[2]) - ex.close_z > 0.1
            self._held = target if held else None
            ex.result.grasped = held
            if not held:
                self._weld(target, False)
                ex.result.message = f"{target} was not lifted"
        elif event == "release":
            if self._held:
                self._weld(self._held, False)
            self._held = None

    def _relpose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Pose of body ``name`` in the hand frame (weld relpose convention)."""
        hand, obj = self.data.body("hand"), self.data.body(name)
        hand_rot = hand.xmat.reshape(3, 3)
        rel_pos = hand_rot.T @ (obj.xpos - hand.xpos)
        neg = np.zeros(4)
        mujoco.mju_negQuat(neg, hand.xquat)
        rel_quat = np.zeros(4)
        mujoco.mju_mulQuat(rel_quat, neg, obj.xquat)
        return rel_pos, rel_quat

    def _weld(self, name: str, active: bool, relpose: tuple[np.ndarray, np.ndarray] | None = None) -> None:
        eq = self.model.equality(f"weld_{name}").id
        if active and relpose is not None:
            self.model.eq_data[eq, 3:6] = relpose[0]
            self.model.eq_data[eq, 6:10] = relpose[1]
        self.data.eq_active[eq] = int(active)


def _scene_dict(scene: SceneSpec) -> dict:
    import json

    return json.loads(scene.to_json())


def home_grip() -> float:
    return GRIPPER_OPEN
