"""Turn ER-2 image-space actions into validated joint trajectories.

ER-2 points are ``[y, x]`` normalised to 0–1000 in the front camera image. Each point is
back-projected with the rendered depth; the segmentation image tells us which object (if
any) it landed on, so a pick snaps to that object's grasp pose and a place into a
container snaps to its centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from er2_demo.planner import CLEAR_Z, JointTrajectory, Planner, PlanningError, grasp_yaw_candidates, wrap
from er2_demo.scene import CONTAINERS, GRIPPER_OPEN, HOME_Q, ObjectSpec
from er2_demo.sim import Capture

SEARCH_RADIUS_PX = 14


def to_pixel(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    y, x = point
    u = int(np.clip(round(x / 1000 * width), 0, width - 1))
    v = int(np.clip(round(y / 1000 * height), 0, height - 1))
    return u, v


def backproject(cap: Capture, u: int, v: int) -> np.ndarray:
    """World point at pixel (u, v) of the front camera using planar depth."""
    h, w = cap.depth.shape
    f = 0.5 * h / math.tan(math.radians(cap.fovy) / 2)
    d = float(cap.depth[v, u])
    local = np.array([(u + 0.5 - w / 2) / f * d, -(v + 0.5 - h / 2) / f * d, -d])
    return cap.cam_pos + cap.cam_mat @ local


def project(cap: Capture, world: np.ndarray) -> tuple[float, float] | None:
    """World point → normalised [y, x] (0–1000) in the front image; None if behind camera."""
    h, w = cap.depth.shape
    f = 0.5 * h / math.tan(math.radians(cap.fovy) / 2)
    local = cap.cam_mat.T @ (np.asarray(world) - cap.cam_pos)
    if local[2] >= 0:
        return None
    u = f * local[0] / -local[2] + w / 2
    v = -f * local[1] / -local[2] + h / 2
    return (v / h * 1000, u / w * 1000)


def object_at(cap: Capture, u: int, v: int, *, prefer: str, exclude: str | None = None) -> str | None:
    """Object under the pixel; if none, the nearest ``prefer`` object within a small radius."""
    h, w = cap.segmentation.shape

    def name_at(uu: int, vv: int) -> str | None:
        return cap.geom_to_object.get(int(cap.segmentation[vv, uu]))

    def ok(name: str | None) -> bool:
        if name is None or name == exclude:
            return False
        spec = cap.state.objects[name].spec
        return spec.graspable if prefer == "graspable" else spec.kind in CONTAINERS

    direct = name_at(u, v)
    if ok(direct):
        return direct
    best, best_d = None, float("inf")
    r = SEARCH_RADIUS_PX
    for vv in range(max(0, v - r), min(h, v + r + 1), 2):
        for uu in range(max(0, u - r), min(w, u + r + 1), 2):
            name = name_at(uu, vv)
            d = (uu - u) ** 2 + (vv - v) ** 2
            if ok(name) and d < best_d and d <= r * r:
                best, best_d = name, d
    return best


@dataclass
class PlannedAction:
    action: str
    trajectory: JointTrajectory | None
    target_world: np.ndarray | None = None
    target_object: str | None = None
    description: str = ""


class SkillPlanner:
    def __init__(self, planner: Planner | None = None) -> None:
        self.planner = planner or Planner()

    def plan(self, action: str, points: list[tuple[float, float]], cap: Capture) -> PlannedAction:
        state = cap.state
        h, w = cap.depth.shape
        if action == "home":
            return PlannedAction("home", self.planner.home(state.q, state.grip), description="return home")
        if action == "done":
            return PlannedAction("done", None, description="task finished")
        if not points:
            raise PlanningError(f"'{action}' needs a point")
        u, v = to_pixel(points[0], w, h)

        if action == "pick":
            if state.held:
                raise PlanningError(f"already holding {state.held}; place it first")
            name = object_at(cap, u, v, prefer="graspable")
            if name is None:
                raise PlanningError("no graspable object at that point")
            obj = state.objects[name]
            grasp_z = max(float(obj.pos[2]), 0.02)
            if obj.spec.kind in ("cube", "cuboid", "cylinder"):
                grasp_z = max(float(obj.pos[2]) + obj.spec.height / 2 - 0.022, 0.018)
            target = np.array([obj.pos[0], obj.pos[1], grasp_z])
            yaw = _best_grasp_yaw(state, name)
            open_width = min(0.04, obj.spec.grip_width / 2 + 0.014)
            traj = self.planner.pick(state.q, state.grip, target, yaw, open_grip=open_width / 0.04 * GRIPPER_OPEN,
                                     clear_z=_clear_z(state))
            return PlannedAction("pick", traj, target, name, f"pick {obj.spec.display_name}")

        if action == "place":
            if not state.held:
                raise PlanningError("not holding anything; pick an object first")
            held_state = state.objects[state.held]
            held = held_state.spec
            clear_z = _clear_z(state)
            container = object_at(cap, u, v, prefer="container", exclude=state.held)
            base = object_at(cap, u, v, prefer="graspable", exclude=state.held)
            direct = cap.geom_to_object.get(int(cap.segmentation[v, u]))
            if base is not None and direct == base:
                # Pointing at an object that already sits in a container means "into that container".
                holder = _container_holding(state, base)
                if holder is not None:
                    container, base = holder, None
            if base is not None and direct == base:
                # Stack: centre on the object below and line the held object up with it.
                b = state.objects[base]
                top = float(b.pos[2]) + b.spec.height / 2
                target = np.array([b.pos[0], b.pos[1], top + held.height / 2 + 0.008])
                yaw = state.tcp_yaw
                if held.kind in ("cube", "cuboid") and b.spec.kind in ("cube", "cuboid"):
                    step = math.pi / 2 if held.kind == "cube" and b.spec.kind == "cube" else math.pi
                    delta = (b.yaw - held_state.yaw + step / 2) % step - step / 2
                    yaw = wrap(state.tcp_yaw + delta)
                traj = self.planner.place(state.q, state.grip, target, yaw=yaw, clear_z=clear_z, release_speed=0.04)
                return PlannedAction("place", traj, target, base, f"stack {held.display_name} on {b.spec.display_name}")
            if container is not None:
                c = state.objects[container]
                release_z = c.spec.height + held.height / 2 + 0.015
                x, y = _free_spot(state, container, state.held)
                target = np.array([x, y, release_z])
                traj = self.planner.place(state.q, state.grip, target, clear_z=clear_z)
                return PlannedAction("place", traj, target, container,
                                     f"place {held.display_name} in {c.spec.display_name}")
            surface = backproject(cap, u, v)
            target = np.array([surface[0], surface[1], max(surface[2], 0.0) + held.height / 2 + 0.012])
            where = f"on {direct.replace('_', ' ')}" if direct else "on the table"
            traj = self.planner.place(state.q, state.grip, target, clear_z=clear_z)
            return PlannedAction("place", traj, target, None, f"place {held.display_name} {where}")

        raise PlanningError(f"unknown action {action!r}")


def _footprint(spec: ObjectSpec) -> float:
    """Horizontal radius of an object's footprint."""
    if spec.kind == "bowl":
        return spec.size[0]
    if spec.kind == "bin":
        return spec.size[0] / 2 + 0.01
    if spec.kind == "mat":
        return 0.0
    if spec.kind == "cuboid":
        return math.hypot(spec.size[0], spec.size[1]) / 2
    return max(spec.grip_width, spec.height if spec.kind == "cube" else 0.0) / 2 * 1.2


def _container_holding(state, name: str) -> str | None:
    obj = state.objects[name]
    for n, o in state.objects.items():
        if o.spec.kind in CONTAINERS and np.linalg.norm(o.pos[:2] - obj.pos[:2]) < _footprint(o.spec) - 0.01:
            return n
    return None


def _clear_z(state) -> float:
    """Transit height that clears every object (plus whatever hangs from the gripper)."""
    held_h = state.objects[state.held].spec.height if state.held else 0.0
    tops = [float(o.pos[2]) + (o.spec.height / 2 if o.spec.graspable else o.spec.height)
            for n, o in state.objects.items() if n != state.held and o.spec.kind != "mat"]
    return max(CLEAR_Z, max(tops, default=0.0) + held_h + 0.07)


def _best_grasp_yaw(state, name: str) -> float:
    """Among equivalent grasp yaws, prefer the one whose fingers stay clear of neighbours."""
    obj = state.objects[name]
    finger_offset = obj.spec.grip_width / 2 + 0.028
    inside = {n for n, o in state.objects.items()
              if o.spec.kind in CONTAINERS and np.linalg.norm(o.pos[:2] - obj.pos[:2]) < _footprint(o.spec)}
    neighbours = [(o.pos[:2], _footprint(o.spec)) for n, o in state.objects.items()
                  if n not in (name, state.held) and n not in inside and o.spec.kind != "mat"
                  and np.linalg.norm(o.pos[:2] - obj.pos[:2]) < 0.25]

    def score(yaw: float) -> float:
        closing = np.array([math.sin(yaw), -math.cos(yaw)])
        fingers = [obj.pos[:2] + closing * finger_offset, obj.pos[:2] - closing * finger_offset]
        clearance = min((float(np.linalg.norm(f - c)) - r for f in fingers for c, r in neighbours), default=1.0)
        return min(clearance, 0.03) - 0.01 * abs(wrap(yaw - state.tcp_yaw))

    return max(grasp_yaw_candidates(obj.spec.kind, obj.yaw), key=score)


def _free_spot(state, container: str, held: str) -> tuple[float, float]:
    """Point inside ``container`` farthest from objects already in it (centre if empty)."""
    c = state.objects[container]
    held_spec = state.objects[held].spec
    inner = (c.spec.size[0] if c.spec.kind == "bowl" else c.spec.size[0] / 2) - 0.012
    reach = max(0.0, inner - max(held_spec.grip_width, held_spec.height) / 2 * 1.2)
    others = [o.pos[:2] for n, o in state.objects.items()
              if n not in (container, held) and np.linalg.norm(o.pos[:2] - c.pos[:2]) < inner + 0.02]
    if not others or reach < 1e-3:
        return float(c.pos[0]), float(c.pos[1])
    best, best_score = c.pos[:2], -1.0
    for dx in np.linspace(-reach, reach, 7):
        for dy in np.linspace(-reach, reach, 7):
            if c.spec.kind == "bowl" and math.hypot(dx, dy) > reach:
                continue
            p = c.pos[:2] + [dx, dy]
            score = min(np.linalg.norm(p - o) for o in others)
            if score > best_score:
                best, best_score = p, score
    return float(best[0]), float(best[1])


def is_home(q: np.ndarray) -> bool:
    return bool(np.max(np.abs(q - HOME_Q)) < 0.05)


def default_grip() -> float:
    return GRIPPER_OPEN
