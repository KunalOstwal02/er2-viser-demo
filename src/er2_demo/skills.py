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

from er2_demo.planner import JointTrajectory, Planner, PlanningError, grasp_yaw_for
from er2_demo.scene import CONTAINERS, GRIPPER_OPEN, HOME_Q
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


def object_at(cap: Capture, u: int, v: int, *, prefer: str) -> str | None:
    """Object under the pixel; if none, the nearest ``prefer`` object within a small radius."""
    h, w = cap.segmentation.shape

    def name_at(uu: int, vv: int) -> str | None:
        return cap.geom_to_object.get(int(cap.segmentation[vv, uu]))

    def ok(name: str | None) -> bool:
        if name is None:
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
            yaw = grasp_yaw_for(obj.spec.kind, obj.yaw, state.tcp_yaw)
            traj = self.planner.pick(state.q, state.grip, target, yaw)
            return PlannedAction("pick", traj, target, name, f"pick {obj.spec.display_name}")

        if action == "place":
            if not state.held:
                raise PlanningError("not holding anything; pick an object first")
            held = state.objects[state.held].spec
            container = object_at(cap, u, v, prefer="container")
            if container is not None:
                c = state.objects[container]
                release_z = c.spec.height + held.height / 2 + 0.015
                x, y = _free_spot(state, container, state.held)
                target = np.array([x, y, release_z])
                desc = f"place {held.display_name} in {c.spec.display_name}"
            else:
                surface = backproject(cap, u, v)
                target = np.array([surface[0], surface[1], max(surface[2], 0.0) + held.height / 2 + 0.012])
                under = cap.geom_to_object.get(int(cap.segmentation[v, u]))
                desc = f"place {held.display_name} " + (f"on {under.replace('_', ' ')}" if under else "on the table")
            traj = self.planner.place(state.q, state.grip, target)
            return PlannedAction("place", traj, target, container, desc)

        raise PlanningError(f"unknown action {action!r}")


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
