"""Smooth Cartesian motion: minimum-jerk waypoints → joint trajectories via mink diff-IK.

Planning runs on its own robot-only model, so it never touches the live simulation.
Every trajectory is fully solved (and validated) before it is shown or executed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mink
import mujoco
import numpy as np

from er2_demo.scene import GRIPPER_CLOSED, GRIPPER_OPEN, HOME_Q, robot_model

DT = 0.01  # trajectory sample period (s) == sim control period
CLEAR_Z = 0.24  # transit height for the fingertip centre above the table
MAX_POS_ERR = 0.008
MAX_ROT_ERR = 0.06
MAX_JOINT_STEP = 0.06  # rad per sample (6 rad/s) – anything larger means a flip/singularity


class PlanningError(RuntimeError):
    """The requested motion cannot be executed (unreachable or unsafe)."""


def min_jerk(t: np.ndarray) -> np.ndarray:
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def down_rotation(yaw: float) -> np.ndarray:
    """TCP orientation pointing straight down, rotated by ``yaw`` about world z."""
    c, s = math.cos(yaw), math.sin(yaw)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rz @ np.diag([1.0, -1.0, -1.0])


def yaw_of(rotation: np.ndarray) -> float:
    return math.atan2(rotation[1, 0], rotation[0, 0])


@dataclass
class Waypoint:
    pos: np.ndarray
    yaw: float
    grip: float = GRIPPER_OPEN
    speed: float | None = None  # m/s override (slow approaches)
    min_duration: float = 0.0
    event: str | None = None  # fired when the segment ends


@dataclass
class JointTrajectory:
    q: np.ndarray  # (N, 7)
    grip: np.ndarray  # (N,)
    tcp: np.ndarray  # (N, 3) fingertip path, for visualisation
    events: dict[int, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.q) * DT


class Planner:
    def __init__(self) -> None:
        self.model = robot_model()
        self.config = mink.Configuration(self.model)
        self.arm_qadr = np.array([self.model.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.finger_qadr = np.array([self.model.joint(n).qposadr[0] for n in ("finger_joint1", "finger_joint2")])
        self.tcp_task = mink.FrameTask("tcp", "site", position_cost=1.0, orientation_cost=0.6, lm_damping=1e-3)
        # Low gain: the null-space pull toward the home posture must be gradual, or the first sample
        # after a far reach jumps (the arm "snaps" back toward home in one 10 ms step).
        self.posture = mink.PostureTask(self.model, cost=1e-2, gain=0.05)
        self.limits = [mink.ConfigurationLimit(self.model)]
        home = self._full_q(HOME_Q)
        self.posture.set_target(home)
        self.home_pos, self.home_yaw = self.fk(HOME_Q)
        self.speed = 0.25  # m/s, adjustable from the UI

    def _full_q(self, arm_q: np.ndarray) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[self.arm_qadr] = arm_q
        q[self.finger_qadr] = 0.04
        return q

    def fk(self, arm_q: np.ndarray) -> tuple[np.ndarray, float]:
        self.config.update(self._full_q(arm_q))
        site = self.config.data.site("tcp")
        return site.xpos.copy(), yaw_of(site.xmat.reshape(3, 3))

    def plan(self, q_start: np.ndarray, grip_start: float, waypoints: list[Waypoint]) -> JointTrajectory:
        self.config.update(self._full_q(q_start))
        site = self.config.data.site("tcp")
        pos, yaw, grip = site.xpos.copy(), yaw_of(site.xmat.reshape(3, 3)), grip_start
        qs, grips, tcps, events = [], [], [], {}
        q_prev = np.asarray(q_start, dtype=float).copy()
        for wp in waypoints:
            dist = float(np.linalg.norm(wp.pos - pos))
            dyaw = wrap(wp.yaw - yaw)
            speed = wp.speed or self.speed
            duration = max(wp.min_duration, dist / speed, abs(dyaw) / 1.2, 0.3 if dist > 1e-4 else 0.0)
            if grip != wp.grip:
                duration = max(duration, 0.6)
            n = max(1, int(round(duration / DT)))
            s = min_jerk(np.arange(1, n + 1) / n)
            for k in range(n):
                p = pos + (wp.pos - pos) * s[k]
                target_yaw = yaw + dyaw * s[k]
                g = grip + (wp.grip - grip) * min(1.0, (k + 1) / max(1, int(0.4 / DT)))
                q = self._solve(p, target_yaw)
                if np.max(np.abs(q - q_prev)) > MAX_JOINT_STEP:
                    raise PlanningError("motion would require a sudden joint flip (near a singularity or limit)")
                q_prev = q
                qs.append(q)
                grips.append(g)
                tcps.append(p)
            self._check(wp.pos, wp.yaw)
            pos, yaw, grip = wp.pos.copy(), wp.yaw, wp.grip
            if wp.event:
                events[len(qs) - 1] = wp.event
        if not qs:
            raise PlanningError("empty trajectory")
        return JointTrajectory(np.array(qs), np.array(grips), np.array(tcps), events)

    def _solve(self, pos: np.ndarray, yaw: float) -> np.ndarray:
        target = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(down_rotation(yaw)), pos)
        self.tcp_task.set_target(target)
        for _ in range(3):
            vel = mink.solve_ik(self.config, [self.tcp_task, self.posture], DT, "daqp", damping=1e-3,
                                limits=self.limits)
            self.config.integrate_inplace(vel, DT)
        return self.config.q[self.arm_qadr].copy()

    def _check(self, pos: np.ndarray, yaw: float) -> None:
        site = self.config.data.site("tcp")
        # A few extra solves at the segment end let IK settle before judging reachability.
        for _ in range(10):
            err = np.linalg.norm(site.xpos - pos)
            if err < MAX_POS_ERR * 0.3:
                break
            self._solve(pos, yaw)
        err = float(np.linalg.norm(site.xpos - pos))
        rot_err = abs(wrap(yaw_of(site.xmat.reshape(3, 3)) - yaw))
        tilt = float(np.arccos(np.clip(-site.xmat.reshape(3, 3)[2, 2], -1, 1)))
        if err > MAX_POS_ERR or rot_err > MAX_ROT_ERR or tilt > MAX_ROT_ERR:
            raise PlanningError(f"target ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) is out of reach "
                                f"(error {err * 100:.1f} cm)")

    # ---- primitives -----------------------------------------------------------------------

    def _rise_first(self, pos: np.ndarray, yaw: float, grip: float, clear_z: float) -> list[Waypoint]:
        if pos[2] < clear_z - 0.02:
            return [Waypoint(np.array([pos[0], pos[1], clear_z]), yaw, grip)]
        return []

    def pick(self, q: np.ndarray, grip: float, target: np.ndarray, grasp_yaw: float, *,
             open_grip: float = GRIPPER_OPEN, clear_z: float = CLEAR_Z) -> JointTrajectory:
        """Approach from above with the fingers pre-shaped to ``open_grip``, close, lift."""
        pos, yaw = self.fk(q)
        x, y, z = target
        clear_z = max(clear_z, z + 0.08)
        wps = self._rise_first(pos, yaw, grip, clear_z)
        wps += [
            Waypoint(np.array([x, y, clear_z]), grasp_yaw, open_grip),
            Waypoint(np.array([x, y, z + 0.06]), grasp_yaw, open_grip),
            Waypoint(np.array([x, y, z]), grasp_yaw, open_grip, speed=0.08),
            Waypoint(np.array([x, y, z]), grasp_yaw, GRIPPER_CLOSED, min_duration=0.8, event="closed"),
            Waypoint(np.array([x, y, z + 0.04]), grasp_yaw, GRIPPER_CLOSED, speed=0.08, event="lift_check"),
            Waypoint(np.array([x, y, clear_z]), grasp_yaw, GRIPPER_CLOSED, event="picked"),
        ]
        return self.plan(q, grip, wps)

    def place(self, q: np.ndarray, grip: float, target: np.ndarray, *, yaw: float | None = None,
              clear_z: float = CLEAR_Z, release_speed: float = 0.08) -> JointTrajectory:
        """Carry the held object over ``target`` (optionally re-orienting it), lower, release, go home."""
        pos, cur_yaw = self.fk(q)
        place_yaw = cur_yaw if yaw is None else yaw
        x, y, z = target
        clear_z = max(clear_z, z + 0.08)
        wps = self._rise_first(pos, cur_yaw, grip, clear_z)
        wps += [
            Waypoint(np.array([x, y, clear_z]), place_yaw, grip),
            Waypoint(np.array([x, y, z + 0.05]), place_yaw, grip),
            Waypoint(np.array([x, y, z]), place_yaw, grip, speed=release_speed, event="release"),
            Waypoint(np.array([x, y, z]), place_yaw, GRIPPER_OPEN, min_duration=0.5),
            Waypoint(np.array([x, y, z + 0.05]), place_yaw, GRIPPER_OPEN, speed=0.1),
            Waypoint(np.array([x, y, clear_z]), place_yaw, GRIPPER_OPEN),
            Waypoint(self.home_pos.copy(), self.home_yaw, GRIPPER_OPEN),
        ]
        return self.plan(q, grip, wps)

    def home(self, q: np.ndarray, grip: float, *, clear_z: float = CLEAR_Z) -> JointTrajectory:
        pos, yaw = self.fk(q)
        wps = self._rise_first(pos, yaw, grip, clear_z)
        wps.append(Waypoint(self.home_pos.copy(), self.home_yaw, grip))
        return self.plan(q, grip, wps)


def grasp_yaw_candidates(kind: str, object_yaw: float) -> list[float]:
    """Wrist yaws that close the fingers across a graspable width of the object."""
    if kind in ("sphere", "cylinder"):
        return [wrap(math.radians(a)) for a in range(-90, 90, 15)]
    if kind == "cube":
        return [wrap(object_yaw + k * math.pi / 2) for k in range(4)]
    return [wrap(object_yaw), wrap(object_yaw + math.pi)]  # cuboid: across the short side


def arm_q_from(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return np.array([data.qpos[model.joint(f"joint{i}").qposadr[0]] for i in range(1, 8)])
