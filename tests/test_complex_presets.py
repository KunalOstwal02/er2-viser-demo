"""Oracle runs of the complex presets: stacking, clutter clearing, multi-container sorting."""

import numpy as np
import pytest

from er2_demo.skills import SkillPlanner, project


def _do(sim, planner, action, world_point, pick_target=None):
    cap = sim.capture()
    planned = planner.plan(action, [project(cap, world_point)], cap)
    result = sim.execute(planned.trajectory, planned.target_object if action == "pick" else None).result(timeout=120)
    assert result.status == "done", result
    return planned, result


def _pick(sim, planner, name):
    pos = sim.state().objects[name].pos
    _, result = _do(sim, planner, "pick", pos)
    assert result.held == name, result


def _top(sim, name):
    o = sim.state().objects[name]
    return o.pos + [0, 0, o.spec.height / 2]


def test_spatial_tower(make_sim):
    sim, planner = make_sim("spatial"), SkillPlanner()
    _pick(sim, planner, "medium_blue_cube")
    planned, _ = _do(sim, planner, "place", _top(sim, "large_red_cube"))
    assert planned.description.startswith("stack")
    _pick(sim, planner, "small_yellow_cube")
    _do(sim, planner, "place", _top(sim, "medium_blue_cube"))
    s = sim.state().objects
    big, mid, small = (s[n].pos for n in ("large_red_cube", "medium_blue_cube", "small_yellow_cube"))
    assert np.linalg.norm(mid[:2] - big[:2]) < 0.015 and np.linalg.norm(small[:2] - mid[:2]) < 0.015
    assert big[2] < mid[2] < small[2]
    assert small[2] == pytest.approx(0.06 + 0.045 + 0.016, abs=0.01)  # resting on the tower, not tilted


def test_clutter_clear_mat(make_sim):
    sim, planner = make_sim("clutter"), SkillPlanner()
    spots = [(0.34, -0.2), (0.34, -0.32), (0.7, -0.2), (0.7, 0.05), (0.7, 0.22), (0.56, 0.3)]
    names = ["red_cube", "blue_cylinder", "yellow_cube", "green_cylinder", "orange_cuboid", "purple_cube"]
    assists = 0
    for name, (x, y) in zip(names, spots, strict=True):
        cap = sim.capture()
        planned = planner.plan("pick", [project(cap, cap.state.objects[name].pos)], cap)
        result = sim.execute(planned.trajectory, planned.target_object).result(timeout=120)
        assert result.held == name, (name, result)
        assists += result.assist_used
        _do(sim, planner, "place", np.array([x, y, 0.0]))
    mat = sim.state().objects["grey_mat"]
    half = np.array(mat.spec.size) / 2
    for name in names:
        p = sim.state().objects[name].pos
        assert np.any(np.abs(p[:2] - mat.pos[:2]) > half), f"{name} still on the mat"
    assert assists <= 2  # clearance-aware grasps should rarely need the assist


def test_multi_container_sort(make_sim):
    sim, planner = make_sim("multi_sort"), SkillPlanner()
    plan = [("red_cube", "red_bin"), ("small_red_cube", "red_bin"), ("blue_cube", "blue_bin"),
            ("red_cylinder", "white_bowl"), ("blue_cylinder", "white_bowl"), ("yellow_cylinder", "white_bowl")]
    for obj, container in plan:
        _pick(sim, planner, obj)
        planned, _ = _do(sim, planner, "place", sim.state().objects[container].pos)
        assert planned.target_object == container
    s = sim.state().objects
    for obj, container in plan:
        c = s[container]
        inner = c.spec.size[0] if c.spec.kind == "bowl" else c.spec.size[0] / 2
        assert np.linalg.norm(s[obj].pos[:2] - c.pos[:2]) < inner, obj
    assert np.linalg.norm(s["yellow_cube"].pos[:2] - [0.66, -0.04]) < 0.02  # distractor untouched
