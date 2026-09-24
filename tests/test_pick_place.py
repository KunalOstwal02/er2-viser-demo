"""Oracle episodes: exact image points for the target objects must complete both presets."""

import numpy as np
import pytest

from er2_demo.skills import SkillPlanner, project


def _run(sim, planner, action, name):
    cap = sim.capture()
    point = project(cap, cap.state.objects[name].pos) if name else None
    planned = planner.plan(action, [point] if point else [], cap)
    result = sim.execute(planned.trajectory, planned.target_object if action == "pick" else None).result(timeout=120)
    assert result.status == "done", result
    return result


def _inside(sim, obj, container, margin=0.0):
    s = sim.state()
    o, c = s.objects[obj].pos, s.objects[container]
    half = (c.spec.size[0] if c.spec.kind == "bowl" else c.spec.size[0] / 2) - margin
    return np.linalg.norm(o[:2] - c.pos[:2]) < half and o[2] < 0.1


@pytest.mark.parametrize("mode", ["physics+assist", "physics"])
def test_pick_place_preset(make_sim, mode):
    sim = make_sim("pick_place")
    sim.grasp_mode = mode
    planner = SkillPlanner()
    pick = _run(sim, planner, "pick", "red_cube")
    assert pick.held == "red_cube"
    _run(sim, planner, "place", "blue_bowl")
    assert sim.state().held is None
    assert _inside(sim, "red_cube", "blue_bowl")


def test_sort_preset(make_sim):
    sim = make_sim("sort")
    planner = SkillPlanner()
    for obj, bin_ in [("red_cube", "red_bin"), ("blue_cube", "blue_bin"),
                      ("red_cylinder", "red_bin"), ("blue_cylinder", "blue_bin")]:
        assert _run(sim, planner, "pick", obj).held == obj
        _run(sim, planner, "place", bin_)
        assert _inside(sim, obj, bin_), obj
