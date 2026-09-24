"""Full closed loop (agent → skills → sim → recorder → replay) with a test-only oracle client."""

import json
import threading

import numpy as np

import er2_demo.agent as agent_mod
from er2_demo.agent import Agent, Recorder, Recording
from er2_demo.er2 import ReplayClient
from er2_demo.skills import project


class OracleClient:
    """Emits ER-2-format JSON from ground truth (TEST ONLY - never shown as ER-2)."""

    def __init__(self, sim, plan):
        self.sim, self.plan = sim, list(plan)

    def decide(self, text, images):
        if not self.plan:
            return json.dumps({"thought": "all sorted", "action": "done", "points": []})
        action, name = self.plan.pop(0)
        cap = self.sim.capture()
        y, x = project(cap, cap.state.objects[name].pos)
        return json.dumps({"thought": f"{action} {name}", "action": action, "points": [[y, x]], "label": name})

    def assess(self, text, image):
        return '{"success": true, "explanation": "looks sorted"}'


class Hooks:
    def __init__(self):
        self.done = threading.Event()
        self.logs, self.final = [], None

    def status(self, text): pass
    def log(self, text): self.logs.append(text)
    def observation(self, cap): pass
    def proposal(self, action, cap): pass
    def planned(self, planned): pass
    def awaiting_confirm(self, waiting): pass

    def finished(self, text):
        self.final = text
        self.done.set()


def _in(sim, obj, container):
    s = sim.state()
    return np.linalg.norm(s.objects[obj].pos[:2] - s.objects[container].pos[:2]) < 0.08


def test_sort_episode_records_and_replays(make_sim, tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "RECORDINGS_DIR", tmp_path)
    sim = make_sim("sort")
    plan = [("pick", "red_cube"), ("place", "red_bin"), ("pick", "blue_cube"), ("place", "blue_bin")]
    hooks = Hooks()
    agent = Agent(sim, hooks)
    agent.auto_run = True
    recorder = Recorder(sim.current_scene(), "sort the cubes", "test-oracle")
    agent.start("sort the cubes", OracleClient(sim, plan), recorder)
    assert hooks.done.wait(300), hooks.logs
    assert "success" in hooks.final, hooks.final
    assert _in(sim, "red_cube", "red_bin") and _in(sim, "blue_cube", "blue_bin")

    rec = Recording.load(recorder.dir)
    assert len(rec.turns) == 5 and rec.assessment
    sim.load_scene(rec.scene).result()
    hooks = Hooks()
    agent = Agent(sim, hooks)
    agent.auto_run = True
    agent.start(rec.task, ReplayClient(rec.turns, rec.assessment), None)
    assert hooks.done.wait(300), hooks.logs
    assert _in(sim, "red_cube", "red_bin") and _in(sim, "blue_cube", "blue_bin")


def test_confirmation_gate_and_skip(make_sim, tmp_path, monkeypatch):
    sim = make_sim("pick_place")
    hooks = Hooks()
    waiting = threading.Event()
    hooks.awaiting_confirm = lambda w: waiting.set() if w else waiting.clear()
    agent = Agent(sim, hooks)
    agent.start("put the red cube in the bowl", OracleClient(sim, [("pick", "red_cube"), ("pick", "red_cube")]), None)
    assert waiting.wait(60)
    before = sim.state().q.copy()
    agent.decide("skip")
    assert waiting.wait(60)  # asked again after the skip, without moving
    assert np.allclose(sim.state().q, before, atol=1e-3)
    agent.stop()
    assert hooks.done.wait(30)
    assert "Stopped" in hooks.final


def test_unreachable_target_is_reported_back(make_sim):
    sim = make_sim("pick_place")
    sim.set_object_pose("red_cube", np.array([1.05, 0.5, 0.03]))

    class Far(OracleClient):
        def assess(self, text, image):
            return '{"success": false, "explanation": "cube out of reach"}'

    hooks = Hooks()
    agent = Agent(sim, hooks)
    agent.auto_run = True
    agent.max_steps = 2
    agent.start("pick the red cube", Far(sim, [("pick", "red_cube")]), None)
    assert hooks.done.wait(120)
    assert any("out of reach" in line or "no graspable" in line for line in hooks.logs), hooks.logs
