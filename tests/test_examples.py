"""The shipped example episodes are real ER-2 outputs: replaying them must still succeed."""

import pytest

from er2_demo.agent import EXAMPLES_DIR, Agent, Recording
from er2_demo.er2 import ReplayClient
from er2_demo.sim import SimRunner
from tests.test_episode import Hooks

EXAMPLES = sorted(p for p in EXAMPLES_DIR.iterdir() if (p / "episode.json").exists())


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_example_replays_cleanly(path):
    rec = Recording.load(path)
    sim = SimRunner(rec.scene, realtime=False)
    sim.start()
    try:
        hooks = Hooks()
        agent = Agent(sim, hooks)
        agent.auto_run = True
        agent.max_steps = len(rec.turns)
        agent.start(rec.task, ReplayClient(rec.turns, rec.assessment), None)
        assert hooks.done.wait(600)
        failures = [line for line in hooks.logs if "❌" in line or "⚠️" in line]
        assert not failures, failures
        assert sim.state().held is None
    finally:
        sim.close()
