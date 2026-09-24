"""Drive the Viser app's handlers directly (no browser)."""

import json
import threading

import numpy as np

import er2_demo.agent as agent_mod
from er2_demo.agent import Recorder
from er2_demo.er2 import Action
from er2_demo.scene import ObjectSpec
from er2_demo.ui import DemoApp
from tests.test_episode import OracleClient


def _wait(pred, timeout=60):
    ev = threading.Event()
    for _ in range(int(timeout / 0.1)):
        if pred():
            return True
        ev.wait(0.1)
    return False


def test_ui_edit_and_replay(make_sim, tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "RECORDINGS_DIR", tmp_path)
    sim = make_sim("pick_place")
    app = DemoApp(sim, port=8123)
    try:
        # spawn + delete rebuild the model and keep the object list in sync
        n0 = len(sim.scene.objects)
        sim.spawn(ObjectSpec("purple_sphere", "sphere", "purple", 0.6, -0.25)).result()
        app._refresh_objects(select="purple_sphere")
        assert len(sim.scene.objects) == n0 + 1 and app.obj_select.value == "purple_sphere"
        # drag via gizmo path
        sim.set_object_pose("purple_sphere", np.array([0.45, -0.25, 0.03]))
        assert _wait(lambda: abs(sim.state().objects["purple_sphere"].pos[0] - 0.45) < 0.01)
        sim.delete("purple_sphere").result()
        app._refresh_objects()
        assert "purple_sphere" not in app.obj_select.options

        # overlays + markers render without errors
        cap = sim.capture()
        app.observation(cap)
        img = app._overlay(cap.images["front"], Action(action="pick", points=[[500, 500]], label="x"))
        assert img.shape == cap.images["front"].shape

        # make a recording with the oracle, then replay it through the UI Run button path
        rec = Recorder(sim.current_scene(), "put the red cube in the blue bowl", "test-oracle")
        app.agent.auto_run = True
        app.agent.start(rec.data["task"], OracleClient(sim, [("pick", "red_cube"), ("place", "blue_bowl")]), rec)
        assert _wait(lambda: not app.agent.running, 300)
        app.mode.value = "Replay recording"
        app.recording.options = [rec.dir.name]
        app.recording.value = rec.dir.name
        app.auto_run.value = True
        app._run()
        assert "REPLAY" in app.banner.content
        assert _wait(lambda: not app.agent.running, 300)
        assert "Self-assessment" in app.banner.content
        assert json.loads((rec.dir / "episode.json").read_text())["turns"]
    finally:
        app.server.stop()
