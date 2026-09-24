import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import pytest

from er2_demo.scene import SceneSpec, list_presets
from er2_demo.sim import SimRunner


@pytest.fixture
def make_sim():
    sims = []

    def make(preset: str) -> SimRunner:
        sim = SimRunner(SceneSpec.load(list_presets()[preset]), realtime=False)
        sim.start()
        sims.append(sim)
        return sim

    yield make
    for sim in sims:
        sim.close()
