"""Entry point: ``er2-demo`` / ``python -m er2_demo``."""

from __future__ import annotations

import argparse
import logging
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

from er2_demo.scene import SceneSpec, list_presets  # noqa: E402
from er2_demo.sim import SimRunner  # noqa: E402
from er2_demo.ui import DemoApp  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemini Robotics ER 2 · Panda Viser playground")
    parser.add_argument("--preset", default="pick_place", choices=sorted(list_presets()))
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it from another machine")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("websockets").setLevel(logging.WARNING)

    sim = SimRunner(SceneSpec.load(list_presets()[args.preset]))
    sim.start()
    DemoApp(sim, port=args.port, host=args.host)
    print(f"\n  ER-2 playground running at http://localhost:{args.port}\n", flush=True)
    if not os.environ.get("GEMINI_API_KEY"):
        print("  GEMINI_API_KEY is not set: live ER-2 runs are disabled (Replay still works).\n", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sim.close()


if __name__ == "__main__":
    main()
