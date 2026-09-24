"""Closed-loop ER-2 episode runner (own thread) plus episode recording/replay."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from PIL import Image

from er2_demo.er2 import ASSESS_PROMPT, Action, Client, ER2Error, parse_action, parse_assessment
from er2_demo.planner import PlanningError
from er2_demo.scene import SceneSpec
from er2_demo.sim import Capture, SimRunner
from er2_demo.skills import PlannedAction, SkillPlanner

log = logging.getLogger(__name__)
RECORDINGS_DIR = Path(os.environ.get("ER2_RECORDINGS_DIR", "recordings"))  # relative to the cwd
EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "recordings"  # shipped in the repo


class Hooks(Protocol):
    """How the agent reports progress to the UI (called from the agent thread)."""

    def status(self, text: str) -> None: ...
    def log(self, text: str) -> None: ...
    def observation(self, cap: Capture) -> None: ...
    def proposal(self, action: Action, cap: Capture) -> None: ...
    def planned(self, planned: PlannedAction | None) -> None: ...
    def awaiting_confirm(self, waiting: bool) -> None: ...
    def finished(self, text: str) -> None: ...


class Recorder:
    """Writes an episode incrementally so partial or failed runs are kept too."""

    def __init__(self, scene: SceneSpec, task: str, model_id: str) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = RECORDINGS_DIR / f"{stamp}_{scene.name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        scene.save(self.dir / "scene.json")
        self.data: dict = {"task": task, "model": model_id, "scene": scene.name, "turns": [], "assessment": None}
        self._write()

    def turn(self, step: int, raw: str, outcome: str, cap: Capture) -> None:
        self.data["turns"].append({"step": step, "raw": raw, "outcome": outcome})
        Image.fromarray(cap.images["front"]).save(self.dir / f"step{step:02d}_front.jpg", quality=85)
        self._write()

    def update_outcome(self, outcome: str) -> None:
        if self.data["turns"]:
            self.data["turns"][-1]["outcome"] = outcome
            self._write()

    def assessment(self, raw: str) -> None:
        self.data["assessment"] = raw
        self._write()

    def _write(self) -> None:
        (self.dir / "episode.json").write_text(json.dumps(self.data, indent=2))


def list_recordings() -> dict[str, Path]:
    """Your recordings (newest first), then the example episodes shipped with the repo."""
    found: dict[str, Path] = {}
    for root, prefix in ((RECORDINGS_DIR, ""), (EXAMPLES_DIR, "example: ")):
        if root.is_dir():
            for p in sorted(root.iterdir(), reverse=True):
                if (p / "episode.json").exists():
                    found[prefix + p.name] = p
    return found


@dataclass
class Recording:
    scene: SceneSpec
    task: str
    turns: list[str]
    assessment: str | None

    @classmethod
    def load(cls, path: Path) -> Recording:
        data = json.loads((path / "episode.json").read_text())
        return cls(SceneSpec.load(path / "scene.json"), data["task"], [t["raw"] for t in data["turns"]],
                   data.get("assessment"))


@dataclass
class _Control:
    stop: threading.Event = field(default_factory=threading.Event)
    decision: threading.Event = field(default_factory=threading.Event)
    choice: str = ""


def turn_prompt(task: str, step: int, max_steps: int, history: list[str], held: str | None,
                last_result: str | None) -> str:
    lines = [f'Task: "{task}"', f"Turn {step} of at most {max_steps}."]
    lines.append(f"The gripper is holding: {held.replace('_', ' ') if held else 'nothing'}.")
    if history:
        lines.append("Actions so far:")
        lines += [f"  {h}" for h in history]
    if last_result:
        lines.append(f"Result of your last action: {last_result}")
    lines.append("Choose the next action. Image 1 = front camera, Image 2 = side camera.")
    return "\n".join(lines)


class Agent:
    def __init__(self, sim: SimRunner, hooks: Hooks) -> None:
        self.sim = sim
        self.hooks = hooks
        self.skills = SkillPlanner()
        self.auto_run = False
        self.max_steps = 12
        self._thread: threading.Thread | None = None
        self._ctl = _Control()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, task: str, client: Client, recorder: Recorder | None) -> None:
        if self.running:
            raise RuntimeError("an episode is already running")
        self._ctl = _Control()
        self._thread = threading.Thread(target=self._safe_loop, args=(task, client, recorder, self._ctl),
                                        name="agent", daemon=True)
        self._thread.start()

    def decide(self, choice: str) -> None:
        """'execute' | 'skip' from the UI while a proposal is awaiting confirmation."""
        self._ctl.choice = choice
        self._ctl.decision.set()

    def stop(self) -> None:
        self._ctl.stop.set()
        self._ctl.choice = "stop"
        self._ctl.decision.set()
        self.sim.stop()

    def _safe_loop(self, task: str, client: Client, recorder: Recorder | None, ctl: _Control) -> None:
        try:
            self._loop(task, client, recorder, ctl)
        except Exception as error:  # noqa: BLE001 - never let the UI hang on a dead episode
            log.exception("episode crashed")
            self.hooks.finished(f"⚠️ Episode error: {error}")
        finally:
            self.hooks.awaiting_confirm(False)

    def _loop(self, task: str, client: Client, recorder: Recorder | None, ctl: _Control) -> None:
        history: list[str] = []
        last_result: str | None = None
        self.hooks.log(f"**Task:** {task}")
        summary = "Stopped."
        for step in range(1, self.max_steps + 1):
            if ctl.stop.is_set():
                break
            cap = self.sim.capture()
            self.hooks.observation(cap)
            self.hooks.planned(None)
            self.hooks.status(f"Step {step}: asking ER-2…")
            t0 = time.perf_counter()
            text = turn_prompt(task, step, self.max_steps, history, cap.state.held, last_result)
            try:
                raw = client.decide(text, [cap.images["front"], cap.images["side"]])
            except ER2Error as error:
                self.hooks.finished(f"⚠️ ER-2 unavailable: {error}. Press Run to retry, or use Replay.")
                return
            if ctl.stop.is_set():
                break
            latency = time.perf_counter() - t0
            if recorder:
                recorder.turn(step, raw, "pending", cap)
            try:
                action = parse_action(raw)
            except ER2Error as error:
                last_result = f"Your reply could not be used ({error}). Reply with the JSON format only."
                self.hooks.log(f"**{step}.** ⚠️ unparseable reply: {error}")
                if recorder:
                    recorder.update_outcome(last_result)
                continue
            self.hooks.proposal(action, cap)
            self.hooks.log(f"**{step}. {action.action}** {action.label} · _{latency:.1f}s_\n\n> {action.thought}")
            if action.action == "done":
                summary = action.thought or "ER-2 reports the task is done."
                if recorder:
                    recorder.update_outcome("done")
                break
            try:
                planned = self.skills.plan(action.action, action.point_tuples(), cap)
            except PlanningError as error:
                last_result = f"{action.action} failed before moving: {error}."
                self.hooks.log(f"↳ ❌ {error}")
                if recorder:
                    recorder.update_outcome(last_result)
                history.append(f"{step}. {action.action} {action.label} → failed ({error})")
                continue
            self.hooks.planned(planned)
            if not self.auto_run:
                self.hooks.status(f"Step {step}: {planned.description} — press Execute (or Skip)")
                self.hooks.awaiting_confirm(True)
                ctl.decision.clear()
                while not ctl.decision.wait(0.1):
                    if self.auto_run:
                        ctl.choice = "execute"
                        break
                self.hooks.awaiting_confirm(False)
                if ctl.choice == "stop" or ctl.stop.is_set():
                    break
                if ctl.choice == "skip":
                    last_result = "The operator rejected that action. Reconsider and choose again."
                    self.hooks.log("↳ ⏭ skipped by operator")
                    if recorder:
                        recorder.update_outcome("skipped")
                    continue
            self.hooks.status(f"Step {step}: {planned.description}…")
            result = self.sim.execute(planned.trajectory,
                                      planned.target_object if action.action == "pick" else None).result()
            if result.status == "stopped":
                break
            if result.status == "error":
                outcome = f"failed: {result.message}"
            elif action.action == "pick" and not result.grasped:
                outcome = f"failed: the gripper closed but {planned.target_object.replace('_', ' ')} was not lifted"
            else:
                outcome = "succeeded" + (" (grasp assist engaged)" if result.assist_used else "")
            last_result = f"{planned.description} {outcome}."
            history.append(f"{step}. {planned.description} → {outcome}")
            self.hooks.log(f"↳ {'✅' if outcome.startswith('succeeded') else '❌'} {planned.description} {outcome}")
            if recorder:
                recorder.update_outcome(outcome)
        else:
            summary = f"Reached the {self.max_steps}-step limit."

        self.hooks.planned(None)
        if ctl.stop.is_set():
            self.hooks.finished("⏹ Stopped.")
            return
        self.hooks.status("Asking ER-2 to check the result…")
        cap = self.sim.capture()
        self.hooks.observation(cap)
        verdict = ""
        try:
            raw = client.assess(ASSESS_PROMPT.format(task=task), cap.images["front"])
            if recorder:
                recorder.assessment(raw)
            assessment = parse_assessment(raw)
            verdict = f"{'✅ success' if assessment.success else '❌ not achieved'} — {assessment.explanation}"
        except ER2Error as error:
            verdict = f"(self-assessment unavailable: {error})"
        self.hooks.finished(f"**ER-2 summary:** {summary}\n\n**Self-assessment:** {verdict}")

