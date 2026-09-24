"""Viser front-end: 3D scene, agent controls, camera overlays and scene editor."""

from __future__ import annotations

import logging
import math
import threading
from collections import deque

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene
from PIL import Image, ImageDraw, ImageFont

from er2_demo.agent import Agent, Recorder, Recording, list_recordings
from er2_demo.er2 import MODEL_ID, Action, ER2Error, GeminiER2, ReplayClient
from er2_demo.planner import PlanningError
from er2_demo.scene import COLORS, DEFAULT_SIZES, PRESET_DIR, ObjectSpec, SceneSpec, list_presets
from er2_demo.sim import Capture, SimRunner
from er2_demo.skills import PlannedAction, to_pixel

log = logging.getLogger(__name__)

ACCENT = (66, 133, 244)
TARGET_RGB = (255, 0, 200)


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


class DemoApp:
    def __init__(self, sim: SimRunner, port: int = 8080, host: str = "127.0.0.1") -> None:
        self.sim = sim
        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        self.server.gui.configure_theme(control_width="large", dark_mode=True, show_logo=False,
                                        brand_color=ACCENT)
        self._scene_lock = threading.Lock()
        self._mj_scene: ViserMujocoScene | None = None
        self._markers: dict[str, object] = {}
        self._gizmo = None
        self._log: deque[str] = deque(maxlen=40)
        self._last_cap: Capture | None = None
        self._last_action: Action | None = None
        self.agent = Agent(sim, self)
        self._build_gui()
        self._set_initial_camera()
        sim.model_listeners.append(self._on_model)
        sim.listeners.append(self._on_frame)
        self._on_model(sim.model)  # scene already built before we subscribed
        self._refresh_objects()

    # ---- 3D scene ---------------------------------------------------------------------------

    def _set_initial_camera(self) -> None:
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            client.camera.position = (1.55, -0.95, 0.95)
            client.camera.look_at = (0.45, 0.0, 0.1)
            client.camera.up_direction = (0.0, 0.0, 1.0)

    def _on_model(self, model: mujoco.MjModel) -> None:
        with self._scene_lock:
            self.server.scene.reset()
            self._markers.clear()
            self._gizmo = None
            self._mj_scene = ViserMujocoScene(self.server, model, num_envs=1)
        # Runs on the sim thread: never call back into the sim from here (deadlock).

    def _on_frame(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        del model
        with self._scene_lock:
            if self._mj_scene is not None:
                self._mj_scene.update_from_mjdata(data)

    def _clear_markers(self) -> None:
        for handle in self._markers.values():
            try:
                handle.remove()
            except Exception:  # noqa: BLE001
                pass
        self._markers.clear()

    # ---- GUI --------------------------------------------------------------------------------

    def _build_gui(self) -> None:
        gui = self.server.gui
        gui.add_markdown("## 🤖 Gemini Robotics ER 2 · Panda playground")
        self.banner = gui.add_markdown("")
        self.status_md = gui.add_markdown("**Status:** ready")
        tabs = gui.add_tab_group()

        with tabs.add_tab("Agent", icon=viser.Icon.ROBOT):
            presets = list(list_presets())
            self.preset = gui.add_dropdown("Scene preset", options=presets, initial_value=self.sim.scene.name
                                           if self.sim.scene.name in presets else presets[0])
            self.task = gui.add_text("Task", self.sim.scene.prompt, multiline=True)
            self.mode = gui.add_dropdown("Mode", options=("Live ER-2", "Replay recording"),
                                         initial_value="Live ER-2")
            recs = list(list_recordings()) or ["(none)"]
            self.recording = gui.add_dropdown("Recording", options=recs, initial_value=recs[0], visible=False)
            with gui.add_folder("Episode", expand_by_default=True):
                self.run_btn = gui.add_button("Run", color="blue", icon=viser.Icon.PLAYER_PLAY)
                self.auto_run = gui.add_checkbox("Auto-run (no confirmation)", False)
                self.execute_btn = gui.add_button("Execute proposed action", color="green",
                                                  icon=viser.Icon.CHECK, disabled=True)
                self.skip_btn = gui.add_button("Skip / ask again", disabled=True, icon=viser.Icon.PLAYER_SKIP_FORWARD)
                self.stop_btn = gui.add_button("Stop", color="red", icon=viser.Icon.PLAYER_STOP, disabled=True)
            with gui.add_folder("Robot & scene", expand_by_default=False):
                self.reset_btn = gui.add_button("Reset scene", icon=viser.Icon.REFRESH)
                self.home_btn = gui.add_button("Arm to home", icon=viser.Icon.HOME)
                self.grasp_mode = gui.add_dropdown("Grasp", options=SimRunner.GRASP_MODES,
                                                   initial_value=self.sim.grasp_mode)
                self.speed = gui.add_slider("Speed (m/s)", min=0.1, max=0.5, step=0.05, initial_value=0.25)
                self.max_steps = gui.add_slider("Max steps", min=3, max=20, step=1, initial_value=12)
            self.log_md = gui.add_markdown("")

        with tabs.add_tab("Cameras", icon=viser.Icon.CAMERA):
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            self.front_img = gui.add_image(blank, label="Front camera (ER-2 image 1)", format="jpeg")
            self.side_img = gui.add_image(blank, label="Side camera (ER-2 image 2)", format="jpeg")
            self.snap_btn = gui.add_button("Refresh images", icon=viser.Icon.CAMERA)

        with tabs.add_tab("Scene", icon=viser.Icon.CUBE):
            gui.add_markdown("Select an object to drag it with the gizmo in the 3D view.")
            self.obj_select = gui.add_dropdown("Object", options=["(none)"], initial_value="(none)")
            self.delete_btn = gui.add_button("Delete selected", icon=viser.Icon.TRASH)
            with gui.add_folder("Add object"):
                self.new_kind = gui.add_dropdown("Shape", options=tuple(DEFAULT_SIZES), initial_value="cube")
                self.new_color = gui.add_dropdown("Color", options=tuple(COLORS), initial_value="green")
                self.spawn_btn = gui.add_button("Add to table", icon=viser.Icon.PLUS)
            with gui.add_folder("Save layout"):
                self.save_name = gui.add_text("Preset name", "my_scene")
                self.save_btn = gui.add_button("Save as preset", icon=viser.Icon.DEVICE_FLOPPY)

        self._wire()

    def _wire(self) -> None:
        @self.preset.on_update
        def _(_) -> None:
            if self._editable():
                scene = SceneSpec.load(list_presets()[self.preset.value])
                self.task.value = scene.prompt
                self._after(self.sim.load_scene(scene))
                self._set_banner("")

        @self.mode.on_update
        def _(_) -> None:
            replay = self.mode.value != "Live ER-2"
            if replay:
                recs = list(list_recordings()) or ["(none)"]
                self.recording.options = recs
                self.recording.value = recs[0]
            self.recording.visible = replay

        self.run_btn.on_click(lambda _: self._run())
        self.execute_btn.on_click(lambda _: self.agent.decide("execute"))
        self.skip_btn.on_click(lambda _: self.agent.decide("skip"))
        self.stop_btn.on_click(lambda _: self.agent.stop() if self.agent.running else self.sim.stop())

        @self.auto_run.on_update
        def _(_) -> None:
            self.agent.auto_run = self.auto_run.value

        @self.grasp_mode.on_update
        def _(_) -> None:
            self.sim.grasp_mode = self.grasp_mode.value

        @self.speed.on_update
        def _(_) -> None:
            self.agent.skills.planner.speed = float(self.speed.value)

        @self.max_steps.on_update
        def _(_) -> None:
            self.agent.max_steps = int(self.max_steps.value)

        @self.reset_btn.on_click
        def _(_) -> None:
            if self._editable():
                self._after(self.sim.reset())
                self._set_banner("")
                self._clear_markers()

        @self.home_btn.on_click
        def _(_) -> None:
            if self._editable():
                threading.Thread(target=self._go_home, daemon=True).start()

        self.snap_btn.on_click(lambda _: threading.Thread(target=self._snapshot, daemon=True).start())

        @self.obj_select.on_update
        def _(_) -> None:
            self._attach_gizmo(self.obj_select.value)

        @self.delete_btn.on_click
        def _(_) -> None:
            name = self.obj_select.value
            if name != "(none)" and self._editable():
                self.sim.delete(name).result()
                self._refresh_objects()

        @self.spawn_btn.on_click
        def _(_) -> None:
            if not self._editable():
                return
            kind, color = self.new_kind.value, self.new_color.value
            x, y = self._free_table_spot()
            obj = ObjectSpec(f"{color}_{kind}", kind, color, x, y)
            self.sim.spawn(obj).result()
            self._refresh_objects(select=obj.name)

        @self.save_btn.on_click
        def _(event: viser.GuiEvent) -> None:
            name = "".join(c for c in self.save_name.value if c.isalnum() or c in "_-") or "my_scene"
            scene = self.sim.current_scene()
            scene.name, scene.prompt = name, self.task.value
            scene.save(PRESET_DIR / f"{name}.json")
            self.preset.options = list(list_presets())
            self._notify(event.client, "Saved", f"Preset '{name}' saved.")

    # ---- actions ------------------------------------------------------------------------------

    def _editable(self) -> bool:
        if self.agent.running or self.sim.busy:
            self._notify(None, "Busy", "Stop the episode (or wait for the arm) before editing.")
            return False
        return True

    def _run(self) -> None:
        if self.agent.running:
            return
        self._clear_markers()
        self._log.clear()
        self.agent.auto_run = self.auto_run.value
        self.agent.max_steps = int(self.max_steps.value)
        if self.mode.value == "Live ER-2":
            try:
                client = GeminiER2()
            except ER2Error as error:
                self._set_banner(f"⚠️ {error} — export it before launching, or use Replay mode.")
                return
            recorder = Recorder(self.sim.current_scene(), self.task.value, MODEL_ID)
            self._set_banner(f"🔴 **LIVE** · {MODEL_ID}")
            self.agent.start(self.task.value, client, recorder)
        else:
            recs = list_recordings()
            if self.recording.value not in recs:
                self._set_banner("⚠️ No recording selected.")
                return
            rec = Recording.load(recs[self.recording.value])
            self.sim.load_scene(rec.scene, as_initial=True).result()
            self._refresh_objects()
            self.task.value = rec.task
            self._set_banner(f"⏪ **REPLAY** of recorded ER-2 run `{self.recording.value}`")
            self.agent.start(rec.task, ReplayClient(rec.turns, rec.assessment), None)
        self._set_running(True)

    def _after(self, future) -> None:
        """Refresh the object list once a scene rebuild completes (off the sim thread)."""
        threading.Thread(target=lambda: (future.result(), self._refresh_objects()), daemon=True).start()

    def _go_home(self) -> None:
        state = self.sim.state()
        try:
            traj = self.agent.skills.planner.home(state.q, state.grip)
        except PlanningError as error:
            self.status(f"Cannot go home: {error}")
            return
        self.sim.execute(traj).result()

    def _snapshot(self) -> None:
        self.observation(self.sim.capture())

    def _set_running(self, running: bool) -> None:
        self.run_btn.disabled = running
        self.stop_btn.disabled = not running
        for handle in (self.preset, self.reset_btn, self.home_btn, self.delete_btn, self.spawn_btn, self.mode):
            handle.disabled = running

    def _set_banner(self, text: str) -> None:
        self.banner.content = text

    def _notify(self, client: viser.ClientHandle | None, title: str, body: str) -> None:
        clients = [client] if client is not None else list(self.server.get_clients().values())
        for c in clients:
            c.add_notification(title, body, auto_close_seconds=4)

    # ---- scene editing ------------------------------------------------------------------------

    def _refresh_objects(self, select: str | None = None) -> None:
        names = [o.name for o in self.sim.scene.objects] or ["(none)"]
        options = ["(none)", *[n for n in names if n != "(none)"]]
        self.obj_select.options = options
        value = select if select in options else "(none)"
        self.obj_select.value = value
        self._attach_gizmo(value)

    def _attach_gizmo(self, name: str) -> None:
        if self._gizmo is not None:
            try:
                self._gizmo.remove()
            except Exception:  # noqa: BLE001
                pass
            self._gizmo = None
        if name == "(none)":
            return
        state = self.sim.state()
        if name not in state.objects:
            return
        obj = state.objects[name]
        gizmo = self.server.scene.add_transform_controls(
            "/gizmo", scale=0.15, position=tuple(obj.pos), wxyz=(math.cos(obj.yaw / 2), 0, 0, math.sin(obj.yaw / 2)),
            disable_rotations=False, depth_test=False)

        @gizmo.on_update
        def _(_) -> None:
            w, x, y, z = gizmo.wxyz
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            pos = np.array(gizmo.position, dtype=float)
            pos[2] = max(pos[2], obj.pos[2])
            self.sim.set_object_pose(name, pos, yaw)

        self._gizmo = gizmo

    def _free_table_spot(self) -> tuple[float, float]:
        state = self.sim.state()
        rng = np.random.default_rng()
        best, best_d = (0.5, 0.0), -1.0
        for _ in range(200):
            p = np.array([rng.uniform(0.35, 0.65), rng.uniform(-0.28, 0.28)])
            d = min((np.linalg.norm(p - o.pos[:2]) for o in state.objects.values()), default=1.0)
            if d > best_d:
                best, best_d = (float(p[0]), float(p[1])), d
        return best

    # ---- agent hooks (called from the agent thread) ---------------------------------------

    def status(self, text: str) -> None:
        self.status_md.content = f"**Status:** {text}"

    def log(self, text: str) -> None:
        self._log.append(text)
        self.log_md.content = "### Episode log\n\n" + "\n\n".join(reversed(self._log))

    def observation(self, cap: Capture) -> None:
        self._last_cap = cap
        self._last_action = None
        self.front_img.image = cap.images["front"]
        self.side_img.image = cap.images["side"]
        if self.obj_select.value != "(none)":
            self._attach_gizmo(self.obj_select.value)

    def proposal(self, action: Action, cap: Capture) -> None:
        self._last_action = action
        self.front_img.image = self._overlay(cap.images["front"], action)

    def planned(self, planned: PlannedAction | None) -> None:
        self._clear_markers()
        if planned is None or planned.trajectory is None:
            return
        path = planned.trajectory.tcp[::4]
        if len(path) >= 2:
            self._markers["path"] = self.server.scene.add_spline_catmull_rom(
                "/er2/path", points=path, color=ACCENT, thickness=4.0, thickness_units="screen")
        if planned.target_world is not None:
            self._markers["target"] = self.server.scene.add_icosphere(
                "/er2/target", radius=0.014, color=TARGET_RGB, position=tuple(planned.target_world))
            self._markers["label"] = self.server.scene.add_label(
                "/er2/target_label", planned.description, position=tuple(planned.target_world + [0, 0, 0.06]))

    def awaiting_confirm(self, waiting: bool) -> None:
        self.execute_btn.disabled = not waiting
        self.skip_btn.disabled = not waiting

    def finished(self, text: str) -> None:
        self.status("episode finished")
        self.log(text)
        self._set_banner(self.banner.content + "\n\n" + text if self.banner.content else text)
        self._set_running(False)
        self.awaiting_confirm(False)

    @staticmethod
    def _overlay(image: np.ndarray, action: Action) -> np.ndarray:
        img = Image.fromarray(image).convert("RGB")
        draw = ImageDraw.Draw(img)
        h, w = image.shape[:2]
        font = _font(18)
        for i, pt in enumerate(action.point_tuples()):
            u, v = to_pixel(pt, w, h)
            r = 11
            draw.ellipse([u - r, v - r, u + r, v + r], outline=TARGET_RGB, width=3)
            draw.line([u - r - 6, v, u + r + 6, v], fill=TARGET_RGB, width=2)
            draw.line([u, v - r - 6, u, v + r + 6], fill=TARGET_RGB, width=2)
            text = f"{action.action}: {action.label}" if i == 0 else f"{i + 1}"
            tb = draw.textbbox((u + 16, v - 26), text, font=font)
            draw.rectangle([tb[0] - 4, tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=(20, 20, 20))
            draw.text((u + 16, v - 26), text, fill=(255, 255, 255), font=font)
        return np.asarray(img)
