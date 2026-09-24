# ER-2 Viser Playground

A MuJoCo + Viser tabletop demo in which **Gemini Robotics ER 2** drives a simulated
**Franka Panda**. ER-2 sees the front and side camera images and replies with one action
per turn: `pick`, `place`, `home` or `done`, with target points in the image. The sim
back-projects those points to 3D, plans a smooth path (minimum-jerk Cartesian waypoints,
mink differential IK), shows the plan, and executes it in real time. Then it sends ER-2
fresh images and repeats.

Unofficial project; not affiliated with Google DeepMind.

## Run

```bash
conda activate er2_viser            # or: pip install -e ".[dev]" in a Python 3.11 env
export GEMINI_API_KEY=...           # only needed for live ER-2 runs
er2-demo                            # → http://localhost:8080   (--preset sort, --port, --host 0.0.0.0)
```

The Panda model is fetched once from MuJoCo Menagerie (see `third_party/README.md`).
On a laptop with a display you can leave `MUJOCO_GL` unset; the app defaults to EGL,
which works on NVIDIA GPUs.

## Using it

**Agent tab**
1. Pick a **Scene preset** (`pick_place`, `sort`, or one you saved). The task text fills in.
   You can edit it freely.
2. Press **Run**. Each turn, ER-2's point appears on the *Cameras* tab. The 3D view shows the
   target (magenta sphere) and the planned gripper path (blue line).
3. Press **Execute proposed action** to run it, or **Skip / ask again**. Tick **Auto-run** to skip
   confirmations.
4. At the end, ER-2 assesses the final image itself; the verdict is shown in the banner.
5. **Stop** halts the arm immediately. **Reset scene** restores the preset.

**Scene tab**: select an object to drag or rotate it with the gizmo. You can do this between
steps of a running episode, and ER-2 will re-plan from the new images. You can also add
cubes, cuboids, cylinders, spheres, bowls and bins in any colour, delete objects, and
**Save as preset**.

**Robot & scene** (folder): grasp mode, arm speed, maximum steps, and an "Arm to home" button.
- *physics+assist* (default): real friction grasp. If the object slips during the first 4 cm
  of lift, a weld "grasp assist" engages and the log says so.
- *physics*: honest mode; slips stay failures.
- *always weld*: attaches whenever the fingers close on an object.

## Record & replay (talk safety net)

Every live run is saved in `recordings/<time>_<scene>/`: the scene, the task, every raw ER-2
response, per-step images and the self-assessment. Switch **Mode → Replay recording**, pick a
run and press **Run**. The same scene is restored and ER-2's recorded responses are replayed
turn by turn; the motion is re-simulated. The banner shows **REPLAY** so the audience knows
it's not live.

### Rehearsal checklist
1. Run `pick_place` and `sort` live 2–3 times each. Note which recording folders were good.
2. Delete bad runs from `recordings/`.
3. Turn Wi-Fi off and replay one good run of each preset end to end.
4. During the talk, go live if the network is fine; otherwise switch to Replay.

## How it works

| Module | Role |
|---|---|
| `scene.py` | JSON scene spec (primitive objects, presets). Builds the MJCF with `MjSpec` around the Menagerie Panda. |
| `sim.py` | The only thread that touches MuJoCo: 500 Hz physics, 100 Hz trajectory playback, grasp assist, camera capture. Streams to Viser. |
| `planner.py` | Minimum-jerk Cartesian segments → mink IK on a separate robot-only model. Every trajectory is fully solved and checked (reach, joint jumps) before it runs. |
| `skills.py` | ER-2 `[y, x]` points → 3D via rendered depth. Segmentation snaps picks to the object and places to a free spot inside a container. |
| `er2.py` | Gemini client (Interactions API with `generate_content` fallback), prompt, and tolerant JSON parsing. |
| `agent.py` | Closed-loop episode thread, confirmation gate, recorder, replay loading. |
| `ui.py` | Viser GUI, overlays, 3D markers, scene editor. |

Environment variables: `ER2_MODEL_ID` (default `gemini-robotics-er-2-preview`),
`ER2_TRANSPORT` (`auto` | `interactions` | `generate`).

## Tests

```bash
MUJOCO_GL=egl pytest -q
```

The tests make no API calls. The end-to-end tests use a test-only oracle client that writes
ER-2-format JSON from ground truth. It exists only to exercise the pipeline; never present
its output as ER-2.
