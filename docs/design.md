# Design (MVP for the 2026-09-25 journal-club talk)

Agreed decisions:
- **Purpose:** a live journal-club demo that is reliable, smooth and quick to reset.
- **Robot:** Franka Panda from MuJoCo Menagerie.
- **Motion:** minimum-jerk Cartesian waypoints tracked with mink differential IK. Each trajectory is
  planned in full on a separate robot-only model and validated before it is shown or executed.
- **ER-2 interface:** structured JSON with one primitive per turn (`pick`, `place`, `home`, `done`)
  and `[y, x]` points (0–1000) in the front image. Closed loop: fresh images after every action.
- **Showcase:** points drawn on the camera images, 3D target and path markers, a thought log,
  plan-then-confirm with an auto-run toggle, and ER-2 self-assessment at the end.
- **Grasping:** physics first. If the object slips, a disclosed weld assist engages; the mode can be toggled.
- **Environment editing:** gizmo drag, spawning primitive shapes, bowls and bins, delete, save/load presets.
  No meshes, no lighting or camera editing.
- **Fallback:** every live run is recorded and can be replayed (re-simulated, clearly labelled REPLAY).
- **Must work:** the `pick_place` and `sort` presets.

Architecture: one process. A sim thread is the sole owner of MuJoCo and runs a command queue.
An agent thread runs the closed loop and does IK planning on its own model. Viser callbacks
only enqueue commands.

Deferred: the push and path primitives, stacking presets, lighting and camera editing, and a workshop/Colab package.
