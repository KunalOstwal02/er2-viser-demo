# Third-party notices

## Franka Emika Panda model (MuJoCo Menagerie)

- Source: https://github.com/google-deepmind/mujoco_menagerie, `franka_emika_panda/`
- Pinned commit: `c96a32d28fb5da84da38c1da4d749e7a13212855`
- License: Apache License 2.0 (a copy is downloaded with the model as `LICENSE`)
- How it is used: the files are **not redistributed** in this repository. On first run,
  `er2_demo/assets.py` downloads `panda.xml`, `LICENSE` and the referenced meshes from the commit
  above, checks them against the SHA-256 hashes in `src/er2_demo/panda_manifest.json`, and caches
  them per user.
- Modifications made at load time (in memory, via `MjSpec`): a fingertip `tcp` site is added,
  gravity compensation is enabled on the arm links, finger-pad friction is raised, the `home`
  keyframe is removed, and the arm is placed in a generated tabletop scene.

## Python dependencies

These are installed from PyPI under their own licenses: MuJoCo (Apache-2.0), mink (Apache-2.0),
Viser (Apache-2.0), mjviser, google-genai (Apache-2.0), NumPy, Pillow, pydantic and platformdirs.

## Example recordings

`examples/recordings/` contains raw text outputs of Google's `gemini-robotics-er-2-preview`
model, plus images rendered by this simulator. Use of model outputs is subject to the Gemini API
terms.
