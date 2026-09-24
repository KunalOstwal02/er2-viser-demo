Normally you don't need this folder: the Panda model is downloaded and verified automatically on
first run (see `src/er2_demo/assets.py`).

For development you can keep a local checkout here, which takes precedence over the cache:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
    git -C third_party/mujoco_menagerie checkout c96a32d28fb5da84da38c1da4d749e7a13212855
    git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda
