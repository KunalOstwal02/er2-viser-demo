The Franka Panda MJCF comes from google-deepmind/mujoco_menagerie
(`franka_emika_panda/`, Apache-2.0, commit c96a32d28fb5da84da38c1da4d749e7a13212855).
Fetch it with:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
    git -C third_party/mujoco_menagerie sparse-checkout set franka_emika_panda
