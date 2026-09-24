"""Locate (and on first use, download) the Franka Panda model from MuJoCo Menagerie.

The files are fetched from a pinned Menagerie commit and checked against SHA-256 hashes, then
cached per user. Resolution order: ``$ER2_PANDA_DIR`` → a repo checkout under ``third_party/`` →
the user cache (downloading into it if needed).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import urllib.request
from pathlib import Path

from platformdirs import user_cache_path

log = logging.getLogger(__name__)

MENAGERIE_COMMIT = "c96a32d28fb5da84da38c1da4d749e7a13212855"
_RAW = "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/{commit}/franka_emika_panda/{path}"
_MANIFEST = json.loads((Path(__file__).with_name("panda_manifest.json")).read_text())
_REPO_COPY = Path(__file__).resolve().parents[2] / "third_party/mujoco_menagerie/franka_emika_panda"


def _complete(directory: Path) -> bool:
    return all((directory / rel).is_file() for rel in _MANIFEST)


def panda_dir() -> Path:
    override = os.environ.get("ER2_PANDA_DIR")
    if override:
        return Path(override)
    if _complete(_REPO_COPY):
        return _REPO_COPY
    cache = user_cache_path("er2-viser-demo") / "menagerie" / MENAGERIE_COMMIT / "franka_emika_panda"
    if not _complete(cache):
        download_panda(cache)
    return cache


def download_panda(target: Path) -> None:
    """Fetch the pinned Panda files (~33 MB), verifying each hash before it is used."""
    log.warning("Downloading the Franka Panda model from MuJoCo Menagerie (one time, ~33 MB)…")
    print("Downloading the Franka Panda model from MuJoCo Menagerie (one time, ~33 MB)…", flush=True)
    target.mkdir(parents=True, exist_ok=True)
    for i, (rel, digest) in enumerate(sorted(_MANIFEST.items()), 1):
        dest = target / rel
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == digest:
            continue
        url = _RAW.format(commit=MENAGERIE_COMMIT, path=rel)
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
            data = response.read()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"checksum mismatch for {rel}; refusing to use it")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
            tmp.write(data)
        os.replace(tmp.name, dest)
        if i % 10 == 0:
            print(f"  {i}/{len(_MANIFEST)} files", flush=True)
    print("Panda model ready.", flush=True)
