"""MediaPipe Pose Landmarker 모델 파일 준비. 앱(frontend assets)과 같은 파일인지 SHA-256으로 확인합니다."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

LITE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
LITE_SHA256 = "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"  # frontend THIRD_PARTY_NOTICES.md
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = REPO_ROOT / "models" / "pose_landmarker_lite.task"
FRONTEND_ASSET = REPO_ROOT.parent / "frontend" / "app" / "src" / "main" / "assets" / "pose_landmarker_lite.task"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_lite_model(path: Path = DEFAULT_PATH) -> Path:
    """모델이 없으면 옆 폴더의 frontend asset을 복사하고, 그것도 없으면 내려받습니다."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if FRONTEND_ASSET.exists():
            shutil.copyfile(FRONTEND_ASSET, path)
        else:
            urllib.request.urlretrieve(LITE_URL, path)
    actual = sha256(path)
    if actual != LITE_SHA256:
        raise RuntimeError(f"{path}: 앱과 다른 모델 파일입니다 (sha256 {actual})")
    return path
