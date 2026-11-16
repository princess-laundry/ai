"""녹화 영상 → 5fps 프레임별 MediaPipe 랜드마크 CSV, 그리고 CSV → 판정기 입력 변환.

CSV 한 행 = 분석 프레임 하나.
  t_ms, width, height, person, roll_deg, 그리고 랜드마크마다 x{i}, y{i}, z{i}, vis{i}, pres{i}
x·y는 MediaPipe 정규화 좌표(0~1)이고, 판정기에 넣을 때 픽셀 좌표로 바꿉니다(앱과 동일).
roll_deg는 영상에는 없으므로 비워 둡니다(합성 데이터만 채움).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd

from .estimator import Observation
from .features import BodySide, Point

NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP = 0, 7, 8, 11, 12, 23, 24
LANDMARK_IDS = (NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP)
FIELDS = ("x", "y", "z", "vis", "pres")
COLUMNS = ["t_ms", "width", "height", "person", "roll_deg"] + [f"{f}{i}" for i in LANDMARK_IDS for f in FIELDS]


def _point(row: pd.Series, index: int, width: float, height: float) -> Point:
    x, y = row[f"x{index}"], row[f"y{index}"]
    vis, pres = row[f"vis{index}"], row[f"pres{index}"]
    if not (np.isfinite(x) and np.isfinite(y)):
        return Point(math.nan, math.nan, 0.0)
    inside = 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    confidence = float(min(vis, pres)) if inside and np.isfinite(vis) and np.isfinite(pres) else 0.0
    return Point(float(x) * width, float(y) * height, confidence)


def row_to_observation(row: pd.Series) -> Observation:
    """앱 CameraSurface.kt와 같은 규칙으로 변환합니다(정규화 좌표 × 이미지 크기, 신뢰도 = min(vis, pres))."""
    width, height = float(row["width"]), float(row["height"])
    person = bool(row["person"])
    roll = row.get("roll_deg", math.nan)
    roll = float(roll) if roll is not None and np.isfinite(roll) else None
    if not person:
        return Observation(int(row["t_ms"]), False, device_roll_deg=roll)

    def side(ear: int, shoulder: int, hip: int) -> BodySide:
        return BodySide(_point(row, ear, width, height), _point(row, shoulder, width, height), _point(row, hip, width, height))

    return Observation(
        timestamp_ms=int(row["t_ms"]),
        person_detected=True,
        left=side(LEFT_EAR, LEFT_SHOULDER, LEFT_HIP),
        right=side(RIGHT_EAR, RIGHT_SHOULDER, RIGHT_HIP),
        nose=_point(row, NOSE, width, height),
        device_roll_deg=roll,
    )


def observations(frame: pd.DataFrame) -> Iterator[Observation]:
    for _, row in frame.iterrows():
        yield row_to_observation(row)


def read_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = [c for c in COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: 랜드마크 CSV 열이 없습니다: {missing[:5]}")
    return frame[COLUMNS]


def empty_row(t_ms: int, width: int, height: int, roll_deg: Optional[float] = None) -> dict:
    row: dict = {c: math.nan for c in COLUMNS}
    row.update(t_ms=t_ms, width=width, height=height, person=0, roll_deg=math.nan if roll_deg is None else roll_deg)
    return row


def extract_video(
    video: str | Path,
    model: str | Path,
    fps: float = 5.0,
    rotate_degrees: int = 0,
) -> pd.DataFrame:
    """영상에서 fps 간격으로 프레임을 골라 MediaPipe PoseLandmarker(VIDEO 모드)를 돌립니다.

    앱과 같은 설정: numPoses=1, 검출·존재·추적 신뢰도 0.5. 영상이 옆으로 누워 있으면 rotate_degrees(90/180/270)를 주세요.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise FileNotFoundError(f"영상을 열 수 없습니다: {video}")
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    interval_ms = 1000.0 / fps
    next_ms = 0.0
    rows: list[dict] = []
    with PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if t_ms + 1e-6 < next_ms:
                continue
            next_ms += interval_ms * max(1, math.floor((t_ms - next_ms) / interval_ms) + 1)
            if rotate_degrees in rotations:
                frame = cv2.rotate(frame, rotations[rotate_degrees])
            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(round(t_ms)))
            row = empty_row(int(round(t_ms)), width, height)
            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                row["person"] = 1
                for i in LANDMARK_IDS:
                    p = landmarks[i]
                    row.update({f"x{i}": p.x, f"y{i}": p.y, f"z{i}": p.z,
                                f"vis{i}": p.visibility if p.visibility is not None else 0.0,
                                f"pres{i}": p.presence if p.presence is not None else 0.0})
            rows.append(row)
    capture.release()
    return pd.DataFrame(rows, columns=COLUMNS)


def write_csv(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.6f")


def concat(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(frames), ignore_index=True)
