"""라벨·임계값 검수용: 영상 위에 귀·어깨·골반 스켈레톤, 특징값, 판정 상태, 정답 라벨을 겹쳐 그린 mp4를 만듭니다."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .estimator import PostureEstimator, Thresholds
from .labels import Interval, label_at
from .landmarks import observations

COLORS = {"NORMAL": (120, 220, 110), "FHP": (60, 170, 255), "LEAN": (60, 170, 255), "AWAY": (160, 160, 160), "UNKNOWN": (200, 200, 200)}


def render(video: Path, landmarks: pd.DataFrame, out: Path, thresholds: Optional[Thresholds] = None,
           labels: Optional[list[Interval]] = None, rotate_degrees: int = 0) -> Path:
    import cv2

    estimator = PostureEstimator(thresholds or Thresholds())
    timeline = []
    for obs in observations(landmarks):
        timeline.append((obs.timestamp_ms, estimator.update(obs)))
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    writer = None
    k = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if rotate_degrees in rotations:
            frame = cv2.rotate(frame, rotations[rotate_degrees])
        t_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        while k + 1 < len(timeline) and timeline[k + 1][0] <= t_ms:
            k += 1
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if timeline:
            _draw(cv2, frame, timeline[k][1], t_ms / 1000.0, labels)
        writer.write(frame)
    capture.release()
    if writer is not None:
        writer.release()
    return out


def _draw(cv2, frame, fb, t_s: float, labels) -> None:
    color = COLORS.get(fb.state, (255, 255, 255))
    if fb.body is not None:
        pts = [(int(p.x), int(p.y)) for p in (fb.body.ear, fb.body.shoulder, fb.body.hip)]
        cv2.line(frame, pts[0], pts[1], color, 4)
        cv2.line(frame, pts[1], pts[2], color, 4)
        for p in pts:
            cv2.circle(frame, p, 7, color, -1)
    lines = [f"t={t_s:6.1f}s  state={fb.state}{'  RESET' if fb.needs_reset else ''}"]
    if fb.head_ratio is not None:
        lines.append(f"axis={fb.axis_angle:+.1f}  head={fb.head_ratio:+.3f}  torso={fb.torso_angle:+.1f}  facing={fb.facing:+d}")
    if labels is not None:
        truth = label_at(labels, t_s, 0.0)
        lines.append(f"label={truth.combo if truth else '-'}{' (exclude)' if truth and truth.exclude else ''}")
    for i, text in enumerate(lines):
        y = 28 + 26 * i
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
