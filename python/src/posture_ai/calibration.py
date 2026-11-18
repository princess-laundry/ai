"""개인 캘리브레이션: 세션 시작 때 바른 자세로 5초 앉은 동안의 특징값을 그 사람의 기준으로 삼습니다.

Kotlin `PostureCalibration.kt`와 1:1로 대응합니다(golden으로 검증).
결과(axis_angle = 바른 자세의 골반→귀 선 기울기, head_pitch = 고개 숙임 각도)에서
axis_angle은 백엔드 user_settings.calib_angle에 대응합니다. head_pitch는 아직 저장할 칸이 없어요.
(2026-11-18 결정: 캘리브레이션은 지금 앱에서 쓰지 않아요. 코드는 다시 켤 때를 위해 남겨 둠)

판정기 입력 흐름: 판정기(캘리브레이션 없이)를 돌리면서 나온 PoseFeedback을 프레임마다 add()에 넣습니다.
5초가 지나면 아래 조건을 확인합니다.
  - 귀·어깨·골반이 보인 프레임이 80% 미만      → NOT_VISIBLE (다시 앉기)
  - 기울기 표준편차 > 4° 또는 고개 숙임 표준편차 > 8°  → MOVING (움직이지 말기)
  - 기준 기울기·고개 숙임이 절대 임계값 + Δ 임계값을 넘음(누가 봐도 이상 자세) → NOT_UPRIGHT (바르게 앉기)
    절대 임계값만으로 막으면 평소 자세가 기준선 근처인 사람은 캘리브레이션을 못 하므로 여유를 둡니다.
  - 그 외                                       → DONE (중앙값을 기준으로 저장)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .estimator import Feedback, Thresholds
from .features import upper_median

CALIBRATION_MS = 5000
MIN_VALID_PERCENT = 80
MIN_VALID_FRAMES = 5
MAX_AXIS_STD = 4.0
MAX_PITCH_STD = 8.0

PENDING, DONE, NOT_VISIBLE, MOVING, NOT_UPRIGHT = "PENDING", "DONE", "NOT_VISIBLE", "MOVING", "NOT_UPRIGHT"


@dataclass(frozen=True)
class Calibration:
    axis_angle: float
    """바른 자세일 때 골반→귀 선 기울기(도)."""
    head_pitch: Optional[float] = None
    """바른 자세일 때 고개 숙임 각도(도). 코가 충분히 안 보였으면 None(그때는 절대 임계값으로 판정)."""


@dataclass(frozen=True)
class CalibrationResult:
    status: str = PENDING
    progress: float = 0.0
    calibration: Optional[Calibration] = None

    @property
    def finished(self) -> bool:
        return self.status != PENDING


def _std(values: list[float]) -> float:
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) * (v - mean) for v in values) / len(values))


class CalibrationCollector:
    def __init__(self, thresholds: Thresholds = Thresholds()) -> None:
        self.thresholds = thresholds
        self._start: Optional[int] = None
        self._total = 0
        self._axes: list[float] = []
        self._pitches: list[float] = []
        self._result: Optional[CalibrationResult] = None

    def add(self, timestamp_ms: int, feedback: Feedback) -> CalibrationResult:
        if self._result is not None:
            return self._result
        if self._start is None:
            self._start = timestamp_ms
        self._total += 1
        if feedback.axis_angle is not None and not feedback.needs_reset:
            self._axes.append(feedback.axis_angle)
            if feedback.head_pitch is not None:
                self._pitches.append(feedback.head_pitch)
        elapsed = timestamp_ms - self._start
        if elapsed < CALIBRATION_MS:
            return CalibrationResult(PENDING, elapsed / CALIBRATION_MS)
        self._result = self._finish()
        return self._result

    def _finish(self) -> CalibrationResult:
        valid = len(self._axes)
        if valid < MIN_VALID_FRAMES or valid * 100 < MIN_VALID_PERCENT * self._total:
            return CalibrationResult(NOT_VISIBLE, 1.0)
        has_pitch = len(self._pitches) >= MIN_VALID_FRAMES
        if _std(self._axes) > MAX_AXIS_STD or (has_pitch and _std(self._pitches) > MAX_PITCH_STD):
            return CalibrationResult(MOVING, 1.0)
        calibration = Calibration(upper_median(self._axes), upper_median(self._pitches) if has_pitch else None)
        t = self.thresholds
        if (calibration.axis_angle > t.lean_degrees + t.lean_delta_degrees
                or (calibration.head_pitch is not None and calibration.head_pitch > t.pitch_degrees + t.pitch_delta_degrees)):
            return CalibrationResult(NOT_UPRIGHT, 1.0, calibration)
        return CalibrationResult(DONE, 1.0, calibration)
