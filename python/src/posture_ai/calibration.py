"""개인 캘리브레이션: 세션 시작 때 바른 자세로 5초 앉은 동안의 특징값을 그 사람의 기준으로 삼습니다.

Kotlin `PostureCalibration.kt`와 1:1로 대응합니다(golden으로 검증).
결과(head_ratio, torso_angle)는 백엔드 user_settings의 calib_ratio, calib_angle에 대응합니다.

판정기 입력 흐름: 판정기(캘리브레이션 없이)를 돌리면서 나온 PoseFeedback을 프레임마다 add()에 넣습니다.
5초가 지나면 아래 조건을 확인합니다.
  - 귀·어깨·골반이 보인 프레임이 80% 미만      → NOT_VISIBLE (다시 앉기)
  - 머리 비율 표준편차 > 0.05 또는 몸통 > 4°    → MOVING (움직이지 말기)
  - 기준값이 절대 임계값 + Δ 임계값을 넘음(누가 봐도 이상 자세) → NOT_UPRIGHT (바르게 앉기)
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
MAX_HEAD_STD = 0.05
MAX_TORSO_STD = 4.0

PENDING, DONE, NOT_VISIBLE, MOVING, NOT_UPRIGHT = "PENDING", "DONE", "NOT_VISIBLE", "MOVING", "NOT_UPRIGHT"


@dataclass(frozen=True)
class Calibration:
    head_ratio: float
    torso_angle: float


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
        self._heads: list[float] = []
        self._torsos: list[float] = []
        self._result: Optional[CalibrationResult] = None

    def add(self, timestamp_ms: int, feedback: Feedback) -> CalibrationResult:
        if self._result is not None:
            return self._result
        if self._start is None:
            self._start = timestamp_ms
        self._total += 1
        if feedback.head_ratio is not None and feedback.torso_angle is not None and not feedback.needs_reset:
            self._heads.append(feedback.head_ratio)
            self._torsos.append(feedback.torso_angle)
        elapsed = timestamp_ms - self._start
        if elapsed < CALIBRATION_MS:
            return CalibrationResult(PENDING, elapsed / CALIBRATION_MS)
        self._result = self._finish()
        return self._result

    def _finish(self) -> CalibrationResult:
        valid = len(self._heads)
        if valid < MIN_VALID_FRAMES or valid * 100 < MIN_VALID_PERCENT * self._total:
            return CalibrationResult(NOT_VISIBLE, 1.0)
        if _std(self._heads) > MAX_HEAD_STD or _std(self._torsos) > MAX_TORSO_STD:
            return CalibrationResult(MOVING, 1.0)
        calibration = Calibration(upper_median(self._heads), upper_median(self._torsos))
        t = self.thresholds
        if (calibration.head_ratio > t.head_ratio + t.head_delta
                or calibration.torso_angle > t.lean_degrees + t.lean_delta_degrees):
            return CalibrationResult(NOT_UPRIGHT, 1.0, calibration)
        return CalibrationResult(DONE, 1.0, calibration)
