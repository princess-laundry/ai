"""실시간 자세 판정 상태 머신. Kotlin `PostureEstimator.kt`와 1:1로 대응합니다.

앱(5fps)과 PC(녹화 영상을 5fps로 추출)가 같은 코드 흐름을 쓰도록 프레임 단위로 동작합니다.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from .features import (
    BodySide,
    Features,
    Point,
    compute_features,
    facing_vote,
    rotate,
    rotate_body,
    select_side,
    upper_median,
)

NORMAL, FHP, LEAN, AWAY, UNKNOWN = "NORMAL", "FHP", "LEAN", "AWAY", "UNKNOWN"

MEDIAN_FRAMES = 5
"""스무딩에 쓰는 최근 프레임 수(5fps에서 약 1초)."""
AGREEMENT_PERCENT = 70
"""2초 창 안에서 같은 판정이 이 비율(%) 이상이면 상태를 바꿉니다."""
RESET_TOLERANCE = 0.30
"""어깨–골반 길이가 기준에서 ±30% 넘게 변하면 재세팅을 안내합니다."""
BASELINE_MS = 1000
"""처음 1초 동안의 길이 중앙값을 기준 길이로 씁니다."""


@dataclass(frozen=True)
class Thresholds:
    version: int = 1
    head_ratio: float = 0.35
    lean_degrees: float = 20.0
    stable_ms: int = 2000
    analysis_fps: int = 5

    def __post_init__(self) -> None:
        if not (self.version > 0 and 0.05 <= self.head_ratio <= 2.0 and 1.0 <= self.lean_degrees <= 89.0):
            raise ValueError("invalid thresholds")
        if not (200 <= self.stable_ms <= 10_000 and 1 <= self.analysis_fps <= 10):
            raise ValueError("invalid thresholds")

    @classmethod
    def from_server(cls, config: dict) -> "Thresholds":
        """GET /api/v1/config 응답 → Thresholds."""
        return cls(
            version=int(config["version"]),
            head_ratio=float(config["t1_ratio"]),
            lean_degrees=float(config["t2_angle"]),
            stable_ms=int(round(float(config["stable_sec"]) * 1000)),
            analysis_fps=int(config["analysis_fps"]),
        )


@dataclass(frozen=True)
class Observation:
    timestamp_ms: int
    person_detected: bool
    left: Optional[BodySide] = None
    right: Optional[BodySide] = None
    nose: Optional[Point] = None
    device_roll_deg: Optional[float] = None


@dataclass(frozen=True)
class Feedback:
    state: str = UNKNOWN
    body: Optional[BodySide] = None
    head_ratio: Optional[float] = None
    torso_angle: Optional[float] = None
    neck_angle: Optional[float] = None
    needs_reset: bool = False
    head_forward: Optional[bool] = None
    torso_lean: Optional[bool] = None
    facing: int = 0

    @property
    def framed(self) -> bool:
        return self.body is not None and not self.needs_reset


@dataclass
class PostureEstimator:
    thresholds: Thresholds = field(default_factory=Thresholds)

    def __post_init__(self) -> None:
        self.current = UNKNOWN
        self.missing_since: Optional[int] = None
        self.baseline_length: Optional[float] = None
        self._baseline_start: Optional[int] = None
        self._baseline_samples: list[float] = []
        self._ratios: deque[float] = deque()
        self._angles: deque[float] = deque()
        self._necks: deque[float] = deque()
        self._lengths: deque[float] = deque()
        self._window: deque[tuple[int, bool, bool]] = deque()
        self._window_start: Optional[int] = None
        self._head: Optional[bool] = None
        self._lean: Optional[bool] = None
        self._nose_votes = 0.0
        self._side_votes = 0.0

    # ------------------------------------------------------------------ 내부 도우미
    def _clear_tracking(self) -> None:
        for buffer in (self._ratios, self._angles, self._necks, self._lengths):
            buffer.clear()
        self._window.clear()
        self._window_start = None
        self._head = None
        self._lean = None

    @staticmethod
    def _smooth(buffer: deque[float], value: float) -> float:
        buffer.append(value)
        if len(buffer) > MEDIAN_FRAMES:
            buffer.popleft()
        return upper_median(list(buffer))

    @property
    def facing(self) -> int:
        if self._nose_votes != 0.0:
            return 1 if self._nose_votes > 0 else -1
        if self._side_votes != 0.0:
            return 1 if self._side_votes > 0 else -1
        return 1

    @staticmethod
    def _decide(previous: Optional[bool], count: int, total: int) -> Optional[bool]:
        if previous is None:
            return count * 2 >= total
        if count * 100 >= AGREEMENT_PERCENT * total:
            return True
        if (total - count) * 100 >= AGREEMENT_PERCENT * total:
            return False
        return previous

    # ------------------------------------------------------------------ 공개 API
    def update(self, obs: Observation) -> Feedback:
        t = self.thresholds
        body, side_code = select_side(obs.left, obs.right)
        if body is None:
            since = self.missing_since
            if since is None:
                since = self.missing_since = obs.timestamp_ms
            if obs.timestamp_ms - since >= t.stable_ms:
                self.current = UNKNOWN if obs.person_detected else AWAY
                self._clear_tracking()
            return Feedback(state=self.current)
        self.missing_since = None

        roll = obs.device_roll_deg if obs.device_roll_deg is not None else 0.0
        upright = rotate_body(body, roll)
        nose = rotate(obs.nose, roll) if obs.nose is not None else None
        nose_vote, side_vote = facing_vote(upright, side_code, nose)
        self._nose_votes += nose_vote
        self._side_votes += side_vote
        facing = self.facing

        raw: Optional[Features] = compute_features(upright, facing)
        if raw is None:
            return Feedback(state=UNKNOWN, facing=facing)

        ratio = self._smooth(self._ratios, raw.head_ratio)
        angle = self._smooth(self._angles, raw.torso_angle)
        neck = self._smooth(self._necks, raw.neck_angle)
        length = self._smooth(self._lengths, raw.length)

        if self.baseline_length is None:
            if self._baseline_start is None:
                self._baseline_start = obs.timestamp_ms
            self._baseline_samples.append(raw.length)
            if obs.timestamp_ms - self._baseline_start >= BASELINE_MS:
                self.baseline_length = upper_median(self._baseline_samples)
        elif abs(length / self.baseline_length - 1.0) > RESET_TOLERANCE:
            self.current = UNKNOWN
            self._clear_tracking()
            return Feedback(state=UNKNOWN, body=body, needs_reset=True, facing=facing)

        head = ratio > t.head_ratio
        lean = angle > t.lean_degrees
        if self._window_start is None:
            self._window_start = obs.timestamp_ms
        self._window.append((obs.timestamp_ms, head, lean))
        while self._window and self._window[0][0] <= obs.timestamp_ms - t.stable_ms:
            self._window.popleft()

        if obs.timestamp_ms - self._window_start >= t.stable_ms and self._window:
            total = len(self._window)
            self._head = self._decide(self._head, sum(1 for e in self._window if e[1]), total)
            self._lean = self._decide(self._lean, sum(1 for e in self._window if e[2]), total)
            self.current = FHP if self._head else LEAN if self._lean else NORMAL

        return Feedback(
            state=self.current,
            body=body,
            head_ratio=ratio,
            torso_angle=angle,
            neck_angle=neck,
            head_forward=self._head,
            torso_lean=self._lean,
            facing=facing,
        )
