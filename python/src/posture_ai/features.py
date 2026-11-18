"""프레임 하나에서 특징값을 계산하는 순수 함수들.

Kotlin `PostureFeatures.kt`와 1:1로 대응합니다. 계산 순서를 바꾸면 두 언어의 결과가
달라질 수 있으니, 수정할 때는 양쪽을 함께 고치고 golden 파일을 다시 만드세요.
정의는 docs/feature-spec.md에 있습니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

MIN_CONFIDENCE = 0.5
"""visibility·presence 중 작은 값이 이보다 낮은 점은 결측으로 봅니다."""

NOSE_MIN_OFFSET = 0.05
"""코–귀 가로 거리가 어깨–골반 길이의 5%보다 작으면 방향 투표에 쓰지 않습니다."""

SIDE_VOTE_WEIGHT = 0.25
"""코가 없을 때 '보이는 쪽'으로 추정한 방향의 투표 가중치(코 투표는 1)."""


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    confidence: float = 1.0


@dataclass(frozen=True)
class BodySide:
    ear: Point
    shoulder: Point
    hip: Point

    @property
    def confidence(self) -> float:
        return min(self.ear.confidence, self.shoulder.confidence, self.hip.confidence)

    @property
    def mean_confidence(self) -> float:
        return (self.ear.confidence + self.shoulder.confidence + self.hip.confidence) / 3

    @property
    def valid(self) -> bool:
        return all(math.isfinite(p.x) and math.isfinite(p.y) for p in (self.ear, self.shoulder, self.hip))


@dataclass(frozen=True)
class Features:
    head_ratio: float
    """(귀−어깨)의 몸통 축 수직 성분 ÷ 어깨–골반 길이. 앞쪽이 +."""
    torso_angle: float
    """골반→어깨 벡터와 수직선 사이 각도(도). 앞으로 숙이면 +."""
    neck_angle: float
    """어깨→귀 벡터 각도 − 몸통 각도(도). 몸통 기준 목 꺾임. 앞쪽이 +."""
    length: float
    """어깨–골반 길이(픽셀)."""
    axis_angle: float
    """골반→귀 벡터와 수직선 사이 각도(도). 앞으로 기울면 +. **판정에 쓰는 값**(2026-11-18 팀 결정).
    머리를 내밀어도, 허리를 숙여도 이 선이 앞으로 기울어서 둘 다 거북목으로 잡혀요."""


def select_side(left: Optional[BodySide], right: Optional[BodySide]) -> tuple[Optional[BodySide], int]:
    """신뢰도 평균이 높은 쪽을 고릅니다. 같으면 왼쪽. 반환: (몸, -1=왼쪽 / +1=오른쪽 / 0=없음)."""
    best: Optional[BodySide] = None
    best_side = 0
    for side, code in ((left, -1), (right, 1)):
        if side is None or not side.valid or side.confidence < MIN_CONFIDENCE:
            continue
        if best is None or side.mean_confidence > best.mean_confidence:
            best, best_side = side, code
    return best, best_side


def rotate(p: Point, degrees: float) -> Point:
    """원점 기준 회전. 이미지 좌표(y 아래)에서 중력 방향이 (sin r, cos r)이면 r만큼 돌려 수직을 맞춥니다."""
    if degrees == 0.0:
        return p
    r = math.radians(degrees)
    c, s = math.cos(r), math.sin(r)
    return Point(p.x * c - p.y * s, p.x * s + p.y * c, p.confidence)


def rotate_body(body: BodySide, degrees: float) -> BodySide:
    return BodySide(rotate(body.ear, degrees), rotate(body.shoulder, degrees), rotate(body.hip, degrees))


def facing_vote(body: BodySide, side_code: int, nose: Optional[Point]) -> tuple[float, float]:
    """이번 프레임의 방향 투표. 반환: (코 투표, 보이는 쪽 투표). 값은 -1·0·+1에 가중치를 곱한 것.

    코가 귀보다 이미지 오른쪽에 있으면 +x가 앞쪽입니다.
    코가 없으면: 사람의 왼쪽 몸이 카메라를 향하면(왼쪽 선택) 앞쪽은 이미지 왼쪽(-x)입니다.
    (앱의 분석 프레임은 전면 카메라라도 좌우 반전되지 않은 원본입니다.)
    """
    length = math.hypot(body.shoulder.x - body.hip.x, body.shoulder.y - body.hip.y)
    nose_vote = 0.0
    if nose is not None and nose.confidence >= MIN_CONFIDENCE and math.isfinite(nose.x):
        dx = nose.x - body.ear.x
        if abs(dx) >= NOSE_MIN_OFFSET * length:
            nose_vote = 1.0 if dx > 0 else -1.0
    side_vote = float(side_code) * SIDE_VOTE_WEIGHT  # 왼쪽(-1) → -x가 앞
    return nose_vote, side_vote


def compute_features(body: BodySide, facing: int) -> Optional[Features]:
    """facing: +1이면 이미지 +x가 앞쪽, -1이면 -x가 앞쪽. 길이가 0에 가까우면 None."""
    # 앞쪽 = +fx, 위쪽 = +up 인 좌표계로 바꿉니다.
    tx = facing * (body.shoulder.x - body.hip.x)
    tu = body.hip.y - body.shoulder.y
    length = math.hypot(tx, tu)
    if not math.isfinite(length) or length < 1e-3:
        return None
    torso_angle = math.degrees(math.atan2(tx, tu))
    hx = facing * (body.ear.x - body.shoulder.x)
    hu = body.shoulder.y - body.ear.y
    # 몸통 축 u=(tx,tu)/L 에 수직이면서 앞쪽을 향하는 단위벡터 n=(tu,-tx)/L
    head_ratio = (hx * tu - hu * tx) / (length * length)
    neck_angle = wrap_degrees(math.degrees(math.atan2(hx, hu)) - torso_angle)
    axis_angle = math.degrees(math.atan2(facing * (body.ear.x - body.hip.x), body.hip.y - body.ear.y))
    return Features(head_ratio, torso_angle, neck_angle, length, axis_angle)


def head_pitch(ear: Point, nose: Optional[Point], facing: int) -> Optional[float]:
    """귀→코 선이 수평에서 아래로 숙인 각도(도). 고개를 숙이면 커져요. 코가 없거나 신뢰도가 낮으면 None.

    골반→귀 기울기로는 '허리는 펴고 고개만 숙인 자세'가 잘 안 잡혀서 함께 봐요(2026-11-18 팀 결정).
    """
    if nose is None or nose.confidence < MIN_CONFIDENCE or not (math.isfinite(nose.x) and math.isfinite(nose.y)):
        return None
    forward = facing * (nose.x - ear.x)
    down = nose.y - ear.y
    if math.hypot(forward, down) < 1e-3:
        return None
    return math.degrees(math.atan2(down, forward))


def wrap_degrees(value: float) -> float:
    """(-180, 180] 범위로 맞춥니다."""
    while value > 180.0:
        value -= 360.0
    while value <= -180.0:
        value += 360.0
    return value


def upper_median(values: list[float]) -> float:
    """정렬 후 size//2 번째 값. 짝수 개면 위쪽 중앙값(Kotlin과 동일)."""
    ordered = sorted(values)
    return ordered[len(ordered) // 2]
