"""합성(가짜) 랜드마크 데이터 생성기.

실험 영상이 없을 때 파이프라인 전체(특징값 → 창 → 튜닝 → 학습 → 평가 → 내보내기)가
돌아가는지 확인하는 용도입니다. 여기서 나온 정확도는 실제 성능이 아닙니다.

막대 인형 모델: 골반에서 몸통 각도 θ만큼 기운 축 u를 따라 어깨, 그 위로 귀를 둡니다.
귀는 몸통 축 방향으로 a·L, 축에 수직(앞쪽)으로 r·L 만큼 떨어져 있어 head_ratio = r 이 됩니다.
출력은 landmarks.extract_video와 같은 CSV 형식이라 뒷단 코드를 그대로 씁니다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .labels import Interval
from .landmarks import (
    COLUMNS,
    LEFT_EAR,
    LEFT_HIP,
    LEFT_SHOULDER,
    NOSE,
    RIGHT_EAR,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    empty_row,
)

WIDTH, HEIGHT = 480, 640  # 세로로 세운 폰(앱의 4:3 분석 해상도를 회전한 크기)

# 데이터 수집용 1회차 대본: 네 자세 × 20초 (발표 3-2). 3회 반복하면 1인 4분.
COLLECTION_SCRIPT: list[tuple[str, float]] = [("NORMAL", 20), ("FHP", 20), ("LEAN", 20), ("BOTH", 20)]
# 앱 테스트용 150초 공통 시나리오 (발표 '테스트 진행' 03)
SCENARIO_150S: list[tuple[str, float]] = [
    ("NORMAL", 20), ("FHP", 20), ("NORMAL", 15), ("LEAN", 20),
    ("NORMAL", 15), ("BOTH", 20), ("AWAY", 15), ("NORMAL", 25),
]


@dataclass(frozen=True)
class Subject:
    """피험자 한 명의 체형·습관·촬영 조건."""

    name: str
    torso_px: float = 200.0       # 어깨–골반 길이(픽셀)
    neck_up: float = 0.42         # 몸통 축 방향 귀 높이(×L)
    head_neutral: float = 0.06    # 바른 자세에서도 있는 귀의 앞쪽 오프셋(×L) — 개인차
    torso_neutral: float = 4.0    # 바른 자세 몸통 각도(도) — 개인차(의자·습관)
    head_shift: float = 0.30      # 전방머리일 때 늘어나는 오프셋(×L)
    lean_shift: float = 24.0      # 상체 기울임일 때 늘어나는 각도(도)
    facing: int = 1               # +1: 이미지 오른쪽을 봄
    roll_deg: float = 0.0         # 폰 거치 기울기
    noise_px: float = 1.5
    dropout: float = 0.02         # 한 프레임에서 랜드마크가 흐려질 확률
    sway: float = 0.6             # 자세 안에서의 흔들림 크기 배율
    pitch_neutral: float = 22.0   # 바른 자세 고개 숙임 각도(귀→코, 수평 아래로, 도) — 개인차
    bow_shift: float = 28.0       # 거북목(머리 전방)일 때 더 숙이는 각도(도)


def random_subject(name: str, rng: np.random.Generator, *, roll: bool = True) -> Subject:
    return Subject(
        name=name,
        torso_px=float(rng.uniform(165, 235)),
        neck_up=float(rng.uniform(0.38, 0.46)),
        head_neutral=float(rng.normal(0.06, 0.09)),
        torso_neutral=float(rng.normal(4.0, 7.0)),
        head_shift=float(rng.uniform(0.12, 0.38)),
        lean_shift=float(rng.uniform(10.0, 30.0)),
        facing=int(rng.choice([-1, 1])),
        roll_deg=float(rng.normal(0.0, 3.0)) if roll else 0.0,
        noise_px=float(rng.uniform(1.0, 2.5)),
        dropout=float(rng.uniform(0.0, 0.04)),
        pitch_neutral=float(rng.normal(22.0, 5.0)),
        bow_shift=float(rng.uniform(15.0, 35.0)),
    )


def script_intervals(script: list[tuple[str, float]], repeats: int = 1) -> list[Interval]:
    intervals: list[Interval] = []
    t = 0.0
    for _ in range(repeats):
        for combo, seconds in script:
            intervals.append(Interval(
                t, t + seconds,
                head_forward=combo in ("FHP", "BOTH"),
                torso_lean=combo in ("LEAN", "BOTH"),
                away=combo == "AWAY",
            ))
            t += seconds
    return intervals


def _targets(subject: Subject, interval: Interval) -> tuple[float, float, float]:
    head = subject.head_neutral + (subject.head_shift if interval.head_forward else 0.0)
    torso = subject.torso_neutral + (subject.lean_shift if interval.torso_lean else 0.0)
    pitch = subject.pitch_neutral + (subject.bow_shift if interval.head_forward else 0.0)
    return head, torso, pitch


def generate(
    subject: Subject,
    intervals: list[Interval],
    *,
    fps: float = 5.0,
    seed: int = 0,
    transition_s: float = 0.8,
) -> pd.DataFrame:
    """라벨 구간대로 움직이는 합성 랜드마크 CSV(DataFrame)를 만듭니다."""
    rng = np.random.default_rng(seed)
    end_s = max(i.end_s for i in intervals)
    rows: list[dict] = []
    last: Optional[tuple[float, ...]] = None   # 직전 프레임의 (head, torso, pitch), 흔들림 제외
    start: Optional[tuple[float, ...]] = None  # 현재 구간이 시작될 때의 값
    current_index = -1
    change_at = 0.0
    hip_x = WIDTH / 2 - subject.facing * 30.0
    hip_y = HEIGHT * 0.80
    phase = rng.uniform(0, 2 * math.pi, size=2)
    for k in range(int(round(end_s * fps))):
        t_s = k / fps
        t_ms = int(round(t_s * 1000))
        index = next(n for n, i in enumerate(intervals) if i.start_s <= t_s < i.end_s)
        interval = intervals[index]
        if index != current_index:
            current_index, change_at, start = index, t_s, last
        if interval.away:
            rows.append(empty_row(t_ms, WIDTH, HEIGHT, subject.roll_deg))
            last = None
            continue
        target = _targets(subject, interval)
        if start is not None and t_s - change_at < transition_s:  # 부드러운 전환
            w = (t_s - change_at) / transition_s
            base = tuple(a + (b - a) * w for a, b in zip(start, target))
        else:
            base = target
        last = base
        head = base[0] + subject.sway * 0.03 * math.sin(2 * math.pi * 0.13 * t_s + phase[0])
        torso = base[1] + subject.sway * 2.5 * math.sin(2 * math.pi * 0.09 * t_s + phase[1])
        pitch = base[2] + subject.sway * 3.0 * math.sin(2 * math.pi * 0.11 * t_s + phase[0])
        rows.append(_frame(subject, t_ms, head, torso, hip_x, hip_y, rng, pitch))
    return pd.DataFrame(rows, columns=COLUMNS)


def _frame(subject: Subject, t_ms: int, head: float, torso: float, hip_x: float, hip_y: float,
           rng: np.random.Generator, pitch: float = 22.0) -> dict:
    L = subject.torso_px
    theta = math.radians(torso)
    u = (math.sin(theta), math.cos(theta))           # 몸통 축 (앞, 위)
    n = (math.cos(theta), -math.sin(theta))          # 몸통에 수직, 앞쪽
    hip = (0.0, 0.0)
    shoulder = (hip[0] + L * u[0], hip[1] + L * u[1])
    ear = (shoulder[0] + subject.neck_up * L * u[0] + head * L * n[0],
           shoulder[1] + subject.neck_up * L * u[1] + head * L * n[1])
    p = math.radians(pitch)                         # 귀→코 선이 수평 아래로 숙인 각도
    nose = (ear[0] + 0.20 * L * math.cos(p), ear[1] - 0.20 * L * math.sin(p))

    def to_image(p: tuple[float, float], depth_dx: float = 0.0) -> tuple[float, float]:
        x = hip_x + subject.facing * (p[0] + depth_dx)
        y = hip_y - p[1]
        # 폰이 roll만큼 기울면 화면 속 좌표는 실제 좌표를 -roll 만큼 돌린 것과 같습니다(중심 기준).
        r = math.radians(-subject.roll_deg)
        cx, cy = WIDTH / 2, HEIGHT / 2
        dx, dy = x - cx, y - cy
        x, y = cx + dx * math.cos(r) - dy * math.sin(r), cy + dx * math.sin(r) + dy * math.cos(r)
        return x + rng.normal(0, subject.noise_px), y + rng.normal(0, subject.noise_px)

    near_conf = lambda: float(np.clip(rng.normal(0.93, 0.04), 0.0, 1.0))  # noqa: E731
    far_conf = lambda: float(np.clip(rng.normal(0.55, 0.12), 0.0, 1.0))   # noqa: E731
    # 사람의 오른쪽(facing=+1이면)이 카메라를 향함 — features.facing_vote의 규칙과 같습니다.
    near_is_right = subject.facing > 0
    blurred = rng.random() < subject.dropout

    row = empty_row(t_ms, WIDTH, HEIGHT, subject.roll_deg)
    row["person"] = 1
    points = {
        NOSE: (to_image(nose), near_conf()),
        (RIGHT_EAR if near_is_right else LEFT_EAR): (to_image(ear), near_conf()),
        (RIGHT_SHOULDER if near_is_right else LEFT_SHOULDER): (to_image(shoulder), near_conf()),
        (RIGHT_HIP if near_is_right else LEFT_HIP): (to_image(hip), near_conf()),
        (LEFT_EAR if near_is_right else RIGHT_EAR): (to_image(ear, -0.05 * L), far_conf()),
        (LEFT_SHOULDER if near_is_right else RIGHT_SHOULDER): (to_image(shoulder, -0.04 * L), far_conf()),
        (LEFT_HIP if near_is_right else RIGHT_HIP): (to_image(hip, -0.03 * L), far_conf()),
    }
    for index, ((x, y), conf) in points.items():
        if blurred and index != NOSE:
            conf = float(rng.uniform(0.1, 0.45))
        row.update({f"x{index}": x / WIDTH, f"y{index}": y / HEIGHT, f"z{index}": 0.0,
                    f"vis{index}": conf, f"pres{index}": min(1.0, conf + 0.03)})
    return row


def team_dataset(seed: int = 7, subjects: int = 3, repeats: int = 3) -> list[tuple[str, Subject, pd.DataFrame, list[Interval]]]:
    """팀원 3명이 수집 대본을 3회 촬영한 상황을 흉내 냅니다. (recording, subject, landmarks, labels)"""
    rng = np.random.default_rng(seed)
    out = []
    for s in range(subjects):
        subject = random_subject(f"S{s + 1}", rng)
        for rep in range(repeats):
            intervals = script_intervals(COLLECTION_SCRIPT)
            frame = generate(subject, intervals, seed=seed * 100 + s * 10 + rep)
            out.append((f"{subject.name}_rep{rep + 1}", subject, frame, intervals))
    return out
