"""정답 라벨 형식. 판정 규칙과 독립적으로, 사람이 영상을 보고 두 축을 기록합니다(docs/labeling-guide.md).

라벨 CSV (영상 1개당 1개):
  start_s,end_s,head_forward,torso_lean,away,exclude
  - head_forward / torso_lean: 0 또는 1 (동시 발생은 둘 다 1)
  - away: 1이면 자리 비움(두 속성 무시)
  - exclude: 1이면 모호한 구간 → 평가·학습에서 제외
구간 사이 전환 지점 ±1초는 windows.py에서 자동으로 제외합니다.

피험자 메타 subjects.csv:
  recording,subject,side,location,labeler
  - recording: 영상/CSV 파일 이름(확장자 제외)
  - subject: 피험자 ID(분할 기준). side: left/right(카메라를 향한 몸 쪽)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

LABEL_COLUMNS = ["start_s", "end_s", "head_forward", "torso_lean", "away", "exclude"]
TRANSITION_MARGIN_S = 1.0


@dataclass(frozen=True)
class Interval:
    start_s: float
    end_s: float
    head_forward: bool
    torso_lean: bool
    away: bool = False
    exclude: bool = False

    @property
    def posture(self) -> str:
        """서버 단일 상태로 바꾼 값(둘 다면 FHP 우선)."""
        if self.away:
            return "AWAY"
        if self.head_forward:
            return "FHP"
        if self.torso_lean:
            return "LEAN"
        return "NORMAL"

    @property
    def combo(self) -> str:
        """네 자세 구분(정상/전방머리/상체 기울임/동시)."""
        if self.away:
            return "AWAY"
        return {(False, False): "NORMAL", (True, False): "FHP", (False, True): "LEAN", (True, True): "BOTH"}[
            (self.head_forward, self.torso_lean)
        ]


def read_labels(path: str | Path) -> list[Interval]:
    frame = pd.read_csv(path)
    missing = [c for c in LABEL_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"{path}: 라벨 열이 없습니다: {missing}")
    intervals = [
        Interval(float(r.start_s), float(r.end_s), bool(r.head_forward), bool(r.torso_lean), bool(r.away), bool(r.exclude))
        for r in frame.itertuples()
    ]
    validate(intervals, str(path))
    return intervals


def write_labels(intervals: list[Interval], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = [
        [i.start_s, i.end_s, int(i.head_forward), int(i.torso_lean), int(i.away), int(i.exclude)] for i in intervals
    ]
    pd.DataFrame(rows, columns=LABEL_COLUMNS).to_csv(path, index=False, float_format="%.3f")


def validate(intervals: list[Interval], name: str = "labels") -> None:
    previous_end = -1.0
    for i in sorted(intervals, key=lambda x: x.start_s):
        if i.end_s <= i.start_s:
            raise ValueError(f"{name}: 끝 시각이 시작보다 빠릅니다 ({i.start_s}~{i.end_s})")
        if i.start_s < previous_end - 1e-9:
            raise ValueError(f"{name}: 구간이 겹칩니다 ({i.start_s})")
        previous_end = i.end_s


def label_at(intervals: list[Interval], t_s: float, margin_s: float = TRANSITION_MARGIN_S) -> Optional[Interval]:
    """시각 t의 라벨. 라벨 없음·제외 구간·구간 경계 ±margin 이내면 None."""
    for i in intervals:
        if i.start_s <= t_s < i.end_s:
            if i.exclude:
                return None
            boundaries = [b for b in _boundaries(intervals) if abs(b - t_s) < margin_s]
            return None if boundaries else i
    return None


def _boundaries(intervals: list[Interval]) -> list[float]:
    """상태가 실제로 바뀌는 지점만 경계로 봅니다(같은 자세가 이어지면 경계 아님)."""
    ordered = sorted(intervals, key=lambda x: x.start_s)
    points: list[float] = []
    for a, b in zip(ordered, ordered[1:]):
        if (a.combo, a.exclude) != (b.combo, b.exclude) or b.start_s - a.end_s > 1e-9:
            points.extend({a.end_s, b.start_s})
    return points


def read_subjects(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    for column in ("recording", "subject"):
        if column not in frame.columns:
            raise ValueError(f"{path}: '{column}' 열이 필요합니다")
    return frame
