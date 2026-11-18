"""랜드마크 CSV + 라벨 → 프레임별 특징값 → 2초 창 샘플.

- 특징값은 앱과 같은 판정기(PostureEstimator)를 돌려 얻습니다. 앱이 보는 스무딩된 값 그대로입니다.
- 샘플 = 비중첩 2초 창(5fps면 10프레임)의 특징값 평균·표준편차.
- 전환 ±1초, 제외 구간, 자리 비움, 라벨이 섞인 창, 특징값이 20% 넘게 빠진 창은 버립니다.
- Δ특징: 녹화 시작 후 첫 바른 자세 구간에서 5초 동안의 중앙값(=앱의 5초 캘리브레이션)을 뺀 값.
  자세별로 나눠 찍어서 그 녹화에 바른 자세 구간이 없으면, 같은 피험자의 다른 녹화에서 구한 기준을 씁니다.
  기준을 구한 5초는 평가 창에서 뺍니다(기준과 같은 구간이라 차이가 0에 가깝게 나와서).
- 정답 `fhp`: 머리 전방 또는 상체 기울임이면 1(거북목), 아니면 0(바른 자세). 2026-11-18 팀 결정.
- 첫 라벨 앞(손뼉·자리 잡기)과 마지막 라벨 뒤 프레임은 버리고 판정기를 돌립니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .estimator import PostureEstimator, Thresholds
from .labels import Interval, TRANSITION_MARGIN_S, label_at
from .landmarks import observations

FEATURES = ("head_ratio", "torso_angle", "neck_angle")
ABS_COLUMNS = ["head_mean", "head_std", "torso_mean", "torso_std", "neck_mean", "neck_std"]
DELTA_COLUMNS = ["d_head", "head_std", "d_torso", "torso_std", "d_neck", "neck_std"]
CALIBRATION_S = 5.0
MAX_MISSING = 0.2


@dataclass(frozen=True)
class Recording:
    name: str
    subject: str
    landmarks: pd.DataFrame
    labels: list[Interval]


def trim_to_labels(landmarks: pd.DataFrame, labels: list[Interval]) -> pd.DataFrame:
    """첫 라벨 시작 전과 마지막 라벨 끝 뒤 프레임을 버립니다. 앱도 세팅이 끝난 뒤부터 판정하니까요."""
    if not labels or landmarks.empty:
        return landmarks
    start_ms = min(i.start_s for i in labels) * 1000.0
    end_ms = max(i.end_s for i in labels) * 1000.0
    t = landmarks["t_ms"]
    return landmarks[(t >= start_ms) & (t < end_ms)].reset_index(drop=True)


def frame_features(landmarks: pd.DataFrame, thresholds: Optional[Thresholds] = None) -> pd.DataFrame:
    """프레임마다 판정기를 돌려 (t_ms, head_ratio, torso_angle, neck_angle, state, needs_reset)를 만듭니다."""
    estimator = PostureEstimator(thresholds or Thresholds())
    rows = []
    for obs in observations(landmarks):
        fb = estimator.update(obs)
        rows.append({
            "t_ms": obs.timestamp_ms,
            "head_ratio": fb.head_ratio if fb.head_ratio is not None else np.nan,
            "torso_angle": fb.torso_angle if fb.torso_angle is not None else np.nan,
            "neck_angle": fb.neck_angle if fb.neck_angle is not None else np.nan,
            "state": fb.state,
            "needs_reset": fb.needs_reset,
        })
    return pd.DataFrame(rows)


Baseline = tuple[dict[str, float], tuple[float, float]]


def calibration(features: pd.DataFrame, labels: list[Interval]) -> Optional[Baseline]:
    """첫 바른 자세 구간(전환 1초 이후)에서 5초 동안의 특징값 중앙값과 그 구간(초)."""
    for interval in sorted(labels, key=lambda i: i.start_s):
        if interval.combo != "NORMAL" or interval.exclude:
            continue
        start = interval.start_s + TRANSITION_MARGIN_S
        end = min(interval.end_s - TRANSITION_MARGIN_S, start + CALIBRATION_S)
        if end - start < CALIBRATION_S * 0.6:
            continue
        t = features["t_ms"] / 1000.0
        part = features[(t >= start) & (t < end)].dropna(subset=list(FEATURES))
        if len(part) >= 5:
            return {f: float(part[f].median()) for f in FEATURES}, (start, end)
    return None


def make_windows(recording: Recording, window_s: float = 2.0, features: Optional[pd.DataFrame] = None,
                 base: Optional[dict[str, float]] = None, skip: Optional[tuple[float, float]] = None) -> pd.DataFrame:
    """base를 안 주면 이 녹화 안에서 기준을 구합니다. skip 구간(기준을 구한 5초)과 겹치는 창은 버립니다."""
    feats = features if features is not None else frame_features(trim_to_labels(recording.landmarks, recording.labels))
    if base is None:
        own = calibration(feats, recording.labels)
        base, skip = own if own is not None else (None, None)
    if feats.empty:
        return pd.DataFrame()
    t = feats["t_ms"].to_numpy() / 1000.0
    end = float(t.max())
    rows = []
    k = 0
    while (k + 1) * window_s <= end + 1e-9:
        lo, hi = k * window_s, (k + 1) * window_s
        k += 1
        if lo < t[0] - 1e-9:  # 앞부분을 잘라 내서 창이 반만 찬 경우
            continue
        if skip is not None and lo < skip[1] and hi > skip[0]:
            continue
        part = feats[(t >= lo) & (t < hi)]
        if part.empty:
            continue
        labs = [label_at(recording.labels, float(x) / 1000.0) for x in part["t_ms"]]
        if any(l is None for l in labs) or len({l.combo for l in labs}) != 1 or labs[0].away:
            continue
        valid = part.dropna(subset=list(FEATURES))
        if len(valid) < (1 - MAX_MISSING) * len(part) or len(valid) < 3:
            continue
        lab = labs[0]
        row = {
            "recording": recording.name,
            "subject": recording.subject,
            "t_start": lo,
            "head_forward": int(lab.head_forward),
            "torso_lean": int(lab.torso_lean),
            "combo": lab.combo,
            "posture": lab.posture,
            "fhp": int(lab.posture == "FHP"),
            "head_mean": valid["head_ratio"].mean(), "head_std": valid["head_ratio"].std(ddof=0),
            "torso_mean": valid["torso_angle"].mean(), "torso_std": valid["torso_angle"].std(ddof=0),
            "neck_mean": valid["neck_angle"].mean(), "neck_std": valid["neck_angle"].std(ddof=0),
        }
        if base is not None:
            row.update(d_head=row["head_mean"] - base["head_ratio"],
                       d_torso=row["torso_mean"] - base["torso_angle"],
                       d_neck=row["neck_mean"] - base["neck_angle"])
        else:
            row.update(d_head=np.nan, d_torso=np.nan, d_neck=np.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def build_samples(recordings: Iterable[Recording]) -> pd.DataFrame:
    recordings = list(recordings)
    feats = {r.name: frame_features(trim_to_labels(r.landmarks, r.labels)) for r in recordings}
    own = {r.name: calibration(feats[r.name], r.labels) for r in recordings}
    subject_base: dict[str, dict[str, float]] = {}
    for r in recordings:
        if own[r.name] is not None:
            subject_base.setdefault(r.subject, own[r.name][0])
    frames = []
    for r in recordings:
        base, skip = own[r.name] if own[r.name] is not None else (subject_base.get(r.subject), None)
        frames.append(make_windows(r, features=feats[r.name], base=base, skip=skip))
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summary(samples: pd.DataFrame) -> pd.DataFrame:
    """피험자별·자세별 유효 창 수(발표 3-2의 '실제 유효 수량과 자세별 분포')."""
    return samples.pivot_table(index="subject", columns="posture", values="t_start", aggfunc="count", fill_value=0)


def quality(recordings: Iterable[Recording]) -> pd.DataFrame:
    """녹화별 데이터 품질: 판정에 쓸 수 있었던 프레임 비율(관절이 보이고 재세팅이 아닌 프레임)."""
    rows = []
    for r in recordings:
        feats = frame_features(trim_to_labels(r.landmarks, r.labels))
        ok = feats.dropna(subset=list(FEATURES))
        ok = ok[~ok["needs_reset"]]
        rows.append({"recording": r.name, "subject": r.subject, "frames": len(feats),
                     "usable": len(ok) / len(feats) if len(feats) else float("nan")})
    return pd.DataFrame(rows)
