"""평가 지표: 거북목/바른 자세 Macro F1, 혼동 행렬, 피험자별 점수, Cohen's κ, 앱 시뮬레이션(오탐·미탐·확정 지연).

2026-11-18 팀 결정으로 앱 상태는 바른 자세(NORMAL) / 거북목(FHP) 두 가지예요(상체 기울임도 거북목).
그래서 평가도 이 두 가지로 해요.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, recall_score

from .calibration import DONE, NOT_VISIBLE, CalibrationCollector, CalibrationResult
from .estimator import FHP, NORMAL, AWAY, PostureEstimator, Thresholds
from .labels import Interval, label_at
from .landmarks import observations
from .windows import Recording, trim_to_labels

CLASSES = ["NORMAL", "FHP"]


def binary_scores(y_true, y_pred) -> dict[str, float]:
    """y: 1=거북목, 0=바른 자세. recall_fhp = 거북목을 놓치지 않은 비율, recall_normal = 바른 자세를 바르다고 한 비율."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    recall = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "f1": float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)),
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else float("nan"),
        "recall_fhp": float(recall[1]) if (y_true == 1).any() else float("nan"),
        "recall_normal": float(recall[0]) if (y_true == 0).any() else float("nan"),
    }


def confusion(y_true, y_pred) -> pd.DataFrame:
    matrix = confusion_matrix(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int), labels=[0, 1])
    return pd.DataFrame(matrix, index=[f"true_{c}" for c in CLASSES], columns=[f"pred_{c}" for c in CLASSES])


def per_subject(samples: pd.DataFrame, pred) -> pd.DataFrame:
    frame = samples.assign(_p=np.asarray(pred))
    rows = []
    for subject, part in frame.groupby("subject"):
        rows.append({"subject": subject, "windows": len(part), **binary_scores(part["fhp"], part["_p"])})
    return pd.DataFrame(rows)


def labeler_agreement(a: list[Interval], b: list[Interval], end_s: float, step_s: float = 0.2) -> dict[str, float]:
    """두 라벨러의 일치도(Cohen's κ). 기준은 fhp_kappa(거북목/바른 자세), head·lean은 참고용.
    어느 한쪽이라도 라벨 없음·제외인 시각은 뺍니다."""
    head_a, head_b, lean_a, lean_b, fhp_a, fhp_b = [], [], [], [], [], []
    for t in np.arange(0, end_s, step_s):
        la, lb = label_at(a, float(t), 0.0), label_at(b, float(t), 0.0)
        if la is None or lb is None or la.away or lb.away:
            continue
        head_a.append(la.head_forward); head_b.append(lb.head_forward)
        lean_a.append(la.torso_lean); lean_b.append(lb.torso_lean)
        fhp_a.append(la.posture == FHP); fhp_b.append(lb.posture == FHP)
    if not head_a:
        return {"fhp_kappa": float("nan"), "head_kappa": float("nan"), "lean_kappa": float("nan"), "frames": 0}
    return {"fhp_kappa": float(cohen_kappa_score(fhp_a, fhp_b)),
            "head_kappa": float(cohen_kappa_score(head_a, head_b)),
            "lean_kappa": float(cohen_kappa_score(lean_a, lean_b)), "frames": len(head_a)}


@dataclass
class AppSimulation:
    """앱과 똑같이 판정기를 돌렸을 때의 결과(프레임 단위)."""

    frames: int
    frame_accuracy: float
    labeled_hours: float
    alerts: int
    false_alerts: int
    false_alerts_per_hour: float
    events: int
    missed_events: int
    miss_rate: float
    latency_median_s: float
    latency_p95_s: float
    calibration_failures: int = 0
    normal_recall: float = float("nan")
    """바른 자세 프레임을 바른 자세로 본 비율(자세별 정확도)."""
    fhp_recall: float = float("nan")
    """거북목 프레임을 거북목으로 본 비율(자세별 정확도)."""
    macro_f1: float = float("nan")
    """프레임 단위 바른 자세/거북목 Macro F1. 확인 중(UNKNOWN)은 '거북목 아님'으로 셉니다."""

    def as_dict(self) -> dict:
        return asdict(self)


def _recall(y_true: list[int], y_hit: list[bool], cls: int) -> float:
    """그 자세 프레임 중 앱 상태가 정확히 그 자세였던 비율."""
    hits = [h for t, h in zip(y_true, y_hit) if t == cls]
    return float(np.mean(hits)) if hits else float("nan")


def _expected_state(interval: Interval) -> str:
    return interval.posture  # 머리 전방·상체 기울임 모두 FHP


def calibrate_recording(rec: Recording, thresholds: Thresholds) -> CalibrationResult:
    """앱처럼 5초 캘리브레이션을 합니다. 시작점은 첫 바른 자세 구간 시작 1초 뒤(진행자가 '바르게 앉기'를 안내한 상황).

    실패하면(움직임·가림·구부정) 앱이 안내 후 다시 재는 것처럼, 바른 자세 구간이 끝날 때까지 다시 시도합니다.
    끝까지 실패하면 마지막 실패 결과를 돌려줍니다.
    """
    normal = next((i for i in sorted(rec.labels, key=lambda i: i.start_s) if i.combo == "NORMAL" and not i.exclude), None)
    if normal is None:
        return CalibrationResult(NOT_VISIBLE, 1.0)
    start, end = normal.start_s + 1.0, normal.end_s
    est = PostureEstimator(thresholds)
    collector = CalibrationCollector(thresholds)
    last = CalibrationResult(NOT_VISIBLE, 1.0)
    for obs in observations(trim_to_labels(rec.landmarks, rec.labels)):
        fb = est.update(obs)
        t = obs.timestamp_ms / 1000.0
        if t < start:
            continue
        if t >= end:
            break
        result = collector.add(obs.timestamp_ms, fb)
        if result.status == DONE:
            return result
        if result.finished:
            last = result
            collector = CalibrationCollector(thresholds)  # 다시 시도
    return last


def simulate_app(runs: Iterable[tuple[Recording, Thresholds]], calibrate: bool = False) -> AppSimulation:
    """녹화마다 (녹화, 적용할 임계값)으로 앱과 같은 판정기를 돌립니다.

    calibrate=True면 촬영 회차(session, 없으면 피험자)마다 5초 캘리브레이션을 하고 Δ 임계값으로 판정합니다.
    캘리브레이션은 그 회차의 녹화 중 바른 자세 구간이 있는 첫 녹화에서 하고, 같은 회차의 다른 녹화에도 씁니다
    (자세별로 나눠 찍은 영상은 거북목 영상에 바른 자세 구간이 없어서). 같은 자리·같은 카메라 위치라고 가정해요.
    캘리브레이션이 실패하면 앱과 같이 절대 임계값으로 돌아가고 calibration_failures(녹화 수)에 셉니다.

    첫 라벨 앞(손뼉·자리 잡기)과 마지막 라벨 뒤 프레임은 버립니다. 앱도 세팅이 끝난 뒤부터 판정해요.

    오탐 = 상태가 FHP로 바뀌었는데 그 시각의 정답이 거북목이 아닌 경우.
    미탐 = 안정화 시간+1초 이상 이어진 이상 자세 구간에서 끝난 뒤 1초까지 한 번도 해당 상태가 안 나온 경우.
    지연 = 이상 자세 구간 시작부터 해당 상태가 처음 나온 시각까지(2초 안정화 포함).
    """
    correct = total = alerts = false_alerts = events = missed = 0
    y_true: list[int] = []
    y_pred: list[int] = []
    y_hit: list[bool] = []
    labeled_s = 0.0
    latencies: list[float] = []
    failures = 0
    runs = list(runs)
    calibrations: dict[str, object] = {}
    if calibrate:
        for rec, thresholds in runs:
            if calibrations.get(rec.calibration_group) is not None or not any(i.posture == NORMAL and not i.exclude for i in rec.labels):
                continue
            result = calibrate_recording(rec, thresholds)
            calibrations[rec.calibration_group] = result.calibration if result.status == DONE else None
    for rec, thresholds in runs:
        stable_s = thresholds.stable_ms / 1000.0
        calibration = calibrations.get(rec.calibration_group) if calibrate else None
        failures += int(calibrate and calibration is None)
        est = PostureEstimator(thresholds, calibration)
        states: list[tuple[float, str]] = []
        previous = None
        for obs in observations(trim_to_labels(rec.landmarks, rec.labels)):
            fb = est.update(obs)
            t = obs.timestamp_ms / 1000.0
            states.append((t, fb.state))
            truth = label_at(rec.labels, t)
            if truth is not None:
                total += 1
                expected_state = _expected_state(truth)
                correct += int(fb.state == expected_state)
                if expected_state in (NORMAL, FHP):
                    y_true.append(int(expected_state == FHP))
                    y_pred.append(int(fb.state == FHP))
                    y_hit.append(fb.state == expected_state)
            if fb.state == FHP and fb.state != previous:
                alerts += 1
                raw = label_at(rec.labels, t, 0.0)
                if raw is None or raw.exclude or raw.posture != fb.state:
                    false_alerts += 1
            previous = fb.state
        step = 1.0 / thresholds.analysis_fps
        labeled_s += sum(step for t, _ in states if (l := label_at(rec.labels, t, 0.0)) is not None and not l.away)
        for interval in rec.labels:
            expected = _expected_state(interval)
            if expected in (NORMAL, AWAY) or interval.exclude or interval.end_s - interval.start_s < stable_s + 1.0:
                continue
            events += 1
            hits = [t for t, s in states if interval.start_s <= t <= interval.end_s + 1.0 and s == expected]
            if hits:
                latencies.append(hits[0] - interval.start_s)
            else:
                missed += 1
    hours = labeled_s / 3600.0
    return AppSimulation(
        frames=total,
        frame_accuracy=correct / total if total else float("nan"),
        normal_recall=_recall(y_true, y_hit, 0),
        fhp_recall=_recall(y_true, y_hit, 1),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", labels=[0, 1], zero_division=0)) if y_true else float("nan"),
        labeled_hours=hours,
        alerts=alerts,
        false_alerts=false_alerts,
        false_alerts_per_hour=false_alerts / hours if hours else float("nan"),
        events=events,
        missed_events=missed,
        miss_rate=missed / events if events else float("nan"),
        latency_median_s=float(np.median(latencies)) if latencies else float("nan"),
        latency_p95_s=float(np.percentile(latencies, 95)) if latencies else float("nan"),
        calibration_failures=failures,
    )
