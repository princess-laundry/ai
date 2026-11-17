"""평가 지표: 속성별 Macro F1, 혼동 행렬, 피험자별 점수, Cohen's κ, 앱 시뮬레이션(오탐·미탐·확정 지연)."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score, recall_score

from .calibration import DONE, NOT_VISIBLE, CalibrationCollector, CalibrationResult
from .estimator import FHP, LEAN, NORMAL, AWAY, PostureEstimator, Thresholds
from .labels import Interval, label_at
from .landmarks import observations
from .windows import Recording

COMBOS = ["NORMAL", "FHP", "LEAN", "BOTH"]


def attribute_scores(head_true, head_pred, lean_true, lean_pred) -> dict[str, float]:
    head = f1_score(head_true, head_pred, average="macro", labels=[0, 1], zero_division=0)
    lean = f1_score(lean_true, lean_pred, average="macro", labels=[0, 1], zero_division=0)
    return {"head_f1": float(head), "lean_f1": float(lean), "mean_f1": float((head + lean) / 2)}


def combo_of(head, lean) -> np.ndarray:
    head, lean = np.asarray(head).astype(bool), np.asarray(lean).astype(bool)
    return np.where(head & lean, "BOTH", np.where(head, "FHP", np.where(lean, "LEAN", "NORMAL")))


def combo_report(true_combo, pred_combo) -> dict:
    matrix = confusion_matrix(true_combo, pred_combo, labels=COMBOS)
    recall = recall_score(true_combo, pred_combo, labels=COMBOS, average=None, zero_division=0)
    return {
        "accuracy": float(np.mean(np.asarray(true_combo) == np.asarray(pred_combo))),
        "confusion": pd.DataFrame(matrix, index=[f"true_{c}" for c in COMBOS], columns=[f"pred_{c}" for c in COMBOS]),
        "recall": dict(zip(COMBOS, map(float, recall))),
    }


def per_subject(samples: pd.DataFrame, head_pred, lean_pred) -> pd.DataFrame:
    frame = samples.assign(_h=np.asarray(head_pred), _l=np.asarray(lean_pred))
    rows = []
    for subject, part in frame.groupby("subject"):
        rows.append({"subject": subject, "windows": len(part),
                     **attribute_scores(part["head_forward"], part["_h"], part["torso_lean"], part["_l"])})
    return pd.DataFrame(rows)


def labeler_agreement(a: list[Interval], b: list[Interval], end_s: float, step_s: float = 0.2) -> dict[str, float]:
    """두 라벨러의 일치도(속성별 Cohen's κ). 어느 한쪽이라도 라벨 없음·제외인 시각은 뺍니다."""
    head_a, head_b, lean_a, lean_b = [], [], [], []
    for t in np.arange(0, end_s, step_s):
        la, lb = label_at(a, float(t), 0.0), label_at(b, float(t), 0.0)
        if la is None or lb is None or la.away or lb.away:
            continue
        head_a.append(la.head_forward); head_b.append(lb.head_forward)
        lean_a.append(la.torso_lean); lean_b.append(lb.torso_lean)
    if not head_a:
        return {"head_kappa": float("nan"), "lean_kappa": float("nan"), "frames": 0}
    return {"head_kappa": float(cohen_kappa_score(head_a, head_b)),
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

    def as_dict(self) -> dict:
        return asdict(self)


def _expected_state(interval: Interval) -> str:
    return interval.posture  # 둘 다면 FHP 우선


def calibrate_recording(rec: Recording, thresholds: Thresholds) -> CalibrationResult:
    """앱처럼 5초 캘리브레이션을 합니다. 시작점은 첫 바른 자세 구간 시작 1초 뒤(진행자가 '바르게 앉기'를 안내한 상황)."""
    start = next((i.start_s + 1.0 for i in sorted(rec.labels, key=lambda i: i.start_s)
                  if i.combo == "NORMAL" and not i.exclude), None)
    if start is None:
        return CalibrationResult(NOT_VISIBLE, 1.0)
    est = PostureEstimator(thresholds)
    collector = CalibrationCollector(thresholds)
    result = CalibrationResult()
    for obs in observations(rec.landmarks):
        fb = est.update(obs)
        if obs.timestamp_ms / 1000.0 >= start:
            result = collector.add(obs.timestamp_ms, fb)
            if result.finished:
                break
    return result if result.finished else CalibrationResult(NOT_VISIBLE, 1.0)


def simulate_app(runs: Iterable[tuple[Recording, Thresholds]], calibrate: bool = False) -> AppSimulation:
    """녹화마다 (녹화, 적용할 임계값)으로 앱과 같은 판정기를 돌립니다.

    calibrate=True면 녹화마다 5초 캘리브레이션을 먼저 하고 Δ 임계값으로 판정합니다.
    캘리브레이션이 실패하면 앱과 같이 절대 임계값으로 돌아가고 calibration_failures에 셉니다.

    오탐 = 상태가 FHP/LEAN으로 바뀌었는데 그 시각의 정답이 그 상태가 아닌 경우.
    미탐 = 안정화 시간+1초 이상 이어진 이상 자세 구간에서 끝난 뒤 1초까지 한 번도 해당 상태가 안 나온 경우.
    지연 = 이상 자세 구간 시작부터 해당 상태가 처음 나온 시각까지(2초 안정화 포함).
    """
    correct = total = alerts = false_alerts = events = missed = 0
    labeled_s = 0.0
    latencies: list[float] = []
    failures = 0
    for rec, thresholds in runs:
        stable_s = thresholds.stable_ms / 1000.0
        calibration = None
        if calibrate:
            result = calibrate_recording(rec, thresholds)
            calibration = result.calibration if result.status == DONE else None
            failures += int(calibration is None)
        est = PostureEstimator(thresholds, calibration)
        states: list[tuple[float, str]] = []
        previous = None
        for obs in observations(rec.landmarks):
            fb = est.update(obs)
            t = obs.timestamp_ms / 1000.0
            states.append((t, fb.state))
            truth = label_at(rec.labels, t)
            if truth is not None:
                total += 1
                correct += int(fb.state == _expected_state(truth))
            if fb.state in (FHP, LEAN) and fb.state != previous:
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
