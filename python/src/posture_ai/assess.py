"""고정 기준값 평가: 파일럿·본 테스트 영상에 앱과 같은 판정기를 **정해 둔 기준값 그대로** 돌려 발표 목표를 확인해요.

`analyze`는 데이터로 기준값을 새로 고르기 때문에, 그 데이터로 성능을 재면 부풀려져요.
외부 테스터 평가는 반드시 이 명령(`python -m posture_ai evaluate`)으로 해요.

발표 목표(3-4·6장)
  - 자세별 정확도 ≥ 85% (바른 자세·거북목 각각, Macro F1 함께 보고)
  - 오탐 ≤ 3회/시간
  - 미탐률 ≤ 20%
  - 처리 지연 p95 ≤ 200ms (실기기 LatencyMeter 값을 --device-p95로 넣음)
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import pandas as pd

from .estimator import Thresholds
from .evaluate import AppSimulation, simulate_app
from .export import thresholds_json
from .windows import Recording

TARGET_ACCURACY = 0.85
TARGET_FALSE_ALERTS_PER_HOUR = 3.0
TARGET_MISS_RATE = 0.20
TARGET_DEVICE_P95_MS = 200.0
MIN_HOURS_FOR_FALSE_ALERTS = 0.25
"""오탐률은 15분 넘게 라벨된 시간이 있어야 의미가 있어요(그보다 짧으면 1회만 나와도 시간당 수치가 커짐)."""


def load_thresholds(config: Optional[Path] = None, t2: Optional[float] = None, pitch: Optional[float] = None) -> Thresholds:
    """기준값: 서버 /config 형식 JSON(analyze가 만든 thresholds.json 등) → 없으면 앱 기본값. --t2·--pitch가 있으면 덮어써요."""
    base = Thresholds.from_server(json.loads(config.read_text(encoding="utf-8"))) if config else Thresholds()
    changes = {}
    if t2 is not None:
        changes["lean_degrees"] = t2
    if pitch is not None:
        changes["pitch_degrees"] = pitch
    return replace(base, **changes) if changes else base


@dataclass
class Check:
    goal: str
    value: str
    passed: Optional[bool]  # None = 판단 불가
    note: str = ""


@dataclass
class Assessment:
    thresholds: Thresholds
    calibrate: bool
    overall: AppSimulation
    per_subject: pd.DataFrame
    checks: list[Check]
    recordings: int
    subjects: int


def _pct(v: float) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.1%}"


def _checks(sim: AppSimulation, device_p95_ms: Optional[float]) -> list[Check]:
    accuracy = min(sim.normal_recall, sim.fhp_recall)
    checks = [
        Check("자세별 정확도 ≥ 85%", f"바른 자세 {_pct(sim.normal_recall)} · 거북목 {_pct(sim.fhp_recall)}",
              None if math.isnan(accuracy) else accuracy >= TARGET_ACCURACY, f"Macro F1 {sim.macro_f1:.3f}"),
    ]
    short = sim.labeled_hours < MIN_HOURS_FOR_FALSE_ALERTS
    checks.append(Check("오탐 ≤ 3회/시간", f"{sim.false_alerts_per_hour:.1f}회/시간 ({sim.false_alerts}회 / {sim.labeled_hours * 60:.0f}분)",
                        None if short else sim.false_alerts_per_hour <= TARGET_FALSE_ALERTS_PER_HOUR,
                        "라벨된 시간이 15분보다 짧아 참고용" if short else ""))
    checks.append(Check("미탐률 ≤ 20%", f"{_pct(sim.miss_rate)} ({sim.missed_events}/{sim.events}개 거북목 구간)",
                        None if sim.events == 0 else sim.miss_rate <= TARGET_MISS_RATE,
                        "거북목 구간(3초 이상)이 없음" if sim.events == 0 else ""))
    if device_p95_ms is None:
        checks.append(Check("처리 지연 p95 ≤ 200ms", "-", None, "실기기 LatencyMeter 값을 --device-p95로 넣어 주세요"))
    else:
        checks.append(Check("처리 지연 p95 ≤ 200ms", f"{device_p95_ms:.0f}ms", device_p95_ms <= TARGET_DEVICE_P95_MS, "실기기 측정값"))
    return checks


def assess(recordings: list[Recording], thresholds: Thresholds, *, calibrate: bool = False,
           device_p95_ms: Optional[float] = None) -> Assessment:
    overall = simulate_app([(r, thresholds) for r in recordings], calibrate=calibrate)
    rows = []
    for subject in sorted({r.subject for r in recordings}):
        sim = simulate_app([(r, thresholds) for r in recordings if r.subject == subject], calibrate=calibrate)
        rows.append({"subject": subject, "frames": sim.frames, "normal_recall": sim.normal_recall,
                     "fhp_recall": sim.fhp_recall, "macro_f1": sim.macro_f1, "false_alerts": sim.false_alerts,
                     "minutes": sim.labeled_hours * 60, "missed_events": sim.missed_events, "events": sim.events})
    return Assessment(thresholds, calibrate, overall, pd.DataFrame(rows), _checks(overall, device_p95_ms),
                      len(recordings), len({r.subject for r in recordings}))


def _mark(passed: Optional[bool]) -> str:
    return "✅ 통과" if passed else ("❌ 미달" if passed is False else "➖ 판단 불가")


def write_report(a: Assessment, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    sim = a.overall
    t = thresholds_json(a.thresholds)
    lines = [
        "# 고정 기준값 평가 리포트\n",
        f"- 녹화 {a.recordings}개 · 피험자 {a.subjects}명 · 라벨된 시간 {sim.labeled_hours * 60:.1f}분 · 프레임 {sim.frames}개",
        f"- 기준값(고정): 몸 기울기 `t2_angle` = {t['t2_angle']}°, 고개 숙임 = {t['t3_pitch']}°"
        + (f", 개인 기준 사용(Δ {t['t2_delta']}° / {t['t3_delta']}°)" if a.calibrate else ", 개인 기준 안 씀"),
        "- 앱과 같은 판정기를 그대로 돌렸어요. 기준값을 이 데이터로 다시 고르지 않아요.\n",
        "## 발표 목표\n",
        "| 목표 | 결과 | 판정 | 비고 |", "|---|---|---|---|",
        *[f"| {c.goal} | {c.value} | {_mark(c.passed)} | {c.note} |" for c in a.checks],
        "",
        "## 참고 지표\n",
        f"- 프레임 정확도(확인 중 상태 포함) {_pct(sim.frame_accuracy)}",
        f"- 상태 확정 지연(2초 안정화 포함): 중앙값 {sim.latency_median_s:.1f}초, p95 {sim.latency_p95_s:.1f}초",
        f"- 경고 {sim.alerts}회 중 오탐 {sim.false_alerts}회"
        + (f" · 캘리브레이션 실패 {sim.calibration_failures}건" if a.calibrate else ""),
        "",
        "## 피험자별\n",
        "| 피험자 | 시간(분) | 바른 자세 | 거북목 | Macro F1 | 오탐 | 놓친 구간 |", "|---|---|---|---|---|---|---|",
        *[f"| {r.subject} | {r.minutes:.1f} | {_pct(r.normal_recall)} | {_pct(r.fhp_recall)} | {r.macro_f1:.3f} | "
          f"{r.false_alerts} | {r.missed_events}/{r.events} |" for r in a.per_subject.itertuples()],
        "",
        "- 자세별 정확도 = 그 자세 프레임 중 앱 상태가 정확히 그 자세였던 비율(시작 2초 '확인 중'은 틀린 것으로 셈).",
        "- 오탐 = 거북목으로 바뀐 순간의 정답이 거북목이 아님. 미탐 = 3초 넘는 거북목 구간 동안 한 번도 못 잡음.",
    ]
    report = out_dir / "evaluation.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "evaluation.json").write_text(json.dumps({
        "thresholds": t, "calibrate": a.calibrate, "overall": sim.as_dict(),
        "checks": [c.__dict__ for c in a.checks],
    }, indent=2, ensure_ascii=False, default=float) + "\n", encoding="utf-8")
    a.per_subject.to_csv(out_dir / "evaluation_per_subject.csv", index=False)
    return report
