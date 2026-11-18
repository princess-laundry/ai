"""Python ↔ Kotlin 일치 검증용 golden 파일 생성.

입력(앱이 판정기에 넘기는 관측값 그대로)과 Python 판정기의 프레임별 출력을 JSON으로 저장합니다.
Kotlin 테스트(GoldenParityTest)가 같은 입력을 넣어 출력이 같은지(실수는 1e-6 이내) 확인합니다.
판정 로직을 고치면 `python -m posture_ai golden`으로 다시 만들고 Kotlin 테스트를 돌리세요.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from . import synthetic
from .calibration import Calibration, CalibrationCollector
from .estimator import Observation, PostureEstimator, Thresholds
from .features import BodySide, Point
from .landmarks import observations

THRESHOLDS = Thresholds(version=1, head_ratio=0.2, lean_degrees=15.0, stable_ms=2000, analysis_fps=5,
                        lean_delta_degrees=10.0)


def _f32(value: float) -> float:
    """Kotlin Point.confidence가 Float라서, 신뢰도는 float32로 표현 가능한 값으로 맞춥니다."""
    return float(np.float32(value))


def _point(p: Optional[Point]):
    if p is None or not (math.isfinite(p.x) and math.isfinite(p.y)):
        return None
    return [p.x, p.y, _f32(p.confidence)]


def _side(b: Optional[BodySide]):
    if b is None:
        return None
    points = [_point(b.ear), _point(b.shoulder), _point(b.hip)]
    return None if any(p is None for p in points) else points


def _normalize(obs: Observation) -> Observation:
    """JSON으로 오가며 바뀌는 부분(float32 신뢰도, NaN 좌표 → 없음)을 Python 입력에도 똑같이 적용합니다."""
    def p(x):
        v = _point(x)
        return None if v is None else Point(v[0], v[1], v[2])

    def s(b):
        v = _side(b)
        return None if v is None else BodySide(*(Point(*q) for q in v))

    return replace(obs, left=s(obs.left), right=s(obs.right), nose=p(obs.nose))


def _frame_json(obs: Observation) -> dict:
    return {"t": obs.timestamp_ms, "person": obs.person_detected, "left": _side(obs.left), "right": _side(obs.right),
            "nose": _point(obs.nose), "roll": obs.device_roll_deg}


def _num(v: Optional[float]):
    return None if v is None else float(v)


def _thresholds_json(t: Thresholds) -> dict:
    return {"version": t.version, "headRatio": t.head_ratio, "leanDegrees": t.lean_degrees, "stableMs": t.stable_ms,
            "analysisFps": t.analysis_fps, "leanDeltaDegrees": t.lean_delta_degrees}


def case(name: str, frames: Iterable[Observation], thresholds: Thresholds = THRESHOLDS,
         calibration: Optional[Calibration] = None) -> dict:
    est = PostureEstimator(thresholds, calibration)
    inputs, expected = [], []
    for obs in frames:
        obs = _normalize(obs)
        fb = est.update(obs)
        inputs.append(_frame_json(obs))
        expected.append({
            "state": fb.state, "needsReset": fb.needs_reset, "framed": fb.framed, "facing": fb.facing,
            "headRatio": _num(fb.head_ratio), "torsoAngle": _num(fb.torso_angle), "neckAngle": _num(fb.neck_angle),
            "axisAngle": _num(fb.axis_angle), "calibrated": fb.calibrated,
        })
    return {
        "name": name,
        "thresholds": _thresholds_json(thresholds),
        "calibration": None if calibration is None else {"axisAngle": calibration.axis_angle},
        "frames": inputs,
        "expected": expected,
    }


def collector_case(name: str, frames: Iterable[Observation], thresholds: Thresholds = THRESHOLDS) -> dict:
    """캘리브레이션 수집기: 판정기(캘리브레이션 없이) 출력을 프레임마다 넣었을 때의 결과."""
    est = PostureEstimator(thresholds)
    collector = CalibrationCollector(thresholds)
    inputs, expected = [], []
    for obs in frames:
        obs = _normalize(obs)
        result = collector.add(obs.timestamp_ms, est.update(obs))
        inputs.append(_frame_json(obs))
        c = result.calibration
        expected.append({"status": result.status, "progress": result.progress,
                         "axisAngle": None if c is None else c.axis_angle})
    return {"name": name, "thresholds": _thresholds_json(thresholds), "frames": inputs, "expected": expected}


def _edge_frames() -> list[Observation]:
    """결측 → 자리 비움, 가림 → 확인 중, 거리 급변 → 재세팅, 복귀를 차례로 겪는 짧은 시퀀스."""
    subject = synthetic.Subject("edge", facing=1, noise_px=1.0, dropout=0.0)
    script = synthetic.script_intervals([("NORMAL", 6), ("FHP", 5), ("AWAY", 3), ("NORMAL", 5)])
    obs = list(observations(synthetic.generate(subject, script, seed=3)))
    out: list[Observation] = []
    for o in obs:
        t = o.timestamp_ms
        if 15_000 <= t < 16_000 and o.left is not None:  # 사람은 있는데 귀·어깨·골반이 가려짐
            dim = lambda b: BodySide(*(Point(p.x, p.y, 0.2) for p in (b.ear, b.shoulder, b.hip)))  # noqa: E731
            o = replace(o, left=dim(o.left), right=dim(o.right))
        if 16_000 <= t < 17_000 and o.right is not None:  # 의자를 당겨 앉음: 길이 1.5배
            big = lambda b: BodySide(*(Point(b.hip.x + (p.x - b.hip.x) * 1.5, b.hip.y + (p.y - b.hip.y) * 1.5, p.confidence)  # noqa: E731
                                       for p in (b.ear, b.shoulder, b.hip)))
            o = replace(o, left=big(o.left), right=big(o.right))
        out.append(o)
    return out


def build_cases() -> list[dict]:
    rng = np.random.default_rng(11)
    right_facing = replace(synthetic.random_subject("G1", rng), facing=1, roll_deg=3.0)
    left_facing = replace(synthetic.random_subject("G2", rng), facing=-1, roll_deg=-5.0)
    one_rep = synthetic.script_intervals(synthetic.COLLECTION_SCRIPT)
    no_nose = [replace(o, nose=None) for o in observations(synthetic.generate(left_facing, one_rep, seed=5))]
    return [
        case("collection_right_facing_with_roll", observations(synthetic.generate(right_facing, one_rep, seed=4))),
        case("collection_left_facing_without_nose", no_nose),
        case("scenario_150s", observations(synthetic.generate(right_facing, synthetic.script_intervals(synthetic.SCENARIO_150S), seed=6))),
        case("edge_away_occlusion_reset", _edge_frames()),
        _calibrated_case(),
    ]


def _calibrated_case() -> dict:
    """평소에도 머리가 앞에 있고 몸통이 숙여진 사람: 절대 임계값이면 늘 이상 자세로 보이지만 개인 기준으로는 바른 자세."""
    subject = synthetic.Subject("slouch", head_neutral=0.15, torso_neutral=11.0, head_shift=0.22, lean_shift=16.0,
                                facing=-1, noise_px=1.2, dropout=0.0)
    script = synthetic.script_intervals(synthetic.COLLECTION_SCRIPT)
    frames = list(observations(synthetic.generate(subject, script, seed=8)))
    est, collector = PostureEstimator(THRESHOLDS), CalibrationCollector(THRESHOLDS)
    result = None
    for o in frames:  # 첫 바른 자세 구간(0~20초)의 1초 뒤부터 5초 캘리브레이션
        fb = est.update(_normalize(o))
        if o.timestamp_ms >= 1000:
            result = collector.add(o.timestamp_ms, fb)
            if result.finished:
                break
    assert result is not None and result.status == "DONE", result
    return case("calibrated_slouching_subject", frames, calibration=result.calibration)


def build_collector_cases() -> dict:
    subject = synthetic.Subject("calib", facing=1, noise_px=1.0, dropout=0.0)

    def run(script, seed):
        return list(observations(synthetic.generate(subject, synthetic.script_intervals(script), seed=seed)))

    return {
        "name": "calibration_collector",
        "cases": [
            collector_case("done", run([("NORMAL", 8)], 21)),
            collector_case("moving", run([("NORMAL", 1.6), ("LEAN", 1.6), ("NORMAL", 1.6), ("LEAN", 1.6), ("NORMAL", 2)], 22)),
            collector_case("not_upright", run([("BOTH", 8)], 23)),
            collector_case("not_visible", run([("AWAY", 8)], 24)),
        ],
    }


def _check_collector_cases(cases: dict) -> None:
    for c in cases["cases"]:
        assert c["expected"][-1]["status"] == c["name"].upper(), (c["name"], c["expected"][-1])


def write(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    collector = build_collector_cases()
    _check_collector_cases(collector)
    for c in build_cases() + [collector]:
        path = out_dir / f"{c['name']}.json"
        path.write_text(json.dumps(c, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
        paths.append(path)
    return paths
