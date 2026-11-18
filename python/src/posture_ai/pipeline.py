"""녹화 묶음 하나를 받아 샘플 → 임계값/SVM/RF × 절대/Δ 비교 → 앱 시뮬레이션 → 내보내기까지 한 번에 돌립니다.

합성 데이터(demo)와 실제 촬영 데이터가 같은 함수를 씁니다.
분류는 바른 자세 / 거북목 두 가지예요(2026-11-18 팀 결정: 상체 기울임도 거북목).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import export, train, tune
from .estimator import Thresholds
from .evaluate import binary_scores, confusion, per_subject, simulate_app
from .windows import Recording, build_samples, quality, summary

SYNTHETIC_BANNER = "> ⚠️ **합성 데이터 결과입니다. 실제 성능이 아닙니다.** 파이프라인이 끝까지 돌아가는지 확인하는 용도예요.\n"
ADOPTION_GAIN = 0.05  # 발표 3-4: 기준선 대비 Macro F1 +0.05 이상


def md_table(frame: pd.DataFrame, digits: int = 3) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in frame.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def analyze(recordings: list[Recording], out_dir: Path, *, synthetic: bool, version: int = 2, seed: int = 0) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    samples = build_samples(recordings)
    if samples.empty:
        raise RuntimeError("유효한 2초 창이 없습니다. 라벨·랜드마크를 확인하세요.")
    samples.to_csv(out_dir / "samples.csv", index=False)
    counts = summary(samples)
    counts.to_csv(out_dir / "sample_counts.csv")
    data_quality = quality(recordings)
    data_quality.to_csv(out_dir / "data_quality.csv", index=False)

    rows, per_subject_tables, predictions = [], {}, {}
    for mode in ("abs", "delta"):
        _, pred, index = tune.leave_one_subject_out(samples, mode)
        data = samples.loc[index]
        rows.append(_row("임계값", mode, data, pred))
        per_subject_tables[("threshold", mode)] = per_subject(data, pred)
        predictions[("threshold", mode)] = (data, pred)
        for kind in ("svm", "rf"):
            result = train.leave_one_subject_out(samples, kind, mode, seed)
            data = samples.loc[result.rows]
            rows.append(_row(kind.upper(), mode, data, result.pred))
            per_subject_tables[(kind, mode)] = per_subject(data, result.pred)
            predictions[(kind, mode)] = (data, result.pred)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    for (kind, mode), table in per_subject_tables.items():
        table.to_csv(out_dir / f"per_subject_{kind}_{mode}.csv", index=False)

    # 앱에 실제로 들어가는 것: 절대 임계값 + 캘리브레이션용 Δ 임계값. 전체 데이터로 다시 고릅니다.
    final = tune.fit(samples, "abs")
    final_delta = tune.fit(samples, "delta") if not tune.usable(samples, "delta").empty else None
    thresholds = _thresholds(version, final, final_delta)
    export.write_thresholds(thresholds, out_dir, "합성 데이터로 고른 값입니다. 서버에 실행하지 마세요." if synthetic else "")

    # 앱 시뮬레이션: 피험자별로 '그 사람을 뺀 나머지'로 고른 임계값을 적용(새 사람 기준).
    runs = []
    for subject in sorted({r.subject for r in recordings}):
        others = samples[samples["subject"] != subject]
        if others.empty:
            t = thresholds
        else:
            delta = tune.fit(others, "delta") if not tune.usable(others, "delta").empty else None
            t = _thresholds(version, tune.fit(others, "abs"), delta)
        runs += [(r, t) for r in recordings if r.subject == subject]
    sim = simulate_app(runs)
    sim_cal = simulate_app(runs, calibrate=True)
    (out_dir / "app_simulation.json").write_text(
        json.dumps({"absolute": sim.as_dict(), "calibrated": sim_cal.as_dict()}, indent=2) + "\n", encoding="utf-8")

    # 가장 좋은 학습형 모델 → ONNX
    ml = comparison[comparison["method"] != "임계값"].sort_values("f1", ascending=False).iloc[0]
    kind, mode = ml["method"].lower(), ml["features"]
    model, params = train.fit_final(samples, kind, mode, seed)
    check = samples.dropna(subset=train.columns(mode))[train.columns(mode)].to_numpy()
    onnx_info = export.to_onnx(model, train.columns(mode), out_dir / f"model_{kind}_{mode}.onnx", check,
                               {"kind": kind, "mode": mode, "params": params, "synthetic": synthetic,
                                "classes_meaning": {"0": "NORMAL", "1": "FHP"}})

    # 기준선 = 같은 특징(abs/delta)을 쓰는 규칙(임계값). 캘리브레이션을 쓰는 앱이면 delta끼리 비교해야 공정해요.
    baseline = comparison[(comparison["method"] == "임계값") & (comparison["features"] == mode)].iloc[0]
    subject_gain = _subject_gain(per_subject_tables[(kind, mode)], per_subject_tables[("threshold", mode)])
    adopt = bool(ml["f1"] - baseline["f1"] >= ADOPTION_GAIN and subject_gain > 0.5)
    data, pred = predictions[(kind, mode)]
    ml_confusion = confusion(data["fhp"], pred)
    ml_confusion.to_csv(out_dir / f"confusion_{kind}_{mode}.csv")
    base_data, base_pred = predictions[("threshold", mode)]
    base_confusion = confusion(base_data["fhp"], base_pred)
    base_confusion.to_csv(out_dir / f"confusion_threshold_{mode}.csv")

    report = _report(synthetic, recordings, samples, counts, data_quality, comparison, per_subject_tables, thresholds,
                     sim, sim_cal, ml, baseline, subject_gain, adopt, onnx_info, base_confusion, ml_confusion, kind, mode)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return {"thresholds": export.thresholds_json(thresholds), "comparison": comparison, "simulation": sim,
            "simulation_calibrated": sim_cal, "adopt_ml": adopt, "onnx": onnx_info}


def _thresholds(version: int, absolute: "tune.ThresholdPair", delta: "tune.ThresholdPair | None") -> Thresholds:
    """튜닝 결과 → 서버 형식. t2_angle = 기울기 기준, t3_pitch = 고개 숙임 기준, *_delta = 개인 기준 Δ.
    t1_ratio는 판정에 안 써서 기본값."""
    default = Thresholds()
    return Thresholds(version=version, lean_degrees=max(1.0, absolute.axis), pitch_degrees=max(1.0, absolute.pitch),
                      lean_delta_degrees=max(1.0, delta.axis) if delta else default.lean_delta_degrees,
                      pitch_delta_degrees=max(1.0, delta.pitch) if delta else default.pitch_delta_degrees)


def _row(method: str, mode: str, data: pd.DataFrame, pred) -> dict:
    return {"method": method, "features": mode, "windows": len(data), **binary_scores(data["fhp"], pred)}


def _fmt(value) -> str:
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def _subject_gain(candidate: pd.DataFrame, baseline: pd.DataFrame) -> float:
    merged = candidate.merge(baseline, on="subject", suffixes=("", "_base"))
    return float(np.mean(merged["f1"] > merged["f1_base"])) if len(merged) else 0.0


def _single_posture(recordings: list[Recording]) -> bool:
    """녹화마다 자세가 하나뿐인지(자세별로 나눠 찍은 영상)."""
    return all(len({i.posture for i in r.labels if not i.exclude}) <= 1 for r in recordings)


def _report(synthetic, recordings, samples, counts, data_quality, comparison, per_subject_tables, thresholds, sim, sim_cal,
            ml, baseline, subject_gain, adopt, onnx_info, base_confusion, ml_confusion, kind, mode) -> str:
    parts = ["# AI 자세 판정 분석 리포트\n"]
    if synthetic:
        parts.append(SYNTHETIC_BANNER)
    single = _single_posture(recordings)
    parts += [
        f"- 녹화 {samples['recording'].nunique()}개 · 피험자 {samples['subject'].nunique()}명 · 유효 2초 창 {len(samples)}개",
        "- 분류: 바른 자세(NORMAL) / 거북목(FHP). 상체 기울임도 거북목이에요(팀 결정).",
        "- 평가 방식: 피험자 1명씩 빼고 나머지로 고른 임계값·모델을 빠진 사람에게 적용(LOSO)\n",
        "## 1. 데이터\n",
        "**유효 창 수 (피험자 × 자세)**\n", md_table(counts.reset_index()), "",
        "**녹화별 품질** (usable = 귀·어깨·골반이 잡히고 재세팅이 아닌 프레임 비율. 0.8 미만이면 촬영 각도·가림을 확인하세요)\n",
        md_table(data_quality), "",
        "## 2. 조건별 비교 (발표 3-1 표)\n",
        "f1 = 두 자세 Macro F1, recall_fhp = 거북목을 놓치지 않은 비율, recall_normal = 바른 자세를 바르다고 한 비율.",
        "features: abs = 절대 특징, delta = 5초 기준 자세 대비.\n",
        md_table(comparison), "",
        "## 3. 피험자별 F1\n",
    ]
    for (k, m), table in per_subject_tables.items():
        parts.append(f"**{k} · {m}**: " + ", ".join(f"{r.subject}={r.f1:.3f}" for r in table.itertuples()))
    parts += [
        "", "## 4. 혼동 행렬\n",
        f"**기준선(임계값 · {mode})**\n", md_table(base_confusion.reset_index(names="")), "",
        f"**{kind.upper()} · {mode}**\n", md_table(ml_confusion.reset_index(names="")), "",
        "## 5. 앱 시뮬레이션 (앱과 같은 판정기)\n",
        "피험자마다 '그 사람을 뺀 나머지'로 고른 임계값을 적용했어요(새 사람 기준).",
        "캘리브레이션 사용: 그 사람의 바른 자세 구간에서 5초 캘리브레이션 → Δ 임계값으로 판정(실패하면 절대 임계값).\n",
        "| 지표 | 절대 임계값 | 캘리브레이션 사용 |", "|---|---|---|",
        *[f"| {k} | {_fmt(v)} | {_fmt(sim_cal.as_dict()[k])} |" for k, v in sim.as_dict().items()], "",
        "- 오탐: 거북목으로 바뀐 순간의 정답이 거북목이 아님 · 미탐: 거북목 구간 동안 한 번도 못 잡음",
        "- 지연: 거북목 시작 → 상태 확정까지(2초 안정화 포함). 기기 처리 지연(ms)은 실기기에서 따로 측정해야 해요.",
    ]
    if single:
        parts += [
            "- ⚠️ 자세별로 나눠 찍은 영상이라 자세가 **바뀌는 순간이 없어요.** 지연은 판정기 준비 시간(약 2~3초)일 뿐이라 의미가 없어요.",
            "- ⚠️ 캘리브레이션은 같은 사람의 바른 자세 영상에서 구한 기준을 거북목 영상에도 썼어요(같은 자리·카메라 위치라고 가정).",
        ]
    parts += [
        "",
        "## 6. 서버에 등록할 임계값 (전체 데이터로 선택)\n",
        "```json", json.dumps(export.thresholds_json(thresholds), indent=2), "```",
        "SQL은 `thresholds.sql`에 있어요."
        + (" **합성 데이터로 고른 값이라 서버에 등록하면 안 돼요.**\n" if synthetic else "\n"),
        "## 7. ML 적용 기준 (발표 3-4)\n",
        f"- 최고 학습형: **{ml['method']} · {ml['features']}** F1 {ml['f1']:.3f} vs 같은 특징의 규칙(임계값 · {mode}) {baseline['f1']:.3f}"
        f" (차이 {ml['f1'] - baseline['f1']:+.3f}, 기준 +{ADOPTION_GAIN})",
        f"- 기준선보다 좋아진 피험자 비율: {subject_gain:.0%} (기준: 과반)",
        "- 처리 p95 ≤ 200ms 조건은 실기기 측정이 필요해요.",
        f"- 판단: **{'ML 적용 후보' if adopt else '규칙 MVP 유지'}**",
        f"- ONNX: `model_{kind}_{mode}.onnx` {onnx_info['onnx_bytes']:,} bytes, sklearn과 예측 일치율 {onnx_info['sklearn_onnx_agreement']:.1%}\n",
    ]
    if samples["subject"].nunique() < 5:
        parts.append(f"> ⚠️ 피험자가 {samples['subject'].nunique()}명뿐이라 수치의 신뢰도가 낮아요. 예비 결과로만 보세요.\n")
    return "\n".join(parts) + "\n"
