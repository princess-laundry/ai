"""녹화 묶음 하나를 받아 샘플 → 임계값/SVM/RF × 절대/Δ 비교 → 앱 시뮬레이션 → 내보내기까지 한 번에 돌립니다.

합성 데이터(demo)와 실제 촬영 데이터가 같은 함수를 씁니다.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import export, train, tune
from .estimator import Thresholds
from .evaluate import attribute_scores, combo_of, combo_report, per_subject, simulate_app
from .windows import Recording, build_samples, summary

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

    rows, per_subject_tables, predictions = [], {}, {}
    for mode in ("abs", "delta"):
        folds, h, l = tune.leave_one_subject_out(samples, mode)
        data = samples.dropna(subset=["d_head", "d_torso"]) if mode == "delta" else samples
        rows.append(_row("임계값", mode, data, h, l))
        per_subject_tables[("threshold", mode)] = per_subject(data, h, l)
        predictions[("threshold", mode)] = (data, h, l)
        for kind in ("svm", "rf"):
            result = train.leave_one_subject_out(samples, kind, mode, seed)
            data = samples.loc[result.rows]
            rows.append(_row(kind.upper(), mode, data, result.head_pred, result.lean_pred))
            per_subject_tables[(kind, mode)] = per_subject(data, result.head_pred, result.lean_pred)
            predictions[(kind, mode)] = (data, result.head_pred, result.lean_pred)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "comparison.csv", index=False)
    for (kind, mode), table in per_subject_tables.items():
        table.to_csv(out_dir / f"per_subject_{kind}_{mode}.csv", index=False)

    # 앱에 실제로 들어가는 것: 절대 임계값(캘리브레이션 없이 동작). 전체 데이터로 다시 고릅니다.
    final = tune.fit(samples, "abs")
    thresholds = Thresholds(version=version, head_ratio=max(0.05, final.head), lean_degrees=max(1.0, final.lean))
    export.write_thresholds(thresholds, out_dir, "합성 데이터로 고른 값입니다. 서버에 실행하지 마세요." if synthetic else "")

    # 앱 시뮬레이션: 피험자별로 '그 사람을 뺀 나머지'로 고른 임계값을 적용(새 사람 기준).
    runs = []
    for subject in sorted({r.subject for r in recordings}):
        others = samples[samples["subject"] != subject]
        pair = tune.fit(others, "abs") if not others.empty else final
        t = replace(thresholds, head_ratio=max(0.05, pair.head), lean_degrees=max(1.0, pair.lean))
        runs += [(r, t) for r in recordings if r.subject == subject]
    sim = simulate_app(runs)
    (out_dir / "app_simulation.json").write_text(json.dumps(sim.as_dict(), indent=2) + "\n", encoding="utf-8")

    # 가장 좋은 학습형 모델 → ONNX
    ml = comparison[comparison["method"] != "임계값"].sort_values("mean_f1", ascending=False).iloc[0]
    kind, mode = ml["method"].lower(), ml["features"]
    model, params = train.fit_final(samples, kind, mode, seed)
    check = samples.dropna(subset=train.columns(mode))[train.columns(mode)].to_numpy()
    onnx_info = export.to_onnx(model, train.columns(mode), out_dir / f"model_{kind}_{mode}.onnx", check,
                               {"kind": kind, "mode": mode, "params": params, "synthetic": synthetic})

    baseline = comparison[(comparison["method"] == "임계값") & (comparison["features"] == "abs")].iloc[0]
    data, h, l = predictions[(kind, mode)]
    _, bh, bl = predictions[("threshold", "abs")]
    subject_gain = _subject_gain(per_subject_tables[(kind, mode)], per_subject_tables[("threshold", "abs")])
    adopt = bool(ml["mean_f1"] - baseline["mean_f1"] >= ADOPTION_GAIN and subject_gain > 0.5)
    combo = combo_report(data["combo"], combo_of(h, l))
    combo["confusion"].to_csv(out_dir / f"confusion_{kind}_{mode}.csv")
    base_data = predictions[("threshold", "abs")][0]
    base_combo = combo_report(base_data["combo"], combo_of(bh, bl))
    base_combo["confusion"].to_csv(out_dir / "confusion_threshold_abs.csv")

    report = _report(synthetic, samples, counts, comparison, per_subject_tables, thresholds, sim, ml, baseline,
                     subject_gain, adopt, onnx_info, base_combo, combo, kind, mode)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    return {"thresholds": export.thresholds_json(thresholds), "comparison": comparison, "simulation": sim,
            "adopt_ml": adopt, "onnx": onnx_info}


def _row(method: str, mode: str, data: pd.DataFrame, h, l) -> dict:
    scores = attribute_scores(data["head_forward"], h, data["torso_lean"], l)
    accuracy = float(np.mean(data["combo"].to_numpy() == combo_of(h, l)))
    return {"method": method, "features": mode, "windows": len(data), **scores, "combo_accuracy": accuracy}


def _subject_gain(candidate: pd.DataFrame, baseline: pd.DataFrame) -> float:
    merged = candidate.merge(baseline, on="subject", suffixes=("", "_base"))
    return float(np.mean(merged["mean_f1"] > merged["mean_f1_base"])) if len(merged) else 0.0


def _report(synthetic, samples, counts, comparison, per_subject_tables, thresholds, sim, ml, baseline, subject_gain,
            adopt, onnx_info, base_combo, combo, kind, mode) -> str:
    parts = ["# AI 자세 판정 분석 리포트\n"]
    if synthetic:
        parts.append(SYNTHETIC_BANNER)
    parts += [
        f"- 녹화 {samples['recording'].nunique()}개 · 피험자 {samples['subject'].nunique()}명 · 유효 2초 창 {len(samples)}개",
        "- 평가 방식: 피험자 1명씩 빼고 나머지로 고른 임계값·모델을 빠진 사람에게 적용(LOSO)\n",
        "## 1. 유효 창 수 (피험자 × 자세)\n", md_table(counts.reset_index()), "",
        "## 2. 조건별 비교 (발표 3-1 표)\n",
        "속성별 Macro F1(머리 전방 / 몸통 기울임)과 평균, 네 자세 정확도. features: abs=절대 특징, delta=5초 기준 자세 대비.\n",
        md_table(comparison), "",
        "## 3. 피험자별 평균 F1\n",
    ]
    for (k, m), table in per_subject_tables.items():
        parts.append(f"**{k} · {m}**: " + ", ".join(f"{r.subject}={r.mean_f1:.3f}" for r in table.itertuples()))
    parts += [
        "", "## 4. 혼동 행렬 (네 자세)\n",
        "**기준선(임계값 · abs)**\n", md_table(base_combo["confusion"].reset_index(names="")), "",
        f"**{kind.upper()} · {mode}**\n", md_table(combo["confusion"].reset_index(names="")), "",
        "## 5. 앱 시뮬레이션 (임계값 · abs, 앱과 같은 판정기)\n",
        "피험자마다 '그 사람을 뺀 나머지'로 고른 임계값을 적용했어요(새 사람 기준).\n",
        "| 지표 | 값 |", "|---|---|",
        *[f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |" for k, v in sim.as_dict().items()], "",
        "- 오탐: FHP/LEAN으로 바뀐 순간의 정답이 그 상태가 아님 · 미탐: 이상 자세 구간 동안 한 번도 못 잡음",
        "- 지연: 이상 자세 시작 → 상태 확정까지(2초 안정화 포함). 기기 처리 지연(ms)은 실기기에서 따로 측정해야 해요.\n",
        "## 6. 서버에 등록할 임계값 (전체 데이터로 선택)\n",
        "```json", json.dumps(export.thresholds_json(thresholds), indent=2), "```",
        "SQL은 `thresholds.sql`에 있어요."
        + (" **합성 데이터로 고른 값이라 서버에 등록하면 안 돼요.**\n" if synthetic else "\n"),
        "## 7. ML 적용 기준 (발표 3-4)\n",
        f"- 최고 학습형: **{ml['method']} · {ml['features']}** 평균 F1 {ml['mean_f1']:.3f} vs 기준선 {baseline['mean_f1']:.3f}"
        f" (차이 {ml['mean_f1'] - baseline['mean_f1']:+.3f}, 기준 +{ADOPTION_GAIN})",
        f"- 기준선보다 좋아진 피험자 비율: {subject_gain:.0%} (기준: 과반)",
        "- 처리 p95 ≤ 200ms 조건은 실기기 측정이 필요해요.",
        f"- 판단: **{'ML 적용 후보' if adopt else '규칙 MVP 유지'}**",
        f"- ONNX: `model_{kind}_{mode}.onnx` {onnx_info['onnx_bytes']:,} bytes, sklearn과 예측 일치율 {onnx_info['sklearn_onnx_agreement']:.1%}\n",
    ]
    return "\n".join(parts) + "\n"
