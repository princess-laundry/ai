"""결과 내보내기: 서버 threshold_configs 형식 JSON·SQL, 학습 모델 ONNX(+ 결과 일치 확인)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .estimator import Thresholds


def thresholds_json(t: Thresholds) -> dict:
    """GET /api/v1/config 응답과 같은 형태."""
    return {
        "version": t.version,
        "t1_ratio": round(float(t.head_ratio), 4),
        "t2_angle": round(float(t.lean_degrees), 2),
        "stable_sec": t.stable_ms / 1000.0,
        "analysis_fps": t.analysis_fps,
    }


def thresholds_sql(t: Thresholds, activate: bool = False) -> str:
    """백엔드 threshold_configs에 새 버전을 등록하는 SQL. 기존 버전은 고치지 않습니다(V2 마이그레이션 주석 참고)."""
    c = thresholds_json(t)
    insert = (
        "INSERT INTO threshold_configs (version, t1_ratio, t2_angle, stable_sec, analysis_fps, is_active)\n"
        f"VALUES ({c['version']}, {c['t1_ratio']}, {c['t2_angle']}, {c['stable_sec']}, {c['analysis_fps']}, false);\n"
    )
    if not activate:
        return insert
    return (
        "BEGIN;\n" + insert
        + "UPDATE threshold_configs SET is_active = false WHERE is_active;\n"
        + f"UPDATE threshold_configs SET is_active = true WHERE version = {c['version']};\n"
        + "COMMIT;\n"
    )


def write_thresholds(t: Thresholds, out_dir: Path, note: str = "") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds_json(t), indent=2) + "\n", encoding="utf-8")
    header = "".join(f"-- {line}\n" for line in note.splitlines())
    (out_dir / "thresholds.sql").write_text(header + thresholds_sql(t, activate=True), encoding="utf-8")


def to_onnx(model, feature_names: list[str], path: Path, check: np.ndarray, meta: dict) -> dict:
    """sklearn 모델 → ONNX. check 입력으로 sklearn과 onnxruntime 예측이 같은지 확인합니다."""
    import onnxruntime as ort
    from skl2onnx import to_onnx as convert
    from skl2onnx.common.data_types import FloatTensorType

    onx = convert(model, initial_types=[("features", FloatTensorType([None, len(feature_names)]))],
                  options={id(model): {"zipmap": False}} if hasattr(model, "predict_proba") else None,
                  target_opset=17)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString())
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    onnx_label = session.run(None, {"features": check.astype(np.float32)})[0]
    sk_label = model.predict(check.astype(np.float32))
    agreement = float(np.mean(np.asarray(onnx_label).astype(str) == np.asarray(sk_label).astype(str)))
    info = {**meta, "features": feature_names, "classes": [str(c) for c in model.classes_],
            "onnx_bytes": path.stat().st_size, "sklearn_onnx_agreement": agreement, "checked_rows": int(len(check))}
    path.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return info
