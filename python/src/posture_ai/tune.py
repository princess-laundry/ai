"""임계값 T₁(머리)·T₂(몸통) 격자 탐색. 두 속성은 서로 독립이라 각각 따로 고릅니다.

평가는 피험자 1명씩 빼고 나머지로 고른 임계값을 빠진 사람에게 적용(LOSO)해서,
'새 사람'에게 얼마나 통하는지를 봅니다. 최종 값은 전체 데이터로 다시 고릅니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .evaluate import attribute_scores

HEAD_GRID = np.round(np.arange(0.05, 0.80 + 1e-9, 0.01), 2)          # Thresholds 허용 범위 0.05~2.0
LEAN_GRID = np.round(np.arange(2.0, 45.0 + 1e-9, 0.5), 1)            # Thresholds 허용 범위 1~89
DELTA_HEAD_GRID = np.round(np.arange(0.02, 0.60 + 1e-9, 0.01), 2)
DELTA_LEAN_GRID = np.round(np.arange(1.0, 40.0 + 1e-9, 0.5), 1)


@dataclass(frozen=True)
class ThresholdPair:
    head: float
    lean: float
    mode: str  # "abs" 또는 "delta"

    def columns(self) -> tuple[str, str]:
        return ("head_mean", "torso_mean") if self.mode == "abs" else ("d_head", "d_torso")

    def predict(self, samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        head_col, lean_col = self.columns()
        return (samples[head_col].to_numpy() > self.head).astype(int), (samples[lean_col].to_numpy() > self.lean).astype(int)


def _best(values: np.ndarray, truth: np.ndarray, grid: np.ndarray) -> float:
    """Macro F1이 가장 높은 임계값. 같은 점수가 여러 개면 그 범위의 가운데(두 자세 사이 여유가 가장 큰 값)."""
    scores = np.array([
        f1_score(truth, (values > t).astype(int), average="macro", labels=[0, 1], zero_division=0) for t in grid
    ])
    tied = grid[scores >= scores.max() - 1e-12]
    return float(tied[len(tied) // 2])


def fit(samples: pd.DataFrame, mode: str = "abs") -> ThresholdPair:
    data = samples.dropna(subset=["d_head", "d_torso"]) if mode == "delta" else samples
    head_grid, lean_grid = (HEAD_GRID, LEAN_GRID) if mode == "abs" else (DELTA_HEAD_GRID, DELTA_LEAN_GRID)
    head_col, lean_col = ("head_mean", "torso_mean") if mode == "abs" else ("d_head", "d_torso")
    return ThresholdPair(
        head=_best(data[head_col].to_numpy(), data["head_forward"].to_numpy(), head_grid),
        lean=_best(data[lean_col].to_numpy(), data["torso_lean"].to_numpy(), lean_grid),
        mode=mode,
    )


def leave_one_subject_out(samples: pd.DataFrame, mode: str = "abs") -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """반환: (fold별 점수표, 전체 샘플에 대한 LOSO 예측 head, lean)."""
    data = samples.dropna(subset=["d_head", "d_torso"]) if mode == "delta" else samples
    head_pred = np.zeros(len(data), dtype=int)
    lean_pred = np.zeros(len(data), dtype=int)
    rows = []
    subjects = sorted(data["subject"].unique())
    for held in subjects:
        test = (data["subject"] == held).to_numpy()
        pair = fit(data[~test], mode)
        h, l = pair.predict(data[test])
        head_pred[test], lean_pred[test] = h, l
        rows.append({"held_out": held, "t1": pair.head, "t2": pair.lean,
                     **attribute_scores(data.loc[test, "head_forward"], h, data.loc[test, "torso_lean"], l)})
    return pd.DataFrame(rows), head_pred, lean_pred
