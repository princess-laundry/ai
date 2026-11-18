"""임계값 격자 탐색: 골반→귀 선 기울기(axis) 하나의 기준값을 고릅니다.

앱은 `기울기 > T`(개인 기준이면 `기울기 − 기준 자세 기울기 > TΔ`)일 때 거북목(FHP)으로 판정해요
(2026-11-18 팀 결정). 정답도 바른 자세 / 거북목 두 가지예요.

- 기준: 거북목/바른 자세 Macro F1이 가장 높은 값
- 같은 점수가 여러 개면 그 범위의 가운데(두 자세 사이 여유가 가장 큰 값)

평가는 피험자 1명씩 빼고 나머지로 고른 임계값을 빠진 사람에게 적용(LOSO)해서,
'새 사람'에게 얼마나 통하는지를 봅니다. 최종 값은 전체 데이터로 다시 고릅니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluate import binary_scores

AXIS_GRID = np.round(np.arange(2.0, 45.0 + 1e-9, 0.5), 1)            # Thresholds 허용 범위 1~89
DELTA_AXIS_GRID = np.round(np.arange(1.0, 40.0 + 1e-9, 0.5), 1)


def feature_column(mode: str) -> str:
    return "axis_mean" if mode == "abs" else "d_axis"


@dataclass(frozen=True)
class AxisThreshold:
    value: float
    mode: str  # "abs" 또는 "delta"

    def predict(self, samples: pd.DataFrame) -> np.ndarray:
        """1 = 거북목, 0 = 바른 자세."""
        return (samples[feature_column(self.mode)].to_numpy() > self.value).astype(int)


def usable(samples: pd.DataFrame, mode: str) -> pd.DataFrame:
    return samples.dropna(subset=[feature_column(mode)])


def fit(samples: pd.DataFrame, mode: str = "abs") -> AxisThreshold:
    data = usable(samples, mode)
    grid = AXIS_GRID if mode == "abs" else DELTA_AXIS_GRID
    values, truth = data[feature_column(mode)].to_numpy(float), data["fhp"].to_numpy().astype(int)
    scores = np.array([binary_scores(truth, (values > g).astype(int))["f1"] for g in grid])
    tied = grid[scores >= scores.max() - 1e-12]
    return AxisThreshold(value=float(tied[len(tied) // 2]), mode=mode)


def leave_one_subject_out(samples: pd.DataFrame, mode: str = "abs") -> tuple[pd.DataFrame, np.ndarray, pd.Index]:
    """반환: (fold별 점수표, LOSO 예측(1=거북목), 예측에 쓴 샘플 행 번호)."""
    data = usable(samples, mode)
    pred = np.zeros(len(data), dtype=int)
    rows = []
    for held in sorted(data["subject"].unique()):
        test = (data["subject"] == held).to_numpy()
        threshold = fit(data[~test], mode)
        pred[test] = threshold.predict(data[test])
        rows.append({"held_out": held, "threshold": threshold.value, **binary_scores(data.loc[test, "fhp"], pred[test])})
    return pd.DataFrame(rows), pred, data.index
