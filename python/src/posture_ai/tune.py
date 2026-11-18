"""임계값 T₁(머리)·T₂(몸통) 격자 탐색.

앱은 `머리 > T₁ 또는 몸통 > T₂`이면 거북목(FHP)으로 판정해요(2026-11-18 팀 결정: 상체 기울임도 거북목).
정답도 바른 자세 / 거북목 두 가지라서, 두 임계값을 **함께** 골라 이 OR 판정이 정답과 가장 잘 맞게 해요.

- 기준: 거북목/바른 자세 Macro F1이 가장 높은 조합
- 같은 점수가 여러 개면: 맞게 판정한 샘플이 임계값에서 가장 멀리 떨어진 조합(두 자세 사이 여유가 가장 큰 값)
  → 거리는 특징값 표준편차로 나눠서 머리·몸통을 같은 척도로 비교해요.

평가는 피험자 1명씩 빼고 나머지로 고른 임계값을 빠진 사람에게 적용(LOSO)해서,
'새 사람'에게 얼마나 통하는지를 봅니다. 최종 값은 전체 데이터로 다시 고릅니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluate import binary_scores

HEAD_GRID = np.round(np.arange(0.05, 0.80 + 1e-9, 0.01), 2)          # Thresholds 허용 범위 0.05~2.0
LEAN_GRID = np.round(np.arange(2.0, 45.0 + 1e-9, 0.5), 1)            # Thresholds 허용 범위 1~89
DELTA_HEAD_GRID = np.round(np.arange(0.02, 0.60 + 1e-9, 0.01), 2)
DELTA_LEAN_GRID = np.round(np.arange(1.0, 40.0 + 1e-9, 0.5), 1)


def feature_columns(mode: str) -> tuple[str, str]:
    return ("head_mean", "torso_mean") if mode == "abs" else ("d_head", "d_torso")


@dataclass(frozen=True)
class ThresholdPair:
    head: float
    lean: float
    mode: str  # "abs" 또는 "delta"

    def columns(self) -> tuple[str, str]:
        return feature_columns(self.mode)

    def predict(self, samples: pd.DataFrame) -> np.ndarray:
        """1 = 거북목(머리 > T₁ 또는 몸통 > T₂), 0 = 바른 자세."""
        head_col, lean_col = self.columns()
        return ((samples[head_col].to_numpy() > self.head) | (samples[lean_col].to_numpy() > self.lean)).astype(int)


def usable(samples: pd.DataFrame, mode: str) -> pd.DataFrame:
    return samples.dropna(subset=list(feature_columns(mode)))


def _scale(values: np.ndarray) -> float:
    s = float(np.std(values))
    return s if s > 1e-9 else 1.0


def fit(samples: pd.DataFrame, mode: str = "abs") -> ThresholdPair:
    data = usable(samples, mode)
    head_grid, lean_grid = (HEAD_GRID, LEAN_GRID) if mode == "abs" else (DELTA_HEAD_GRID, DELTA_LEAN_GRID)
    head_col, lean_col = feature_columns(mode)
    head, lean = data[head_col].to_numpy(float), data[lean_col].to_numpy(float)
    truth = data["fhp"].to_numpy().astype(bool)
    lean_d = (lean[None, :] - lean_grid[:, None]) / _scale(lean)          # (b, n), > 0이면 몸통 기준 초과
    head_scale = _scale(head)

    f1 = np.empty((len(head_grid), len(lean_grid)))
    margin = np.empty_like(f1)
    for a, t1 in enumerate(head_grid):
        strength = np.maximum((head - t1)[None, :] / head_scale, lean_d)   # (b, n), > 0이면 거북목으로 판정
        pred = strength > 0
        tp = (pred & truth).sum(1); fp = (pred & ~truth).sum(1)
        fn = (~pred & truth).sum(1); tn = (~pred & ~truth).sum(1)
        f_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
        f_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1), 0.0)
        f1[a] = (f_pos + f_neg) / 2
        signed = np.where(truth[None, :], strength, -strength)          # > 0이면 맞게 판정
        margin[a] = np.where(signed > 0, signed, np.inf).min(1)

    tied = f1 >= f1.max() - 1e-12
    best_margin = np.where(tied, margin, -np.inf)
    cells = np.argwhere(best_margin >= best_margin.max() - 1e-9)
    a, b = cells[len(cells) // 2]
    return ThresholdPair(head=float(head_grid[a]), lean=float(lean_grid[b]), mode=mode)


def leave_one_subject_out(samples: pd.DataFrame, mode: str = "abs") -> tuple[pd.DataFrame, np.ndarray, pd.Index]:
    """반환: (fold별 점수표, LOSO 예측(1=거북목), 예측에 쓴 샘플 행 번호)."""
    data = usable(samples, mode)
    pred = np.zeros(len(data), dtype=int)
    rows = []
    for held in sorted(data["subject"].unique()):
        test = (data["subject"] == held).to_numpy()
        pair = fit(data[~test], mode)
        pred[test] = pair.predict(data[test])
        rows.append({"held_out": held, "t1": pair.head, "t2": pair.lean, **binary_scores(data.loc[test, "fhp"], pred[test])})
    return pd.DataFrame(rows), pred, data.index
