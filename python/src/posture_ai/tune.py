"""임계값 격자 탐색: 몸 기울기(axis) 기준과 고개 숙임(pitch) 기준을 함께 고릅니다.

앱은 `기울기 > T 또는 고개 숙임 > P`일 때 거북목(FHP)으로 판정해요(2026-11-18 팀 결정).
개인 기준이면 각각 기준 자세 대비 증가량(Δ)으로 비교해요. 정답은 바른 자세 / 거북목 두 가지예요.

- 기준: 거북목/바른 자세 Macro F1이 가장 높은 조합
- 같은 점수가 여러 개면: 맞게 판정한 샘플이 임계값에서 가장 멀리 떨어진 조합(두 자세 사이 여유가 가장 큰 값)
  → 거리는 특징값 표준편차로 나눠서 기울기·고개 숙임을 같은 척도로 비교해요.
- 코가 안 보여 고개 숙임 값이 없는 창은 기울기로만 판정해요(앱과 같음).

평가는 피험자 1명씩 빼고 나머지로 고른 임계값을 빠진 사람에게 적용(LOSO)해서,
'새 사람'에게 얼마나 통하는지를 봅니다. 최종 값은 전체 데이터로 다시 고릅니다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evaluate import binary_scores

AXIS_GRID = np.round(np.arange(2.0, 45.0 + 1e-9, 0.5), 1)            # Thresholds 허용 범위 1~89
PITCH_GRID = np.round(np.arange(10.0, 85.0 + 1e-9, 1.0), 1)
DELTA_AXIS_GRID = np.round(np.arange(1.0, 40.0 + 1e-9, 0.5), 1)
DELTA_PITCH_GRID = np.round(np.arange(2.0, 60.0 + 1e-9, 1.0), 1)


def feature_columns(mode: str) -> tuple[str, str]:
    return ("axis_mean", "pitch_mean") if mode == "abs" else ("d_axis", "d_pitch")


@dataclass(frozen=True)
class ThresholdPair:
    axis: float
    pitch: float
    mode: str  # "abs" 또는 "delta"

    def predict(self, samples: pd.DataFrame) -> np.ndarray:
        """1 = 거북목(기울기 > T 또는 고개 숙임 > P), 0 = 바른 자세. 고개 숙임 값이 없으면 기울기만 봐요."""
        axis_col, pitch_col = feature_columns(self.mode)
        axis = samples[axis_col].to_numpy(float)
        pitch = samples[pitch_col].to_numpy(float)
        with np.errstate(invalid="ignore"):
            return ((axis > self.axis) | (pitch > self.pitch)).astype(int)


def usable(samples: pd.DataFrame, mode: str) -> pd.DataFrame:
    return samples.dropna(subset=[feature_columns(mode)[0]])


def _scale(values: np.ndarray) -> float:
    s = float(np.nanstd(values)) if np.isfinite(values).any() else 0.0
    return s if s > 1e-9 else 1.0


def fit(samples: pd.DataFrame, mode: str = "abs") -> ThresholdPair:
    data = usable(samples, mode)
    axis_grid, pitch_grid = (AXIS_GRID, PITCH_GRID) if mode == "abs" else (DELTA_AXIS_GRID, DELTA_PITCH_GRID)
    axis_col, pitch_col = feature_columns(mode)
    axis, pitch = data[axis_col].to_numpy(float), data[pitch_col].to_numpy(float)
    truth = data["fhp"].to_numpy().astype(bool)
    pitch_d = (pitch[None, :] - pitch_grid[:, None]) / _scale(pitch)       # (b, n), > 0이면 고개 숙임 기준 초과
    axis_scale = _scale(axis)

    f1 = np.empty((len(axis_grid), len(pitch_grid)))
    margin = np.empty_like(f1)
    for a, t in enumerate(axis_grid):
        strength = np.fmax((axis - t)[None, :] / axis_scale, pitch_d)      # NaN(코 없음)은 무시
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
    return ThresholdPair(axis=float(axis_grid[a]), pitch=float(pitch_grid[b]), mode=mode)


def leave_one_subject_out(samples: pd.DataFrame, mode: str = "abs") -> tuple[pd.DataFrame, np.ndarray, pd.Index]:
    """반환: (fold별 점수표, LOSO 예측(1=거북목), 예측에 쓴 샘플 행 번호)."""
    data = usable(samples, mode)
    pred = np.zeros(len(data), dtype=int)
    rows = []
    for held in sorted(data["subject"].unique()):
        test = (data["subject"] == held).to_numpy()
        pair = fit(data[~test], mode)
        pred[test] = pair.predict(data[test])
        rows.append({"held_out": held, "axis": pair.axis, "pitch": pair.pitch,
                     **binary_scores(data.loc[test, "fhp"], pred[test])})
    return pd.DataFrame(rows), pred, data.index
