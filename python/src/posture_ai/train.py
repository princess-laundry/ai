"""SVM·Random Forest 학습과 LOSO 비교 (발표 3-1의 4가지 조건).

- 분류 대상: 바른 자세(0) / 거북목(1). 2026-11-18 팀 결정으로 상체 기울임도 거북목이에요.
  점수는 두 자세의 Macro F1입니다.
- SVM: 표준화는 학습 데이터로만(파이프라인 안), RBF 커널, class_weight=balanced.
- RF: class_weight=balanced, 트리 수·깊이는 학습 피험자 안에서 다시 LOSO로 고릅니다(개발군 선택).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .evaluate import binary_scores
from .windows import ABS_COLUMNS, DELTA_COLUMNS

SVM_GRID = [{"C": c} for c in (0.3, 1.0, 3.0, 10.0)]
RF_GRID = [{"n_estimators": n, "max_depth": d} for n, d in product((100, 300), (3, 5, None))]


def columns(mode: str) -> list[str]:
    return ABS_COLUMNS if mode == "abs" else DELTA_COLUMNS


def make_model(kind: str, params: dict, seed: int = 0):
    if kind == "svm":
        return Pipeline([("scale", StandardScaler()),
                         ("svm", SVC(kernel="rbf", gamma="scale", class_weight="balanced", random_state=seed, **params))])
    if kind == "rf":
        return RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=1, **params)
    raise ValueError(kind)


def _score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return binary_scores(y_true, y_pred)["f1"]


def select_params(kind: str, data: pd.DataFrame, mode: str, seed: int = 0) -> dict:
    """학습 피험자들 안에서 다시 1명씩 빼며 하이퍼파라미터를 고릅니다. 피험자가 1명이면 기본값."""
    grid = SVM_GRID if kind == "svm" else RF_GRID
    subjects = sorted(data["subject"].unique())
    if len(subjects) < 2:
        return grid[0]
    X, y = data[columns(mode)].to_numpy(), data["fhp"].to_numpy().astype(int)
    best, best_score = grid[0], -1.0
    for params in grid:
        scores = []
        for held in subjects:
            test = (data["subject"] == held).to_numpy()
            if len(set(y[~test])) < 2:
                continue
            model = make_model(kind, params, seed).fit(X[~test], y[~test])
            scores.append(_score(y[test], model.predict(X[test])))
        score = float(np.mean(scores)) if scores else -1.0
        if score > best_score + 1e-12:
            best, best_score = params, score
    return best


@dataclass
class LosoResult:
    folds: pd.DataFrame
    pred: np.ndarray  # 1=거북목
    rows: pd.Index


def leave_one_subject_out(samples: pd.DataFrame, kind: str, mode: str, seed: int = 0) -> LosoResult:
    data = samples.dropna(subset=columns(mode))
    X, y = data[columns(mode)].to_numpy(), data["fhp"].to_numpy().astype(int)
    pred = np.zeros(len(data), dtype=int)
    rows = []
    for held in sorted(data["subject"].unique()):
        test = (data["subject"] == held).to_numpy()
        params = select_params(kind, data[~test], mode, seed)
        model = make_model(kind, params, seed).fit(X[~test], y[~test])
        pred[test] = model.predict(X[test])
        rows.append({"held_out": held, "params": params, **binary_scores(y[test], pred[test])})
    return LosoResult(pd.DataFrame(rows), pred, data.index)


def fit_final(samples: pd.DataFrame, kind: str, mode: str, seed: int = 0):
    """전체 데이터로 최종 모델을 학습합니다(앱 탑재 후보)."""
    data = samples.dropna(subset=columns(mode))
    params = select_params(kind, data, mode, seed)
    return make_model(kind, params, seed).fit(data[columns(mode)].to_numpy(), data["fhp"].to_numpy().astype(int)), params
