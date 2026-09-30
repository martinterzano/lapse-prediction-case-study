"""
Isotonic calibration wrapper + decile lift analysis — the pattern.

Two ideas illustrated together because they live at the same operational seam:

1. A ranking model (LightGBM in this project, but the pattern is model-agnostic)
   is separated from its probability calibration. The base model produces a score;
   an independently trained CalibratedClassifierCV wraps it and produces a
   well-calibrated probability. Both are persisted as separate artefacts.

2. Business consumption of the calibrated probability is via decile bins, not raw
   thresholds. This module computes the decile ranking, capture rate, and lift
   over random selection — the numbers a retention team needs to decide how deep
   into the ranked list to go.

Neither of these is novel. Both are shown here as a reference for the discipline
of keeping ranking and calibration as separate artefacts, and of exposing model
output to the business as decile bands with a lift interpretation, not as raw
probabilities.

The example at the bottom uses synthetic data with a deliberately over-confident
base classifier — the shape of the isotonic correction is visible in the printed
reliability comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score


# --------------------------------------------------------------------- protocol


class RankingModel(Protocol):
    """Anything that can produce a probability-like score for the positive class."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RankingModel": ...
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...


# ---------------------------------------------------------------- calibration


def fit_isotonic_calibration(
    base_model: RankingModel,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
) -> CalibratedClassifierCV:
    """
    Wrap a ranking model with isotonic probability calibration.

    The calibration is fit with cross-validation so that no fold sees the same
    data for both base training and calibration fitting. This is what
    CalibratedClassifierCV does under the hood — spelled out here for clarity.

    Why isotonic over Platt:
    - Larger validation sets (>1k positive samples): isotonic's flexibility earns
      its extra parameters.
    - Non-sigmoid miscalibration patterns (piecewise, plateaus, spikes): isotonic
      reproduces the shape; Platt would flatten it into one sigmoid curve.

    Persist the returned object alongside the base model, not on top of it.
    Ranking model and calibration evolve independently.
    """
    calibrated = CalibratedClassifierCV(
        estimator=base_model,
        method="isotonic",
        cv=cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def save_calibrated_model(model: CalibratedClassifierCV, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


# -------------------------------------------------------- decile lift analysis


@dataclass
class DecileReport:
    """Per-decile capture rate and lift against random selection."""

    df: pd.DataFrame  # one row per decile, sorted with decile 1 = highest risk

    def top_k_recall(self, k: int) -> float:
        """Recall achieved by targeting the top-k deciles."""
        return float(self.df.head(k)["captured_share_cum"].iloc[-1])

    def print_summary(self) -> None:
        print(self.df.to_string(index=False))


def compute_decile_report(
    y_true: np.ndarray | pd.Series,
    y_proba: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> DecileReport:
    """
    Rank a scored dataset into `n_bins` bands and compute per-band statistics
    a retention team can act on.

    Columns produced (one row per decile, decile 1 = highest risk):

    - `n`                    number of policies in the decile
    - `n_positive`           number of true positives (lapses) in the decile
    - `lapse_rate`           positive rate in this decile
    - `captured_share`       share of *all* positives that fall in this decile
    - `captured_share_cum`   cumulative share of positives from decile 1 down
    - `lift`                 lapse_rate / overall_positive_rate — how many times
                             better than random this decile is
    """
    df = pd.DataFrame({"y_true": np.asarray(y_true), "y_proba": np.asarray(y_proba)})

    # decile 1 = highest risk (lowest rank number, highest score)
    df["rank"] = df["y_proba"].rank(method="first", ascending=False)
    df["decile"] = pd.qcut(df["rank"], q=n_bins, labels=range(1, n_bins + 1))

    total_positives = df["y_true"].sum()
    overall_rate = df["y_true"].mean()

    report = (
        df.groupby("decile", observed=True)
        .agg(n=("y_true", "size"), n_positive=("y_true", "sum"))
        .reset_index()
        .sort_values("decile")
    )

    report["lapse_rate"] = report["n_positive"] / report["n"]
    report["captured_share"] = report["n_positive"] / total_positives
    report["captured_share_cum"] = report["captured_share"].cumsum()
    report["lift"] = report["lapse_rate"] / overall_rate

    return DecileReport(df=report)


# ---------------------------------------------------------------------- usage

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    # Synthetic problem — the pattern is what matters, not the numbers.
    X, y = make_classification(
        n_samples=20_000,
        n_features=20,
        weights=[0.97, 0.03],  # 3% prevalence, similar to the case study
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="lapse")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

    base = LogisticRegression(max_iter=1000)
    base.fit(X_train, y_train)

    calibrated = fit_isotonic_calibration(base, X_train, y_train, cv=5)

    p_base = base.predict_proba(X_test)[:, 1]
    p_cal = calibrated.predict_proba(X_test)[:, 1]

    print(f"AUC (base)       : {roc_auc_score(y_test, p_base):.4f}")
    print(f"AUC (calibrated) : {roc_auc_score(y_test, p_cal):.4f}")
    print("(AUC is invariant to monotonic transforms — should be nearly identical.)")
    print()

    print("Decile report on calibrated probabilities:")
    report = compute_decile_report(y_test.values, p_cal)
    report.print_summary()
    print()
    print(f"Top-1 decile recall : {report.top_k_recall(1):.1%}")
    print(f"Top-2 decile recall : {report.top_k_recall(2):.1%}")
    print(f"Top-3 decile recall : {report.top_k_recall(3):.1%}")
