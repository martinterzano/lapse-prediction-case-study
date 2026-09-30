"""
Versioned preprocessing state — the pattern.

The single most important integrity guarantee of any batch-scoring pipeline is
that the preprocessing applied at scoring time is exactly the preprocessing
applied at training time. Not "the same code" — the same *state*: the same
winsorization caps, the same imputation medians, the same category mappings.

This module illustrates the pattern used in the lapse case study:

    fit_time  → PreprocessingDict.fit(train_df) → saves .json to model-artefacts store
    score_time → PreprocessingDict.load(path)  → applies to production rows

The dictionary itself is a plain JSON payload. It is stored alongside the model
artefacts (Cloud Storage, S3, MLFlow artefact store — the store doesn't matter,
the discipline does) and versioned by whatever mechanism versions the model.

The point is NOT that this class is the right abstraction for every project —
plenty of teams use scikit-learn pipelines, MLflow custom flavours, or feast-style
feature stores instead. The point is that ANY of those work as long as you commit
to loading the same object at fit time and at score time. Ad-hoc "well, the median
is roughly 34.5" numbers copy-pasted into scoring scripts is where accidents happen.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class PreprocessingDict:
    """Versioned preprocessing state for a tabular model.

    Holds four kinds of state, all of which are learned from the training set
    and applied identically at scoring time:

    - numeric_medians    → for median imputation of numeric features
    - categorical_modes  → for mode imputation of categorical features
    - p99_caps           → for winsorization at the 99th percentile
    - category_mappings  → for stable string→int encoding of categoricals

    Extend or replace with what your model needs. The invariant to preserve is
    that fit(...) is the only method that mutates the object, and every subsequent
    apply(...) call is deterministic.
    """

    numeric_medians: dict[str, float] = field(default_factory=dict)
    categorical_modes: dict[str, str] = field(default_factory=dict)
    p99_caps: dict[str, float] = field(default_factory=dict)
    category_mappings: dict[str, dict[str, int]] = field(default_factory=dict)

    # ------------------------------------------------------------------ fit

    def fit(
        self,
        df: pd.DataFrame,
        numeric_cols: list[str],
        categorical_cols: list[str],
        p99_cols: list[str] | None = None,
    ) -> "PreprocessingDict":
        """Learn all state from a training DataFrame."""
        p99_cols = p99_cols or []

        self.numeric_medians = {
            c: float(df[c].median()) for c in numeric_cols
        }

        self.categorical_modes = {
            c: str(df[c].mode(dropna=True).iloc[0]) if df[c].notna().any() else "__missing__"
            for c in categorical_cols
        }

        self.p99_caps = {
            c: float(df[c].quantile(0.99)) for c in p99_cols
        }

        # Fixed alphabetical mapping so that adding a new category later gets a new
        # index without shifting existing ones as long as the fit is not re-run.
        self.category_mappings = {
            c: {v: i for i, v in enumerate(sorted(df[c].dropna().unique().astype(str).tolist()))}
            for c in categorical_cols
        }

        return self

    # ---------------------------------------------------------------- apply

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the stored state to any DataFrame (train, validation, or production)."""
        out = df.copy()

        for col, median in self.numeric_medians.items():
            if col in out.columns:
                out[col] = out[col].fillna(median)

        for col, mode in self.categorical_modes.items():
            if col in out.columns:
                out[col] = out[col].fillna(mode)

        for col, cap in self.p99_caps.items():
            if col in out.columns:
                out[col] = out[col].clip(upper=cap)

        for col, mapping in self.category_mappings.items():
            if col in out.columns:
                # Unknown categories at scoring time collapse to -1. Log-and-monitor
                # this rate in production; if it grows, refit.
                out[col] = out[col].astype(str).map(mapping).fillna(-1).astype(int)

        return out

    # ----------------------------------------------------- persist / load

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @classmethod
    def load(cls, path: str | Path) -> "PreprocessingDict":
        data: dict[str, Any] = json.loads(Path(path).read_text())
        return cls(**data)


# --------------------------------------------------------------------- usage

if __name__ == "__main__":
    # Toy example — the pattern is what matters, not the numbers.
    train_df = pd.DataFrame(
        {
            "age": [25, 35, 45, 55, 65, None, 999],
            "premium": [100, 200, 150, 300, 250, 175, 220],
            "channel": ["web", "agent", "web", "agent", None, "web", "phone"],
        }
    )

    pp = PreprocessingDict().fit(
        train_df,
        numeric_cols=["age", "premium"],
        categorical_cols=["channel"],
        p99_cols=["age", "premium"],
    )

    # In production this file lives next to the model artefact, loaded at scoring time.
    pp.save("artefacts/preprocessing_dict.json")

    scoring_df = pd.DataFrame(
        {
            "age": [30, None, 500],
            "premium": [None, 400, 180],
            "channel": ["web", "unknown_new_channel", None],
        }
    )

    print(PreprocessingDict.load("artefacts/preprocessing_dict.json").apply(scoring_df))
