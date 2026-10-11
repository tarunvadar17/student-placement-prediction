"""Schema validation, dataset loading, and feature preprocessing."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Raise ValueError if required columns are missing. Returns the input unchanged."""
    required = set(ALL_FEATURES)
    if require_target:
        required.add(TARGET)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    return df


def load_dataset(path: Path | str) -> pd.DataFrame:
    """Load and validate a training dataset, dropping rows without a target label."""
    df = pd.read_csv(path)
    validate_schema(df, require_target=True)
    df = df.dropna(subset=[TARGET]).copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build the feature transformer shared by training and inference."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
