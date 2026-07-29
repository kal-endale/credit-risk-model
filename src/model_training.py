"""Leakage-safe model preparation and evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


TARGET_COLUMN = "is_high_risk"

MODEL_NUMERIC_FEATURES = [
    "average_signed_amount",
    "std_signed_amount",
    "minimum_signed_amount",
    "maximum_signed_amount",
    "average_transaction_value",
    "std_transaction_value",
    "minimum_transaction_value",
    "maximum_transaction_value",
    "debit_transaction_count",
    "debit_ratio",
    "unique_products",
    "unique_product_categories",
    "unique_channels",
    "unique_providers",
    "active_days",
    "active_span_days",
    "average_transaction_hour",
    "weekend_transaction_ratio",
    "night_transaction_ratio",
]

MODEL_CATEGORICAL_FEATURES = [
    "dominant_product_category",
    "dominant_channel",
    "dominant_provider",
    "dominant_pricing_strategy",
]

MODEL_FEATURES = (
    MODEL_NUMERIC_FEATURES
    + MODEL_CATEGORICAL_FEATURES
)

EXCLUDED_PROXY_COLUMNS = [
    "CustomerId",
    "transaction_count",
    "signed_total_amount",
    "total_transaction_value",
    "recency",
    "frequency",
    "monetary_value",
    "recency_scaled",
    "frequency_scaled",
    "monetary_scaled",
    "rfm_cluster",
    "is_high_risk",
    "first_transaction",
    "last_transaction_x",
    "last_transaction_y",
]


def prepare_model_data(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Select leakage-safe predictors and binary target."""

    required_columns = set(MODEL_FEATURES + [TARGET_COLUMN])
    missing_columns = required_columns.difference(dataset.columns)

    if missing_columns:
        raise ValueError(
            f"Required modelling columns missing: "
            f"{sorted(missing_columns)}"
        )

    features = dataset[MODEL_FEATURES].copy()
    target = dataset[TARGET_COLUMN].astype(int).copy()

    if set(target.unique()) != {0, 1}:
        raise ValueError(
            "The target must contain both binary classes."
        )

    return features, target


def evaluate_classifier(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calculate required binary classification metrics."""

    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(
            accuracy_score(target, predictions)
        ),
        "precision": float(
            precision_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(target, probabilities)
        ),
    }