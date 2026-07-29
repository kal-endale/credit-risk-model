"""Tests for model-data preparation and evaluation."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_processing import build_preprocessor
from src.model_training import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    evaluate_classifier,
    prepare_model_data,
)


def create_sample_modeling_data(
    row_count: int = 30,
) -> pd.DataFrame:
    """Create a complete synthetic modelling dataset."""

    target = np.array(
        [0, 1] * (row_count // 2)
    )

    data = {}

    for index, feature in enumerate(
        MODEL_NUMERIC_FEATURES
    ):
        data[feature] = (
            np.arange(row_count, dtype=float)
            + index
            + target * 0.5
        )

    categories = np.where(
        target == 1,
        "category_b",
        "category_a",
    )

    for feature in MODEL_CATEGORICAL_FEATURES:
        data[feature] = categories

    data["CustomerId"] = [
        f"C{index}" for index in range(row_count)
    ]
    data["recency"] = target * 20 + 1
    data["frequency"] = 20 - target * 15
    data["monetary_value"] = 1_000 - target * 800
    data["rfm_cluster"] = target
    data["is_high_risk"] = target

    return pd.DataFrame(data)


def test_prepare_model_data_excludes_proxy_variables() -> None:
    dataset = create_sample_modeling_data()

    features, target = prepare_model_data(dataset)

    assert list(features.columns) == MODEL_FEATURES
    assert "recency" not in features.columns
    assert "frequency" not in features.columns
    assert "monetary_value" not in features.columns
    assert "rfm_cluster" not in features.columns
    assert target.name == "is_high_risk"


def test_complete_pipeline_can_fit_and_predict() -> None:
    dataset = create_sample_modeling_data()
    features, target = prepare_model_data(dataset)

    preprocessor = build_preprocessor(
        numeric_features=MODEL_NUMERIC_FEATURES,
        categorical_features=MODEL_CATEGORICAL_FEATURES,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(features, target)
    probabilities = pipeline.predict_proba(features)[:, 1]

    assert len(probabilities) == len(dataset)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_evaluation_returns_required_metrics() -> None:
    dataset = create_sample_modeling_data()
    features, target = prepare_model_data(dataset)

    preprocessor = build_preprocessor(
        numeric_features=MODEL_NUMERIC_FEATURES,
        categorical_features=MODEL_CATEGORICAL_FEATURES,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(features, target)
    metrics = evaluate_classifier(
        pipeline,
        features,
        target,
    )

    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
    }

    assert all(
        0 <= value <= 1
        for value in metrics.values()
    )