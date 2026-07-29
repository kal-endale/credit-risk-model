"""Tests for customer feature engineering and preprocessing."""

import numpy as np
import pandas as pd

from src.data_processing import (
    WeightOfEvidenceEncoder,
    build_customer_features,
    build_preprocessor,
)


def create_sample_transactions() -> pd.DataFrame:
    """Create deterministic synthetic transactions."""

    return pd.DataFrame(
        {
            "TransactionId": ["T1", "T2", "T3", "T4"],
            "CustomerId": ["C1", "C1", "C2", "C2"],
            "ProductId": ["P1", "P2", "P1", "P1"],
            "ProductCategory": [
                "airtime",
                "financial_services",
                "airtime",
                "airtime",
            ],
            "ChannelId": [
                "web",
                "mobile",
                "mobile",
                "mobile",
            ],
            "ProviderId": ["A", "B", "A", "A"],
            "Amount": [100.0, -40.0, 200.0, 50.0],
            "Value": [100.0, 40.0, 200.0, 50.0],
            "TransactionStartTime": [
                "2019-01-01T10:00:00Z",
                "2019-01-02T02:00:00Z",
                "2019-01-03T12:00:00Z",
                "2019-01-05T14:00:00Z",
            ],
            "PricingStrategy": [1, 2, 1, 1],
        }
    )


def test_customer_feature_aggregation() -> None:
    features = build_customer_features(
        create_sample_transactions()
    )

    assert len(features) == 2
    assert features["CustomerId"].is_unique

    customer_one = features.loc[
        features["CustomerId"] == "C1"
    ].iloc[0]

    assert customer_one["transaction_count"] == 2
    assert customer_one["signed_total_amount"] == 60.0
    assert customer_one["total_transaction_value"] == 140.0
    assert customer_one["debit_transaction_count"] == 1
    assert customer_one["debit_ratio"] == 0.5


def test_customer_features_have_no_missing_values() -> None:
    features = build_customer_features(
        create_sample_transactions()
    )

    assert features.isna().sum().sum() == 0


def test_woe_encoder_returns_finite_values() -> None:
    X = pd.DataFrame(
        {
            "category": [
                "a",
                "a",
                "b",
                "b",
                "c",
                "c",
            ]
        }
    )
    y = pd.Series([0, 0, 0, 1, 1, 1])

    encoder = WeightOfEvidenceEncoder()
    encoded = encoder.fit_transform(X, y)

    unseen = encoder.transform(
        pd.DataFrame({"category": ["unknown"]})
    )

    assert encoded.shape == (6, 1)
    assert np.isfinite(encoded).all()
    assert unseen[0, 0] == 0.0


def test_preprocessor_combines_numeric_and_categorical() -> None:
    X = pd.DataFrame(
        {
            "numeric_feature": [1.0, 2.0, np.nan, 4.0],
            "category_feature": ["a", "a", "b", "b"],
        }
    )
    y = pd.Series([0, 0, 1, 1])

    preprocessor = build_preprocessor(
        numeric_features=["numeric_feature"],
        categorical_features=["category_feature"],
    )

    transformed = preprocessor.fit_transform(X, y)

    assert transformed.shape == (4, 2)
    assert np.isfinite(transformed).all()