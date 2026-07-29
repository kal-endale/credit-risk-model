"""Tests for RFM calculation and proxy-target creation."""

import pandas as pd

from src.proxy_target import calculate_rfm, create_risk_proxy


def test_calculate_rfm_values() -> None:
    transactions = pd.DataFrame(
        {
            "CustomerId": ["C1", "C1", "C2"],
            "TransactionId": ["T1", "T2", "T3"],
            "TransactionStartTime": [
                "2019-01-09T10:00:00Z",
                "2019-01-10T10:00:00Z",
                "2019-01-01T10:00:00Z",
            ],
            "Value": [10.0, 20.0, 5.0],
        }
    )

    rfm = calculate_rfm(
        transactions,
        reference_date="2019-01-11",
    )

    customer_one = rfm.loc[
        rfm["CustomerId"] == "C1"
    ].iloc[0]

    customer_two = rfm.loc[
        rfm["CustomerId"] == "C2"
    ].iloc[0]

    assert customer_one["recency"] == 1
    assert customer_one["frequency"] == 2
    assert customer_one["monetary_value"] == 30.0

    assert customer_two["recency"] == 10
    assert customer_two["frequency"] == 1
    assert customer_two["monetary_value"] == 5.0


def create_sample_rfm() -> pd.DataFrame:
    """Create three distinct behavioral customer groups."""

    return pd.DataFrame(
        {
            "CustomerId": [
                "C1",
                "C2",
                "C3",
                "C4",
                "C5",
                "C6",
                "C7",
                "C8",
                "C9",
            ],
            "recency": [1, 2, 3, 12, 15, 18, 45, 55, 65],
            "frequency": [30, 25, 20, 12, 10, 8, 1, 2, 1],
            "monetary_value": [
                3000,
                2500,
                2000,
                900,
                700,
                600,
                40,
                60,
                30,
            ],
            "last_transaction": pd.to_datetime(
                [
                    "2019-02-12",
                    "2019-02-11",
                    "2019-02-10",
                    "2019-02-01",
                    "2019-01-29",
                    "2019-01-26",
                    "2018-12-30",
                    "2018-12-20",
                    "2018-12-10",
                ],
                utc=True,
            ),
        }
    )


def test_proxy_creates_three_clusters_and_binary_target() -> None:
    labeled, profile, _, _, high_cluster = (
        create_risk_proxy(create_sample_rfm())
    )

    assert labeled["rfm_cluster"].nunique() == 3
    assert set(labeled["is_high_risk"].unique()) == {0, 1}
    assert profile["rfm_cluster"].nunique() == 3
    assert high_cluster in labeled["rfm_cluster"].unique()


def test_high_risk_group_has_riskier_behavior() -> None:
    labeled, _, _, _, _ = create_risk_proxy(
        create_sample_rfm()
    )

    high_risk = labeled.loc[labeled["is_high_risk"] == 1]
    lower_risk = labeled.loc[labeled["is_high_risk"] == 0]

    assert high_risk["recency"].mean() > lower_risk["recency"].mean()
    assert high_risk["frequency"].mean() < lower_risk["frequency"].mean()
    assert (
        high_risk["monetary_value"].mean()
        < lower_risk["monetary_value"].mean()
    )


def test_proxy_creation_is_reproducible() -> None:
    first, _, _, _, _ = create_risk_proxy(
        create_sample_rfm(),
        random_state=42,
    )
    second, _, _, _, _ = create_risk_proxy(
        create_sample_rfm(),
        random_state=42,
    )

    assert first["rfm_cluster"].equals(second["rfm_cluster"])
    assert first["is_high_risk"].equals(second["is_high_risk"])