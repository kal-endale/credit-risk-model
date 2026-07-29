"""RFM-based proxy target creation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


RFM_COLUMNS = [
    "recency",
    "frequency",
    "monetary_value",
]


def calculate_rfm(
    transactions: pd.DataFrame,
    reference_date: str | datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Calculate customer-level recency, frequency, and monetary value."""

    required_columns = {
        "CustomerId",
        "TransactionId",
        "TransactionStartTime",
        "Value",
    }

    missing_columns = required_columns.difference(
        transactions.columns
    )

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {sorted(missing_columns)}"
        )

    data = transactions.copy()

    data["TransactionStartTime"] = pd.to_datetime(
        data["TransactionStartTime"],
        errors="coerce",
        utc=True,
    )

    if data["TransactionStartTime"].isna().any():
        raise ValueError("Invalid transaction timestamps detected.")

    maximum_transaction_date = (
        data["TransactionStartTime"].max().normalize()
    )

    if reference_date is None:
        reference_timestamp = (
            maximum_transaction_date + pd.Timedelta(days=1)
        )
    else:
        reference_timestamp = pd.Timestamp(reference_date)

        if reference_timestamp.tzinfo is None:
            reference_timestamp = reference_timestamp.tz_localize(
                "UTC"
            )
        else:
            reference_timestamp = reference_timestamp.tz_convert(
                "UTC"
            )

        reference_timestamp = reference_timestamp.normalize()

    if reference_timestamp <= maximum_transaction_date:
        raise ValueError(
            "Reference date must be after the last transaction date."
        )

    customer_rfm = (
        data.groupby("CustomerId", as_index=False)
        .agg(
            last_transaction=(
                "TransactionStartTime",
                "max",
            ),
            frequency=("TransactionId", "count"),
            monetary_value=("Value", "sum"),
        )
    )

    customer_rfm["recency"] = (
        reference_timestamp
        - customer_rfm["last_transaction"].dt.normalize()
    ).dt.days

    customer_rfm = customer_rfm[
        [
            "CustomerId",
            "recency",
            "frequency",
            "monetary_value",
            "last_transaction",
        ]
    ]

    if (customer_rfm[RFM_COLUMNS] < 0).any().any():
        raise ValueError("RFM features cannot contain negative values.")

    return customer_rfm


def create_risk_proxy(
    rfm_data: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    StandardScaler,
    KMeans,
    int,
]:
    """Cluster RFM behavior and identify the high-risk cluster."""

    missing_columns = {
        "CustomerId",
        *RFM_COLUMNS,
    }.difference(rfm_data.columns)

    if missing_columns:
        raise ValueError(
            f"Required RFM columns missing: "
            f"{sorted(missing_columns)}"
        )

    if len(rfm_data) < n_clusters:
        raise ValueError(
            "The number of customers must be at least equal "
            "to the number of clusters."
        )

    result = rfm_data.copy()

    # Log transformation reduces the influence of extreme customers.
    log_rfm = np.log1p(result[RFM_COLUMNS])

    scaler = StandardScaler()
    scaled_rfm = scaler.fit_transform(log_rfm)

    scaled_columns = [
        "recency_scaled",
        "frequency_scaled",
        "monetary_scaled",
    ]

    result[scaled_columns] = scaled_rfm

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=20,
        algorithm="lloyd",
    )

    with threadpool_limits(limits=1):
        result["rfm_cluster"] = kmeans.fit_predict(scaled_rfm)

    cluster_profile = (
        result.groupby("rfm_cluster", as_index=False)
        .agg(
            customer_count=("CustomerId", "count"),
            mean_recency=("recency", "mean"),
            median_recency=("recency", "median"),
            mean_frequency=("frequency", "mean"),
            median_frequency=("frequency", "median"),
            mean_monetary_value=("monetary_value", "mean"),
            median_monetary_value=("monetary_value", "median"),
            mean_recency_scaled=("recency_scaled", "mean"),
            mean_frequency_scaled=("frequency_scaled", "mean"),
            mean_monetary_scaled=("monetary_scaled", "mean"),
        )
    )

    cluster_profile["customer_percentage"] = (
        cluster_profile["customer_count"]
        .div(len(result))
        .mul(100)
    )

    # Higher recency and lower frequency/monetary imply higher risk.
    cluster_profile["risk_score"] = (
        cluster_profile["mean_recency_scaled"]
        - cluster_profile["mean_frequency_scaled"]
        - cluster_profile["mean_monetary_scaled"]
    )

    high_risk_cluster = int(
        cluster_profile.loc[
            cluster_profile["risk_score"].idxmax(),
            "rfm_cluster",
        ]
    )

    cluster_profile["is_high_risk_cluster"] = (
        cluster_profile["rfm_cluster"] == high_risk_cluster
    ).astype(int)

    result["is_high_risk"] = (
        result["rfm_cluster"] == high_risk_cluster
    ).astype(int)

    cluster_profile = cluster_profile.sort_values(
        "risk_score",
        ascending=False,
    ).reset_index(drop=True)

    return (
        result,
        cluster_profile,
        scaler,
        kmeans,
        high_risk_cluster,
    )