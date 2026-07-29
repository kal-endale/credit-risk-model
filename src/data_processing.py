"""Data loading, customer aggregation, and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


REQUIRED_TRANSACTION_COLUMNS = {
    "TransactionId",
    "CustomerId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "ProviderId",
    "Amount",
    "Value",
    "TransactionStartTime",
    "PricingStrategy",
}

CANDIDATE_NUMERIC_FEATURES = [
    "transaction_count",
    "signed_total_amount",
    "average_signed_amount",
    "std_signed_amount",
    "minimum_signed_amount",
    "maximum_signed_amount",
    "total_transaction_value",
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

CANDIDATE_CATEGORICAL_FEATURES = [
    "dominant_product_category",
    "dominant_channel",
    "dominant_provider",
    "dominant_pricing_strategy",
]


def load_transaction_data(path: str | Path) -> pd.DataFrame:
    """Load and validate the raw transaction dataset."""

    data_path = Path(path)

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    data = pd.read_csv(data_path)

    missing_columns = REQUIRED_TRANSACTION_COLUMNS.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            f"Required columns missing: {sorted(missing_columns)}"
        )

    data["TransactionStartTime"] = pd.to_datetime(
        data["TransactionStartTime"],
        errors="coerce",
        utc=True,
    )

    invalid_dates = int(
        data["TransactionStartTime"].isna().sum()
    )

    if invalid_dates:
        raise ValueError(
            f"Invalid transaction timestamps: {invalid_dates}"
        )

    duplicate_ids = int(
        data["TransactionId"].duplicated().sum()
    )

    if duplicate_ids:
        raise ValueError(
            f"Duplicate transaction IDs: {duplicate_ids}"
        )

    return data


def _safe_mode(series: pd.Series) -> object:
    """Return a deterministic mode or a missing-value label."""

    cleaned = series.dropna()

    if cleaned.empty:
        return "missing"

    modes = cleaned.mode()

    if modes.empty:
        return cleaned.iloc[0]

    return modes.iloc[0]


def build_customer_features(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate transaction-level records into customer features."""

    missing_columns = REQUIRED_TRANSACTION_COLUMNS.difference(
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
        raise ValueError("TransactionStartTime contains invalid values.")

    data["absolute_amount"] = data["Amount"].abs()
    data["is_debit"] = (data["Amount"] < 0).astype(int)
    data["transaction_date"] = data[
        "TransactionStartTime"
    ].dt.date
    data["transaction_hour"] = data[
        "TransactionStartTime"
    ].dt.hour
    data["is_weekend"] = (
        data["TransactionStartTime"].dt.dayofweek >= 5
    ).astype(int)
    data["is_night"] = data["transaction_hour"].between(
        0,
        5,
        inclusive="both",
    ).astype(int)

    customer_features = (
        data.groupby("CustomerId", as_index=False)
        .agg(
            transaction_count=("TransactionId", "count"),
            signed_total_amount=("Amount", "sum"),
            average_signed_amount=("Amount", "mean"),
            std_signed_amount=("Amount", "std"),
            minimum_signed_amount=("Amount", "min"),
            maximum_signed_amount=("Amount", "max"),
            total_transaction_value=("Value", "sum"),
            average_transaction_value=("Value", "mean"),
            std_transaction_value=("Value", "std"),
            minimum_transaction_value=("Value", "min"),
            maximum_transaction_value=("Value", "max"),
            debit_transaction_count=("is_debit", "sum"),
            debit_ratio=("is_debit", "mean"),
            unique_products=("ProductId", "nunique"),
            unique_product_categories=(
                "ProductCategory",
                "nunique",
            ),
            unique_channels=("ChannelId", "nunique"),
            unique_providers=("ProviderId", "nunique"),
            active_days=("transaction_date", "nunique"),
            average_transaction_hour=(
                "transaction_hour",
                "mean",
            ),
            weekend_transaction_ratio=("is_weekend", "mean"),
            night_transaction_ratio=("is_night", "mean"),
            first_transaction=("TransactionStartTime", "min"),
            last_transaction=("TransactionStartTime", "max"),
            dominant_product_category=(
                "ProductCategory",
                _safe_mode,
            ),
            dominant_channel=("ChannelId", _safe_mode),
            dominant_provider=("ProviderId", _safe_mode),
            dominant_pricing_strategy=(
                "PricingStrategy",
                _safe_mode,
            ),
        )
    )

    customer_features["active_span_days"] = (
        customer_features["last_transaction"]
        - customer_features["first_transaction"]
    ).dt.total_seconds().div(86_400)

    standard_deviation_columns = [
        "std_signed_amount",
        "std_transaction_value",
    ]

    customer_features[standard_deviation_columns] = (
        customer_features[standard_deviation_columns].fillna(0)
    )

    return customer_features


class WeightOfEvidenceEncoder(
    BaseEstimator,
    TransformerMixin,
):
    """Supervised Weight of Evidence encoder for binary targets."""

    def __init__(
        self,
        smoothing: float = 0.5,
        unknown_value: float = 0.0,
    ) -> None:
        self.smoothing = smoothing
        self.unknown_value = unknown_value

    @staticmethod
    def _to_frame(
        values: pd.DataFrame | np.ndarray,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        if isinstance(values, pd.DataFrame):
            return values.copy()

        array = np.asarray(values)

        if array.ndim == 1:
            array = array.reshape(-1, 1)

        if columns is None:
            columns = [
                f"feature_{index}"
                for index in range(array.shape[1])
            ]

        return pd.DataFrame(array, columns=list(columns))

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "WeightOfEvidenceEncoder":
        """Learn category-to-WoE mappings."""

        if y is None:
            raise ValueError("WoE encoding requires a binary target.")

        frame = self._to_frame(X)
        target = pd.Series(
            np.asarray(y).reshape(-1),
            index=frame.index,
            name="target",
        )

        target_values = set(target.dropna().unique())

        if not target_values.issubset({0, 1}):
            raise ValueError("WoE target must contain only 0 and 1.")

        if target.nunique() != 2:
            raise ValueError(
                "WoE target must contain both target classes."
            )

        self.feature_names_in_ = np.asarray(
            frame.columns,
            dtype=object,
        )
        self.woe_mappings_: dict[object, dict[object, float]] = {}

        total_events = float(target.sum())
        total_non_events = float(len(target) - target.sum())

        for column in frame.columns:
            feature = frame[column].astype(object).fillna("__missing__")

            grouped = (
                pd.DataFrame(
                    {
                        "feature": feature,
                        "target": target,
                    }
                )
                .groupby("feature", dropna=False)["target"]
                .agg(["count", "sum"])
            )

            category_count = len(grouped)
            events = grouped["sum"]
            non_events = grouped["count"] - events

            event_distribution = (
                events + self.smoothing
            ) / (
                total_events
                + self.smoothing * category_count
            )

            non_event_distribution = (
                non_events + self.smoothing
            ) / (
                total_non_events
                + self.smoothing * category_count
            )

            woe = np.log(
                non_event_distribution / event_distribution
            ).clip(-20, 20)

            self.woe_mappings_[column] = woe.to_dict()

        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Replace categories with fitted WoE values."""

        check_is_fitted(
            self,
            attributes=[
                "feature_names_in_",
                "woe_mappings_",
            ],
        )

        frame = self._to_frame(
            X,
            columns=self.feature_names_in_,
        )

        encoded_columns = []

        for column in self.feature_names_in_:
            encoded = (
                frame[column]
                .astype(object)
                .fillna("__missing__")
                .map(self.woe_mappings_[column])
                .fillna(self.unknown_value)
                .astype(float)
            )

            encoded_columns.append(encoded.to_numpy())

        return np.column_stack(encoded_columns)

    def get_feature_names_out(
        self,
        input_features: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Return output feature names."""

        check_is_fitted(self, attributes=["feature_names_in_"])

        features = (
            self.feature_names_in_
            if input_features is None
            else np.asarray(input_features, dtype=object)
        )

        return np.asarray(
            [f"woe_{feature}" for feature in features],
            dtype=object,
        )


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str],
) -> ColumnTransformer:
    """Create a reusable preprocessing pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "woe",
                WeightOfEvidenceEncoder(),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                list(numeric_features),
            ),
            (
                "categorical",
                categorical_pipeline,
                list(categorical_features),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )