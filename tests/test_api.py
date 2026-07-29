"""Tests for the FastAPI prediction service."""

import numpy as np
from fastapi.testclient import TestClient

from src.api import main as api_main


client = TestClient(api_main.app)


class DummyModel:
    """Deterministic model used by API unit tests."""

    def predict_proba(self, features):
        """Return a fixed high-risk probability."""

        return np.array([[0.20, 0.80]])


def sample_request() -> dict:
    """Return a valid prediction request."""

    return {
        "customer_id": "CustomerId_1",
        "average_signed_amount": 1_000.0,
        "std_signed_amount": 200.0,
        "minimum_signed_amount": -100.0,
        "maximum_signed_amount": 2_000.0,
        "average_transaction_value": 1_200.0,
        "std_transaction_value": 300.0,
        "minimum_transaction_value": 100.0,
        "maximum_transaction_value": 3_000.0,
        "debit_transaction_count": 2,
        "debit_ratio": 0.25,
        "unique_products": 3,
        "unique_product_categories": 2,
        "unique_channels": 2,
        "unique_providers": 2,
        "active_days": 5,
        "active_span_days": 20,
        "average_transaction_hour": 13.5,
        "weekend_transaction_ratio": 0.20,
        "night_transaction_ratio": 0.10,
        "dominant_product_category": "airtime",
        "dominant_channel": "ChannelId_3",
        "dominant_provider": "ProviderId_4",
        "dominant_pricing_strategy": 2,
    }


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert "status" in response.json()
    assert "model_available" in response.json()


def test_prediction_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(
        api_main,
        "get_model",
        lambda: DummyModel(),
    )

    response = client.post(
        "/predict",
        json=sample_request(),
    )

    assert response.status_code == 200

    result = response.json()

    assert result["risk_probability"] == 0.8
    assert result["is_high_risk"] == 1
    assert result["risk_label"] == "high_risk"
    assert result["customer_id"] == "CustomerId_1"


def test_request_validation_rejects_invalid_ratio() -> None:
    request = sample_request()
    request["debit_ratio"] = 1.50

    response = client.post(
        "/predict",
        json=request,
    )

    assert response.status_code == 422