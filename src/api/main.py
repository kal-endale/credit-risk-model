"""FastAPI service for credit-risk proxy predictions."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.api.schemas import (
    RiskPredictionRequest,
    RiskPredictionResponse,
)
from src.model_training import MODEL_FEATURES



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "best_model.joblib"
)
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH))
)

CLASSIFICATION_THRESHOLD = 0.5
MODEL_NAME = "credit-risk-proxy-model"


app = FastAPI(
    title="Credit Risk Proxy API",
    description=(
        "Predicts RFM-derived customer risk using "
        "alternative transaction behavior."
    ),
    version="1.0.0",
)


@lru_cache(maxsize=1)
def get_model() -> Any:
    """Load and cache the trained model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def root() -> dict[str, str]:
    """Return basic service information."""

    return {
        "service": "Credit Risk Proxy API",
        "documentation": "/docs",
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report application and model availability."""

    model_available = MODEL_PATH.exists()

    return {
        "status": (
            "healthy" if model_available else "degraded"
        ),
        "model_available": model_available,
        "model_path": str(MODEL_PATH),
    }


@app.post(
    "/predict",
    response_model=RiskPredictionResponse,
)
def predict_risk(
    request: RiskPredictionRequest,
) -> RiskPredictionResponse:
    """Predict a customer's high-risk proxy probability."""

    try:
        model = get_model()
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error

    feature_values = request.model_dump(
        exclude={"customer_id"}
    )

    feature_frame = pd.DataFrame(
        [feature_values],
        columns=MODEL_FEATURES,
    )

    try:
        probability = float(
            model.predict_proba(feature_frame)[0, 1]
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed.",
        ) from error

    is_high_risk = int(
        probability >= CLASSIFICATION_THRESHOLD
    )

    return RiskPredictionResponse(
        customer_id=request.customer_id,
        risk_probability=round(probability, 6),
        is_high_risk=is_high_risk,
        risk_label=(
            "high_risk"
            if is_high_risk
            else "lower_risk"
        ),
        threshold=CLASSIFICATION_THRESHOLD,
        model_name=MODEL_NAME,
    )