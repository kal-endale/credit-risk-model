"""Pydantic request and response schemas."""

from pydantic import BaseModel, ConfigDict, Field


class RiskPredictionRequest(BaseModel):
    """Customer behavior used by the risk model."""

    model_config = ConfigDict(extra="forbid")

    customer_id: str | None = None

    average_signed_amount: float
    std_signed_amount: float = Field(ge=0)
    minimum_signed_amount: float
    maximum_signed_amount: float

    average_transaction_value: float = Field(ge=0)
    std_transaction_value: float = Field(ge=0)
    minimum_transaction_value: float = Field(ge=0)
    maximum_transaction_value: float = Field(ge=0)

    debit_transaction_count: float = Field(ge=0)
    debit_ratio: float = Field(ge=0, le=1)

    unique_products: float = Field(ge=0)
    unique_product_categories: float = Field(ge=0)
    unique_channels: float = Field(ge=0)
    unique_providers: float = Field(ge=0)
    active_days: float = Field(ge=0)
    active_span_days: float = Field(ge=0)

    average_transaction_hour: float = Field(ge=0, le=23)
    weekend_transaction_ratio: float = Field(ge=0, le=1)
    night_transaction_ratio: float = Field(ge=0, le=1)

    dominant_product_category: str
    dominant_channel: str
    dominant_provider: str
    dominant_pricing_strategy: int


class RiskPredictionResponse(BaseModel):
    """Risk prediction returned by the API."""

    customer_id: str | None
    risk_probability: float
    is_high_risk: int
    risk_label: str
    threshold: float
    model_name: str