"""Send an actual customer record to the local API."""

import json
from pathlib import Path

import httpx
import pandas as pd

from src.model_training import MODEL_FEATURES


def main() -> None:
    """Submit one customer to the prediction endpoint."""

    project_root = Path(__file__).resolve().parents[1]
    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "modeling_dataset.csv"
    )

    dataset = pd.read_csv(dataset_path)
    customer = dataset.iloc[0]

    payload = json.loads(
        customer[MODEL_FEATURES].to_json()
    )
    payload["customer_id"] = str(
        customer["CustomerId"]
    )

    response = httpx.post(
        "http://127.0.0.1:8000/predict",
        json=payload,
        timeout=30,
    )

    print("TASK 6 API SMOKE TEST")
    print("-" * 72)
    print(f"HTTP status: {response.status_code}")
    print(f"Customer ID: {payload['customer_id']}")
    print(json.dumps(response.json(), indent=2))

    response.raise_for_status()


if __name__ == "__main__":
    main()