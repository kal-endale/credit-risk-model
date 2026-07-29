"""Build customer-level features from the Xente transactions."""

from pathlib import Path

from src.data_processing import (
    build_customer_features,
    load_transaction_data,
)


def main() -> None:
    """Execute the feature-building workflow."""

    project_root = Path(__file__).resolve().parents[1]
    input_path = project_root / "data" / "raw" / "data.csv"
    output_path = (
        project_root
        / "data"
        / "processed"
        / "customer_features.csv"
    )

    transactions = load_transaction_data(input_path)
    customer_features = build_customer_features(transactions)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    customer_features.to_csv(output_path, index=False)

    print("TASK 3 FEATURE ENGINEERING VALIDATION")
    print("-" * 72)
    print(f"Transaction rows loaded: {len(transactions):,}")
    print(
        "Unique transaction customers: "
        f"{transactions['CustomerId'].nunique():,}"
    )
    print(f"Customer feature rows: {len(customer_features):,}")
    print(f"Features created: {customer_features.shape[1] - 1}")
    print(
        "Duplicate customer IDs: "
        f"{customer_features['CustomerId'].duplicated().sum():,}"
    )
    print(
        "Missing feature values: "
        f"{customer_features.isna().sum().sum():,}"
    )
    print(
        "Customer coverage passed: "
        f"{len(customer_features) == transactions['CustomerId'].nunique()}"
    )
    print(
        "Output: "
        f"{output_path.relative_to(project_root)}"
    )


if __name__ == "__main__":
    main()