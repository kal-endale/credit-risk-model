"""Create the RFM-based credit-risk proxy target."""

from pathlib import Path

import pandas as pd

from src.data_processing import load_transaction_data
from src.proxy_target import calculate_rfm, create_risk_proxy


def main() -> None:
    """Generate and validate the customer modelling dataset."""

    project_root = Path(__file__).resolve().parents[1]

    transaction_path = (
        project_root / "data" / "raw" / "data.csv"
    )
    feature_path = (
        project_root
        / "data"
        / "processed"
        / "customer_features.csv"
    )
    output_path = (
        project_root
        / "data"
        / "processed"
        / "modeling_dataset.csv"
    )
    profile_path = (
        project_root
        / "data"
        / "processed"
        / "rfm_cluster_profile.csv"
    )

    transactions = load_transaction_data(transaction_path)

    if not feature_path.exists():
        raise FileNotFoundError(
            "Customer features do not exist. Run "
            "'python -m scripts.build_features' first."
        )

    customer_features = pd.read_csv(feature_path)

    customer_rfm = calculate_rfm(transactions)

    (
        labeled_rfm,
        cluster_profile,
        _,
        _,
        high_risk_cluster,
    ) = create_risk_proxy(customer_rfm)

    modeling_dataset = customer_features.merge(
        labeled_rfm,
        on="CustomerId",
        how="inner",
        validate="one_to_one",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    modeling_dataset.to_csv(output_path, index=False)
    cluster_profile.to_csv(profile_path, index=False)

    print("TASK 4 RFM PROXY VALIDATION")
    print("-" * 78)
    print(f"Customers analyzed: {len(customer_rfm):,}")
    print(
        f"Reference date: "
        f"{transactions['TransactionStartTime'].max().date()}"
        f"+ pd.Timedelta(days=1)"
    )
    print(
        f"RFM clusters created: "
        f"{labeled_rfm['rfm_cluster'].nunique()}"
    )
    print(f"High-risk cluster: {high_risk_cluster}")
    print(
        f"High-risk customers: "
        f"{labeled_rfm['is_high_risk'].sum():,}"
    )
    print(
        f"High-risk percentage: "
        f"{labeled_rfm['is_high_risk'].mean() * 100:.2f}%"
    )
    print(
        f"Duplicate customer IDs: "
        f"{modeling_dataset['CustomerId'].duplicated().sum():,}"
    )
    print(
        f"Missing values: "
        f"{modeling_dataset.isna().sum().sum():,}"
    )
    print(
        f"Customer coverage passed: "
        f"{len(modeling_dataset) == len(customer_features)}"
    )

    print("\nRFM CLUSTER PROFILE")
    print("-" * 78)
    print(
        cluster_profile[
            [
                "rfm_cluster",
                "customer_count",
                "customer_percentage",
                "mean_recency",
                "mean_frequency",
                "mean_monetary_value",
                "risk_score",
                "is_high_risk_cluster",
            ]
        ].round(4).to_string(index=False)
    )

    print(f"\nOutput: {output_path.relative_to(project_root)}")
    print(
        f"Cluster profile: "
        f"{profile_path.relative_to(project_root)}"
    )


if __name__ == "__main__":
    main()