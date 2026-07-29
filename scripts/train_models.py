"""Train, tune, compare, and register credit-risk models."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.data_processing import build_preprocessor
from src.model_training import (
    MODEL_CATEGORICAL_FEATURES,
    MODEL_NUMERIC_FEATURES,
    evaluate_classifier,
    prepare_model_data,
)


RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "credit-risk-proxy-model"


def build_search_spaces() -> dict[str, dict]:
    """Return the candidate models and parameter grids."""

    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                max_iter=2_000,
                random_state=RANDOM_STATE,
            ),
            "parameters": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [
                    None,
                    "balanced",
                ],
                "classifier__solver": ["liblinear"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=1,
            ),
            "parameters": {
                "classifier__n_estimators": [200, 350],
                "classifier__max_depth": [8, None],
                "classifier__min_samples_leaf": [1, 3],
                "classifier__class_weight": [
                    None,
                    "balanced",
                ],
            },
        },
    }


def main() -> None:
    """Run model training, comparison, and registration."""

    project_root = Path(__file__).resolve().parents[1]

    dataset_path = (
        project_root
        / "data"
        / "processed"
        / "modeling_dataset.csv"
    )
    model_path = (
        project_root / "models" / "best_model.joblib"
    )
    metadata_path = (
        project_root / "models" / "model_metadata.json"
    )
    comparison_path = (
        project_root / "reports" / "model_comparison.csv"
    )
    tracking_database = project_root / "mlflow.db"

    if not dataset_path.exists():
        raise FileNotFoundError(
            "modeling_dataset.csv was not found. Run "
            "'python -m scripts.create_proxy_target' first."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset = pd.read_csv(dataset_path)
    features, target = prepare_model_data(dataset)

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    tracking_uri = (
        f"sqlite:///{tracking_database.resolve().as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(
        "credit-risk-proxy-classification"
    )

    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    comparison_records = []
    trained_models = {}
    run_ids = {}

    for model_name, configuration in build_search_spaces().items():
        print(
            f"\nStarting model training: {model_name}",
            flush=True,
        )

        preprocessor = build_preprocessor(
            numeric_features=MODEL_NUMERIC_FEATURES,
            categorical_features=MODEL_CATEGORICAL_FEATURES,
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    configuration["estimator"],
                ),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=configuration["parameters"],
            scoring="roc_auc",
            cv=cross_validation,
            n_jobs=1,
            refit=True,
            return_train_score=True,
            verbose=1,
        )

        with mlflow.start_run(run_name=model_name) as run:
            search.fit(X_train, y_train)

            best_model = search.best_estimator_

            metrics = evaluate_classifier(
                best_model,
                X_test,
                y_test,
            )

            mlflow.log_param("model_name", model_name)
            mlflow.log_param(
                "training_rows",
                len(X_train),
            )
            mlflow.log_param(
                "testing_rows",
                len(X_test),
            )
            mlflow.log_param(
                "feature_count",
                len(features.columns),
            )
            mlflow.log_param(
                "cross_validation_folds",
                5,
            )

            for parameter, value in search.best_params_.items():
                parameter_name = parameter.replace(
                    "classifier__",
                    "",
                )

                mlflow.log_param(
                    f"best_{parameter_name}",
                    str(value),
                )

            mlflow.log_metric(
                "best_cv_roc_auc",
                float(search.best_score_),
            )

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(
                    f"test_{metric_name}",
                    metric_value,
                )

            signature_input = X_train.head(50)
            signature_output = best_model.predict_proba(
                signature_input
            )[:, 1]

            signature = infer_signature(
                signature_input,
                signature_output,
            )

            mlflow.sklearn.log_model(
                sk_model=best_model,
                name="model",
                signature=signature,
                input_example=X_train.head(3),
                skops_trusted_types=[
                    "numpy.dtype",
                    "src.data_processing.WeightOfEvidenceEncoder",
                ],
            )

            mlflow.log_dict(
                {
                    "numeric_features": (
                        MODEL_NUMERIC_FEATURES
                    ),
                    "categorical_features": (
                        MODEL_CATEGORICAL_FEATURES
                    ),
                },
                "feature_definition.json",
            )

            comparison_records.append(
                {
                    "model": model_name,
                    "cv_roc_auc": float(
                        search.best_score_
                    ),
                    **metrics,
                    "best_parameters": json.dumps(
                        search.best_params_,
                        sort_keys=True,
                    ),
                    "run_id": run.info.run_id,
                }
            )

            trained_models[model_name] = best_model
            run_ids[model_name] = run.info.run_id

    comparison = (
        pd.DataFrame(comparison_records)
        .sort_values(
            ["roc_auc", "f1_score"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    best_model_name = str(
        comparison.iloc[0]["model"]
    )
    best_model = trained_models[best_model_name]
    best_run_id = run_ids[best_model_name]

    joblib.dump(best_model, model_path)
    comparison.to_csv(comparison_path, index=False)

    metadata = {
        "model_name": best_model_name,
        "registered_model_name": REGISTERED_MODEL_NAME,
        "target": "is_high_risk",
        "classification_threshold": 0.5,
        "numeric_features": MODEL_NUMERIC_FEATURES,
        "categorical_features": MODEL_CATEGORICAL_FEATURES,
        "excluded_proxy_features": [
            "recency",
            "frequency",
            "monetary_value",
            "rfm_cluster",
            "transaction_count",
            "total_transaction_value",
        ],
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    registered_model = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name=REGISTERED_MODEL_NAME,
    )

    client = MlflowClient()
    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias="champion",
        version=registered_model.version,
    )

    print("\nTASK 5 MODEL TRAINING VALIDATION")
    print("-" * 100)
    print(f"Customers: {len(dataset):,}")
    print(f"Training customers: {len(X_train):,}")
    print(f"Testing customers: {len(X_test):,}")
    print(f"Predictor count: {features.shape[1]}")
    print(f"Target rate: {target.mean() * 100:.2f}%")
    print(f"Best model: {best_model_name}")
    print(
        f"Registered model: {REGISTERED_MODEL_NAME} "
        f"version {registered_model.version}"
    )
    print(
        f"Model artifact: "
        f"{model_path.relative_to(project_root)}"
    )

    print("\nMODEL COMPARISON")
    print("-" * 100)
    print(
        comparison[
            [
                "model",
                "cv_roc_auc",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "roc_auc",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()