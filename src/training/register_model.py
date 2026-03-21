import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


DEFAULT_EXPERIMENT_NAME = "/Users/akshaykr9531@gmail.com/biometric-training"
DEFAULT_MODEL_NAME = "biometric_model"
DEFAULT_RUN_NAME = "biometric-simple-model"
DEFAULT_TRACKING_URI = "databricks"
DEFAULT_REGISTRY_URI = "databricks-uc"


def resolve_model_name(registry_uri: str) -> str:
    model_name = os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
    if registry_uri != "databricks-uc":
        return model_name

    if model_name.count(".") == 2:
        return model_name

    uc_catalog = os.getenv("MLFLOW_UC_CATALOG")
    uc_schema = os.getenv("MLFLOW_UC_SCHEMA")
    if not uc_catalog or not uc_schema:
        raise ValueError(
            "Unity Catalog registration requires either "
            "`MLFLOW_MODEL_NAME=catalog.schema.model_name` or both "
            "`MLFLOW_UC_CATALOG` and `MLFLOW_UC_SCHEMA`."
        )

    return f"{uc_catalog}.{uc_schema}.{model_name}"


def resolve_run_id(client: MlflowClient) -> str:
    explicit_run_id = os.getenv("MLFLOW_RUN_ID")
    if explicit_run_id:
        return explicit_run_id

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"MLflow experiment not found: {experiment_name}")

    run_name = os.getenv("MLFLOW_RUN_NAME", DEFAULT_RUN_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(
            f"No MLflow runs found in experiment '{experiment_name}' with run_name '{run_name}'."
        )

    return runs[0].info.run_id


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", DEFAULT_REGISTRY_URI)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = MlflowClient()

    run_id = resolve_run_id(client)
    model_name = resolve_model_name(registry_uri)
    model_uri = f"runs:/{run_id}/model"

    if registry_uri == "databricks-uc" and model_name.count(".") != 2:
        raise ValueError(
            "Unity Catalog model registration requires "
            "`catalog_name.schema_name.model_name`."
        )

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(
        f"Registered model '{model_name}' from run_id={run_id} as version={result.version} "
        f"using registry_uri={registry_uri}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
