import logging
import os
import sys
import warnings
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
import yaml


DEFAULT_EXPERIMENT_NAME = "/Users/akshaykr9531@gmail.com/biometric-training"
DEFAULT_MODEL_NAME = "biometric_model"
DEFAULT_RUN_NAME = "biometric-simple-model"
DEFAULT_TRACKING_URI = "databricks"
DEFAULT_REGISTRY_URI = "databricks-uc"
DEFAULT_UC_CATALOG = "iakshaykr"
DEFAULT_UC_SCHEMA = "default"


def _suppress_spark_connect_noise() -> None:
    """Silence stale Spark-Connect / gRPC session warnings."""
    logging.getLogger("pyspark.sql.connect.logging").setLevel(logging.CRITICAL)
    logging.getLogger("pyspark.sql.connect.client.core").setLevel(logging.CRITICAL)
    warnings.filterwarnings(
        "ignore",
        message=".*Spark Connect Session expired.*",
        category=UserWarning,
    )


def resolve_model_name(registry_uri: str) -> str:
    model_name = os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
    if registry_uri != "databricks-uc":
        return model_name

    if model_name.count(".") == 2:
        return model_name

    uc_catalog = os.getenv("MLFLOW_UC_CATALOG", DEFAULT_UC_CATALOG)
    uc_schema = os.getenv("MLFLOW_UC_SCHEMA", DEFAULT_UC_SCHEMA)
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
        max_results=20,
    )
    if not runs:
        raise ValueError(
            f"No MLflow runs found in experiment '{experiment_name}' "
            f"with run_name '{run_name}'."
        )

    for run in runs:
        if run_model_has_signature(run.info.run_id):
            return run.info.run_id

    raise ValueError(
        f"Found {len(runs)} run(s) in experiment '{experiment_name}' with "
        f"run_name '{run_name}', but none contain a model signature at "
        f"artifact path 'model'. Unity Catalog requires all models to "
        f"include a signature. Please re-run train.py to generate a new "
        f"run with signature logging enabled."
    )


def run_model_has_signature(run_id: str) -> bool:
    try:
        mlmodel_path = Path(
            mlflow.artifacts.download_artifacts(artifact_uri=f"runs:/{run_id}/model/MLmodel")
        )
    except Exception:
        return False

    if not mlmodel_path.is_file():
        return False

    with open(mlmodel_path, "r", encoding="utf-8") as mlmodel_file:
        mlmodel_data = yaml.safe_load(mlmodel_file)

    signature = mlmodel_data.get("signature")
    return signature is not None


def main() -> int:
    _suppress_spark_connect_noise()

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
        rc = main()
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        rc = 1
    # Only raise SystemExit when running as a real script, not inside IPython/notebook
    if not hasattr(__builtins__, "__IPYTHON__") and "IPython" not in sys.modules:
        raise SystemExit(rc)
