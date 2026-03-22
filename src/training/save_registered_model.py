import logging
import os
import sys
import warnings
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

DEFAULT_MODEL_NAME = "biometric_model"
DEFAULT_TRACKING_URI = "databricks"
DEFAULT_REGISTRY_URI = "databricks-uc"
DEFAULT_OUTPUT_DIR = "/Volumes/iakshaykr/default/prod"


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


def _suppress_spark_connect_noise() -> None:
    """Silence stale Spark-Connect / gRPC session warnings."""
    logging.getLogger("pyspark.sql.connect.logging").setLevel(logging.CRITICAL)
    logging.getLogger("pyspark.sql.connect.client.core").setLevel(logging.CRITICAL)
    warnings.filterwarnings(
        "ignore",
        message=".*Spark Connect Session expired.*",
        category=UserWarning,
    )


def resolve_latest_model_uri(client: MlflowClient, model_name: str) -> tuple[str, str]:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered model versions found for {model_name}")

    latest_version = max(versions, key=lambda version: int(version.version))
    return f"models:/{model_name}/{latest_version.version}", str(latest_version.version)


def main() -> int:
    _suppress_spark_connect_noise()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", DEFAULT_REGISTRY_URI)
    base_output_dir = Path(os.getenv("MODEL_DOWNLOAD_DIR", DEFAULT_OUTPUT_DIR))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = MlflowClient()

    model_name = resolve_model_name(registry_uri)
    model_uri, version = resolve_latest_model_uri(client, model_name)

    # Organize as: <base>/<model_short_name>/<version>/
    model_short_name = model_name.rsplit(".", 1)[-1]
    output_dir = base_output_dir / model_short_name / version
    output_dir.mkdir(parents=True, exist_ok=True)

    local_model_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(output_dir),
    )

    print(
        f"Saved registered model '{model_name}' version={version} "
        f"to {local_model_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        rc = main()
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        rc = 1
    if not hasattr(__builtins__, "__IPYTHON__") and "IPython" not in sys.modules:
        raise SystemExit(rc)
