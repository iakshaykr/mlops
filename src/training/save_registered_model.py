import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from src.training.register_model import resolve_model_name


DEFAULT_MODEL_NAME = "biometric_model"
DEFAULT_TRACKING_URI = "databricks"
DEFAULT_REGISTRY_URI = "databricks-uc"
DEFAULT_OUTPUT_DIR = "saved_model"


def resolve_latest_model_uri(client: MlflowClient, model_name: str) -> tuple[str, str]:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No registered model versions found for {model_name}")

    latest_version = max(versions, key=lambda version: int(version.version))
    return f"models:/{model_name}/{latest_version.version}", str(latest_version.version)


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", DEFAULT_REGISTRY_URI)
    output_dir = Path(os.getenv("MODEL_DOWNLOAD_DIR", DEFAULT_OUTPUT_DIR))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = MlflowClient()

    model_name = resolve_model_name(registry_uri)
    model_uri, version = resolve_latest_model_uri(client, model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_model_path = mlflow.artifacts.download_artifacts(
        artifact_uri=model_uri,
        dst_path=str(output_dir),
    )

    print(f"Saved registered model '{model_name}' version={version} to {local_model_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
