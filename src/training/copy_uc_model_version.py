import os
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient


DEFAULT_TRACKING_URI = "databricks"
DEFAULT_REGISTRY_URI = "databricks-uc"
DEFAULT_SOURCE_MODEL_NAME = "biometric_model"
DEFAULT_DESTINATION_MODEL_NAME = "iakshaykr.default.prod"
DEFAULT_DOWNLOAD_DIR = "promoted_model_artifacts"


def resolve_uc_model_name(
    model_name_env: str,
    catalog_env: str,
    schema_env: str,
    default_model_name: str,
) -> str:
    model_name = os.getenv(model_name_env, default_model_name)
    if model_name.count(".") == 2:
        return model_name

    catalog = os.getenv(catalog_env)
    schema = os.getenv(schema_env)
    if not catalog or not schema:
        raise ValueError(
            f"{model_name_env} must be a full Unity Catalog model name "
            f"or both {catalog_env} and {schema_env} must be set."
        )

    return f"{catalog}.{schema}.{model_name}"


def resolve_latest_version(client: MlflowClient, model_name: str) -> str:
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No model versions found for {model_name}")

    latest_version = max(versions, key=lambda version: int(version.version))
    return str(latest_version.version)


def configure_mlflow(tracking_uri: str, registry_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    return MlflowClient()


def validate_downloaded_model(local_model_path: Path) -> None:
    mlmodel_file = local_model_path / "MLmodel"
    if not mlmodel_file.is_file():
        raise FileNotFoundError(f"Downloaded model does not contain MLmodel: {local_model_path}")

    loaded_model = mlflow.pyfunc.load_model(str(local_model_path))
    if loaded_model is None:
        raise ValueError(f"Failed to load downloaded model from {local_model_path}")


def main() -> int:
    source_tracking_uri = os.getenv("SOURCE_MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    source_registry_uri = os.getenv("SOURCE_MLFLOW_REGISTRY_URI", DEFAULT_REGISTRY_URI)
    destination_tracking_uri = os.getenv("DESTINATION_MLFLOW_TRACKING_URI", source_tracking_uri)
    destination_registry_uri = os.getenv("DESTINATION_MLFLOW_REGISTRY_URI", DEFAULT_REGISTRY_URI)

    source_model_name = resolve_uc_model_name(
        model_name_env="SOURCE_MODEL_NAME",
        catalog_env="SOURCE_UC_CATALOG",
        schema_env="SOURCE_UC_SCHEMA",
        default_model_name=DEFAULT_SOURCE_MODEL_NAME,
    )
    destination_model_name = os.getenv(
        "DESTINATION_MODEL_NAME",
        DEFAULT_DESTINATION_MODEL_NAME,
    )
    if destination_model_name.count(".") != 2:
        raise ValueError(
            "DESTINATION_MODEL_NAME must be a full Unity Catalog model name "
            "in the form catalog.schema.model."
        )

    source_client = configure_mlflow(source_tracking_uri, source_registry_uri)
    source_version = os.getenv("SOURCE_MODEL_VERSION") or resolve_latest_version(
        source_client, source_model_name
    )
    source_uri = f"models:/{source_model_name}/{source_version}"
    download_root = Path(os.getenv("MODEL_DOWNLOAD_DIR", DEFAULT_DOWNLOAD_DIR))
    download_root.mkdir(parents=True, exist_ok=True)
    local_model_path = Path(
        mlflow.artifacts.download_artifacts(
            artifact_uri=source_uri,
            dst_path=str(download_root),
        )
    )
    if not local_model_path.exists():
        raise FileNotFoundError(f"Downloaded model path not found: {local_model_path}")

    validate_downloaded_model(local_model_path)

    configure_mlflow(destination_tracking_uri, destination_registry_uri)
    registered_model = mlflow.register_model(
        model_uri=str(local_model_path),
        name=destination_model_name,
    )

    print(
        f"Promoted {source_uri} to {destination_model_name} as version={registered_model.version} "
        f"using local artifacts at {local_model_path}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
