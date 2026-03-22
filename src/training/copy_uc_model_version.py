import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


DEFAULT_TRACKING_URI = "databricks"
DEFAULT_REGISTRY_URI = "databricks-uc"
DEFAULT_SOURCE_MODEL_NAME = "biometric_model"
DEFAULT_DESTINATION_MODEL_NAME = "iakshaykr.default.prod"


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


def main() -> int:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    registry_uri = os.getenv("MLFLOW_REGISTRY_URI", DEFAULT_REGISTRY_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)
    client = MlflowClient()

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

    source_version = os.getenv("SOURCE_MODEL_VERSION")
    if not source_version:
        source_version = resolve_latest_version(client, source_model_name)

    source_uri = f"models:/{source_model_name}/{source_version}"
    copied_version = client.copy_model_version(source_uri, destination_model_name)

    print(
        f"Copied {source_uri} to {destination_model_name} "
        f"as version={copied_version.version}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
