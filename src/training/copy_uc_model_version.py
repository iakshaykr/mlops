import os
import shutil
import sys
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path("/Workspace/Users/akshaykr9531@gmail.com/mlops/src/training/copy_uc_model_version.py")

PROJECT_ROOT = _this_file.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_TRACKING_URI = "databricks"
DEFAULT_REGISTRY_URI = "databricks-uc"
DEFAULT_SOURCE_MODEL_NAME = "biometric_model"
DEFAULT_DESTINATION_MODEL_NAME = "iakshaykr.default.prod"
DEFAULT_DOWNLOAD_DIR = "promoted_model_artifacts"
NORMALIZED_MODEL_DIR_NAME = "model"


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


def write_github_output(name: str, value: str) -> None:
    github_output = os.getenv("GITHUB_OUTPUT")
    if not github_output:
        return

    with open(github_output, "a", encoding="utf-8") as output_file:
        output_file.write(f"{name}={value}\n")


def resolve_downloaded_model_dir(download_root: Path, downloaded_path: Path) -> Path:
    if downloaded_path.is_dir() and (downloaded_path / "MLmodel").is_file():
        return downloaded_path

    if downloaded_path.is_file() and downloaded_path.name == "MLmodel":
        return downloaded_path.parent

    search_root = downloaded_path if downloaded_path.is_dir() else downloaded_path.parent
    candidate_paths = sorted(path.parent for path in search_root.rglob("MLmodel"))
    if not candidate_paths and search_root != download_root:
        candidate_paths = sorted(path.parent for path in download_root.rglob("MLmodel"))

    if not candidate_paths:
        raise FileNotFoundError(
            f"No MLmodel file found under downloaded artifacts. "
            f"downloaded_path={downloaded_path}, download_root={download_root}"
        )

    return candidate_paths[0]


def normalize_downloaded_model_path(download_root: Path, downloaded_path: Path) -> Path:
    source_model_dir = resolve_downloaded_model_dir(download_root, downloaded_path)
    normalized_path = download_root / NORMALIZED_MODEL_DIR_NAME
    if normalized_path.exists():
        if normalized_path.is_dir():
            shutil.rmtree(normalized_path)
        else:
            normalized_path.unlink()
    shutil.copytree(source_model_dir, normalized_path)
    return normalized_path


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

    source_client = configure_mlflow(source_tracking_uri, source_registry_uri)
    source_version = os.getenv("SOURCE_MODEL_VERSION") or resolve_latest_version(
        source_client, source_model_name
    )
    write_github_output("source_model_version", source_version)
    source_uri = f"models:/{source_model_name}/{source_version}"
    if os.getenv("SKIP_DOWNLOAD", "false").lower() == "true":
        local_model_path = Path(os.getenv("LOCAL_MODEL_PATH", ""))
        if not local_model_path.exists():
            raise FileNotFoundError(f"Provided LOCAL_MODEL_PATH not found: {local_model_path}")
    else:
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
        local_model_path = normalize_downloaded_model_path(download_root, local_model_path)

    validate_downloaded_model(local_model_path)
    write_github_output("downloaded_model_path", str(local_model_path))

    if os.getenv("VALIDATION_ONLY", "false").lower() == "true":
        print(f"Validated source model successfully at {local_model_path}")
        return 0

    destination_model_name = os.getenv(
        "DESTINATION_MODEL_NAME",
        DEFAULT_DESTINATION_MODEL_NAME,
    )
    if destination_model_name.count(".") != 2:
        raise ValueError(
            "DESTINATION_MODEL_NAME must be a full Unity Catalog model name "
            "in the form catalog.schema.model."
        )

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
