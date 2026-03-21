import os
import sys

import mlflow
from mlflow.tracking import MlflowClient


DEFAULT_EXPERIMENT_NAME = "/Users/akshaykr9531@gmail.com/biometric-training"
DEFAULT_MODEL_NAME = "biometric_model"
DEFAULT_RUN_NAME = "biometric-simple-model"


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
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "databricks"))
    client = MlflowClient()

    run_id = resolve_run_id(client)
    model_name = os.getenv("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri=model_uri, name=model_name)
    print(
        f"Registered model '{model_name}' from run_id={run_id} as version={result.version}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
