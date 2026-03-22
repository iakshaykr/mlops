import os
import sys
from pathlib import Path

import mlflow.pytorch
import torch


DEFAULT_MODEL_DIR = "saved_model"
DEFAULT_MODEL_NAME = "catalog.schema.prod_model"
DEFAULT_INPUT_SIZE = 3072


def resolve_model_path(model_dir: Path) -> Path:
    if (model_dir / "MLmodel").is_file():
        return model_dir

    candidate_paths = sorted(path for path in model_dir.rglob("MLmodel"))
    if not candidate_paths:
        raise FileNotFoundError(f"No MLmodel file found under {model_dir}")
    return candidate_paths[0].parent


def resolve_model_source() -> str:
    model_uri = os.getenv("MODEL_URI")
    if model_uri:
        return model_uri

    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    model_version = os.getenv("MODEL_VERSION")
    model_stage = os.getenv("MODEL_STAGE")

    if model_version:
        return f"models:/{model_name}/{model_version}"
    if model_stage:
        return f"models:/{model_name}@{model_stage}"

    model_dir = Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Saved model directory not found: {model_dir}. "
            "Set MODEL_URI, MODEL_NAME with MODEL_VERSION, or MODEL_DIR."
        )
    return str(resolve_model_path(model_dir))


def main() -> int:
    input_size = int(os.getenv("PREDICTION_INPUT_SIZE", str(DEFAULT_INPUT_SIZE)))
    model_source = resolve_model_source()
    model = mlflow.pytorch.load_model(model_source)
    model.eval()

    sample_input = torch.zeros((1, input_size), dtype=torch.float32)
    with torch.no_grad():
        prediction = model(sample_input)

    predicted_class = int(torch.argmax(prediction, dim=1).item())
    print(
        f"Prediction successful. model_source={model_source}, "
        f"output_shape={tuple(prediction.shape)}, "
        f"predicted_class={predicted_class}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
