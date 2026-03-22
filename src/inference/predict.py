import json
import logging
import os
from pathlib import Path

import mlflow.pytorch
import torch
from torch import nn

DEFAULT_MODEL_DIR = "saved_model"
DEFAULT_MODEL_NAME = "catalog.schema.prod_model"
DEFAULT_INPUT_SIZE = 3072

logger = logging.getLogger(__name__)


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


def resolve_sample_input(input_size: int) -> torch.Tensor:
    prediction_values = os.getenv("PREDICTION_VALUES")
    if not prediction_values:
        return torch.zeros((1, input_size), dtype=torch.float32)

    try:
        parsed_values = json.loads(prediction_values)
    except json.JSONDecodeError:
        parsed_values = [value.strip() for value in prediction_values.split(",") if value.strip()]

    if not isinstance(parsed_values, list):
        raise ValueError("PREDICTION_VALUES must be a JSON array or comma-separated list.")

    if len(parsed_values) != input_size:
        raise ValueError(
            f"PREDICTION_VALUES length {len(parsed_values)} does not match "
            f"PREDICTION_INPUT_SIZE {input_size}."
        )

    try:
        numeric_values = [float(value) for value in parsed_values]
    except ValueError as exc:
        raise ValueError("PREDICTION_VALUES must contain only numeric values.") from exc

    return torch.tensor([numeric_values], dtype=torch.float32)


def load_model() -> tuple[nn.Module, str]:
    model_source = resolve_model_source()
    model = mlflow.pytorch.load_model(model_source)
    model.eval()
    return model, model_source


def predict_tensor(model: nn.Module, sample_input: torch.Tensor) -> tuple[torch.Tensor, int]:
    with torch.no_grad():
        prediction = model(sample_input)
    predicted_class = int(torch.argmax(prediction, dim=1).item())
    return prediction, predicted_class


def predict_values(values: list[float], input_size: int) -> dict[str, object]:
    if len(values) != input_size:
        raise ValueError(
            f"Input feature length {len(values)} does not match expected input size {input_size}."
        )

    model, model_source = load_model()
    sample_input = torch.tensor([values], dtype=torch.float32)
    prediction, predicted_class = predict_tensor(model, sample_input)
    return {
        "model_source": model_source,
        "output_shape": list(prediction.shape),
        "predicted_class": predicted_class,
        "scores": prediction.squeeze(0).tolist(),
    }


def main() -> int:
    input_size = int(os.getenv("PREDICTION_INPUT_SIZE", str(DEFAULT_INPUT_SIZE)))
    model, model_source = load_model()
    sample_input = resolve_sample_input(input_size)
    prediction, predicted_class = predict_tensor(model, sample_input)
    logger.info(
        "Prediction successful. model_source=%s output_shape=%s predicted_class=%s",
        model_source,
        tuple(prediction.shape),
        predicted_class,
    )
    return 0


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        raise SystemExit(1) from exc
