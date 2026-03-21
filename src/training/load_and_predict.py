import os
import sys
from pathlib import Path

import mlflow.pytorch
import torch


DEFAULT_MODEL_DIR = "saved_model"
DEFAULT_INPUT_SIZE = 3072


def resolve_model_path(model_dir: Path) -> Path:
    if (model_dir / "MLmodel").is_file():
        return model_dir

    candidate_paths = sorted(path for path in model_dir.rglob("MLmodel"))
    if not candidate_paths:
        raise FileNotFoundError(f"No MLmodel file found under {model_dir}")
    return candidate_paths[0].parent


def main() -> int:
    model_dir = Path(os.getenv("MODEL_DIR", DEFAULT_MODEL_DIR))
    input_size = int(os.getenv("PREDICTION_INPUT_SIZE", str(DEFAULT_INPUT_SIZE)))

    if not model_dir.exists():
        raise FileNotFoundError(f"Saved model directory not found: {model_dir}")

    resolved_model_path = resolve_model_path(model_dir)
    model = mlflow.pytorch.load_model(str(resolved_model_path))
    model.eval()

    sample_input = torch.zeros((1, input_size), dtype=torch.float32)
    with torch.no_grad():
        prediction = model(sample_input)

    predicted_class = int(torch.argmax(prediction, dim=1).item())
    print(
        f"Prediction successful. output_shape={tuple(prediction.shape)}, "
        f"predicted_class={predicted_class}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
