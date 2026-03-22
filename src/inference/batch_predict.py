import json
import logging
import os
from pathlib import Path

import mlflow.pytorch
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_MODEL_DIR = "saved_model"
DEFAULT_MODEL_NAME = "catalog.schema.biometric_model"
DEFAULT_INPUT_SIZE = 3072
DEFAULT_BATCH_SIZE = 32

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


def load_input_data(input_path: str, input_size: int) -> pd.DataFrame:
    """Load input data from CSV, Parquet, or JSON file."""
    if input_path.endswith(".csv"):
        df = pd.read_csv(input_path)
    elif input_path.endswith(".parquet"):
        df = pd.read_parquet(input_path)
    elif input_path.endswith(".json"):
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    logger.info(f"Loaded data with shape {df.shape} from {input_path}")

    # Validate that we have enough features
    feature_cols = [col for col in df.columns if col not in [
        "id", "label", "target"]]
    if len(feature_cols) < input_size:
        raise ValueError(
            f"Input file has {len(feature_cols)} feature columns, but expected {input_size}"
        )

    return df


def extract_features(df: pd.DataFrame, input_size: int, feature_cols: list[str] | None = None) -> torch.Tensor:
    """Extract feature tensor from dataframe."""
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in [
            "id", "label", "target"]][:input_size]

    features_array = df[feature_cols].values.astype("float32")
    return torch.tensor(features_array, dtype=torch.float32)


def load_model() -> tuple[nn.Module, str]:
    model_source = resolve_model_source()
    model = mlflow.pytorch.load_model(model_source)
    model.eval()
    return model, model_source


def batch_predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> list[dict[str, object]]:
    """Run batch predictions on dataloader."""
    results = []
    model.to(device)

    with torch.no_grad():
        for batch_idx, batch_tensor in enumerate(dataloader):
            batch_tensor = batch_tensor[0].to(device)
            predictions = model(batch_tensor)
            predicted_classes = torch.argmax(predictions, dim=1).cpu().numpy()
            scores = predictions.cpu().numpy()

            for idx, (pred_class, score) in enumerate(zip(predicted_classes, scores)):
                results.append(
                    {
                        "batch_idx": batch_idx,
                        "sample_idx": idx,
                        "predicted_class": int(pred_class),
                        "scores": score.tolist(),
                        "confidence": float(score[int(pred_class)]),
                    }
                )

    return results


def save_predictions_to_volume(
    results: list[dict[str, object]],
    input_df: pd.DataFrame,
    output_volume_path: str,
    model_source: str,
) -> None:
    """Save batch predictions directly to Databricks volume."""
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError(
            "databricks-sdk not available. Install with: pip install databricks-sdk")

    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df["id"] = input_df.get("id", range(len(input_df))).values
    results_df["model_source"] = model_source
    results_df["batch_timestamp"] = pd.Timestamp.now()

    # Save to local CSV first
    local_file = "batch_predictions.csv"
    results_df.to_csv(local_file, index=False)

    # Upload directly to Databricks volume
    try:
        # Initialize Databricks client
        client = WorkspaceClient(
            host=os.getenv("DATABRICKS_HOST"),
            token=os.getenv("DATABRICKS_TOKEN")
        )

        # Generate unique filename
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        volume_file_path = f"{output_volume_path}/batch_predictions_{timestamp}.csv"

        # Upload file to volume
        with open(local_file, "rb") as f:
            client.files.upload(volume_file_path, f, overwrite=True)

        logger.info(f"Batch predictions saved to volume: {volume_file_path}")

    except Exception as e:
        logger.error(f"Could not save to volume: {e}")
        # Fallback: keep local file
        logger.info(f"Predictions saved locally as: {local_file}")
        raise


def save_predictions_to_csv(
    results: list[dict[str, object]],
    input_df: pd.DataFrame,
    output_path: str,
    model_source: str,
) -> None:
    """Save batch predictions to CSV file."""
    results_df = pd.DataFrame(results)
    results_df["id"] = input_df.get("id", range(len(input_df))).values
    results_df["model_source"] = model_source
    results_df["batch_timestamp"] = pd.Timestamp.now()

    results_df.to_csv(output_path, index=False)
    logger.info(f"Batch predictions saved to CSV: {output_path}")


def main() -> int:
    # Configuration from environment
    input_path = os.getenv("BATCH_INPUT_PATH")
    output_volume_path = os.getenv(
        "BATCH_OUTPUT_VOLUME_PATH", "/Volumes/iakshaykr/default/prod/prod_predictions/")
    output_csv = os.getenv("BATCH_OUTPUT_CSV")
    batch_size = int(os.getenv("BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    input_size = int(
        os.getenv("PREDICTION_INPUT_SIZE", str(DEFAULT_INPUT_SIZE)))
    device = os.getenv("DEVICE", "cpu")

    if not input_path:
        raise ValueError("BATCH_INPUT_PATH environment variable is required")

    # Load data
    logger.info(f"Loading input data from {input_path}")
    input_df = load_input_data(input_path, input_size)

    # Extract features
    features_tensor = extract_features(input_df, input_size)
    dataset = TensorDataset(features_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model, model_source = load_model()
    logger.info(f"Loaded model from {model_source}")

    # Run batch predictions
    logger.info(f"Running batch predictions with batch size {batch_size}")
    results = batch_predict(model, dataloader, device=device)

    logger.info(f"Generated {len(results)} predictions")

    # Save results
    if output_volume_path and output_volume_path != "":
        try:
            save_predictions_to_volume(
                results, input_df, output_volume_path, model_source)
        except ImportError as e:
            logger.warning(
                f"Could not save to volume: {e}. Try CSV export instead.")

    if output_csv:
        save_predictions_to_csv(results, input_df, output_csv, model_source)

    logger.info("Batch prediction complete")
    return 0


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO,
                            format="%(levelname)s %(name)s: %(message)s")
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch prediction failed: %s", exc)
        raise SystemExit(1) from exc
