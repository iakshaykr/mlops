import sys
from pathlib import Path

try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path("/Workspace/Users/akshaykr9531@gmail.com/mlops/src/training/train.py")

PROJECT_ROOT = _this_file.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
import yaml
import mlflow
import mlflow.pytorch

from src.biometric.download import prepare_local_data
from src.biometric.loader import BiometricDataset
from src.biometric.model import SimpleModel


def load_config(config_path: Path | None = None) -> dict:
    config_path = config_path or PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_dataset_root(config: dict) -> Path:
    """Prefer UC volume path (ADLS) if available, else fall back to local + Kaggle download."""
    volume_path = config["data"].get("volume_path")
    if volume_path:
        vp = Path(volume_path)
        if vp.exists():
            print(f"Using dataset from UC volume: {vp}")
            return vp
        print(f"UC volume path not found ({vp}), falling back to local path.")

    dataset_root = PROJECT_ROOT / config["data"]["raw_path"]
    if not dataset_root.exists():
        print("Local dataset not found. Downloading from Kaggle...")
        prepare_local_data(
            dataset_ref=config["data"]["kaggle_dataset"],
            raw_target=config["data"]["raw_path"],
        )
    return dataset_root


def main() -> None:
    config = load_config()
    dataset_root = resolve_dataset_root(config)

    dataset = BiometricDataset(
        dataset_root=str(dataset_root),
        image_size=config["data"]["image_size"],
    )
    loader = DataLoader(dataset, batch_size=config["data"]["batch_size"], shuffle=True)

    if len(dataset) == 0:
        raise ValueError(
            "Dataset is empty. Download the Kaggle dataset or verify `data.raw_path`."
        )

    sample_features, _ = dataset[0]
    input_size = sample_features.numel()

    if input_size != config["model"]["input_size"]:
        raise ValueError(
            f"Configured input_size={config['model']['input_size']} does not match "
            f"dataset feature size={input_size}."
        )

    mlflow.set_experiment("/Users/akshaykr9531@gmail.com/biometric-training")

    with mlflow.start_run(run_name="biometric-simple-model") as run:
        # Log all config parameters
        mlflow.log_params({
            "seed": config.get("seed", 42),
            "image_size": config["data"]["image_size"],
            "batch_size": config["data"]["batch_size"],
            "epochs": config["training"]["epochs"],
            "learning_rate": config["training"]["lr"],
            "input_size": config["model"]["input_size"],
            "hidden_size": config["model"]["hidden_size"],
            "output_size": config["model"]["output_size"],
            "dataset_size": len(dataset),
            "data_source": str(dataset_root),
        })

        model = SimpleModel(
            input_size=input_size,
            hidden_size=config["model"]["hidden_size"],
            output_size=config["model"]["output_size"],
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])
        loss_fn = torch.nn.CrossEntropyLoss()

        for epoch in range(config["training"]["epochs"]):
            epoch_loss = 0.0
            for x, y in loader:
                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            average_loss = epoch_loss / max(len(loader), 1)
            mlflow.log_metric("train_loss", average_loss, step=epoch + 1)
            print(f"Epoch {epoch + 1}: loss={average_loss:.4f}")

        # Log final loss and the trained model
        mlflow.log_metric("final_loss", average_loss)
        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"Training done \u2014 MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    torch.manual_seed(load_config().get("seed", 42))
    main()
