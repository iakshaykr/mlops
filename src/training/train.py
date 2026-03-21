import os
import random
import sys
import time
from pathlib import Path

import numpy as np

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
from src.biometric.loader import BiometricDataset, PreprocessedBiometricDataset
from src.biometric.model import SimpleModel
from src.biometric.preprocess import preprocess_dataset


def load_config(config_path: Path | None = None) -> dict:
    config_path = config_path or PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset: BiometricDataset, config: dict) -> DataLoader:
    data_config = config["data"]
    seed = config.get("seed", 42)
    num_workers = data_config.get("num_workers", 0)
    pin_memory = data_config.get("pin_memory", False) and torch.cuda.is_available()

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=data_config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def prepare_training_dataset(config: dict, dataset_root: Path):
    preprocessing_config = config["data"].get("preprocessing", {})
    if not preprocessing_config.get("enabled", False):
        return BiometricDataset(
            dataset_root=str(dataset_root),
            image_size=config["data"]["image_size"],
        )

    cache_path = Path(preprocessing_config["cache_path"])
    cache_root = cache_path if cache_path.is_absolute() else PROJECT_ROOT / cache_path
    mode = preprocessing_config.get("mode", "local")
    if mode == "spark":
        raise ValueError(
            "Spark preprocessing must be run as a separate Databricks preprocessing job "
            f"before training. Expected cache at {cache_root}."
        )

    preprocess_dataset(
        dataset_root=dataset_root,
        output_root=cache_root,
        image_size=config["data"]["image_size"],
        num_workers=preprocessing_config.get("num_workers", 4),
        force_rebuild=preprocessing_config.get("force_rebuild", False),
    )

    return PreprocessedBiometricDataset(preprocessed_root=str(cache_root))


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
    seed_everything(config.get("seed", 42))
    dataset_root = resolve_dataset_root(config)

    dataset = prepare_training_dataset(config, dataset_root)
    loader = build_loader(dataset, config)

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
            "num_workers": config["data"].get("num_workers", 0),
            "pin_memory": config["data"].get("pin_memory", False),
            "preprocessing_enabled": config["data"].get("preprocessing", {}).get("enabled", False),
            "preprocessing_num_workers": config["data"].get("preprocessing", {}).get("num_workers", 0),
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
            epoch_start_time = time.perf_counter()
            epoch_loss = 0.0
            for x, y in loader:
                optimizer.zero_grad()
                out = model(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            average_loss = epoch_loss / max(len(loader), 1)
            epoch_duration = max(time.perf_counter() - epoch_start_time, 1e-8)
            samples_per_second = len(dataset) / epoch_duration
            mlflow.log_metric("train_loss", average_loss, step=epoch + 1)
            mlflow.log_metric("epoch_duration_seconds", epoch_duration, step=epoch + 1)
            mlflow.log_metric("samples_per_second", samples_per_second, step=epoch + 1)
            print(f"Epoch {epoch + 1}: loss={average_loss:.4f}")
            print(
                f"Epoch {epoch + 1}: duration={epoch_duration:.2f}s, "
                f"throughput={samples_per_second:.2f} samples/s"
            )

        # Log final loss and the trained model
        mlflow.log_metric("final_loss", average_loss)
        mlflow.pytorch.log_model(model, artifact_path="model")

        print(f"Training done \u2014 MLflow run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
