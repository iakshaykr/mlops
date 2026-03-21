import torch
from torch.utils.data import DataLoader
import yaml
from pathlib import Path

from src.biometric.download import prepare_local_data
from src.biometric.loader import BiometricDataset
from src.biometric.model import SimpleModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: Path | None = None) -> dict:
    config_path = config_path or PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def main() -> None:
    config = load_config()
    dataset_root = PROJECT_ROOT / config["data"]["raw_path"]

    if not dataset_root.exists():
        print("Local dataset not found. Downloading from Kaggle...")
        prepare_local_data(dataset_ref=config["data"]["kaggle_dataset"], raw_target=config["data"]["raw_path"])

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
        print(f"Epoch {epoch + 1}: loss={average_loss:.4f}")

    print("Training done")


if __name__ == "__main__":
    torch.manual_seed(load_config().get("seed", 42))
    main()
