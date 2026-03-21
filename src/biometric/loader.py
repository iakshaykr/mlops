from pathlib import Path
import json

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

from src.biometric.preprocess import discover_samples


class BiometricDataset(Dataset):
    def __init__(self, dataset_root: str, image_size: int = 32):
        self.dataset_root = Path(dataset_root)
        self.image_size = image_size

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.dataset_root}"
            )

        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError("No valid multimodal samples were found in the dataset.")

    def _build_samples(self) -> list[tuple[Path, Path, Path, int]]:
        return discover_samples(self.dataset_root)

    def _load_image(self, image_path: Path) -> np.ndarray:
        with Image.open(image_path) as image:
            image = image.convert("L")
            image = image.resize((self.image_size, self.image_size))
            return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        left_iris_path, right_iris_path, fingerprint_path, label = self.samples[idx]

        left_iris = self._load_image(left_iris_path)
        right_iris = self._load_image(right_iris_path)
        fingerprint = self._load_image(fingerprint_path)
        data = np.concatenate([left_iris, right_iris, fingerprint]).astype(np.float32)

        data = torch.from_numpy(data)
        label = torch.tensor(label, dtype=torch.long)
        return data, label


class PreprocessedBiometricDataset(Dataset):
    def __init__(self, preprocessed_root: str):
        self.preprocessed_root = Path(preprocessed_root)
        metadata_path = self.preprocessed_root / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Preprocessed dataset metadata not found: {metadata_path}"
            )

        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)

        self.records = metadata.get("records", [])
        if not self.records:
            raise ValueError("No preprocessed records were found in the dataset cache.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        data = np.load(record["feature_path"]).astype(np.float32)
        label = int(record["label"])
        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)
