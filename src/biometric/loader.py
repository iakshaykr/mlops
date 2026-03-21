from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


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
        samples: list[tuple[Path, Path, Path, int]] = []
        subject_dirs = sorted(
            [
                path
                for path in self.dataset_root.iterdir()
                if path.is_dir() and path.name.isdigit()
            ],
            key=lambda path: int(path.name),
        )

        for label, subject_dir in enumerate(subject_dirs):
            left_dir = subject_dir / "left"
            right_dir = subject_dir / "right"
            fingerprint_dir = subject_dir / "Fingerprint"
            if not left_dir.is_dir() or not right_dir.is_dir() or not fingerprint_dir.is_dir():
                continue

            left_images = sorted(
                [path for path in left_dir.iterdir() if path.suffix.lower() == ".bmp"]
            )
            right_images = sorted(
                [path for path in right_dir.iterdir() if path.suffix.lower() == ".bmp"]
            )
            fingerprint_images = sorted(
                [path for path in fingerprint_dir.iterdir() if path.suffix.lower() == ".bmp"]
            )

            sample_count = min(len(left_images), len(right_images), len(fingerprint_images))
            for idx in range(sample_count):
                samples.append(
                    (left_images[idx], right_images[idx], fingerprint_images[idx], label)
                )

        return samples

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
