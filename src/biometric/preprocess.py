import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image


def discover_samples(dataset_root: Path) -> list[tuple[Path, Path, Path, int]]:
    samples: list[tuple[Path, Path, Path, int]] = []
    subject_dirs = sorted(
        [
            path
            for path in dataset_root.iterdir()
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


def _load_image(image_path: Path, image_size: int) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("L")
        image = image.resize((image_size, image_size))
        return np.asarray(image, dtype=np.float32).reshape(-1) / 255.0


def compute_dataset_signature(samples: list[tuple[Path, Path, Path, int]]) -> str:
    signature = hashlib.sha256()
    for left_path, right_path, fingerprint_path, label in samples:
        for path in (left_path, right_path, fingerprint_path):
            stat = path.stat()
            signature.update(str(path).encode("utf-8"))
            signature.update(str(stat.st_mtime_ns).encode("utf-8"))
            signature.update(str(stat.st_size).encode("utf-8"))
        signature.update(str(label).encode("utf-8"))
    return signature.hexdigest()


def build_metadata(
    dataset_root: Path,
    image_size: int,
    samples: list[tuple[Path, Path, Path, int]],
    records: list[dict],
    mode: str,
) -> dict:
    return {
        "dataset_root": str(dataset_root.resolve()),
        "image_size": image_size,
        "num_samples": len(records),
        "dataset_signature": compute_dataset_signature(samples),
        "mode": mode,
        "records": records,
    }


def load_metadata(metadata_path: Path) -> dict | None:
    if not metadata_path.is_file():
        return None
    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def cache_is_valid(
    dataset_root: Path,
    output_root: Path,
    image_size: int,
    samples: list[tuple[Path, Path, Path, int]],
) -> bool:
    metadata = load_metadata(output_root / "metadata.json")
    if metadata is None:
        return False

    if metadata.get("dataset_root") != str(dataset_root.resolve()):
        return False
    if metadata.get("image_size") != image_size:
        return False
    if metadata.get("num_samples") != len(samples):
        return False
    if metadata.get("dataset_signature") != compute_dataset_signature(samples):
        return False

    records = metadata.get("records", [])
    return all(Path(record["feature_path"]).is_file() for record in records)


def reset_cache(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)


def preprocess_sample_to_file(
    sample_index: int,
    left_path: str,
    right_path: str,
    fingerprint_path: str,
    label: int,
    image_size: int,
    features_dir: str,
) -> dict:
    left_path_obj = Path(left_path)
    right_path_obj = Path(right_path)
    fingerprint_path_obj = Path(fingerprint_path)
    features_dir_path = Path(features_dir)

    left_iris = _load_image(left_path_obj, image_size)
    right_iris = _load_image(right_path_obj, image_size)
    fingerprint = _load_image(fingerprint_path_obj, image_size)
    features = np.concatenate([left_iris, right_iris, fingerprint]).astype(np.float32)

    output_path = features_dir_path / f"sample_{sample_index:06d}.npy"
    np.save(output_path, features)
    return {
        "feature_path": str(output_path),
        "label": label,
    }


def _preprocess_sample(task: tuple[int, tuple[Path, Path, Path, int], int, str]) -> dict:
    sample_index, sample, image_size, features_dir = task
    left_path, right_path, fingerprint_path, label = sample
    return preprocess_sample_to_file(
        sample_index=sample_index,
        left_path=str(left_path),
        right_path=str(right_path),
        fingerprint_path=str(fingerprint_path),
        label=label,
        image_size=image_size,
        features_dir=features_dir,
    )


def preprocess_dataset(
    dataset_root: Path,
    output_root: Path,
    image_size: int,
    num_workers: int = 4,
    force_rebuild: bool = False,
) -> Path:
    samples = discover_samples(dataset_root)
    if not samples:
        raise ValueError(f"No samples found for preprocessing in {dataset_root}")

    if not force_rebuild and cache_is_valid(dataset_root, output_root, image_size, samples):
        return output_root

    reset_cache(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    features_dir = output_root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.json"

    tasks = [
        (sample_index, sample, image_size, str(features_dir))
        for sample_index, sample in enumerate(samples)
    ]

    max_workers = max(1, num_workers)
    if max_workers == 1:
        records = [_preprocess_sample(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            records = list(executor.map(_preprocess_sample, tasks))

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(build_metadata(dataset_root, image_size, samples, records, mode="local"), metadata_file)

    return output_root
