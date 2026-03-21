from pathlib import Path
import shutil

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def download_dataset(dataset_ref: str) -> Path:
    path = Path(kagglehub.dataset_download(dataset_ref))
    print(f"Path to dataset files: {path}")
    return path


def prepare_local_data(
    dataset_ref: str, raw_target: str = "data/IRIS and FINGERPRINT DATASET"
) -> Path:
    source_root = download_dataset(dataset_ref)
    dataset_dir = source_root / "IRIS and FINGERPRINT DATASET"
    target_path = PROJECT_ROOT / raw_target

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            "Downloaded dataset does not contain the expected `IRIS and FINGERPRINT DATASET` folder."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        shutil.rmtree(target_path)

    shutil.copytree(dataset_dir, target_path)
    print(f"Copied dataset to: {target_path}")
    return target_path

if __name__ == "__main__":
    prepare_local_data("ninadmehendale/multimodal-iris-fingerprint-biometric-data")
