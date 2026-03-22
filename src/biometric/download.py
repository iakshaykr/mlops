import logging
import shutil
from pathlib import Path

import kagglehub

try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path("/Workspace/Users/<databricks-user>/mlops/src/biometric/download.py")

PROJECT_ROOT = _this_file.parents[2]

VOLUME_PATH = Path("/Volumes/<catalog>/<schema>/biometric_data")
DEFAULT_DATASET_REF = "ninadmehendale/multimodal-iris-fingerprint-biometric-data"
DEFAULT_RAW_TARGET = "data/IRIS and FINGERPRINT DATASET"

logger = logging.getLogger(__name__)


def download_dataset(dataset_ref: str) -> Path:
    path = Path(kagglehub.dataset_download(dataset_ref))
    logger.info("Downloaded dataset to temporary path: %s", path)
    return path


def prepare_local_data(dataset_ref: str, raw_target: str = DEFAULT_RAW_TARGET) -> Path:
    # Prefer UC volume (ADLS) if available
    if VOLUME_PATH.exists():
        logger.info("Using dataset from UC volume: %s", VOLUME_PATH)
        return VOLUME_PATH

    # Fall back to Kaggle download
    logger.info("UC volume not found, downloading dataset from Kaggle.")
    source_root = download_dataset(dataset_ref)
    dataset_dir = source_root / "IRIS and FINGERPRINT DATASET"
    target_path = PROJECT_ROOT / raw_target

    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            "Downloaded dataset does not contain the expected "
            "`IRIS and FINGERPRINT DATASET` folder."
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if target_path.exists():
        shutil.rmtree(target_path)

    shutil.copytree(dataset_dir, target_path)
    logger.info("Copied dataset to: %s", target_path)
    return target_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    prepare_local_data(DEFAULT_DATASET_REF)
