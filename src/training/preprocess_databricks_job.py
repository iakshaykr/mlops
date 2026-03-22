import logging
import sys
from pathlib import Path
from typing import Any

import yaml

try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path(
        "/Workspace/Users/<databricks-user>/mlops/src/training/preprocess_databricks_job.py"
    )

PROJECT_ROOT = _this_file.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.biometric.preprocess_spark import preprocess_dataset_spark  # noqa: E402

logger = logging.getLogger(__name__)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or PROJECT_ROOT / "configs" / "config.yaml"
    with open(config_path, encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def resolve_dataset_root(config: dict[str, Any]) -> Path:
    volume_path = config["data"].get("volume_path")
    if volume_path:
        volume_root = Path(volume_path)
        if volume_root.exists():
            return volume_root

    return PROJECT_ROOT / config["data"]["raw_path"]


def resolve_cache_root(config: dict[str, Any]) -> Path:
    preprocessing_config = config["data"]["preprocessing"]
    cache_path = Path(preprocessing_config["cache_path"])
    return cache_path if cache_path.is_absolute() else PROJECT_ROOT / cache_path


def main() -> int:
    config = load_config()
    preprocessing_config = config["data"].get("preprocessing", {})
    if preprocessing_config.get("mode", "local") != "spark":
        logger.info("Spark preprocessing mode is disabled; skipping distributed preprocessing.")
        return 0

    preprocess_dataset_spark(
        dataset_root=resolve_dataset_root(config),
        output_root=resolve_cache_root(config),
        image_size=config["data"]["image_size"],
        num_partitions=preprocessing_config.get("num_partitions", 8),
        force_rebuild=preprocessing_config.get("force_rebuild", False),
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
