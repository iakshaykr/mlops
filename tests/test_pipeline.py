import sys
from pathlib import Path
import subprocess
import textwrap

try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path("/Workspace/Users/<databricks-user>/mlops/tests/test_pipeline.py")

sys.path.insert(0, str(_this_file.parents[1]))

from PIL import Image

import numpy as np
import pytest

from src.biometric.preprocess import load_metadata, preprocess_dataset


def _create_image(path: Path, pixel_value: int) -> None:
    image = Image.new("L", (16, 16), color=pixel_value)
    image.save(path)


def _torch_runtime_available() -> bool:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import torch; print(torch.__version__)",
        ],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def _build_sample_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "IRIS and FINGERPRINT DATASET" / "1"
    left_dir = dataset_root / "left"
    right_dir = dataset_root / "right"
    fingerprint_dir = dataset_root / "Fingerprint"

    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)
    fingerprint_dir.mkdir(parents=True)

    _create_image(left_dir / "left_1.bmp", 10)
    _create_image(right_dir / "right_1.bmp", 20)
    _create_image(fingerprint_dir / "finger_1.bmp", 30)
    return tmp_path / "IRIS and FINGERPRINT DATASET"


def test_dataset_and_model_smoke(tmp_path: Path) -> None:
    if not _torch_runtime_available():
        pytest.skip("Skipping torch-dependent smoke test because local torch runtime is unavailable.")

    dataset_root = _build_sample_dataset(tmp_path)
    smoke_script = textwrap.dedent(
        f"""
        import sys
        from pathlib import Path

        sys.path.insert(0, {str(_this_file.parents[1])!r})
        from src.biometric.loader import BiometricDataset
        from src.biometric.model import SimpleModel

        dataset = BiometricDataset(dataset_root={str(dataset_root)!r}, image_size=32)
        features, label = dataset[0]
        assert tuple(features.shape) == (3072,)
        assert int(label) == 0

        model = SimpleModel(input_size=3072, hidden_size=64, output_size=1)
        output = model(features.unsqueeze(0))
        assert tuple(output.shape) == (1, 1)
        """
    )
    result = subprocess.run([sys.executable, "-c", smoke_script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_parallel_preprocessing_cache(tmp_path: Path) -> None:
    dataset_root = _build_sample_dataset(tmp_path)
    output_root = tmp_path / "preprocessed-cache"
    try:
        preprocess_dataset(
            dataset_root=dataset_root,
            output_root=output_root,
            image_size=32,
            num_workers=2,
        )
    except (NotImplementedError, PermissionError) as exc:
        pytest.skip(f"Skipping multiprocessing preprocessing test in restricted environment: {exc}")

    metadata = load_metadata(output_root / "metadata.json")
    assert metadata is not None
    assert metadata["image_size"] == 32
    assert metadata["num_samples"] == 1
    feature_path = Path(metadata["records"][0]["feature_path"])
    features = np.load(feature_path)
    assert features.shape == (3072,)
    assert int(metadata["records"][0]["label"]) == 0


def test_preprocessing_cache_invalidates_on_image_size_change(tmp_path: Path) -> None:
    dataset_root = _build_sample_dataset(tmp_path)
    output_root = tmp_path / "preprocessed-cache"
    preprocess_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        image_size=32,
        num_workers=1,
    )
    first_metadata = load_metadata(output_root / "metadata.json")

    preprocess_dataset(
        dataset_root=dataset_root,
        output_root=output_root,
        image_size=16,
        num_workers=1,
    )
    second_metadata = load_metadata(output_root / "metadata.json")

    assert first_metadata["image_size"] == 32
    assert second_metadata["image_size"] == 16
