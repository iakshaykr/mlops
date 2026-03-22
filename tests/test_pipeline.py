import sys
from pathlib import Path

try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path("/Workspace/Users/<databricks-user>/mlops/tests/test_pipeline.py")

sys.path.insert(0, str(_this_file.parents[1]))

from PIL import Image

from src.biometric.loader import BiometricDataset, PreprocessedBiometricDataset
from src.biometric.model import SimpleModel
from src.biometric.preprocess import load_metadata, preprocess_dataset


def _create_image(path: Path, pixel_value: int) -> None:
    image = Image.new("L", (16, 16), color=pixel_value)
    image.save(path)


def test_dataset_and_model_smoke(tmp_path: Path) -> None:
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

    dataset = BiometricDataset(
        dataset_root=str(tmp_path / "IRIS and FINGERPRINT DATASET"),
        image_size=32,
    )

    features, label = dataset[0]
    assert tuple(features.shape) == (3072,)
    assert int(label) == 0

    model = SimpleModel(input_size=3072, hidden_size=64, output_size=1)
    output = model(features.unsqueeze(0))
    assert tuple(output.shape) == (1, 1)


def test_parallel_preprocessing_cache(tmp_path: Path) -> None:
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

    output_root = tmp_path / "preprocessed-cache"
    preprocess_dataset(
        dataset_root=tmp_path / "IRIS and FINGERPRINT DATASET",
        output_root=output_root,
        image_size=32,
        num_workers=2,
    )

    dataset = PreprocessedBiometricDataset(preprocessed_root=str(output_root))
    features, label = dataset[0]
    assert tuple(features.shape) == (3072,)
    assert int(label) == 0


def test_preprocessing_cache_invalidates_on_image_size_change(tmp_path: Path) -> None:
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

    output_root = tmp_path / "preprocessed-cache"
    preprocess_dataset(
        dataset_root=tmp_path / "IRIS and FINGERPRINT DATASET",
        output_root=output_root,
        image_size=32,
        num_workers=1,
    )
    first_metadata = load_metadata(output_root / "metadata.json")

    preprocess_dataset(
        dataset_root=tmp_path / "IRIS and FINGERPRINT DATASET",
        output_root=output_root,
        image_size=16,
        num_workers=1,
    )
    second_metadata = load_metadata(output_root / "metadata.json")

    assert first_metadata["image_size"] == 32
    assert second_metadata["image_size"] == 16
