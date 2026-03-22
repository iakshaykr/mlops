import json
from pathlib import Path

from pyspark.sql import SparkSession

from src.biometric.preprocess import (
    build_metadata,
    cache_is_valid,
    discover_samples,
    preprocess_sample_to_file,
    reset_cache,
)


def preprocess_dataset_spark(
    dataset_root: Path,
    output_root: Path,
    image_size: int,
    num_partitions: int = 8,
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

    spark = SparkSession.builder.appName("biometric-preprocessing").getOrCreate()
    spark_context = spark.sparkContext

    tasks = [
        (
            sample_index,
            str(sample[0]),
            str(sample[1]),
            str(sample[2]),
            int(sample[3]),
            image_size,
            str(features_dir),
        )
        for sample_index, sample in enumerate(samples)
    ]

    records = (
        spark_context.parallelize(tasks, max(1, num_partitions))
        .map(
            lambda task: preprocess_sample_to_file(
                sample_index=task[0],
                left_path=task[1],
                right_path=task[2],
                fingerprint_path=task[3],
                label=task[4],
                image_size=task[5],
                features_dir=task[6],
            )
        )
        .collect()
    )

    with open(metadata_path, "w", encoding="utf-8") as metadata_file:
        json.dump(
            build_metadata(
                dataset_root,
                image_size,
                samples,
                records,
                mode="spark",
            ),
            metadata_file,
        )

    return output_root


if __name__ == "__main__":
    raise SystemExit(
        "Import `preprocess_dataset_spark` from a Databricks job or notebook "
        "and call it with explicit paths."
    )
