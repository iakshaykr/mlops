# mlops

Simple multimodal biometric training project using iris and fingerprint images.

## Project Structure

- `src/biometric/download.py`: Downloads the Kaggle dataset and copies it into the repo-local `data/` folder.
- `src/biometric/loader.py`: Builds a PyTorch dataset from the downloaded image folders.
- `src/biometric/model.py`: Defines a small feed-forward classifier.
- `src/training/train.py`: Trains the model using the local dataset and `configs/config.yaml`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download Data

```bash
python src/biometric/download.py
```

This project uses the Kaggle dataset `ninadmehendale/multimodal-iris-fingerprint-biometric-data`.

## Train

```bash
python src/training/train.py
```

## Multimodal Fusion

The model combines iris and fingerprint data through early feature-level fusion.

- one left-iris image is loaded
- one right-iris image is loaded
- one fingerprint image is loaded
- each image is converted to grayscale, resized, and flattened
- the three flattened vectors are concatenated into a single feature vector

Fusion shape with the current defaults:

- image size = `32 x 32`
- one image = `1024` features
- left iris + right iris + fingerprint = `1024 + 1024 + 1024 = 3072` features

That combined `3072`-dimensional vector is the direct input to the classifier.

## Parallel Data Loading

The training pipeline does support multiprocessing during data loading and preprocessing.

- `data.num_workers` in [`config.yaml`](/Users/akshaykumar/mlops/mlops/configs/config.yaml) controls PyTorch `DataLoader` worker processes
- the current default is `2`, so training does not run in a single-worker mode
- `persistent_workers` is enabled automatically when `num_workers > 0`
- `seed_worker(...)` keeps worker-level randomness reproducible across runs

There are two separate concurrency knobs:

- `data.num_workers`: parallel batch loading during model training
- `data.preprocessing.num_workers`: parallel local preprocessing when building the cached feature dataset

So the repo does use multiprocessing; it is just configured conservatively for a small portfolio-sized dataset and GitHub runner friendliness.

## Architecture

See [`ARCHITECTURE.md`](/Users/akshaykumar/mlops/mlops/ARCHITECTURE.md) for the full system diagram, component responsibilities, scalability discussion, and current design limits.

```text
MLOps pipeline -> candidate -> QA
Prod release workflow -> prod -> prod_live prediction
```

Architecture explanation:

- GitHub Actions orchestrates the release flow
- ADLS stores data and validated model artifacts
- Databricks handles heavier preprocessing and training
- MLflow and Unity Catalog manage experiment tracking and versioned model promotion
- smoke tests and `prod_live` verify that the promoted model can actually predict

## Workflows

[`mlops_pipeline.yml`](/Users/akshaykumar/mlops/mlops/.github/workflows/mlops_pipeline.yml) runs:

- Python dependency installation
- Python bytecode compilation
- Ruff lint checks
- Ruff formatting checks
- A small pytest smoke test for the dataset/model pipeline
- Kaggle dataset download and ADLS upload
- MLflow model registration via `src/training/register_model.py`
- Unity Catalog model promotion via `src/training/copy_uc_model_version.py`
- QA promotion and QA smoke test

[`prod_release.yml`](/Users/akshaykumar/mlops/mlops/.github/workflows/prod_release.yml) runs:

- production promotion from a validated ADLS artifact path
- production smoke test
- optional final `prod_live` prediction using a supplied feature vector

## Tests

The repo does include automated tests in [`tests/test_pipeline.py`](/Users/akshaykumar/mlops/mlops/tests/test_pipeline.py).

Current test coverage focuses on:

- dataset and model smoke validation
- preprocessing cache creation
- cache invalidation when preprocessing settings change

Run locally with:

```bash
pytest -q
```

The current suite is intentionally small, but it is real and runs in CI.

## Linting and Formatting

Code quality checks use Ruff via [`pyproject.toml`](/Users/akshaykumar/mlops/mlops/pyproject.toml).

Run locally with:

```bash
ruff check .
ruff format --check .
```

This is also enforced in the main GitHub Actions workflow.

## Inference Pipeline

Inference is implemented through [`load_and_predict.py`](/Users/akshaykumar/mlops/mlops/src/training/load_and_predict.py).

The supported inference paths are:

- local artifact inference from `saved_model/`
- registry-backed inference from a specific Unity Catalog model version
- post-promotion smoke inference in QA and production
- optional final `prod_live` inference with a real feature vector provided at manual workflow dispatch time

Inference flow:

```text
Promoted model version in Unity Catalog
        ->
load_and_predict.py resolves models:/<model_name>/<version>
        ->
mlflow.pytorch.load_model(...)
        ->
feature vector input
        ->
PyTorch forward pass
        ->
predicted class
```

For production-style inference, the workflow uses:

- `qa_smoke_test` to verify the promoted QA version loads and predicts
- `prod_smoke_test` in [`prod_release.yml`](/Users/akshaykumar/mlops/mlops/.github/workflows/prod_release.yml) to verify the promoted production version loads and predicts
- `prod_live` in [`prod_release.yml`](/Users/akshaykumar/mlops/mlops/.github/workflows/prod_release.yml) to run one final manual prediction against the exact promoted production version

## Scalability

- ADLS separates storage from compute, so raw data and validated model artifacts do not depend on the runner filesystem.
- Databricks moves expensive preprocessing and training off GitHub-hosted runners.
- The repo supports both local preprocessing and Spark-based preprocessing, which gives a clean path from small experiments to larger workloads.
- Candidate, QA, and production model stages make the release flow safer as the number of model versions grows.

Scalability reasoning:

- storage can grow independently in ADLS
- compute can scale independently in Databricks
- preprocessing can move from local multiprocessing to Spark without changing the overall pipeline shape
- staged promotion scales operationally better than replacing one live model directly

## Trade-offs

- this design is more realistic than a local-only pipeline, but it is also more complex to configure
- staged QA and prod promotion is safer, but it creates more workflow steps and registry objects
- lightweight smoke inference is useful for release validation, but it is not a full serving architecture

## Performance Metrics

The training pipeline logs both model and throughput metrics to MLflow.

Metrics logged in [`train.py`](/Users/akshaykumar/mlops/mlops/src/training/train.py):

- `train_loss`: per-epoch training loss
- `final_loss`: final loss at the end of training
- `epoch_duration_seconds`: wall-clock time per epoch
- `samples_per_second`: approximate training throughput per epoch

The pipeline also logs useful run parameters such as:

- `batch_size`
- `num_workers`
- `preprocessing_num_workers`
- `dataset_size`
- `image_size`
- `input_size`
- `hidden_size`
- `output_size`

This means the repo already captures both learning behavior and basic operational performance, even though it does not yet include richer evaluation metrics like accuracy, precision, recall, or latency dashboards.

## Model Versioning

This repo uses explicit model stages rather than a single mutable model target.

- `biometric_model` is the candidate model registered from training
- `QA_MODEL_NAME` receives a promoted version after validation
- `PROD_MODEL_NAME` receives a promoted version only after QA passes

Versioning behavior:

- training creates an MLflow run and logs the model artifact
- [`register_model.py`](/Users/akshaykumar/mlops/mlops/src/training/register_model.py) registers that artifact as a new version of `biometric_model`
- [`copy_uc_model_version.py`](/Users/akshaykumar/mlops/mlops/src/training/copy_uc_model_version.py) validates and promotes the candidate artifact into the QA or prod registry model
- each promotion creates a new version in the destination registry model and writes `promoted_model_version` back to the workflow
- the QA and prod smoke tests use that exact promoted version instead of relying on an implicit latest alias

That means version lineage is environment-specific:

- candidate version in `biometric_model`
- corresponding QA version in `QA_MODEL_NAME`
- corresponding production version in `PROD_MODEL_NAME`

The current design gives safe release progression even though the numeric version in QA or prod does not have to match the numeric version in the candidate model.

## Databricks Job Trigger

This repo includes `src/training/trigger_databricks_job.py` to call the Databricks Jobs API `run-now` endpoint for a configured Databricks job.
It also includes `src/training/register_model.py` to register the latest trained MLflow model in a separate pipeline stage.
`src/training/preprocess_databricks_job.py` runs distributed Spark preprocessing in Databricks when preprocessing mode is set to `spark`.

Required GitHub secrets:

- `AZURE_CREDENTIALS`: JSON credentials for the Azure service principal used by the workflow

Required GitHub Actions variables:

- `AZURE_STORAGE_ACCOUNT`: `<storage-account>`
- `AZURE_STORAGE_CONTAINER`: `<container>`
- `DATABRICKS_HOST`: `https://<databricks-workspace-host>`
- `MLFLOW_EXPERIMENT_NAME`: `/Users/<databricks-user>/biometric-training`
- `QA_MODEL_NAME`: `<catalog>.<schema>.qa_model`
- `PROD_MODEL_NAME`: `<catalog>.<schema>.prod_model`
- `DATABRICKS_JOB_ID`: `<databricks-job-id>`

Set these in GitHub under `Settings` -> `Secrets and variables` -> `Actions` -> `Variables`.

Environment progression:

- `biometric_model`: candidate model registered from training
- `QA_MODEL_NAME`: validated model promoted to QA
- `PROD_MODEL_NAME`: QA-approved model promoted to production

Workflow progression:

- `model_registration`
- `model_validation`
- `promote_to_qa`
- `qa_smoke_test`

Prod release workflow progression:

- `promote_to_prod`
- `prod_smoke_test`
- `prod_live` when `prod_live_input_values` is provided on manual dispatch

For `prod_live`, provide `prod_live_input_values` as either:

- a JSON array like `[0.1, 0.2, 0.3, ...]`
- or a comma-separated list like `0.1,0.2,0.3,...`

The vector length must match `PREDICTION_INPUT_SIZE` in the workflow, currently `3072`.

For the production release workflow, provide `validated_artifact_path` as the ADLS path created by validation, for example:

- `prod/123456789-1`

The workflow signs in to Azure, gets a Microsoft Entra access token for Azure Databricks, and uses that token to call the Databricks Jobs API.

Model registration runs after the Databricks training job completes and registers the latest run from experiment `/Users/<databricks-user>/biometric-training` with run name `biometric-simple-model` as model `biometric_model`.
