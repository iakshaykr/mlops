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

## Architecture

```text
          ┌──────────────────────┐
          │   Kaggle Dataset     │
          └─────────┬────────────┘
                    │
                    ▼
      ┌────────────────────────────┐
      │ GitHub Actions (CI/CD)     │
      │ - Download data            │
      │ - Validate                 │
      │ - Upload to ADLS           │
      └─────────┬──────────────────┘
                │
                ▼
     ┌─────────────────────────────┐
     │ ADLS Gen2 (Data Storage)    │
     │ - Scalable storage          │
     │ - Central data layer        │
     └─────────┬───────────────────┘
               │
               ▼
     ┌─────────────────────────────┐
     │ Training Pipeline (.py)     │
     │ - Data Loader               │
     │ - Model Training            │
     │ - Caching                   │
     └─────────┬───────────────────┘
               │
               ▼
     ┌─────────────────────────────┐
     │ MLflow Tracking             │
     │ - Params                    │
     │ - Metrics                   │
     │ - Model artifacts           │
     └─────────┬───────────────────┘
               │
               ▼
     ┌─────────────────────────────┐
     │ Model Registry (Future)     │
     └─────────────────────────────┘
```

## CI

GitHub Actions runs:

- Python dependency installation
- Python bytecode compilation
- A small pytest smoke test for the dataset/model pipeline
- Kaggle dataset download and ADLS upload
- Databricks job trigger via `src/training/trigger_databricks_job.py`
- MLflow model registration via `src/training/register_model.py`
- Unity Catalog model promotion via `src/training/copy_uc_model_version.py`

## Databricks Job Trigger

This repo includes `src/training/trigger_databricks_job.py` to call the Databricks Jobs API `run-now` endpoint for a configured Databricks job.
It also includes `src/training/register_model.py` to register the latest trained MLflow model in a separate pipeline stage.
For Databricks jobs, `src/training/bootstrap_databricks_env.py` installs the uploaded wheel and `requirements-databricks.txt` from the configured dependency path before training starts.
`src/training/databricks_job_entrypoint.py` bootstraps the Databricks environment and then starts training.
`src/training/preprocess_databricks_job.py` runs distributed Spark preprocessing in Databricks when preprocessing mode is set to `spark`.

## Databricks Dependency Artifacts

The repo now builds a wheel from `pyproject.toml` and uploads these dependency artifacts to ADLS:

- `dist/biometric_mlops-0.1.0-py3-none-any.whl`
- `requirements-databricks.txt`

Current upload target in ADLS:

- `datacontainer/databricks-libs/`

Recommended Databricks usage:

1. Run `python src/training/bootstrap_databricks_env.py` as the first Databricks task step
2. Keep `data.libraries_path` in [config.yaml](/Users/akshaykumar/mlops/mlops/configs/config.yaml) pointed at the mounted volume or storage path that contains the uploaded artifacts
3. Run the training task after bootstrap completes
4. Keep dataset files and dependency artifacts in separate ADLS paths

Example Databricks job command sequence:

```bash
python src/training/bootstrap_databricks_env.py
python src/training/train.py
```

Required GitHub secrets:

- `AZURE_CREDENTIALS`: JSON credentials for the Azure service principal used by the workflow

Required GitHub Actions variables:

- `AZURE_STORAGE_ACCOUNT`: `<storage-account>`
- `AZURE_STORAGE_CONTAINER`: `<container>`
- `DATABRICKS_HOST`: `https://<databricks-workspace-host>`
- `MLFLOW_EXPERIMENT_NAME`: `/Users/<databricks-user>/biometric-training`
- `PROD_MODEL_NAME`: `<catalog>.<schema>.prod_model`
- `DATABRICKS_JOB_ID`: `<databricks-job-id>`

Set these in GitHub under `Settings` -> `Secrets and variables` -> `Actions` -> `Variables`.

The workflow signs in to Azure, gets a Microsoft Entra access token for Azure Databricks, and uses that token to call the Databricks Jobs API.

Model registration runs after the Databricks training job completes and registers the latest run from experiment `/Users/<databricks-user>/biometric-training` with run name `biometric-simple-model` as model `biometric_model`.
