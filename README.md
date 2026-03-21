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

## CI

GitHub Actions runs:

- Python dependency installation
- Python bytecode compilation
- A small pytest smoke test for the dataset/model pipeline
- Kaggle dataset download and ADLS upload
- Databricks wheel and requirements artifact build/upload to ADLS
- Databricks job trigger via `src/training/trigger_databricks_job.py`
- MLflow model registration via `src/training/register_model.py`

## Databricks Job Trigger

This repo includes `src/training/trigger_databricks_job.py` to call the Databricks Jobs API `run-now` endpoint for job `286717033859672`.
It also includes `src/training/register_model.py` to register the latest trained MLflow model in a separate pipeline stage.
For Databricks jobs, `src/training/bootstrap_databricks_env.py` installs the uploaded wheel and `requirements-databricks.txt` from the configured dependency path before training starts.

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

- `AZURE_CREDENTIALS`: JSON credentials for service principal `e59ca002-8cc0-4bc0-ab19-a9aac456b2d3`

The workflow signs in to Azure with that service principal, gets a Microsoft Entra access token for Azure Databricks resource `2ff814a6-3304-4ab8-85cb-cd0e6f879c1d`, and uses that token to call the Databricks Jobs API.

Model registration runs after the Databricks training job completes and registers the latest run from experiment `/Users/akshaykr9531@gmail.com/biometric-training` with run name `biometric-simple-model` as model `biometric_model`.
