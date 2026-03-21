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
