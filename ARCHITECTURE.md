# System Architecture

## Overview

This repository implements a multimodal biometric MLOps pipeline around four main concerns:

- data ingestion into Azure Data Lake Storage Gen2
- model training and experiment tracking with Databricks and MLflow
- controlled promotion through candidate, QA, and production model stages
- post-promotion prediction checks against the promoted registry model

## End-to-End Diagram

```text
                              +------------------------------+
                              | GitHub Actions               |
                              | mlops_pipeline.yml           |
                              +--------------+---------------+
                                             |
             +-------------------------------+--------------------------------+
             |                               |                                |
             v                               v                                v
 +------------------------+      +------------------------+      +------------------------+
 | test                   |      | data_pipeline          |      | databricks_job         |
 | - install deps         |      | - download from Kaggle |      | - trigger Databricks   |
 | - compile python       |      | - upload to ADLS       |      |   training job         |
 | - run pytest           |      +-----------+------------+      +-----------+------------+
 +-----------+------------+                  |                               |
             |                               v                               v
             |                    +------------------------+      +------------------------+
             |                    | ADLS Gen2              |      | Databricks training    |
             |                    | raw data + validated   |      | - preprocessing        |
             |                    | model artifacts        |      | - model training       |
             |                    +-----------+------------+      | - MLflow logging       |
             |                                |                   +-----------+------------+
             |                                |                               |
             +--------------------------------+-------------------------------+
                                              |
                                              v
                               +-------------------------------+
                               | model_registration            |
                               | src/training/register_model.py|
                               | registers candidate model     |
                               +---------------+---------------+
                                               |
                                               v
                               +-------------------------------+
                               | model_validation              |
                               | copy_uc_model_version.py      |
                               | validates registered model    |
                               | uploads artifacts to ADLS     |
                               +---------------+---------------+
                                               |
                           +-------------------+-------------------+
                           |                                       |
                           v                                       v
              +----------------------------+          +----------------------------+
              | promote_to_qa              |          | qa_smoke_test              |
              | promote candidate to QA    |--------->| load QA model and predict  |
              +-------------+--------------+          +-------------+--------------+
                            |                                       |
                            +-------------------+-------------------+
                                                |
                                                v
                                   +----------------------------+
                                   | Prod Release Workflow      |
                                   | prod_release.yml           |
                                   +-------------+--------------+
                                                 |
                           +---------------------+----------------------+
                           |                                            |
                           v                                            v
              +----------------------------+               +----------------------------+
              | prod_smoke_test            |               | prod_live                  |
              | load promoted prod model   |               | optional final prediction  |
              | with synthetic input       |               | with supplied feature      |
              +----------------------------+               | vector                     |
                                                           +----------------------------+
```

## Architecture Explanation

The architecture is split deliberately so each layer has one job:

- GitHub Actions handles orchestration, validation, promotion, and release checks.
- ADLS is the persistent storage layer for uploaded data and validated model artifacts.
- Databricks is the compute layer for heavier preprocessing and training.
- MLflow and Unity Catalog are the experiment-tracking and model-governance layer.
- The prediction scripts are the inference verification layer used after promotion.

This separation is the main design choice in the repo. It keeps orchestration, storage, training, registry operations, and inference checks decoupled enough to evolve independently.

## Component Responsibilities

### GitHub Actions

- Orchestrates the candidate and QA pipeline in `mlops_pipeline.yml`.
- Uses a separate `prod_release.yml` workflow for production release actions.
- Provides manual control through `workflow_dispatch`.

### ADLS Gen2

- Stores uploaded raw dataset files from the ingestion stage.
- Stores validated model artifacts used later by QA and production promotion jobs.
- Decouples compute-heavy jobs from artifact persistence.

### Databricks

- Runs the training workload outside GitHub-hosted runners.
- Supports Spark-based preprocessing through `src/training/preprocess_databricks_job.py`.
- Produces MLflow-tracked runs and model artifacts for downstream registration.

### MLflow and Unity Catalog

- Tracks experiments, parameters, metrics, and model artifacts.
- Registers the candidate model version.
- Supports QA and production model names as separate controlled registry targets.

### Prediction Scripts

- `src/training/load_and_predict.py` validates that promoted models can be loaded.
- The smoke tests use synthetic input for low-cost checks.
- `prod_live` supports a real feature vector for a final manual production-style prediction.

### Parallelism and Throughput

- PyTorch training uses `DataLoader(..., num_workers=...)` for parallel batch preparation.
- Local preprocessing has its own `num_workers` setting, separate from training-time loading.
- Spark preprocessing provides a distributed path when local multiprocessing is no longer enough.
- Training logs throughput-style metrics such as epoch duration and samples per second to MLflow.

## Inference Pipeline

The repository does include an inference path, but it is intentionally lightweight and validation-oriented rather than a deployed online service.

```text
Unity Catalog model version
        ->
load_and_predict.py
        ->
resolve model URI or local artifact path
        ->
mlflow.pytorch.load_model(...)
        ->
input tensor creation
        ->
model forward pass
        ->
predicted class and output shape
```

There are three inference-style checkpoints in the current design:

- `qa_smoke_test` loads the promoted QA version and runs a low-cost prediction
- `prod_smoke_test` in the prod release workflow loads the promoted production version and runs a low-cost prediction
- `prod_live` in the prod release workflow optionally runs a final prediction against the exact promoted production version using a manually supplied feature vector

This is enough to demonstrate post-promotion inference verification, but it is not yet a deployed serving endpoint or batch inference service.

## Model Versioning Strategy

Versioning is handled at two levels:

- MLflow run versioning for training artifacts
- Unity Catalog model versioning for release stages

Release path:

```text
MLflow run artifact
      ->
candidate registry model: biometric_model
      ->
QA registry model: QA_MODEL_NAME
      ->
production registry model: PROD_MODEL_NAME
```

Important details:

- registering a trained artifact creates a new candidate version in `biometric_model`
- promoting to QA creates a new version in the QA model namespace
- promoting to production creates a new version in the production model namespace
- the smoke-test jobs consume the explicit `promoted_model_version` output from the promotion job, which avoids ambiguity about which model version was tested

Because QA and production are separate registry models, their version numbers do not need to match the source candidate version numerically. The important invariant is artifact lineage, not equal version numbers across model names.

## Scalability Discussion

This repo now has an explicit scalability story:

- Storage scales independently from compute.
  ADLS holds raw data and validated artifacts, so training and promotion jobs do not depend on local runner disks.

- Compute can move from local Python to distributed Databricks execution.
  The training path is already separated from GitHub runners via the Databricks job trigger, which is the right direction once data size or preprocessing time grows.

- Preprocessing has a local and distributed path.
  Local preprocessing is fine for small experiments, while Spark-based preprocessing allows scaling to larger volumes without changing the overall pipeline shape.

- Model lifecycle scales through staged promotion.
  Candidate, QA, and production model names reduce the risk of directly replacing a live model and make approvals or manual checks easier to insert.

- CI/CD remains lightweight.
  GitHub Actions is used mainly for orchestration, validation, and promotion, not for heavyweight model training, which avoids hitting runner limits too early.

## Scalability Reasoning

The scalability reasoning behind this design is straightforward:

- put storage in ADLS so data size can grow without depending on runner-local disks
- keep training outside GitHub-hosted runners so compute can scale separately from orchestration
- support local preprocessing first, then Spark preprocessing when the dataset or preprocessing cost grows
- use staged registry models so model-release complexity scales more safely than replacing one live model directly

In other words, the repo is designed so the first bottlenecks can be relieved by swapping compute strategy, not by redesigning the entire pipeline.

## Trade-offs

- GitHub Actions plus Databricks is more operationally complex than a purely local workflow, but it is much closer to a real production setup.
- Staged promotion through candidate, QA, and prod is safer than a single registry target, but it adds more jobs and registry objects to manage.
- ADLS-backed artifacts improve durability and reproducibility, but they add storage operations and cloud dependency to the promotion path.
- Lightweight smoke inference is cheap and useful for release validation, but it is not the same as a full serving or batch-inference system.
- Separate orchestration and training layers scale better, but they make provenance more important and require clearer run-to-run linkage.

## Current Limits

- `model_registration` should ideally be tied to the exact Databricks run ID from the same workflow for stronger provenance.
- `prod_live` is useful only when you provide a real feature vector shaped like the trained model input.
- The QA and production jobs still repeat some setup logic that could later be extracted into a reusable action.

## Why This Design Is Reasonable

- It separates storage, orchestration, training, registry, and inference checks.
- It supports gradual hardening from candidate to QA to production.
- It is simple enough for a portfolio repository but still shows a credible production-oriented path.
