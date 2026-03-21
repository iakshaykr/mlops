import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml


DEFAULT_DATABRICKS_HOST = "https://adb-6701353639688524.4.azuredatabricks.net"
DEFAULT_JOB_NAME = "Mlops"
DEFAULT_GIT_URL = "https://github.com/iakshaykr/mlops.git"
DEFAULT_GIT_PROVIDER = "gitHub"
DEFAULT_GIT_BRANCH = "main"
DEFAULT_ENTRYPOINT = "src/training/databricks_job_entrypoint.py"
DEFAULT_PREPROCESS_ENTRYPOINT = "src/training/preprocess_databricks_job.py"
DEFAULT_LIBRARIES_PATH = "/Volumes/iakshaykr/default/biometric_data/databricks-libs"
DEFAULT_WAIT_TIMEOUT_SECONDS = 7200
DEFAULT_POLL_INTERVAL_SECONDS = 30


def databricks_api_request(
    host: str,
    token: str,
    endpoint: str,
    method: str = "GET",
    payload: dict | None = None,
) -> dict:
    request_data = None
    if payload is not None:
        request_data = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url=f"{host}{endpoint}",
        data=request_data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Databricks API returned HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Databricks host {host}: {exc.reason}") from exc


def get_auth() -> tuple[str, str]:
    host = os.getenv("DATABRICKS_HOST", DEFAULT_DATABRICKS_HOST).rstrip("/")
    token = os.getenv("DATABRICKS_AAD_TOKEN") or os.getenv("DATABRICKS_TOKEN")
    if not token:
        raise ValueError("Set DATABRICKS_AAD_TOKEN or DATABRICKS_TOKEN.")
    return host, token


def load_config() -> dict:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def write_github_output(name: str, value: str) -> None:
    github_output = os.getenv("GITHUB_OUTPUT")
    if not github_output:
        return
    with open(github_output, "a", encoding="utf-8") as output_file:
        output_file.write(f"{name}={value}\n")


def build_job_settings() -> dict:
    config = load_config()
    preprocessing_config = config.get("data", {}).get("preprocessing", {})
    preprocessing_mode = preprocessing_config.get("mode", "local")

    tasks = []
    if preprocessing_config.get("enabled", False) and preprocessing_mode == "spark":
        tasks.append(
            {
                "task_key": "preprocess",
                "run_if": "ALL_SUCCESS",
                "spark_python_task": {
                    "python_file": os.getenv(
                        "DATABRICKS_PREPROCESS_ENTRYPOINT",
                        DEFAULT_PREPROCESS_ENTRYPOINT,
                    )
                },
                "environment_key": "train_environment",
                "timeout_seconds": 0,
            }
        )

    train_task = {
        "task_key": "train",
        "run_if": "ALL_SUCCESS",
        "spark_python_task": {
            "python_file": os.getenv(
                "DATABRICKS_JOB_ENTRYPOINT",
                DEFAULT_ENTRYPOINT,
            )
        },
        "environment_key": "train_environment",
        "timeout_seconds": 0,
    }
    if tasks:
        train_task["depends_on"] = [{"task_key": "preprocess"}]
    tasks.append(train_task)

    return {
        "name": os.getenv("DATABRICKS_JOB_NAME", DEFAULT_JOB_NAME),
        "max_concurrent_runs": 1,
        "tasks": tasks,
        "git_source": {
            "git_url": os.getenv("DATABRICKS_GIT_URL", DEFAULT_GIT_URL),
            "git_provider": os.getenv("DATABRICKS_GIT_PROVIDER", DEFAULT_GIT_PROVIDER),
            "git_branch": os.getenv("DATABRICKS_GIT_BRANCH", DEFAULT_GIT_BRANCH),
        },
        "environments": [
            {
                "environment_key": "train_environment",
                "spec": {
                    "client": "1",
                    "dependencies": [
                        "pip",
                        "setuptools",
                        "wheel",
                    ],
                    "environment_version": "5",
                },
            }
        ],
        "queue": {"enabled": True},
        "performance_target": "STANDARD",
    }


def find_job_by_name(host: str, token: str, job_name: str) -> dict | None:
    has_more = True
    page_token = None

    while has_more:
        query_params = {"limit": "25", "name": job_name}
        if page_token:
            query_params["page_token"] = page_token
        endpoint = f"/api/2.1/jobs/list?{urllib.parse.urlencode(query_params)}"
        response = databricks_api_request(host=host, token=token, endpoint=endpoint)

        for job in response.get("jobs", []):
            settings = job.get("settings", {})
            if settings.get("name") == job_name:
                return job

        has_more = response.get("has_more", False)
        page_token = response.get("next_page_token")

    return None


def ensure_job(host: str, token: str) -> int:
    job_settings = build_job_settings()
    existing_job = find_job_by_name(host, token, job_settings["name"])

    if existing_job:
        job_id = existing_job["job_id"]
        databricks_api_request(
            host=host,
            token=token,
            endpoint="/api/2.1/jobs/update",
            method="POST",
            payload={
                "job_id": job_id,
                "new_settings": job_settings,
            },
        )
        print(f"Updated existing Databricks job '{job_settings['name']}' with job_id={job_id}")
        return int(job_id)

    response = databricks_api_request(
        host=host,
        token=token,
        endpoint="/api/2.1/jobs/create",
        method="POST",
        payload=job_settings,
    )
    job_id = int(response["job_id"])
    print(f"Created Databricks job '{job_settings['name']}' with job_id={job_id}")
    return job_id


def wait_for_run_completion(host: str, token: str, run_id: int) -> None:
    poll_interval = int(
        os.getenv("DATABRICKS_POLL_INTERVAL_SECONDS", str(DEFAULT_POLL_INTERVAL_SECONDS))
    )
    timeout_seconds = int(
        os.getenv("DATABRICKS_WAIT_TIMEOUT_SECONDS", str(DEFAULT_WAIT_TIMEOUT_SECONDS))
    )
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        run_response = databricks_api_request(
            host=host,
            token=token,
            endpoint=f"/api/2.1/jobs/runs/get?run_id={run_id}",
        )
        state = run_response.get("state", {})
        life_cycle_state = state.get("life_cycle_state")
        result_state = state.get("result_state")
        state_message = state.get("state_message", "")

        print(
            f"Databricks run {run_id}: "
            f"life_cycle_state={life_cycle_state}, result_state={result_state}, "
            f"message={state_message}"
        )

        if life_cycle_state == "TERMINATED":
            if result_state == "SUCCESS":
                return
            raise RuntimeError(
                f"Databricks run {run_id} terminated with result_state={result_state}: "
                f"{state_message}"
            )

        if life_cycle_state in {"SKIPPED", "INTERNAL_ERROR", "BLOCKED"}:
            raise RuntimeError(
                f"Databricks run {run_id} ended with life_cycle_state={life_cycle_state}: "
                f"{state_message}"
            )

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Timed out waiting for Databricks run {run_id} after {timeout_seconds} seconds."
    )


def run_job(host: str, token: str, job_id: int) -> int:
    response = databricks_api_request(
        host=host,
        token=token,
        endpoint="/api/2.1/jobs/run-now",
        method="POST",
        payload={"job_id": job_id},
    )
    run_id = int(response["run_id"])
    print(f"Triggered Databricks job_id={job_id}, run_id={run_id}")
    write_github_output("databricks_run_id", str(run_id))
    wait_for_run_completion(host=host, token=token, run_id=run_id)
    return run_id


def main() -> int:
    host, token = get_auth()
    job_id = ensure_job(host=host, token=token)
    write_github_output("databricks_job_id", str(job_id))
    run_id = run_job(host=host, token=token, job_id=job_id)
    print(f"Databricks job completed successfully. job_id={job_id}, run_id={run_id}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
