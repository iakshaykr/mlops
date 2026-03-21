import json
import os
import sys
import time
import urllib.error
import urllib.request


DEFAULT_DATABRICKS_HOST = "https://adb-6701353639688524.4.azuredatabricks.net"
DEFAULT_JOB_ID = 286717033859672
DEFAULT_POLL_INTERVAL_SECONDS = 30
DEFAULT_TIMEOUT_SECONDS = 7200


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


def write_github_output(name: str, value: str) -> None:
    github_output = os.getenv("GITHUB_OUTPUT")
    if not github_output:
        return

    with open(github_output, "a", encoding="utf-8") as output_file:
        output_file.write(f"{name}={value}\n")


def wait_for_run_completion(host: str, token: str, run_id: int) -> None:
    poll_interval = int(
        os.getenv("DATABRICKS_POLL_INTERVAL_SECONDS", str(DEFAULT_POLL_INTERVAL_SECONDS))
    )
    timeout_seconds = int(
        os.getenv("DATABRICKS_WAIT_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
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


def trigger_job() -> int:
    host = os.getenv("DATABRICKS_HOST", DEFAULT_DATABRICKS_HOST).rstrip("/")
    token = os.getenv("DATABRICKS_AAD_TOKEN") or os.getenv("DATABRICKS_TOKEN")
    job_id = int(os.getenv("DATABRICKS_JOB_ID", str(DEFAULT_JOB_ID)))

    if not token:
        raise ValueError(
            "DATABRICKS_AAD_TOKEN is required to trigger a Databricks job."
        )

    response_body = databricks_api_request(
        host=host,
        token=token,
        endpoint="/api/2.1/jobs/run-now",
        method="POST",
        payload={"job_id": job_id},
    )

    run_id = response_body.get("run_id")
    number_in_job = response_body.get("number_in_job")
    wait_for_completion = os.getenv("DATABRICKS_WAIT_FOR_COMPLETION", "true").lower() == "true"

    print(
        f"Triggered Databricks job {job_id}. "
        f"run_id={run_id}, number_in_job={number_in_job}"
    )
    write_github_output("databricks_run_id", str(run_id))

    if wait_for_completion:
        wait_for_run_completion(host=host, token=token, run_id=int(run_id))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(trigger_job())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
