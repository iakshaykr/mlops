import json
import os
import sys
import urllib.error
import urllib.request


DEFAULT_DATABRICKS_HOST = "https://adb-6701353639688524.4.azuredatabricks.net"
DEFAULT_JOB_ID = 286717033859672


def trigger_job() -> int:
    host = os.getenv("DATABRICKS_HOST", DEFAULT_DATABRICKS_HOST).rstrip("/")
    token = os.getenv("DATABRICKS_AAD_TOKEN") or os.getenv("DATABRICKS_TOKEN")
    job_id = int(os.getenv("DATABRICKS_JOB_ID", str(DEFAULT_JOB_ID)))

    if not token:
        raise ValueError(
            "DATABRICKS_AAD_TOKEN is required to trigger a Databricks job."
        )

    payload = json.dumps({"job_id": job_id}).encode("utf-8")
    request = urllib.request.Request(
        url=f"{host}/api/2.1/jobs/run-now",
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Databricks API returned HTTP {exc.code}: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to reach Databricks host {host}: {exc.reason}") from exc

    run_id = response_body.get("run_id")
    number_in_job = response_body.get("number_in_job")

    print(
        f"Triggered Databricks job {job_id}. "
        f"run_id={run_id}, number_in_job={number_in_job}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(trigger_job())
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
