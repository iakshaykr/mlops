import os
import subprocess
import sys
from pathlib import Path

try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path("/Workspace/Users/<databricks-user>/mlops/src/training/bootstrap_databricks_env.py")

PROJECT_ROOT = _this_file.parents[2]
DEFAULT_LIBRARIES_PATH = "/Volumes/<catalog>/<schema>/biometric_data/databricks-libs"


def run_pip_install(*args: str) -> None:
    command = [sys.executable, "-m", "pip", "install", *args]
    print(f"Running: {' '.join(command)}")
    subprocess.run(command, check=True)


def resolve_libraries_path() -> Path:
    return Path(os.getenv("DATABRICKS_LIBRARIES_PATH", DEFAULT_LIBRARIES_PATH))


def install_dependencies() -> int:
    libraries_path = resolve_libraries_path()
    if not libraries_path.exists():
        raise FileNotFoundError(
            f"Databricks dependency path not found: {libraries_path}"
        )

    requirements_path = libraries_path / "requirements-databricks.txt"
    if requirements_path.is_file():
        run_pip_install("-r", str(requirements_path))
    else:
        print(f"Requirements file not found, skipping: {requirements_path}")

    wheel_candidates = sorted(libraries_path.glob("*.whl"))
    if not wheel_candidates:
        raise FileNotFoundError(
            f"No wheel files found in Databricks dependency path: {libraries_path}"
        )

    latest_wheel = max(wheel_candidates, key=lambda path: path.stat().st_mtime)
    run_pip_install("--force-reinstall", str(latest_wheel))
    print(f"Installed Databricks project wheel: {latest_wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(install_dependencies())
