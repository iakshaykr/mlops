import os
import sys
from pathlib import Path


try:
    _this_file = Path(__file__).resolve()
except NameError:
    _this_file = Path("/Workspace/Users/<databricks-user>/mlops/src/training/databricks_job_entrypoint.py")

PROJECT_ROOT = _this_file.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.bootstrap_databricks_env import install_dependencies


def main() -> int:
    if os.getenv("DATABRICKS_SKIP_BOOTSTRAP", "false").lower() != "true":
        install_dependencies()
        os.environ["DATABRICKS_SKIP_BOOTSTRAP"] = "true"

    import torch
    from src.training.train import load_config, main as train_main

    torch.manual_seed(load_config().get("seed", 42))
    train_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
