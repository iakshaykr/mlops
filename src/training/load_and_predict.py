import logging

from src.inference.predict import main

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        raise SystemExit(1) from exc
