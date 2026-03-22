import argparse
import json
import math
from collections.abc import Sequence

DEFAULT_INPUT_SIZE = 3072


def build_values(size: int, mode: str, value: float) -> list[float]:
    if mode == "zeros":
        return [0.0] * size
    if mode == "constant":
        return [value] * size
    if mode == "ramp":
        if size == 1:
            return [0.0]
        return [index / (size - 1) for index in range(size)]
    if mode == "sin":
        return [math.sin(index / 32.0) for index in range(size)]
    raise ValueError(f"Unsupported mode: {mode}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a valid JSON payload for 3072-feature model inference."
    )
    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_INPUT_SIZE,
        help=f"Feature vector length. Default: {DEFAULT_INPUT_SIZE}",
    )
    parser.add_argument(
        "--mode",
        choices=["zeros", "constant", "ramp", "sin"],
        default="zeros",
        help="Pattern used to generate the feature vector.",
    )
    parser.add_argument(
        "--value",
        type=float,
        default=0.1,
        help="Constant value to use when --mode constant is selected.",
    )
    parser.add_argument(
        "--wrapped",
        action="store_true",
        help='Wrap the output as {"features": [...]} for JSON request-style usage.',
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    values = build_values(size=args.size, mode=args.mode, value=args.value)
    payload = {"features": values} if args.wrapped else values
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
