"""Evaluation suite scaffold for Quant-Singularity.

This module is intentionally lightweight for the initial repository commit.
It provides a stable entrypoint for future evaluation logic and keeps the
project structure ready for training-time instrumentation.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


DEFAULT_EVAL_THRESHOLD = 0.0


@dataclass(frozen=True)
class EvaluationResult:
    """Container for evaluation outputs."""

    name: str
    score: float
    threshold: float = DEFAULT_EVAL_THRESHOLD
    passed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_text(path: str | Path) -> str:
    """Read a UTF-8 text file."""

    return Path(path).read_text(encoding="utf-8")


def main() -> int:
    """CLI entrypoint for the evaluation scaffold."""

    print("Quant-Singularity eval suite scaffold is in place.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
