#!/usr/bin/env python3
"""Validate llama.cpp total/per-slot context math and optional live slot JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{number}: expected NAME=value")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def positive(values: dict[str, str], name: str) -> int:
    raw = values.get(name)
    if raw is None:
        raise ValueError(f"{name} is missing")
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error
    if value < 1:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def validate(values: dict[str, str], child_concurrency: int, noob_ctx: int) -> tuple[int, int, int]:
    per_slot = positive(values, "LLM_CTX_PER_SLOT")
    parallel = positive(values, "LLM_PARALLEL")
    total = positive(values, "LLM_CTX_TOTAL")
    expected = per_slot * parallel
    if total != expected:
        raise ValueError(
            f"LLM_CTX_TOTAL is {total}, expected {per_slot} * {parallel} = {expected}"
        )
    needed = child_concurrency + 1
    if parallel < needed:
        raise ValueError(
            f"LLM_PARALLEL is {parallel}; need at least {needed} to reserve one parent "
            f"slot beside {child_concurrency} child slots"
        )
    if per_slot != noob_ctx:
        raise ValueError(
            f"LLM_CTX_PER_SLOT is {per_slot}, but noob context is {noob_ctx}; align them"
        )
    return per_slot, parallel, total


def validate_slots(source: str, per_slot: int, parallel: int) -> None:
    if source == "-":
        slots = json.load(sys.stdin)
    else:
        slots = json.loads(Path(source).read_text())
    if not isinstance(slots, list):
        raise ValueError("slot JSON must be an array")
    if len(slots) != parallel:
        raise ValueError(f"server reports {len(slots)} slots, expected {parallel}")
    wrong = [slot.get("n_ctx") for slot in slots if slot.get("n_ctx") != per_slot]
    if wrong:
        raise ValueError(
            f"server slot n_ctx values do not all equal {per_slot}: {wrong}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("env_file", type=Path)
    parser.add_argument(
        "--child-concurrency",
        type=int,
        default=4,
        help="noob detached-child limit (default: 4)",
    )
    parser.add_argument(
        "--noob-context",
        type=int,
        default=131072,
        help="noob per-request context (default: 131072)",
    )
    parser.add_argument("--slots-json", help="GET /slots JSON path, or - for stdin")
    args = parser.parse_args()
    try:
        values = load_env(args.env_file)
        per_slot, parallel, total = validate(
            values, args.child_concurrency, args.noob_context
        )
        if args.slots_json:
            validate_slots(args.slots_json, per_slot, parallel)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"context config error: {error}", file=sys.stderr)
        return 1
    print(
        f"ok: {parallel} slots, {per_slot} tokens per slot, {total} total; "
        f"one parent plus up to {args.child_concurrency} children"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
