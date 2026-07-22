#!/usr/bin/env python3
"""Measure served prefill/decode throughput by context depth on llama-server.

Sends fresh prompts (cache_prompt=false) of roughly the requested token depths
to the native /completion endpoint and reads the server's own timings back.
One request per run gives both numbers: prompt_per_second (prefill) and
predicted_per_second (decode, with n_predict forced tokens). Best of N runs,
matching the tables in docs/.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

WORDS = (
    "unified memory bandwidth kernel scheduler cache tensor batch decode "
    "prefill vulkan shader queue latency throughput allocator device driver "
    "context slot token weight quantize layer expert router attention head"
).split()


def build_prompt(target_tokens: int, run: int) -> str:
    """Deterministic filler prose of about target_tokens tokens.

    The run number is woven in so no two runs share a prefix; cache_prompt is
    already false, this also defeats any slot-level prompt reuse.
    """
    words = []
    for i in range(max(1, int(target_tokens * 0.75))):
        words.append(WORDS[(i * 7 + run * 13) % len(WORDS)])
        if i % 17 == 16:
            words.append(f"run{run}sample{i}.")
    return f"Benchmark run {run}. " + " ".join(words)


def post_completion(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{url.rstrip('/')}/completion",
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def run_depth(
    url: str, depth: int, runs: int, decode_tokens: int, timeout: float
) -> dict:
    """Best-of-N prefill and decode rates at one prompt depth."""
    prompt_n = 0
    prefill = decode = 0.0
    for run in range(runs):
        result = post_completion(
            url,
            {
                "prompt": build_prompt(depth, run),
                "n_predict": decode_tokens,
                "ignore_eos": True,
                "cache_prompt": False,
            },
            timeout,
        )
        timings = result["timings"]
        prompt_n = timings["prompt_n"]
        prefill = max(prefill, timings["prompt_per_second"])
        decode = max(decode, timings["predicted_per_second"])
    return {"depth": depth, "prompt_n": prompt_n, "prefill": prefill, "decode": decode}


def render(rows: list[dict]) -> str:
    lines = [
        "| Context depth | Prompt tokens | Prefill (t/s) | Decode (t/s) |",
        "|--------------:|--------------:|--------------:|-------------:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['depth'] // 1024}k | {row['prompt_n']} "
            f"| {row['prefill']:.0f} | {row['decode']:.1f} |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument(
        "--depths",
        default="2048,8192,16384,32768",
        help="comma-separated target prompt token depths",
    )
    parser.add_argument("--runs", type=int, default=3, help="best-of-N (default 3)")
    parser.add_argument(
        "--decode-tokens",
        type=int,
        default=128,
        help="forced generation length per run (default 128)",
    )
    parser.add_argument(
        "--timeout", type=float, default=600.0, help="per-request timeout seconds"
    )
    parser.add_argument("--json", action="store_true", help="emit JSON, not markdown")
    args = parser.parse_args()

    rows = []
    for depth in (int(d) for d in args.depths.split(",")):
        try:
            row = run_depth(args.url, depth, args.runs, args.decode_tokens, args.timeout)
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as error:
            print(f"bench error at depth {depth}: {error}", file=sys.stderr)
            return 1
        print(f"depth {depth}: prefill {row['prefill']:.0f} t/s, "
              f"decode {row['decode']:.1f} t/s ({row['prompt_n']} prompt tokens)",
              file=sys.stderr)
        rows.append(row)

    print(json.dumps(rows, indent=2) if args.json else render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
