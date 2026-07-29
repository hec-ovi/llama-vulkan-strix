#!/usr/bin/env python3
"""Full served bench for one loaded model (no reload mid-suite).

1) Prefill table: ~no prefill (64), 16k, 32k. Best-of-N prefill + decode.
2) Decode average, no prefill: 10 short-prompt runs, mean/median decode t/s.
3) Decode average at 32k with cache: one 32k fill (cache_prompt=true), then
   10 more with the same prompt and cache_prompt=true; mean/median decode.

Writes one JSON object to stdout. Progress on stderr.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request

WORDS = (
    "unified memory bandwidth kernel scheduler cache tensor batch decode "
    "prefill vulkan shader queue latency throughput allocator device driver "
    "context slot token weight quantize layer expert router attention head"
).split()


def build_prompt(target_tokens: int, run: int) -> str:
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


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def run_depth(
    url: str, depth: int, runs: int, decode_tokens: int, timeout: float
) -> dict:
    prompt_n = 0
    prefill = decode = 0.0
    samples = []
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
        p = float(timings["prompt_per_second"])
        d = float(timings["predicted_per_second"])
        prefill = max(prefill, p)
        decode = max(decode, d)
        samples.append({"prompt_n": prompt_n, "prefill": p, "decode": d})
    return {
        "depth": depth,
        "label": "no_prefill" if depth <= 128 else f"{depth // 1024}k",
        "prompt_n": prompt_n,
        "prefill_best": prefill,
        "decode_best": decode,
        "samples": samples,
    }


def decode_avg_short(
    url: str, depth: int, runs: int, decode_tokens: int, timeout: float
) -> dict:
    """10 (or N) independent short prompts: pure decode average, no long prefill."""
    rates = []
    prompt_n = 0
    for run in range(runs):
        result = post_completion(
            url,
            {
                "prompt": build_prompt(depth, 1000 + run),
                "n_predict": decode_tokens,
                "ignore_eos": True,
                "cache_prompt": False,
            },
            timeout,
        )
        timings = result["timings"]
        prompt_n = timings["prompt_n"]
        rates.append(float(timings["predicted_per_second"]))
        print(
            f"  short decode {run + 1}/{runs}: {rates[-1]:.1f} t/s "
            f"(prompt_n={prompt_n})",
            file=sys.stderr,
        )
    return {
        "mode": "no_prefill_decode_avg",
        "prompt_n": prompt_n,
        "decode_tokens": decode_tokens,
        "rates": rates,
        **{f"decode_{k}": v for k, v in stats(rates).items()},
    }


def decode_avg_cached_32k(
    url: str, depth: int, runs: int, decode_tokens: int, timeout: float
) -> dict:
    """Fill 32k once with cache_prompt, then N more same-prompt cached decodes.

    If the server reports cache_n=0 after the fill (common with --cache-ram 0),
    fall back to a few cold 32k runs so we still get a decode-at-32k average
    without re-prefiling ten full 32k prompts.
    """
    prompt = build_prompt(depth, 7777)

    # Fill: pay prefill, enable cache for later reuse.
    fill = post_completion(
        url,
        {
            "prompt": prompt,
            "n_predict": decode_tokens,
            "ignore_eos": True,
            "cache_prompt": True,
        },
        timeout,
    )
    fill_t = fill["timings"]
    fill_cache = int(fill_t.get("cache_n") or 0)
    print(
        f"  32k cache fill: prompt_n={fill_t['prompt_n']} "
        f"cache_n={fill_cache} "
        f"prefill={fill_t['prompt_per_second']:.0f} "
        f"decode={fill_t['predicted_per_second']:.1f}",
        file=sys.stderr,
    )

    # Probe whether the next same-prompt request hits cache.
    probe = post_completion(
        url,
        {
            "prompt": prompt,
            "n_predict": decode_tokens,
            "ignore_eos": True,
            "cache_prompt": True,
        },
        timeout,
    )
    probe_t = probe["timings"]
    probe_cache = int(probe_t.get("cache_n") or 0)
    rates = [float(probe_t["predicted_per_second"])]
    cache_ns = [probe_cache]
    prompt_ns = [int(probe_t["prompt_n"])]
    print(
        f"  32k cached probe: {rates[0]:.1f} t/s "
        f"(prompt_n={prompt_ns[0]} cache_n={probe_cache})",
        file=sys.stderr,
    )

    cache_works = probe_cache > 0
    # Full N only when cache actually skips prefill; otherwise a few cold runs.
    remaining = (runs - 1) if cache_works else min(2, max(0, runs - 1))
    if not cache_works:
        print(
            "  note: prompt cache not hitting (cache_n=0); "
            f"only {remaining + 1} cold 32k decode samples "
            "(server likely has --cache-ram 0)",
            file=sys.stderr,
        )

    for run in range(remaining):
        result = post_completion(
            url,
            {
                "prompt": prompt,
                "n_predict": decode_tokens,
                "ignore_eos": True,
                "cache_prompt": True,
            },
            timeout,
        )
        timings = result["timings"]
        rates.append(float(timings["predicted_per_second"]))
        cache_ns.append(int(timings.get("cache_n") or 0))
        prompt_ns.append(int(timings["prompt_n"]))
        print(
            f"  32k cached decode {run + 2}/{remaining + 1}: {rates[-1]:.1f} t/s "
            f"(prompt_n={prompt_ns[-1]} cache_n={cache_ns[-1]})",
            file=sys.stderr,
        )

    # Include fill decode so the average always has the first 32k measurement.
    all_rates = [float(fill_t["predicted_per_second"])] + rates
    return {
        "mode": "cached_32k_decode_avg",
        "cache_works": cache_works,
        "fill": {
            "prompt_n": fill_t["prompt_n"],
            "cache_n": fill_cache,
            "prefill": float(fill_t["prompt_per_second"]),
            "decode": float(fill_t["predicted_per_second"]),
        },
        "decode_tokens": decode_tokens,
        "rates": all_rates,
        "cache_n_samples": cache_ns,
        "prompt_n_samples": prompt_ns,
        "cache_hit_mean": statistics.fmean(cache_ns) if cache_ns else 0.0,
        **{f"decode_{k}": v for k, v in stats(all_rates).items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument(
        "--depths",
        default="64,16384,32768",
        help="prefill depths (default: no-prefill, 16k, 32k)",
    )
    parser.add_argument("--prefill-runs", type=int, default=3, help="best-of-N prefill")
    parser.add_argument(
        "--decode-avg-runs", type=int, default=10, help="N for decode averages"
    )
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--name", default="model", help="label in output JSON")
    args = parser.parse_args()

    out: dict = {"name": args.name, "url": args.url, "prefill_table": [], "decode_avg": {}}

    # 1) Prefill table
    for depth in (int(d) for d in args.depths.split(",")):
        print(f"prefill depth {depth} ...", file=sys.stderr)
        try:
            row = run_depth(
                args.url, depth, args.prefill_runs, args.decode_tokens, args.timeout
            )
        except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as err:
            print(f"bench error at depth {depth}: {err}", file=sys.stderr)
            return 1
        print(
            f"  best prefill={row['prefill_best']:.0f} "
            f"decode={row['decode_best']:.1f} prompt_n={row['prompt_n']}",
            file=sys.stderr,
        )
        out["prefill_table"].append(row)

    # 2) 10x decode, no prefill
    print(f"decode avg no-prefill x{args.decode_avg_runs} ...", file=sys.stderr)
    try:
        out["decode_avg"]["no_prefill"] = decode_avg_short(
            args.url, 64, args.decode_avg_runs, args.decode_tokens, args.timeout
        )
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as err:
        print(f"decode avg no-prefill error: {err}", file=sys.stderr)
        return 1
    np = out["decode_avg"]["no_prefill"]
    print(
        f"  no-prefill decode mean={np['decode_mean']:.1f} "
        f"median={np['decode_median']:.1f} "
        f"min={np['decode_min']:.1f} max={np['decode_max']:.1f}",
        file=sys.stderr,
    )

    # 3) 10x decode at 32k with cache
    print(f"decode avg 32k-cached x{args.decode_avg_runs} ...", file=sys.stderr)
    try:
        out["decode_avg"]["cached_32k"] = decode_avg_cached_32k(
            args.url, 32768, args.decode_avg_runs, args.decode_tokens, args.timeout
        )
    except (urllib.error.URLError, OSError, KeyError, json.JSONDecodeError) as err:
        print(f"decode avg cached-32k error: {err}", file=sys.stderr)
        return 1
    c32 = out["decode_avg"]["cached_32k"]
    print(
        f"  cached-32k decode mean={c32['decode_mean']:.1f} "
        f"median={c32['decode_median']:.1f} "
        f"cache_hit_mean={c32['cache_hit_mean']:.0f}",
        file=sys.stderr,
    )

    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
