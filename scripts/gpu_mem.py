#!/usr/bin/env python3
"""Report (and optionally assert) AMD iGPU memory split: VRAM carve-out vs GTT.

On Strix Halo the dedicated "VRAM" is a small BIOS carve-out; the 128 GB of
unified RAM is reachable by the GPU as GTT. For LLM inference you want the model
weights in GTT, with VRAM near idle. amdgpu exposes both as sysfs counters per
card:

  /sys/class/drm/card<N>/device/mem_info_vram_used   (bytes)
  /sys/class/drm/card<N>/device/mem_info_vram_total
  /sys/class/drm/card<N>/device/mem_info_gtt_used
  /sys/class/drm/card<N>/device/mem_info_gtt_total

Usage:
  python3 gpu_mem.py                      # human table
  python3 gpu_mem.py --json               # machine-readable
  python3 gpu_mem.py --verify --min-gtt-mib 4096 --max-vram-mib 1024
                                          # exit 1 unless the load is in GTT, not VRAM

--sysfs-root overrides /sys/class/drm (used by the tests).
"""
import argparse
import json
import sys
from pathlib import Path

MIB = 1024 * 1024
FIELDS = (
    "mem_info_vram_used",
    "mem_info_vram_total",
    "mem_info_gtt_used",
    "mem_info_gtt_total",
)


def _read_int(path: Path):
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def read_cards(sysfs_root: Path):
    """Return one dict per amdgpu card that exposes the mem_info counters."""
    cards = []
    for card_dir in sorted(sysfs_root.glob("card[0-9]*")):
        dev = card_dir / "device"
        vram_total = dev / "mem_info_vram_total"
        if not vram_total.exists():
            continue  # not an amdgpu card (no mem_info counters)
        vals = {f: _read_int(dev / f) for f in FIELDS}
        if vals["mem_info_vram_used"] is None or vals["mem_info_gtt_used"] is None:
            continue
        cards.append(
            {
                "card": card_dir.name,
                "vram_used_mib": vals["mem_info_vram_used"] / MIB,
                "vram_total_mib": (vals["mem_info_vram_total"] or 0) / MIB,
                "gtt_used_mib": vals["mem_info_gtt_used"] / MIB,
                "gtt_total_mib": (vals["mem_info_gtt_total"] or 0) / MIB,
            }
        )
    return cards


def verify(cards, min_gtt_mib, max_vram_mib):
    """A card passes if GTT carries the load and VRAM stays near idle.

    Returns (ok, lines). Checks across all cards combined so a multi-node
    machine still answers "is the model in GTT?".
    """
    lines = []
    if not cards:
        return False, ["FAIL: no amdgpu card with mem_info counters found"]
    gtt_used = sum(c["gtt_used_mib"] for c in cards)
    vram_used = sum(c["vram_used_mib"] for c in cards)
    ok = True
    if gtt_used < min_gtt_mib:
        ok = False
        lines.append(
            f"FAIL: GTT used {gtt_used:.0f} MiB < expected >= {min_gtt_mib} MiB "
            "(model does not appear to be loaded into GTT)"
        )
    else:
        lines.append(f"PASS: GTT used {gtt_used:.0f} MiB >= {min_gtt_mib} MiB")
    if vram_used > max_vram_mib:
        ok = False
        lines.append(
            f"FAIL: VRAM used {vram_used:.0f} MiB > allowed {max_vram_mib} MiB "
            "(weights leaked into the VRAM carve-out)"
        )
    else:
        lines.append(f"PASS: VRAM used {vram_used:.0f} MiB <= {max_vram_mib} MiB")
    return ok, lines


def _table(cards):
    if not cards:
        return "no amdgpu card with mem_info counters found"
    head = f"{'card':<8}{'vram_used':>12}{'vram_total':>12}{'gtt_used':>12}{'gtt_total':>12}"
    rows = [head]
    for c in cards:
        rows.append(
            f"{c['card']:<8}"
            f"{c['vram_used_mib']:>10.0f}M"
            f"{c['vram_total_mib']:>10.0f}M"
            f"{c['gtt_used_mib']:>10.0f}M"
            f"{c['gtt_total_mib']:>10.0f}M"
        )
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sysfs-root", default="/sys/class/drm")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--min-gtt-mib", type=float, default=1024.0)
    ap.add_argument("--max-vram-mib", type=float, default=1024.0)
    args = ap.parse_args(argv)

    cards = read_cards(Path(args.sysfs_root))

    if args.verify:
        ok, lines = verify(cards, args.min_gtt_mib, args.max_vram_mib)
        out = {"ok": ok, "cards": cards, "checks": lines}
        print(json.dumps(out, indent=2) if args.json else "\n".join(lines))
        return 0 if ok else 1

    print(json.dumps(cards, indent=2) if args.json else _table(cards))
    return 0


if __name__ == "__main__":
    sys.exit(main())
