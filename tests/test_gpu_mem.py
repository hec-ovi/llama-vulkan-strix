"""End-to-end tests for scripts/gpu_mem.py against a fake amdgpu sysfs tree.

We build a temp /sys/class/drm with the real counter filenames, then drive the
script through its CLI (subprocess) and assert both the parsed numbers and the
verify exit codes. No GPU needed.
"""
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "gpu_mem.py"
MIB = 1024 * 1024


def make_card(root: Path, name: str, vram_used_mib, vram_total_mib, gtt_used_mib, gtt_total_mib):
    dev = root / name / "device"
    dev.mkdir(parents=True)
    (dev / "mem_info_vram_used").write_text(str(int(vram_used_mib * MIB)))
    (dev / "mem_info_vram_total").write_text(str(int(vram_total_mib * MIB)))
    (dev / "mem_info_gtt_used").write_text(str(int(gtt_used_mib * MIB)))
    (dev / "mem_info_gtt_total").write_text(str(int(gtt_total_mib * MIB)))


def run(root: Path, *args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--sysfs-root", str(root), *args],
        capture_output=True,
        text=True,
    )


def test_json_reports_card_split(tmp_path):
    make_card(tmp_path, "card0", vram_used_mib=256, vram_total_mib=512,
              gtt_used_mib=8192, gtt_total_mib=65536)
    r = run(tmp_path, "--json")
    assert r.returncode == 0, r.stderr
    cards = json.loads(r.stdout)
    assert len(cards) == 1
    c = cards[0]
    assert c["card"] == "card0"
    assert round(c["vram_used_mib"]) == 256
    assert round(c["gtt_used_mib"]) == 8192


def test_ignores_non_amdgpu_card(tmp_path):
    # a card with no mem_info_* files (e.g. a display-only node) is skipped
    (tmp_path / "card1" / "device").mkdir(parents=True)
    make_card(tmp_path, "card0", 256, 512, 8192, 65536)
    cards = json.loads(run(tmp_path, "--json").stdout)
    assert [c["card"] for c in cards] == ["card0"]


def test_verify_pass_when_model_in_gtt(tmp_path):
    make_card(tmp_path, "card0", vram_used_mib=300, vram_total_mib=512,
              gtt_used_mib=16000, gtt_total_mib=65536)
    r = run(tmp_path, "--verify", "--min-gtt-mib", "8000", "--max-vram-mib", "1024")
    assert r.returncode == 0, r.stdout
    assert "PASS" in r.stdout


def test_verify_fail_when_weights_leak_to_vram(tmp_path):
    make_card(tmp_path, "card0", vram_used_mib=12000, vram_total_mib=16000,
              gtt_used_mib=200, gtt_total_mib=65536)
    r = run(tmp_path, "--verify", "--min-gtt-mib", "8000", "--max-vram-mib", "1024")
    assert r.returncode == 1
    assert "leaked into the VRAM" in r.stdout


def test_verify_fail_when_model_not_loaded(tmp_path):
    make_card(tmp_path, "card0", vram_used_mib=200, vram_total_mib=512,
              gtt_used_mib=100, gtt_total_mib=65536)
    r = run(tmp_path, "--verify", "--min-gtt-mib", "8000", "--max-vram-mib", "1024")
    assert r.returncode == 1
    assert "does not appear to be loaded into GTT" in r.stdout


def test_verify_fail_when_no_card(tmp_path):
    r = run(tmp_path, "--verify")
    assert r.returncode == 1
    assert "no amdgpu card" in r.stdout
