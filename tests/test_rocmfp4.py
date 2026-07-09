"""Invariants of the ROCmFP4 + MTP stack (docker-compose.rocmfp4.yml + its
Dockerfile). This stack deliberately does what the base stack must NOT: it builds
the custom fork from source and mounts /dev/kfd. These tests pin that intent so a
future edit cannot quietly turn it back into (or away from) what it needs to be.

Parsed with PyYAML for structure; the Dockerfile is checked as text. Both run
with only pytest + pyyaml, no Docker and no GPU.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE_PATH = ROOT / "docker-compose.rocmfp4.yml"
BASE_COMPOSE_PATH = ROOT / "docker-compose.yml"
DOCKERFILE = (ROOT / "tools" / "Dockerfile.rocmfp4").read_text()

COMPOSE_TEXT = COMPOSE_PATH.read_text()
COMPOSE = yaml.safe_load(COMPOSE_TEXT)
SVC = COMPOSE["services"]["rocmfp4-llm"]
CMD = SVC["command"]


def _after(flag):
    """The token following `flag` in the command list (its value)."""
    i = CMD.index(flag)
    return CMD[i + 1]


def _default_host_ports(compose_text):
    """Host ports a compose file publishes, resolving ${VAR:-default} to default."""
    ports = set()
    for m in re.finditer(r'"\$\{[A-Z0-9_]+:-(\d+)\}:\d+"', compose_text):
        ports.add(int(m.group(1)))
    return ports


def test_builds_the_fork_not_a_stock_image():
    # The whole point: this cannot be a pulled stock image; it must build the fork.
    assert SVC["build"]["dockerfile"] == "tools/Dockerfile.rocmfp4"
    assert "git clone" in DOCKERFILE
    assert "charlie12345/rocmfp4-llama" in DOCKERFILE
    assert "build-strix-rocmfp4-mtp.sh" in DOCKERFILE


def test_runs_compute_on_vulkan_not_rocm():
    # -dev Vulkan0: the math runs on RADV, even though ROCm is present.
    assert _after("-dev") == "Vulkan0"
    assert _after("--spec-draft-device") == "Vulkan0"


def test_needs_both_dri_and_kfd():
    devices = SVC["devices"]
    assert any(d.startswith("/dev/dri") for d in devices)
    # The deliberate opposite of the base stack: the HIP-linked binary needs kfd.
    assert any(d.startswith("/dev/kfd") for d in devices)


def test_mounts_models_read_only():
    assert any(v.endswith(":/models:ro") for v in SVC["volumes"])


def test_model_path_fails_fast_when_blank():
    # Same guard as the base stack: blank ROCMFP4_MODEL errors at compose time.
    assert "${ROCMFP4_MODEL:?" in COMPOSE_TEXT


def test_mtp_self_speculation_enabled():
    # draft-mtp is the reason these builds exist; losing it silently is a regression.
    assert _after("--spec-type") == "draft-mtp"
    assert "--spec-draft-ngl" in CMD


def test_forces_gtt_on_both_backends():
    env = "\n".join(SVC["environment"])
    assert "GGML_HIP_ENABLE_UNIFIED_MEMORY=1" in env  # HIP allocations -> GTT
    assert "GGML_VK_PREFER_HOST_MEMORY=1" in env      # Vulkan allocations -> GTT


def test_base_image_overridable_and_defaults_to_fork_pin():
    args = SVC["build"]["args"]
    base = args["BASE_ROCM_DEV_CONTAINER"]
    assert "${ROCMFP4_BASE_IMAGE:-" in base           # one knob to move the base
    assert "rocm/dev-ubuntu-24.04:7.2.1-complete" in base  # fork-tested default


def test_compiles_for_gfx1151():
    assert "gfx1151" in SVC["build"]["args"]["CMAKE_HIP_ARCHITECTURES"]
    assert "gfx1151" in DOCKERFILE  # base-image comment / default arg


def test_server_entrypoint_and_health():
    assert 'ENTRYPOINT ["/app/llama-server"]' in DOCKERFILE
    assert "/health" in COMPOSE_TEXT


def test_default_port_does_not_clash_with_base_stack():
    base_ports = _default_host_ports(BASE_COMPOSE_PATH.read_text())
    rocmfp4_ports = _default_host_ports(COMPOSE_TEXT)
    assert rocmfp4_ports  # sanity: we actually parsed one
    assert base_ports.isdisjoint(rocmfp4_ports), (base_ports, rocmfp4_ports)
