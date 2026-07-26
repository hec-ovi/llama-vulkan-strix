"""Invariants for the Laguna ROCmFPX Runtime V2 stack."""

import hashlib
import os
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
COMPOSE_PATH = ROOT / "docker-compose.laguna-rocmfpx.yml"
DOCKERFILE_PATH = ROOT / "tools" / "Dockerfile.laguna-rocmfpx"
ENTRYPOINT_PATH = ROOT / "tools" / "entrypoint.laguna-rocmfpx.sh"
BASE_COMPOSE_PATH = ROOT / "docker-compose.yml"
QWEN_COMPOSE_PATH = ROOT / "docker-compose.rocmfp4.yml"

COMPOSE_TEXT = COMPOSE_PATH.read_text()
DOCKERFILE = DOCKERFILE_PATH.read_text()
COMPOSE = yaml.safe_load(COMPOSE_TEXT)
SVC = COMPOSE["services"]["laguna-llm"]
CMD = SVC["command"]


def _after(flag):
    return CMD[CMD.index(flag) + 1]


def _default_host_ports(compose_text):
    return {
        int(match.group(1))
        for match in re.finditer(r"\$\{[A-Z0-9_]+:-(\d+)\}:\d+", compose_text)
    }


def test_builds_exact_model_card_runtime():
    assert SVC["build"] == {
        "context": "./tools",
        "dockerfile": "Dockerfile.laguna-rocmfpx",
    }
    assert "https://github.com/ciru-ai/ROCmFPX.git" in DOCKERFILE
    assert "090e317b4e2f998a9470faeb076cf841ba72b739" in DOCKERFILE
    assert 'test "$(git rev-parse HEAD)" = "${ROCMFPX_REF}"' in DOCKERFILE
    assert "scripts/build-laguna-strix-vulkan.sh" in DOCKERFILE
    assert 'ENTRYPOINT ["/app/entrypoint.sh"]' in DOCKERFILE


def test_uses_vulkan_without_rocm_device():
    assert _after("--device") == "Vulkan0"
    assert SVC["devices"] == ["/dev/dri:/dev/dri"]


def test_model_mount_and_path_are_safe():
    assert any(volume.endswith(":/models:ro") for volume in SVC["volumes"])
    assert "${LAGUNA_ROCMFPX_MODEL:?" in COMPOSE_TEXT
    assert SVC["build"]["context"] == "./tools"


def test_runtime_is_unprivileged_and_read_only():
    assert SVC["read_only"] is True
    assert SVC["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in SVC["security_opt"]
    assert any(item.startswith("/tmp:") for item in SVC["tmpfs"])
    assert "USER 10001:10001" in DOCKERFILE


def test_uses_validated_128k_profile():
    assert _after("--ctx-size") == "${LAGUNA_ROCMFPX_CTX:-131072}"
    assert _after("--split-mode") == "row"
    assert _after("--flash-attn") == "on"
    assert _after("--cache-type-k") == "f16"
    assert _after("--cache-type-v") == "f16"
    assert _after("--batch-size") == "2048"
    assert _after("--ubatch-size") == "512"
    assert _after("--parallel") == "1"

    env = "\n".join(SVC["environment"])
    assert "GGML_VK_MAX_NODES_PER_SUBMIT=10" in env
    assert "GGML_VK_FA_MAX_WORKGROUPS_X_PER_DISPATCH=4" in env


def test_disables_features_the_model_does_not_have():
    assert _after("--spec-type") == "none"
    assert _after("--reasoning") == "off"
    assert _after("--reasoning-format") == "none"
    assert _after("--reasoning-budget") == "0"
    assert "--no-mmproj" in CMD
    assert "--chat-template-file" not in CMD


def test_uses_standard_llamacpp_host_port():
    base = _default_host_ports(BASE_COMPOSE_PATH.read_text())
    qwen = _default_host_ports(QWEN_COMPOSE_PATH.read_text())
    laguna = _default_host_ports(COMPOSE_TEXT)
    assert base == qwen == laguna == {8080}
    assert "${LAGUNA_ROCMFPX_BIND:-127.0.0.1}" in SVC["ports"][0]


def test_entrypoint_verifies_model_before_starting_server(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"verified model fixture")
    expected = hashlib.sha256(model.read_bytes()).hexdigest()

    server = tmp_path / "server"
    server.write_text("#!/bin/sh\nprintf '%s\\n' \"$@\"\n")
    server.chmod(0o755)

    env = os.environ | {
        "EXPECTED_MODEL_SHA256": expected,
        "LLAMA_SERVER_BIN": str(server),
    }
    result = subprocess.run(
        ["/bin/sh", ENTRYPOINT_PATH, "--model", str(model), "--alias", "laguna"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert str(model) in result.stdout


def test_entrypoint_rejects_wrong_model_hash(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"tampered")
    server = tmp_path / "server"
    server.write_text("#!/bin/sh\nexit 99\n")
    server.chmod(0o755)

    env = os.environ | {
        "EXPECTED_MODEL_SHA256": "0" * 64,
        "LLAMA_SERVER_BIN": str(server),
    }
    result = subprocess.run(
        ["/bin/sh", ENTRYPOINT_PATH, "--model", str(model)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 3
    assert "SHA-256 mismatch" in result.stderr
