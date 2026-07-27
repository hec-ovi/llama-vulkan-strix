"""The plain Docker Compose command starts the stock Vulkan service, and the
repo ships no default model: the user picks one in .env."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
COMPOSE_TEXT = (ROOT / "docker-compose.yml").read_text()
COMPOSE = yaml.safe_load(COMPOSE_TEXT)
ENV_EXAMPLE = {
    key: value
    for line in (ROOT / ".env.example").read_text().splitlines()
    if line and not line.startswith("#") and "=" in line
    for key, value in [line.split("=", 1)]
}


def test_plain_compose_uses_stock_vulkan_service():
    assert set(COMPOSE["services"]) == {"llm"}
    service = COMPOSE["services"]["llm"]
    assert service["image"] == "ghcr.io/ggml-org/llama.cpp:server-vulkan"
    assert "build" not in service
    assert "--spec-type" not in service["command"]


def test_example_env_ships_no_default_model():
    # No model is "the standard" on the repo: .env.example leaves the model
    # and template paths blank so the compose :? guards fail loud until the
    # user picks one. Examples live in comments above the blank lines.
    assert ENV_EXAMPLE["LLM_MODEL"] == ""
    assert ENV_EXAMPLE["LLM_CHAT_TEMPLATE"] == ""
    assert ENV_EXAMPLE["ROCMFP4_MODEL"] == ""
    assert ENV_EXAMPLE["ROCMFP4_CHAT_TEMPLATE"] == ""
    assert "${LLM_MODEL:?" in COMPOSE_TEXT
    assert "${LLM_CHAT_TEMPLATE:?" in COMPOSE_TEXT


def test_rocmfp4_stack_is_not_the_default():
    # The ROCmFP4 + MTP fork moved out of the default file into its own.
    assert "Dockerfile.rocmfp4" not in COMPOSE_TEXT
    assert "/dev/kfd" not in "\n".join(COMPOSE["services"]["llm"]["devices"])
    rocmfp4 = yaml.safe_load((ROOT / "docker-compose.rocmfp4.yml").read_text())
    service = rocmfp4["services"]["llm"]
    assert service["build"]["dockerfile"] == "tools/Dockerfile.rocmfp4"
    assert "draft-mtp" in service["command"]
