"""The plain Docker Compose command starts the selected 27B ROCmFP4 model."""

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

MODEL = (
    "Qwen3.6-27b/"
    "Qwen3.6-27B-OBLITERATED-MTP-ROCmFP4-STRIX-embF16-imatrix-headQ6.gguf"
)


def test_plain_compose_uses_rocmfp4_mtp_service():
    assert set(COMPOSE["services"]) == {"llm"}
    service = COMPOSE["services"]["llm"]
    assert service["build"]["dockerfile"] == "tools/Dockerfile.rocmfp4"
    assert "image" not in service
    assert "--spec-type" in service["command"]
    assert "draft-mtp" in service["command"]


def test_example_env_selects_27b_obliterated_at_native_max_context():
    assert ENV_EXAMPLE["ROCMFP4_MODEL"] == MODEL
    assert ENV_EXAMPLE["ROCMFP4_CHAT_TEMPLATE"] == "Qwen3.6-27b/chat_template.jinja"
    assert ENV_EXAMPLE["ROCMFP4_ALIAS"] == "qwen3.6-27b-obliterated-mtp"
    per_slot = int(ENV_EXAMPLE["ROCMFP4_CTX_PER_SLOT"])
    parallel = int(ENV_EXAMPLE["ROCMFP4_PARALLEL"])
    total = int(ENV_EXAMPLE["ROCMFP4_CTX_TOTAL"])
    assert (per_slot, parallel, total) == (262144, 5, 1310720)
    assert total == per_slot * parallel
    assert "${ROCMFP4_CTX_TOTAL:-1310720}" in COMPOSE_TEXT
    assert "${ROCMFP4_PARALLEL:-5}" in COMPOSE_TEXT
    assert "--no-kv-unified" in COMPOSE["services"]["llm"]["command"]


def test_default_profile_stays_below_host_memory_budget():
    service = COMPOSE["services"]["llm"]
    command = service["command"]
    assert service["mem_limit"] == "${ROCMFP4_MEMORY_LIMIT:-90g}"
    assert service["memswap_limit"] == "${ROCMFP4_MEMORY_LIMIT:-90g}"
    assert ENV_EXAMPLE["ROCMFP4_MEMORY_LIMIT"] == "90g"
    assert ENV_EXAMPLE["ROCMFP4_KV_TYPE"] == "q8_0"
    assert ENV_EXAMPLE["ROCMFP4_DRAFT_KV_TYPE"] == "q4_0"
    assert command[command.index("-ctk") + 1] == "${ROCMFP4_KV_TYPE:-q8_0}"
    assert command[command.index("-ctv") + 1] == "${ROCMFP4_KV_TYPE:-q8_0}"
    assert command[command.index("--spec-draft-type-k") + 1] == (
        "${ROCMFP4_DRAFT_KV_TYPE:-q4_0}"
    )
    assert command[command.index("--spec-draft-type-v") + 1] == (
        "${ROCMFP4_DRAFT_KV_TYPE:-q4_0}"
    )
    assert command[command.index("--cache-ram") + 1] == "0"
