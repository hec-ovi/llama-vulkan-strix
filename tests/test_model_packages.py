"""Per-model compose packages under compose/models/ and .env.example blocks."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "compose" / "models"
ENV_TEXT = (ROOT / ".env.example").read_text()
ENV_ACTIVE = {
    key: value
    for line in ENV_TEXT.splitlines()
    if line and not line.startswith("#") and "=" in line
    for key, value in [line.split("=", 1)]
}

EXPECTED = {
    "qwen3.6-27b-heretic.yml": {
        "mtp": True,
        "model_hint": "Qwen3.6-27B-heretic-v2",
    },
    "qwen3.6-35b-heretic.yml": {
        "mtp": False,
        "model_hint": "Qwen3.6-35B-A3B-heretic",
    },
    "laguna-s-2.1.yml": {
        "mtp": False,
        "model_hint": "laguna-s-2.1",
        "sampling": True,
    },
    "gemma-4-26b-a4b.yml": {
        "mtp": False,
        "model_hint": "gemma-4-26b-a4b-abliterated",
    },
}


def test_four_model_packages_exist():
    names = sorted(p.name for p in MODELS_DIR.glob("*.yml"))
    assert names == sorted(EXPECTED)


def test_each_package_replaces_full_command_and_keeps_guards():
    for name, meta in EXPECTED.items():
        text = (MODELS_DIR / name).read_text()
        data = yaml.safe_load(text)
        cmd = data["services"]["llm"]["command"]
        assert "${LLM_MODEL:?" in " ".join(cmd)
        assert "${LLM_CHAT_TEMPLATE:?" in " ".join(cmd)
        assert "--ctx-size" in cmd
        assert "--parallel" in cmd
        assert "--flash-attn" in cmd
        has_mtp = "--spec-type" in cmd and "draft-mtp" in cmd
        assert has_mtp is meta["mtp"], name
        if meta.get("sampling"):
            assert "--temp" in cmd
            assert "0.7" in cmd
            assert "--cache-type-k" in cmd


def test_env_example_documents_each_package_block():
    for name, meta in EXPECTED.items():
        assert f"compose/models/{name}" in ENV_TEXT
        assert meta["model_hint"] in ENV_TEXT


def test_env_example_ships_blank_model_until_package_uncommented():
    assert ENV_ACTIVE["LLM_MODEL"] == ""
    assert ENV_ACTIVE["LLM_CHAT_TEMPLATE"] == ""
    assert ENV_ACTIVE["ROCMFP4_MODEL"] == ""
    assert ENV_ACTIVE["ROCMFP4_CHAT_TEMPLATE"] == ""


def test_env_example_has_rocmfp4_section_and_hf_token_knob():
    assert "ROCmFP4" in ENV_TEXT or "ROCMFP4" in ENV_TEXT
    assert "slow" in ENV_TEXT.lower() or "tens of minutes" in ENV_TEXT
    assert "HF_TOKEN" in ENV_ACTIVE
    assert "Gemma-4-26B-A4B-It-Abliterated-Q5_K_M" in ENV_TEXT


def test_default_ctx_arithmetic_is_five_slots():
    per = int(ENV_ACTIVE["LLM_CTX_PER_SLOT"])
    parallel = int(ENV_ACTIVE["LLM_PARALLEL"])
    total = int(ENV_ACTIVE["LLM_CTX_TOTAL"])
    assert parallel == 5
    assert total == per * parallel
