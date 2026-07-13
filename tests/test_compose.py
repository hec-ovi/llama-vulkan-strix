"""Invariants of docker-compose.yml: the Vulkan/GTT wiring is what we expect.

Text-based so it runs with only pytest. CI additionally runs
`docker compose config -q` (base and the tools profile) for full schema validation.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = (ROOT / "docker-compose.yml").read_text()
# Active config only, so prose in comments (which mention /dev/kfd, privileged,
# etc. to say we DON'T use them) never satisfies or trips an assertion.
CODE = "\n".join(
    ln for ln in COMPOSE.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
)
ENV_EXAMPLE = {
    key: value
    for line in (ROOT / ".env.example").read_text().splitlines()
    if line and not line.startswith("#") and "=" in line
    for key, value in [line.split("=", 1)]
}


def test_uses_prebuilt_llamacpp_vulkan_image():
    assert "image: ghcr.io/ggml-org/llama.cpp:server-vulkan" in CODE


def test_vulkan_needs_only_dri_not_kfd():
    assert "/dev/dri:/dev/dri" in CODE
    # Vulkan path must not pull in the ROCm/KFD node; that would be scope creep.
    assert "/dev/kfd" not in CODE
    assert "privileged: true" not in CODE


def test_forces_gtt_over_vram():
    assert "GGML_VK_PREFER_HOST_MEMORY=1" in CODE


def test_mounts_models_read_only():
    assert ":/models:ro" in CODE


def test_server_knobs_present():
    for flag in ("--n-gpu-layers", "--ctx-size", "--parallel", "--model"):
        assert flag in CODE, flag


def test_disables_reasoning_with_supported_server_flag():
    assert "--reasoning" in CODE
    assert '- "off"' in CODE
    assert "--chat-template-kwargs" not in CODE


def test_parallel_defaults_reserve_a_parent_slot_and_divide_exactly():
    per_slot = int(ENV_EXAMPLE["LLM_CTX_PER_SLOT"])
    parallel = int(ENV_EXAMPLE["LLM_PARALLEL"])
    total = int(ENV_EXAMPLE["LLM_CTX_TOTAL"])
    assert parallel >= 5  # one parent plus noob's default four children
    assert total == per_slot * parallel
    assert "${LLM_CTX_TOTAL:-655360}" in CODE
    assert "${LLM_PARALLEL:-5}" in CODE
    assert "--no-kv-unified" in CODE


def test_llm_model_fails_fast_when_blank():
    # A blank/unset LLM_MODEL must fail at compose time (the :? guard), not pass
    # llama-server a bare /models/ directory that errors cryptically at load.
    assert "${LLM_MODEL:?" in CODE


def test_websearch_sidecar_is_opt_in_and_http():
    assert 'profiles: ["tools"]' in CODE
    assert "tools/Dockerfile.websearch" in CODE
    # published on 8000 (the HTTP /mcp port the wrapper binds)
    assert "8000" in CODE
