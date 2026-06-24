<h1 align="center">llama-vulkan-strix</h1>

<p align="center">
  <strong>llama.cpp OpenAI-compatible server on the Vulkan backend, for testing GGUF models on an AMD Strix Halo APU (gfx1151). Weights load into GTT (unified RAM), not the small VRAM carve-out, and there is a script to prove it. Optional keyless web-search MCP sidecar.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AMD-Strix_Halo-ED1C24?logo=amd&logoColor=white" alt="AMD Strix Halo" />
  <img src="https://img.shields.io/badge/backend-Vulkan-AC162C?logo=vulkan&logoColor=white" alt="Vulkan" />
  <img src="https://img.shields.io/badge/llama.cpp-server-000000" alt="llama.cpp" />
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License" />
</p>

---

## What this is

A lean Docker Compose wrapper around the prebuilt `ghcr.io/ggml-org/llama.cpp:server-vulkan` image. Point it at a folder of GGUF files, set one in `.env`, bring it up, and you have an OpenAI-compatible endpoint on `:8080`. Swap the model in `.env` and restart to test the next one.

No ROCm, no `/dev/kfd`, no privileged container. The Vulkan backend needs only `/dev/dri` and your GPU group IDs. It builds nothing: the base stack is a single pulled image.

## Quick start

Prerequisites: an AMD Strix Halo box (Ryzen AI Max+, gfx1151) on a recent amdgpu kernel, Docker + Compose, and some GGUF models on disk.

```bash
cp .env.example .env
# edit .env: set MODELS_DIR, LLM_MODEL, and your RENDER_GID / VIDEO_GID
#   getent group render | cut -d: -f3
#   getent group video  | cut -d: -f3

docker compose up -d
docker compose logs -f llm        # watch it load
```

Call it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"llm","messages":[{"role":"user","content":"hi"}]}'
```

Test a different model: edit `LLM_MODEL` in `.env`, then `docker compose up -d` again.

## GTT, not VRAM

On Strix Halo the dedicated "VRAM" is a small BIOS carve-out; the 128 GB of unified RAM is reachable by the GPU as GTT. You want model weights in GTT, with VRAM near idle.

The compose file sets `GGML_VK_PREFER_HOST_MEMORY=1` on the server. In llama.cpp's Vulkan backend this is a presence check (any value enables it), and it makes the allocator request host-visible/GTT memory first, with VRAM only as a fallback. On gfx1151 the backend is UMA and already prefers GTT by default; setting it makes that explicit and guaranteed.

Prove it after the model loads:

```bash
scripts/verify-gtt.sh --min-gtt-mib 14000     # ~14 GB model; adjust to yours
```

It waits for `/health`, then reads the kernel's amdgpu counters under `/sys/class/drm/card*/device/` (`mem_info_gtt_used`, `mem_info_vram_used`, in bytes) and asserts GTT carries the load while VRAM stays idle. `scripts/gpu_mem.py` is the underlying tool (`--json` for a raw snapshot). These sysfs counters are the source of truth on Strix Halo, where `rocm-smi` can misreport against the tiny VRAM pool. `amdgpu_top` and `radeontop` show the same split live.

## Web search (optional)

An opt-in sidecar runs [hec-ovi/websearch-skill](https://github.com/hec-ovi/websearch-skill) (keyless web search + page fetch to clean Markdown) as an MCP server over HTTP, so the models you test, or any MCP client, can call `web_search` / `web_fetch` as a tool.

```bash
docker compose --profile tools up -d
# MCP endpoint (streamable-http): http://localhost:8000/mcp
```

Point an MCP client at that URL. Tools: `web_search`, `web_fetch`, `web_open`, `arxiv_search`, `github_search`.

The bundled `websearch mcp` command speaks stdio only; `tools/websearch_http.py` flips the same FastMCP server to HTTP so it is reachable over the network. If you do not want a sidecar at all, call it directly instead:

```bash
uvx --from git+https://github.com/hec-ovi/websearch-skill websearch web-search "your query" --json
```

## Layout

```text
docker-compose.yml          llm service (+ optional websearch sidecar)
.env.example                model, ports, GPU group IDs
scripts/gpu_mem.py          read amdgpu VRAM vs GTT counters; --verify mode
scripts/verify-gtt.sh       wait for /health, then assert model is in GTT
tools/websearch_http.py     websearch MCP server over HTTP (sidecar entry point)
tools/Dockerfile.websearch  the sidecar image (built only with --profile tools)
tests/                      compose invariants, gpu_mem parser, wrapper
```

## License

[MIT](LICENSE) for the build glue here. This repo pulls prebuilt images and mounts your models read-only; llama.cpp is MIT, and the GGUF weights you mount carry their own licenses (Gemma, Llama, Qwen, etc.). You are responsible for complying with each model's terms.
