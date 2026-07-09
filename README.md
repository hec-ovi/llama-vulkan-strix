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

For the custom ROCmFP4 quants (Qwen3.6 MTP builds), the stock image cannot read the weights and there is a second, heavier stack in `docker-compose.rocmfp4.yml`. See [ROCmFP4 + MTP](#rocmfp4--mtp-separate-stack) below.

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

## ROCmFP4 + MTP (separate stack)

[plunderstruck](https://huggingface.co/collections/plunderstruck/rocmfp4-mtp-strix-halo)'s Qwen3.6 GGUFs use custom `Q4_0_ROCMFP4` tensor types that upstream llama.cpp does not know about, so the stock `server-vulkan` image cannot load them. Running them means building the [charlie12345/rocmfp4-llama](https://github.com/charlie12345/rocmfp4-llama) fork (branch `mtp-rocmfp4-strix`) from source. That is a different animal from the base stack, so it lives in its own file, `docker-compose.rocmfp4.yml`, and leaves the base stack and its "no ROCm" guarantee untouched.

The math still runs on Vulkan. The run command uses `-dev Vulkan0` (RADV / mesa), which the model card says beats the ROCm backend for ROCmFP4 on Strix Halo. ROCm is in the image only because the fork compiles the HIP backend in and the binary initializes it at startup, which is why this stack mounts `/dev/kfd` where the base stack does not. The ROCm version therefore does not touch inference speed here; it is a build and init dependency, not the compute path. The default base is the fork-tested `rocm/dev-ubuntu-24.04:7.2.1-complete`, overridable with `ROCMFP4_BASE_IMAGE`.

The point of these builds is MTP self-speculative decoding: the model drafts its own tokens through a built-in MTP head (`--spec-type draft-mtp`), running on the same Vulkan device.

Get the model (about 20 GB with the vision projector) into `MODELS_DIR`:

```bash
hf download plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF \
  --local-dir "$MODELS_DIR/Qwen3.6-35B-A3B-MTP-ROCmFP4"
```

Then build and run (first build compiles the fork, so it is slow):

```bash
docker compose -f docker-compose.rocmfp4.yml up -d --build
docker compose -f docker-compose.rocmfp4.yml logs -f rocmfp4-llm
```

It serves the OpenAI API on `:8081` (host), separate from the base stack's port. `ROCMFP4_MODEL`, `ROCMFP4_CTX` (model max is 262144), the base image, and the gfx target are all in `.env`. Vision is off by default; add `--mmproj /models/Qwen3.6-35B-A3B-MTP-ROCmFP4/mmproj-F32.gguf` to the command to enable the Qwen3-VL projector. `scripts/verify-gtt.sh --min-gtt-mib 18000` proves the load is in GTT here too (pass `LLM_PORT=8081` so it polls the right health endpoint).

Measured throughput at 2k to 32k context, next to the other models on this box, is in [Benchmarks](#benchmarks) below (full per-model detail in [docs/](docs/qwen3.6-35b-a3b-mtp-rocmfp4.md)).

## Benchmarks

All on the same idle Strix Halo box (Radeon 8060S, RADV GFX1151), Vulkan compute (`-dev Vulkan0`), f16 KV, best-of-N per point, generation forced to 128 tokens (`ignore_eos`). Prefill and decode are shown as their value at 2k context and at 32k context. MTP decode is content-dependent (draft acceptance), so treat it as a band, not a fixed number. Each model was the only active GPU user during its run.

| Model | Active / total | Quant | MTP | Prefill 2k → 32k (t/s) | Decode 2k → 32k (t/s) |
|---|---|---|:--:|--:|--:|
| Qwen3.6-35B-A3B | 3B / 35B MoE | ROCmFP4 | yes | 551 → 520 | 110 → 93 |

More rows (gemma-4-26B-A4B-heretic, Qwen3.6-27B dense, Qwen3.6-27B-OBLITERATED) are being measured and added. Per-model detail lives in [docs/](docs/qwen3.6-35b-a3b-mtp-rocmfp4.md).

## Layout

```text
docker-compose.yml          base llm service (+ optional websearch sidecar)
docker-compose.rocmfp4.yml  ROCmFP4 + MTP service (builds the fork; ROCm + Vulkan)
.env.example                model, ports, GPU group IDs, ROCmFP4 knobs
scripts/gpu_mem.py          read amdgpu VRAM vs GTT counters; --verify mode
scripts/verify-gtt.sh       wait for /health, then assert model is in GTT
tools/websearch_http.py     websearch MCP server over HTTP (sidecar entry point)
tools/Dockerfile.websearch  the sidecar image (built only with --profile tools)
tools/Dockerfile.rocmfp4    the ROCmFP4 fork build (server target, gfx1151)
docs/                       per-model benchmarks and run notes (ROCmFP4)
tests/                      compose invariants, gpu_mem parser, wrapper, rocmfp4
```

## License

[MIT](LICENSE) for the build glue here. The base stack pulls a prebuilt image; the ROCmFP4 stack builds the charlie12345/rocmfp4-llama fork, itself an MIT llama.cpp derivative. Models are mounted read-only and the GGUF weights carry their own licenses (Gemma, Llama, Qwen, etc.). You are responsible for complying with each model's terms.
