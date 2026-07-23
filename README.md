<h1 align="center">llama-vulkan-strix</h1>

<p align="center">
  <strong>llama.cpp OpenAI-compatible server on the Vulkan backend, for testing GGUF models on an AMD Strix Halo APU (gfx1151). Weights load into GTT (unified RAM), not the small VRAM carve-out, and there is a script to prove it.</strong>
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

## Parallel agent capacity

`llama-server` treats `--ctx-size` as the total KV cache shared by its server
slots, while `--parallel` selects the number of slots. See the
[official server option reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
This stack names both sides of that relationship in `.env`:

- `LLM_CTX_PER_SLOT` is the context available to one request.
- `LLM_PARALLEL` is the number of concurrent request slots.
- `LLM_CTX_TOTAL` must equal `LLM_CTX_PER_SLOT * LLM_PARALLEL`.

The base stack uses fixed, non-unified slots so the division is exact. Keep one
slot free for the main agent. For noob's defaults of four detached children and
a 131,072-token context, use:

```dotenv
LLM_CTX_PER_SLOT=131072
LLM_PARALLEL=5
LLM_CTX_TOTAL=655360
```

The standard model, laguna-s-2.1 (48 layers, 8 KV heads, 128-wide K/V,
sliding-window attention on most layers), loads five 131k slots at about 101 GiB
of GTT: 70 GiB of weights plus roughly 31 GiB of KV cache and compute buffers,
leaving ~15 GiB of the 116 GiB GTT window. The sliding-window layers are what
make that possible; full attention on every layer at this geometry would cost
about 24 GiB of f16 KV per slot and five slots would not fit. Do not copy this
total to a model with a larger KV architecture
without recalculating memory. If a model cannot fit five 131k slots, lower both
noob's `NOOB_CTX` and `LLM_CTX_PER_SLOT`, or lower noob's
`NOOB_TASK_CONCURRENCY` and retain `LLM_PARALLEL >= NOOB_TASK_CONCURRENCY + 1`.

Validate the configured arithmetic before starting the service:

```bash
python3 scripts/check_context_config.py .env
docker compose config -q
```

After startup, verify the runtime slots, not just the Compose text:

```bash
curl -fsS http://localhost:${LLM_PORT:-8080}/slots |
  python3 scripts/check_context_config.py .env --slots-json -
```

## GTT, not VRAM

On Strix Halo the dedicated "VRAM" is a small BIOS carve-out; the 128 GB of unified RAM is reachable by the GPU as GTT. You want model weights in GTT, with VRAM near idle.

The compose file sets `GGML_VK_PREFER_HOST_MEMORY=1` on the server. In llama.cpp's Vulkan backend this is a presence check (any value enables it), and it makes the allocator request host-visible/GTT memory first, with VRAM only as a fallback. On gfx1151 the backend is UMA and already prefers GTT by default; setting it makes that explicit and guaranteed.

Prove it after the model loads:

```bash
scripts/verify-gtt.sh --min-gtt-mib 14000     # ~14 GB model; adjust to yours
```

It waits for `/health`, then reads the kernel's amdgpu counters under `/sys/class/drm/card*/device/` (`mem_info_gtt_used`, `mem_info_vram_used`, in bytes) and asserts GTT carries the load while VRAM stays idle. `scripts/gpu_mem.py` is the underlying tool (`--json` for a raw snapshot). These sysfs counters are the source of truth on Strix Halo, where `rocm-smi` can misreport against the tiny VRAM pool. `amdgpu_top` and `radeontop` show the same split live.

## ROCmFP4 + MTP (separate stack)

[plunderstruck](https://huggingface.co/collections/plunderstruck/rocmfp4-mtp-strix-halo)'s Qwen3.6 GGUFs use custom `Q4_0_ROCMFP4` tensor types that upstream llama.cpp does not know about, so the stock `server-vulkan` image cannot load them. Running them means building the [charlie12345/rocmfp4-llama](https://github.com/charlie12345/rocmfp4-llama) fork (branch `mtp-rocmfp4-strix`) from source. That is a different animal from the base stack, so it lives in its own file, `docker-compose.rocmfp4.yml`, and leaves the base stack and its "no ROCm" guarantee untouched.

The image is built from `ubuntu:26.04` LTS with a pinned [TheRock](https://github.com/ROCm/TheRock) ROCm 7.13 dist tarball (`ROCMFP4_THEROCK_VERSION` in `.env`, default `7.13.0a20260515`, the last 7.13 nightly). 7.13 is the first ROCm line with gfx1151 in the support matrix, so the old `HSA_OVERRIDE_GFX_VERSION` workaround is gone. The 26.04 toolchain also matters for speed: its current `glslc` compiles the Vulkan integer-dot shader variants the old 24.04 base silently skipped (the binary now reports `int dot: 1`), and the runtime carries mesa 26.0.3 RADV. Both backends are compiled in, so `-dev Vulkan0` and `-dev ROCm0` both work at runtime; the compose file mounts `/dev/dri` and `/dev/kfd` because the HIP-linked binary initializes ROCm at startup either way. The runtime image keeps the pruned ROCm libs it actually loads (~2.5 GB total), not the full SDK.

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

It serves the OpenAI API on `:8081` (host), separate from the base stack's port. `ROCMFP4_MODEL`, `ROCMFP4_CTX` (model max is 262144), the TheRock pin, and the gfx target are all in `.env`. Vision is off by default; add `--mmproj /models/Qwen3.6-35B-A3B-MTP-ROCmFP4/mmproj-F32.gguf` to the command to enable the Qwen3-VL projector. `scripts/verify-gtt.sh --min-gtt-mib 18000` proves the load is in GTT here too (pass `LLM_PORT=8081` so it polls the right health endpoint).

Measured throughput at 2k to 32k context, next to the other models on this box, is in [Benchmarks](#benchmarks) below (full per-model detail in [docs/](docs/qwen3.6-35b-a3b-mtp-rocmfp4.md)).

## Benchmarks

All on the same idle Strix Halo box (Radeon 8060S, RADV `STRIX_HALO`), through the actual served stacks: fresh prompts against `/completion`, generation forced to 128 tokens, best of 3 per point (`scripts/bench_server.py`). The laguna row is the base Vulkan stack (stock image, five 131k slots, no MTP, measured 2026-07-22); the Qwen rows are the ROCmFP4 + MTP stack (`-dev Vulkan0`, f16 KV, `-ub 1024`, MTP on, measured 2026-07-09). The arrow spans 2k context to the deepest depth measured for that model (in parentheses). MTP decode is content-dependent (draft acceptance), so treat it as a band, not a fixed number: the 27B swung 23 to 39 t/s across reps of the same config, and real chat (reasoning plus code generation, natural stop) lands the 35B at 77-86 t/s versus the table's 101-119 on predictable prose.

| Model | Active / total | Quant | MTP | Prefill (t/s) | Decode (t/s) |
|---|---|---|:--:|--:|--:|
| laguna-s-2.1 | 8B / 118B MoE | Q4_K_M | no | 293 → 196 (32k) | 22.7 → 19.5 (32k) |
| Qwen3.6-35B-A3B | 3B / 35B MoE | ROCmFP4 | yes | 714 → 707 (32k) | 119 → 101 (32k) |
| Qwen3.6-27B | 27B dense | ROCmFP4 | yes | 217 → 212 (16k) | 39 → 39 (16k) |
| Qwen3.6-27B-OBLITERATED | 27B dense | ROCmFP4 | yes | 213 → 221 (8k) | 37 → 39 (8k) |

Pure batch throughput is higher than the served numbers (MTP's draft context re-processes the prompt, ~15% prefill toll): llama-bench pp2048 for the 35B is 1195 t/s on Vulkan and 1411 t/s on ROCm at `-ub 2048`. Per-model detail, backend and MTP A/Bs, and advertised-vs-measured tables live in [docs/](docs/): [laguna-s-2.1](docs/laguna-s-2.1.md), [35B-A3B](docs/qwen3.6-35b-a3b-mtp-rocmfp4.md), [27B + OBLITERATED](docs/qwen3.6-27b-mtp-rocmfp4.md). A gemma-4-26B-A4B-heretic row (base Vulkan stack) is still to be measured.

## Layout

```text
docker-compose.yml          base llm service
docker-compose.rocmfp4.yml  ROCmFP4 + MTP service (builds the fork; ROCm + Vulkan)
.env.example                model, ports, GPU group IDs, ROCmFP4 knobs
scripts/gpu_mem.py          read amdgpu VRAM vs GTT counters; --verify mode
scripts/verify-gtt.sh       wait for /health, then assert model is in GTT
scripts/bench_server.py     served prefill/decode by context depth (docs/ tables)
tools/Dockerfile.rocmfp4    the ROCmFP4 fork build (server target, gfx1151)
docs/                       per-model benchmarks and run notes
tests/                      compose invariants, gpu_mem parser, wrapper, rocmfp4, bench
```

Run the tests (they need pytest and PyYAML, no Docker and no GPU):

```bash
uvx --with pyyaml pytest tests/ -q
```

## Credits

The ROCmFP4 stack here is packaging and measurement; the actual work belongs to:

- [charlie12345](https://github.com/charlie12345/rocmfp4-llama), creator of the ROCmFP4 format and the rocmfp4-llama fork. In his words, "essentially a 5bit quant cosplaying as a 4bit quant", built for AMD GPUs (RDNA2 to current) with QAT, MTP and Eagle 3 support.
- [plunderstruck](https://huggingface.co/plunderstruck), who quantizes and publishes the Qwen3.6 ROCmFP4 GGUFs served here, including the STRIX hybrids (f16 embeddings, Q6_K head). The [r/StrixHalo announcement thread](https://www.reddit.com/r/StrixHalo/comments/1u0muh0/experimental_amd_strix_halo_gfx1151_quant_of/) for the 27B is a good starting point.
- [wendell / Level1Techs](https://forum.level1techs.com/t/n5-max-proxmox-strix-halo-with-docker-rocm-fp4-and-mtp-ultimate-setup-guide/251182), whose N5 Max guide supplied the host tuning homework and the Strix Halo reference benchmarks this repo compares against.
- [kyuz0 / Donato](https://github.com/kyuz0/amd-strix-halo-toolboxes), whose toolboxes established the TheRock-tarball-into-a-container pattern this Dockerfile follows, and who added ROCmFP4 to the toolboxes.
- The [TheRock](https://github.com/ROCm/TheRock) team at AMD, for shipping ROCm with gfx1151 in the support matrix.

## License

[MIT](LICENSE) for the build glue here. The base stack pulls a prebuilt image; the ROCmFP4 stack builds the charlie12345/rocmfp4-llama fork, itself an MIT llama.cpp derivative. Models are mounted read-only and the GGUF weights carry their own licenses (Gemma, Llama, Qwen, etc.). You are responsible for complying with each model's terms.
