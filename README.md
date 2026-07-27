<h1 align="center">llama-vulkan-strix</h1>

<p align="center">
  <strong>llama.cpp GGUF servers for AMD Strix Halo. Standard GGUFs on the stock Vulkan image by default; custom ROCmFP4 + MTP and Laguna ROCmFPX stacks as options. No model is the standard , you pick yours in .env.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AMD-Strix_Halo-ED1C24?logo=amd&logoColor=white" alt="AMD Strix Halo" />
  <img src="https://img.shields.io/badge/backend-Vulkan-AC162C?logo=vulkan&logoColor=white" alt="Vulkan" />
  <img src="https://img.shields.io/badge/llama.cpp-server-000000" alt="llama.cpp" />
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License" />
</p>

---

## What this is

A Docker Compose setup for serving GGUF models on gfx1151. Plain `docker compose up -d` pulls the stock `ghcr.io/ggml-org/llama.cpp:server-vulkan` image and serves whichever standard GGUF you point `LLM_MODEL` at in `.env`, on `:8080`. There is no default model in the repo; `.env.example` lists tested examples and the compose file fails fast until you pick one.

Two optional stacks cover models the stock image cannot load. `docker-compose.rocmfp4.yml` builds the custom ROCmFP4 + MTP fork for plunderstruck's Qwen3.6 ROCmFP4 GGUFs (it mounts `/dev/kfd` because the HIP-linked binary needs ROCm at startup, though compute runs on Vulkan). `docker-compose.laguna-rocmfpx.yml` is the separate Laguna Runtime V2 stack. All three are alternatives on port `8080` , run one at a time.

## Supported models

| Stack | Compose file | Models |
|---|---|---|
| Stock Vulkan (default) | `docker-compose.yml` | Any standard GGUF upstream llama.cpp supports. Tested here: `llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF` (Q8_0), `llmfan46/Qwen3.6-35B-A3B-uncensored-heretic-GGUF` (Q8_0 and the other quants), `Qwen3.6-35B-A3B-Q4_K_M`, `laguna-s-2.1-Q4_K_M`. |
| ROCmFP4 + MTP | `docker-compose.rocmfp4.yml` | [plunderstruck](https://huggingface.co/collections/plunderstruck/rocmfp4-mtp-strix-halo)'s Qwen3.6 ROCmFP4 GGUFs only: 27B, 27B-OBLITERATED, 35B-A3B-MTP. The custom `Q4_0_ROCMFP4` tensors and MTP head need the fork. |
| Laguna ROCmFPX | `docker-compose.laguna-rocmfpx.yml` | The [Laguna S 2.1 Chadrock ROCmFP4 V4 GGUF](https://huggingface.co/jcbtc/Laguna-S-2.1-Chadrock-ROCmFP4-StrixKVSpine-V4-GGUF) only; it needs the pinned Ciru Runtime V2 build. |

## Quick start

Prerequisites: an AMD Strix Halo box (Ryzen AI Max+, gfx1151) on a recent amdgpu kernel, Docker + Compose, and some GGUF models on disk.

```bash
cd ~/workspace/llama-vulkan-strix
cp .env.example .env
# edit .env: set MODELS_DIR, pick a model (LLM_MODEL + LLM_CHAT_TEMPLATE),
# and set your RENDER_GID / VIDEO_GID
#   getent group render | cut -d: -f3
#   getent group video  | cut -d: -f3

docker compose up -d
docker compose logs -f llm
```

Run Compose from this repository. `~/workspace/noob-cli/workspace` belongs to a different Compose project.

Example: the Qwen3.6-27B heretic v2 Q8 (standard GGUF, default stack; three 128k slots):

```bash
hf download llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-GGUF \
  Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q8_0.gguf \
  --local-dir "$MODELS_DIR/Qwen3.6-27B-heretic-v2"
# then in .env:
#   LLM_MODEL=Qwen3.6-27B-heretic-v2/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q8_0.gguf
#   LLM_CHAT_TEMPLATE=Qwen3.6-27B-heretic-v2/chat_template.jinja
#   LLM_CTX_PER_SLOT=131072
#   LLM_PARALLEL=3
#   LLM_CTX_TOTAL=393216
```

The repo does not ship a `chat_template.jinja` with the GGUF; extract the embedded one with gguf-py or drop a vendor-refreshed template next to the model.

Call it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"llm","messages":[{"role":"user","content":"hi"}]}'
```

Serve another standard GGUF by changing `LLM_MODEL`, `LLM_CHAT_TEMPLATE`, and `LLM_ALIAS` in `.env`.

## Context and parallel slots

`llama-server` treats `--ctx-size` as the total KV cache shared by its server
slots, while `--parallel` selects the number of slots. See the
[official server option reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).
Both server stacks use fixed slots, with the same arithmetic under different
env prefixes:

- `LLM_CTX_PER_SLOT` (or `ROCMFP4_CTX_PER_SLOT`) is the context available to one request.
- `LLM_PARALLEL` (or `ROCMFP4_PARALLEL`) is the number of concurrent slots.
- `LLM_CTX_TOTAL` (or `ROCMFP4_CTX_TOTAL`) must equal their product.

```dotenv
LLM_CTX_PER_SLOT=131072
LLM_PARALLEL=5
LLM_CTX_TOTAL=655360
```

The Qwen3.6 model cards set the native limit to 262,144 tokens (see the [upstream Qwen config](https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/config.json)). That limit includes input and generated output. The ROCmFP4 example profile uses it: five slots of 262,144 (`ROCMFP4_CTX_TOTAL=1310720`).

Five f16 slots at 256k can use about 80 GiB for full-attention KV before weights and runtime buffers. The ROCmFP4 stack instead uses q8 target KV and q4 draft KV, disables the extra prompt cache, and caps container memory plus swap at 90 GiB (`ROCMFP4_MEMORY_LIMIT`). This keeps five full slots without letting the server consume the host.

Validate the configured arithmetic and Compose model:

```bash
python3 scripts/check_context_config.py .env --noob-context 131072
python3 scripts/check_context_config.py .env \
  --prefix ROCMFP4 --noob-context 262144
docker compose config -q
```

After startup, verify the runtime slots:

```bash
curl -fsS http://localhost:${LLM_PORT:-8080}/slots |
  python3 scripts/check_context_config.py .env \
    --noob-context 131072 --slots-json -
```

## GTT, not VRAM

On Strix Halo the dedicated "VRAM" is a small BIOS carve-out; the 128 GB of unified RAM is reachable by the GPU as GTT. You want model weights in GTT, with VRAM near idle.

The default Compose file sets `GGML_VK_PREFER_HOST_MEMORY=1` (the ROCmFP4 stack adds `GGML_HIP_ENABLE_UNIFIED_MEMORY=1` for its HIP side). The Vulkan setting is a presence check and makes the allocator request host-visible/GTT memory first.

Prove it after the model loads:

```bash
scripts/verify-gtt.sh --min-gtt-mib 16000
```

It waits for `/health`, then reads the kernel's amdgpu counters under `/sys/class/drm/card*/device/` (`mem_info_gtt_used`, `mem_info_vram_used`, in bytes) and asserts GTT carries the load while VRAM stays idle. `scripts/gpu_mem.py` is the underlying tool (`--json` for a raw snapshot). These sysfs counters are the source of truth on Strix Halo, where `rocm-smi` can misreport against the tiny VRAM pool. `amdgpu_top` and `radeontop` show the same split live.

## The GTT pool (raise it once in GRUB)

The GTT window is not the whole 128 GB by default. amdgpu sizes it from `ttm.pages_limit`, which defaults to half of RAM: about 62 GiB on this box (`cat /sys/module/ttm/parameters/pages_limit` reads 16182224 pages of 4 KiB). The five-slot laguna load is ~101 GiB, so it does not fit a stock pool, and `--n-gpu-layers 99` drives the allocator past the ceiling. On unified memory that is not a graceful OOM: the box hard-freezes and needs a power cycle.

Raise the pool on the kernel command line and reboot. Edit `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=off amdgpu.gttsize=114688 ttm.pages_limit=29360128"
```

Then `sudo update-grub` and reboot. `ttm.pages_limit` is in 4 KiB pages, so 29360128 x 4096 = 112 GiB of GTT (the benchmark box ran 116; 112 leaves a bit more for the host). `amd_iommu=off` lets the GPU address the full pool; with IOMMU on, only a few GiB is allocatable at load time. `amdgpu.gttsize` (MiB) is honored on kernels that still expose the param and ignored on newer ones, so `ttm.pages_limit` is the value that actually moves the pool.

This lives on the host, outside the stack, and a GRUB reset silently drops it back to the 62 GiB default. If the box starts freezing on model load again, check the pool before anything else:

```bash
cat /sys/module/ttm/parameters/pages_limit   # want 29360128, not 16182224
python3 scripts/gpu_mem.py                    # gtt_total should read ~114688M
```

The five-slot Qwen profile still needs the raised pool. Laguna's ~70 GiB of weights alone also overflow the stock pool, and so does a 37 GB Q8_0 model once KV and runtime buffers join in.

## ROCmFP4 + MTP (optional stack)

[plunderstruck](https://huggingface.co/collections/plunderstruck/rocmfp4-mtp-strix-halo)'s Qwen3.6 GGUFs use custom `Q4_0_ROCMFP4` tensor types that upstream llama.cpp does not know about, so the stock `server-vulkan` image cannot load them. `docker-compose.rocmfp4.yml` builds [charlie12345/rocmfp4-llama](https://github.com/charlie12345/rocmfp4-llama), branch `mtp-rocmfp4-strix`.

The image is built from `ubuntu:26.04` LTS with a pinned [TheRock](https://github.com/ROCm/TheRock) ROCm 7.13 dist tarball (`ROCMFP4_THEROCK_VERSION` in `.env`, default `7.13.0a20260515`, the last 7.13 nightly). 7.13 is the first ROCm line with gfx1151 in the support matrix, so the old `HSA_OVERRIDE_GFX_VERSION` workaround is gone. The 26.04 toolchain also matters for speed: its current `glslc` compiles the Vulkan integer-dot shader variants the old 24.04 base silently skipped (the binary now reports `int dot: 1`), and the runtime carries mesa 26.0.3 RADV. Both backends are compiled in, so `-dev Vulkan0` and `-dev ROCm0` both work at runtime; the compose file mounts `/dev/dri` and `/dev/kfd` because the HIP-linked binary initializes ROCm at startup either way. The runtime image keeps the pruned ROCm libs it actually loads (~2.5 GB total), not the full SDK.

The point of these builds is MTP self-speculative decoding: the model drafts its own tokens through a built-in MTP head (`--spec-type draft-mtp`), running on the same Vulkan device.

Get a model and template into `MODELS_DIR`:

```bash
hf download plunderstruck/Qwen3.6-27B-OBLITERATED-MTP-ROCmFP4-GGUF \
  Qwen3.6-27B-OBLITERATED-MTP-ROCmFP4-STRIX-embF16-imatrix-headQ6.gguf \
  chat_template.jinja \
  --local-dir "$MODELS_DIR/Qwen3.6-27b"
# then set ROCMFP4_MODEL and ROCMFP4_CHAT_TEMPLATE in .env (examples are
# commented in .env.example)
```

The first start compiles the fork:

```bash
docker compose -f docker-compose.rocmfp4.yml up -d --build
docker compose -f docker-compose.rocmfp4.yml logs -f llm
```

There is no compatible stock llama.cpp image for this custom GGUF. With an empty Docker image store, the first start must download Ubuntu and TheRock layers and compile the fork. Later starts reuse the local build.

It serves the OpenAI API on `:8080`. Model, template, alias, five-slot context profile, TheRock pin, and gfx target are in `.env`. Vision is off by default; the model repository also provides `mmproj-F32.gguf`.

Measured throughput at 2k to 32k context is in [Benchmarks](#benchmarks) below.

## Laguna ROCmFPX (optional stack)

The [Laguna S 2.1 Chadrock ROCmFP4 V4 GGUF](https://huggingface.co/jcbtc/Laguna-S-2.1-Chadrock-ROCmFP4-StrixKVSpine-V4-GGUF) uses Laguna architecture support that is absent from the Qwen ROCmFP4 fork above. `docker-compose.laguna-rocmfpx.yml` builds the model card's exact Ciru ROCmFPX Runtime V2 commit. The runtime is Vulkan-only, so the container gets `/dev/dri`, not `/dev/kfd`.

Download the 60.945 GiB file beside the other Laguna models:

```bash
HF_TOKEN="$HF_TOKEN" hf download \
  jcbtc/Laguna-S-2.1-Chadrock-ROCmFP4-StrixKVSpine-V4-GGUF \
  laguna-s-2.1-ROCmFP4-StrixKVSpine-v4.gguf \
  --local-dir "$MODELS_DIR/laguna-s-2.1"
```

Then build and run it:

```bash
docker compose -f docker-compose.laguna-rocmfpx.yml up -d --build
docker compose -f docker-compose.laguna-rocmfpx.yml logs -f laguna-llm
```

It serves on `127.0.0.1:8080`. Its env settings are separate from the Qwen stack. The default 131072-token profile uses the Runtime V2 submission limits validated by the model author. The container verifies the published model SHA-256 before launch, runs unprivileged with a read-only root filesystem and no Linux capabilities, and mounts model files read-only.

## Benchmarks

All on the same idle Strix Halo box (Radeon 8060S, RADV `STRIX_HALO`), through the actual served stacks: fresh prompts against `/completion`, generation forced to 128 tokens, best of 3 per point (`scripts/bench_server.py`). The laguna row used the default stock Vulkan stack. The Qwen rows used the ROCmFP4 + MTP stack (`-dev Vulkan0`, f16 KV, `-ub 1024`, MTP on).

| Model | Active / total | Quant | MTP | Prefill (t/s) | Decode (t/s) |
|---|---|---|:--:|--:|--:|
| laguna-s-2.1 | 8B / 118B MoE | Q4_K_M | no | 293 → 196 (32k) | 22.7 → 19.5 (32k) |
| Qwen3.6-35B-A3B | 3B / 35B MoE | ROCmFP4 | yes | 714 → 707 (32k) | 119 → 101 (32k) |
| Qwen3.6-27B | 27B dense | ROCmFP4 | yes | 217 → 212 (16k) | 39 → 39 (16k) |
| Qwen3.6-27B-OBLITERATED | 27B dense | ROCmFP4 | yes | 213 → 221 (8k) | 37 → 39 (8k) |

Pure batch throughput is higher than the served numbers (MTP's draft context re-processes the prompt, ~15% prefill toll): llama-bench pp2048 for the 35B is 1195 t/s on Vulkan and 1411 t/s on ROCm at `-ub 2048`. Per-model detail, backend and MTP A/Bs, and advertised-vs-measured tables live in [docs/](docs/): [laguna-s-2.1](docs/laguna-s-2.1.md), [35B-A3B](docs/qwen3.6-35b-a3b-mtp-rocmfp4.md), [27B + OBLITERATED](docs/qwen3.6-27b-mtp-rocmfp4.md). A gemma-4-26B-A4B-heretic row (default Vulkan stack) is still to be measured.

## Layout

```text
docker-compose.yml          default stock llama.cpp Vulkan service (standard GGUFs)
docker-compose.vulkan.yml   compatibility entry point for the default service
docker-compose.rocmfp4.yml  optional ROCmFP4 + MTP fork service (plunderstruck GGUFs)
docker-compose.laguna-rocmfpx.yml  Laguna ROCmFPX Runtime V2 service (Vulkan)
.env.example                model examples, ports, GPU group IDs, custom-runtime knobs
scripts/gpu_mem.py          read amdgpu VRAM vs GTT counters; --verify mode
scripts/verify-gtt.sh       wait for /health, then assert model is in GTT
scripts/bench_server.py     served prefill/decode by context depth (docs/ tables)
tools/Dockerfile.rocmfp4    the ROCmFP4 fork build (server target, gfx1151)
tools/Dockerfile.laguna-rocmfpx  pinned Laguna ROCmFPX Runtime V2 build
docs/                       per-model benchmarks and run notes
tests/                      compose invariants, gpu_mem parser, custom stacks, bench
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

[MIT](LICENSE) for the build glue here. The default stack pulls a prebuilt image; the ROCmFP4 stack builds the charlie12345/rocmfp4-llama fork, itself an MIT llama.cpp derivative. Models are mounted read-only and the GGUF weights carry their own licenses (Gemma, Llama, Qwen, etc.). You are responsible for complying with each model's terms.
