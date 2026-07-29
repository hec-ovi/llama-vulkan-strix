<h1 align="center">llama-vulkan-strix</h1>

<p align="center">
  <strong>Docker Compose for llama.cpp on AMD Strix Halo: Qwen, Gemma, and Laguna GGUFs (abliterated and quantized), with stock Vulkan, ROCmFP4 + MTP, and ROCmFPX. Parallel slots and context are per package in .env. Prefill, decode, and quality metrics below are measured on this rig so you can pick what fits.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/AMD-Strix_Halo-ED1C24?logo=amd&logoColor=white" alt="AMD Strix Halo" />
  <img src="https://img.shields.io/badge/backend-Vulkan-AC162C?logo=vulkan&logoColor=white" alt="Vulkan" />
  <img src="https://img.shields.io/badge/llama.cpp-server-000000" alt="llama.cpp" />
  <img src="https://img.shields.io/badge/License-MIT-blue" alt="License" />
</p>

---

## What this is

Docker Compose packages for llama.cpp on gfx1151 (Strix Halo). Each model block in `.env` wires path, template, max context, and parallel concurrency. Stock stack: `docker compose up -d` pulls `ghcr.io/ggml-org/llama.cpp:server-vulkan` and serves the GGUF you pick on `:8080`. Packages cover abliterated/heretic and quantized GGUFs (Qwen3.6, Gemma 4 26B-A4B, Laguna S 2.1).

Speed vs quality is your call. This README has **prefill** and **decode** on this box, plus quality vs original (KL, refusals) and capability scores from the model cards (SWE, Terminal, tools, STEM). Use those to choose a package.

Custom formats the stock image cannot load: `docker-compose.rocmfp4.yml` (plunderstruck Qwen3.6 ROCmFP4 + MTP, long first build) and `docker-compose.laguna-rocmfpx.yml` (Laguna Runtime V2 / ROCmFPX). One stack at a time on port `8080`.

## Supported models

| Stack | Compose | Models |
|---|---|---|
| Stock Vulkan | `docker-compose.yml` + `compose/models/<package>.yml` | Packages in `.env.example`: Qwen3.6-27B heretic-v2 Q8_0 (MTP), Qwen3.6-35B-A3B heretic Q8_0, Laguna S 2.1 IQ4_XS, Gemma 4 26B-A4B abliterated Q5_K_M. Any other standard GGUF works if you set paths yourself. |
| ROCmFP4 + MTP | `docker-compose.rocmfp4.yml` | plunderstruck Qwen3.6 ROCmFP4 GGUFs only (27B, 27B-OBLITERATED, 35B-A3B-MTP). Custom `Q4_0_ROCMFP4` tensors. |
| Laguna ROCmFPX | `docker-compose.laguna-rocmfpx.yml` | Chadrock Laguna S 2.1 ROCmFP4 V4 GGUF only (pinned Ciru Runtime V2). |

## Quick start

Prerequisites: AMD Strix Halo (Ryzen AI Max+, gfx1151), Docker + Compose, GGUFs on disk.

```bash
cd ~/workspace/llama-vulkan-strix
cp .env.example .env
# edit .env: MODELS_DIR, RENDER_GID / VIDEO_GID, then uncomment ONE model package
# (COMPOSE_FILE + LLM_MODEL + LLM_CHAT_TEMPLATE + alias + ctx + parallel)

docker compose up -d
docker compose logs -f llm
```

`COMPOSE_FILE` is what wires the per-model package (MTP, flash-attn, Laguna sampling, q8 KV). Example for Gemma:

```dotenv
COMPOSE_FILE=docker-compose.yml:compose/models/gemma-4-26b-a4b.yml
LLM_MODEL=gemma-4-26b-a4b-abliterated/Gemma-4-26B-A4B-It-Abliterated-Q5_K_M.gguf
LLM_CHAT_TEMPLATE=gemma-4-26b-a4b-abliterated/chat_template.jinja
LLM_ALIAS=gemma-4-26b-a4b-abliterated-q5
LLM_CTX_PER_SLOT=131072
LLM_PARALLEL=5
LLM_CTX_TOTAL=655360
```

Download Gemma (use `HF_TOKEN` from `.env` for authenticated speed):

```bash
set -a && source .env && set +a
HF_TOKEN="$HF_TOKEN" hf download \
  SevenOfNine/Gemma-4-26B-A4B-It-Abliterated-GGUF \
  Gemma-4-26B-A4B-It-Abliterated-Q5_K_M.gguf \
  --local-dir "$MODELS_DIR/gemma-4-26b-a4b-abliterated"
# drop chat_template.jinja next to the GGUF if it is not already there
```

Call it:

```bash
curl http://localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"llm","messages":[{"role":"user","content":"hi"}]}'
```

## Model packages (comment / uncomment)

Each package is a full block in `.env.example`: model path, chat template, alias, per-slot context, parallel slots, total context, and `COMPOSE_FILE` pointing at `compose/models/<name>.yml`.

| Package | GGUF (under `MODELS_DIR`) | Active / total | Quant | MTP | Native max ctx | Package file |
|---|---|---|---|:--:|---:|---|
| A | `Qwen3.6-27B-heretic-v2/...-Q8_0.gguf` | 27B dense | Q8_0 | yes | 262144 | `compose/models/qwen3.6-27b-heretic.yml` |
| B | `Qwen3.6-35B-A3B-heretic/...-Q8_0.gguf` | 3B / 35B MoE | Q8_0 | no* | 262144 | `compose/models/qwen3.6-35b-heretic.yml` |
| C | `laguna-s-2.1/Laguna-S-2.1-IQ4_XS-00001-of-00002.gguf` | 8B / 118B MoE | IQ4_XS | no | 262144 | `compose/models/laguna-s-2.1.yml` |
| D | `gemma-4-26b-a4b-abliterated/...-Q5_K_M.gguf` | ~4B / 26B MoE | Q5_K_M | no | 262144 | `compose/models/gemma-4-26b-a4b.yml` |

\* Package B is the non-MTP-preserved heretic quant. For MTP on 35B, use the ROCmFP4 stack or a Native-MTP-Preserved GGUF with a matching package.

Default profile for all four: **5 parallel slots x 131072 = 655360** total KV. That is half the native 256k per slot, chosen so five concurrent requests fit Strix Halo GTT after weights. You can raise `LLM_CTX_PER_SLOT` toward 262144 if you drop parallel or free GTT.

## Context and parallel slots

`llama-server` treats `--ctx-size` as the total KV cache shared by its slots; `--parallel` is the slot count. See the [server option reference](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

- `LLM_CTX_PER_SLOT`: context available to one request
- `LLM_PARALLEL`: concurrent slots
- `LLM_CTX_TOTAL` must equal their product

```bash
python3 scripts/check_context_config.py .env --noob-context 131072
docker compose config -q
curl -fsS http://localhost:${LLM_PORT:-8080}/slots |
  python3 scripts/check_context_config.py .env --noob-context 131072 --slots-json -
```

### Max context with 5 parallel slots

| Model | Architecture max | Package default (5 slots) | 5 x full native max? |
|---|---:|---:|---|
| Qwen3.6-27B heretic Q8 | 262144 | 131072 per slot (total 655360) | Possible in principle (total 1310720); heavy on GTT with ~28 GB weights |
| Qwen3.6-35B-A3B heretic Q8 | 262144 | 131072 per slot | Same math; ~35 GB Q8 weights |
| Laguna S 2.1 IQ4_XS | 262144 (YaRN in GGUF) | 131072 per slot, q8 KV | 5 x 256k f16 KV alone is ~80+ GiB class; package uses q8 KV |
| Gemma 4 26B-A4B Q5 | 262144 (medium Gemma 4) | 131072 per slot, q8 KV | Same; E2B/E4B are 128k, **26B A4B is 256k** |

Sources: [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) / [35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) `config` (262144), [Gemma 4](https://ai.google.dev/gemma/docs/core) medium models 256k, Laguna GGUF YaRN to 262144. All four packages use the same 5 x 131072 default on this repo.

## GTT, not VRAM

On Strix Halo the dedicated "VRAM" is a small BIOS carve-out; unified RAM is GTT. Weights should land in GTT.

The default compose sets `GGML_VK_PREFER_HOST_MEMORY=1`. Prove it after load:

```bash
scripts/verify-gtt.sh --min-gtt-mib 16000
```

### Raise the GTT pool once in GRUB

amdgpu sizes GTT from `ttm.pages_limit` (default ~half of RAM). Large multi-slot loads need more. Edit `GRUB_CMDLINE_LINUX_DEFAULT` in `/etc/default/grub`:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amd_iommu=off amdgpu.gttsize=114688 ttm.pages_limit=29360128"
```

Then `sudo update-grub` and reboot. `ttm.pages_limit=29360128` is 112 GiB of GTT. Check:

```bash
cat /sys/module/ttm/parameters/pages_limit   # want 29360128, not ~16182224
python3 scripts/gpu_mem.py
```

## ROCmFP4 + MTP (optional, slow first install)

[plunderstruck](https://huggingface.co/collections/plunderstruck/rocmfp4-mtp-strix-halo)'s Qwen3.6 GGUFs use custom `Q4_0_ROCMFP4` types. Stock `server-vulkan` cannot load them. `docker-compose.rocmfp4.yml` builds [charlie12345/rocmfp4-llama](https://github.com/charlie12345/rocmfp4-llama) (`mtp-rocmfp4-strix`).

**First install cost:** empty Docker store means pulling `ubuntu:26.04`, downloading the TheRock ROCm 7.13 dist tarball, and compiling Vulkan + HIP for gfx1151. That is multi-GB of downloads and a long compile (often tens of minutes). Later starts reuse the image. The container needs `/dev/kfd` and `/dev/dri` even though compute runs on Vulkan.

```bash
docker compose -f docker-compose.rocmfp4.yml up -d --build
docker compose -f docker-compose.rocmfp4.yml logs -f llm
```

MTP (`--spec-type draft-mtp`) is on by default in that file. Details and measured ROCmFP4 throughput: [docs/qwen3.6-35b-a3b-mtp-rocmfp4.md](docs/qwen3.6-35b-a3b-mtp-rocmfp4.md), [docs/qwen3.6-27b-mtp-rocmfp4.md](docs/qwen3.6-27b-mtp-rocmfp4.md).

## Laguna ROCmFPX (optional)

Separate stack for the [Chadrock ROCmFP4 V4 GGUF](https://huggingface.co/jcbtc/Laguna-S-2.1-Chadrock-ROCmFP4-StrixKVSpine-V4-GGUF). First build also compiles a pinned Ciru Runtime V2 commit.

```bash
docker compose -f docker-compose.laguna-rocmfpx.yml up -d --build
```

## Benchmarks

Use this section to pick a package. Nothing is ranked "best"; each table answers a different question.

1. **Quality vs original (ablation / quant):** how close the uncensored or quantized weights stay to the full-precision base (KL, refusals, MMLU). Low KL means the brain is mostly intact.
2. **Capability (model cards):** what the *original* models score on agentic coding (SWE, Terminal), tools (MCPMark), and hard STEM (LiveCodeBench, AIME). Approximate ceiling for the GGUF you serve.
3. **Served speed on this box:** **prefill** = tokens/s while chewing the prompt (matters for long context and tools); **decode** = tokens/s while streaming the reply (what you feel in chat). Measured here on stock Vulkan and, separately, on the ROCmFP4 stack.

### Quality: ablation and quant vs original

KL and refusal counts are from the heretic/abliteration authors. MMLU is their re-run of original vs ablated at full precision before quant. Quant KL against BF16 for these exact files was not re-measured on this box; Q8_0 is near-lossless in community tables, IQ4_XS / Q5_K_M trade more.

| Model (served quant) | Ablation KL vs original | Refusals ablated / original | MMLU ablated / original | Quant note |
|---|---:|---|---|---|
| Qwen3.6-27B heretic-v2 Q8_0 | **0.0021** | **6 / 92** | **85.67% / 86.65%** | Q8_0; MTP tensors preserved (15/15) |
| Qwen3.6-35B-A3B heretic Q8_0 | **0.0015** | **10 / 83** | **83.22% / 83.71%** | Q8_0; this file is non-MTP-preserved |
| Laguna S 2.1 IQ4_XS | n/a (not abliterated) | n/a | n/a | Official imatrix IQ4_XS; no public KL vs BF16 for this file on this box |
| Gemma 4 26B-A4B abliterated Q5_K_M | **0.0845** | **18 / 100** | not published | Q5_K_M of abliterated BF16; separate quant KL not published |

Sources: [27B heretic-v2](https://huggingface.co/llmfan46/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved), [35B heretic](https://huggingface.co/llmfan46/Qwen3.6-35B-A3B-uncensored-heretic), [Gemma abliterated](https://huggingface.co/SevenOfNine/Gemma-4-26B-A4B-It-Abliterated).

### Capability benchmarks (original model cards)

These are the **upstream / original** scores (full precision harnesses). Heretic/abliteration cards above show tiny MMLU/KL deltas; they do not republish full SWE/Terminal tables for the ablated weights. Use the original numbers as the capability ceiling the quant approximates.

| Benchmark | Qwen3.6-27B | Qwen3.6-35B-A3B | Gemma 4 26B-A4B | Laguna S 2.1 |
|---|---:|---:|---:|---:|
| SWE-bench Verified | 77.2 | 73.4 | 17.4* | (see Multilingual / Pro) |
| SWE-bench Multilingual | 71.3 | 67.2 | 17.3* | **78.5** |
| SWE-bench Pro | 53.5 | 49.5 | 13.8* | **59.4** |
| Terminal-Bench | 59.3 (2.0) | 51.5 (2.0) | 34.2* (2.0) | **70.2** (2.1) |
| MCPMark (tool use) | (see 27B blog) | **37.0** | 14.2* | n/a |
| LiveCodeBench v6 | 83.9 | 80.4 | **77.1** | n/a |
| AIME 2026 | 94.1 | 92.7 | 88.3 | n/a |
| MMLU-Pro | 86.2 | 85.2 | 82.6 | n/a |

\* Gemma 4 **26B-A4B** agentic coding scores from the Qwen3.6-35B comparison table (Qwen blog / model card). Gemma **31B** is much stronger on SWE (e.g. 52.0 Verified) but is a different model. Laguna numbers from [Poolside Laguna S 2.1](https://poolside.ai/blog/introducing-laguna-s-2-1). Qwen numbers from the [27B](https://huggingface.co/Qwen/Qwen3.6-27B) and [35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) cards.

### Served speed on this Strix Halo box (stock packages)

Measured 2026-07-29, idle Radeon 8060S (RADV `STRIX_HALO`), one request in flight, stock `server-vulkan`, 5 slots x 131072. Method: `scripts/bench_full.py` (one load per model). Prefill: best-of-3, fresh prompts. Decode averages: mean of 10 runs. Higher prefill helps long prompts and tool dumps; higher decode feels snappier in interactive use. Raw JSON: [docs/bench-results/](docs/bench-results/).

**Prefill + decode at no-prefill / 16k / 32k** (best-of-3):

| Model | Quant | MTP | Prefill no / 16k / 32k (t/s) | Decode no / 16k / 32k (t/s) |
|---|---|:--:|---|---|
| Qwen3.6-27B heretic-v2 | Q8_0 | yes | **143 / 250 / 185** | **28.5 / 26.6 / 24.1** |
| Laguna S 2.1 | IQ4_XS | no | **125 / 352 / 307** | **40.6 / 34.3 / 30.7** |
| Qwen3.6-35B-A3B heretic | Q8_0 | no | **143 / 940 / 816** | **55.4 / 49.4 / 45.3** |
| Gemma 4 26B-A4B abliterated | Q5_K_M | no | **361 / 828 / 642** | **61.0 / 50.4 / 45.0** |

**Decode averages** (10 runs, mean):

| Model | Decode mean, no prefill (t/s) | Decode mean at 32k cached (t/s) | Cache hit? |
|---|---:|---:|---|
| Qwen3.6-27B heretic-v2 Q8 | **24.5** (median 25.2) | **23.7** (3 cold samples)* | no (`--cache-ram 0` for MTP) |
| Laguna S 2.1 IQ4_XS | **38.8** (median 38.7) | **30.3** (median 30.3) | yes (`cache_n` ~41k) |
| Qwen3.6-35B-A3B heretic Q8 | **55.2** (median 55.1) | **44.4** (median 44.4) | yes |
| Gemma 4 26B-A4B Q5 | **60.5** (median 60.6) | **43.6** (median 43.6) | yes |

\* 27B keeps `--cache-ram 0` (MTP + prompt-cache path has crashed this class of builds). Its 32k decode average is from cold 32k samples, not true cache hits. Laguna / 35B / Gemma packages use `--cache-ram 8192` so 32k re-decode reuses KV.

Reproduce one package (full suite, no reload mid-run):

```bash
# pick a package in .env, then:
docker compose up -d
# wait for /health
python3 scripts/bench_full.py --url http://localhost:8080 --name my-run
# or all stock packages in order (27B, Laguna, 35B, Gemma; no ROCmFP4):
bash scripts/run_model_benches.sh
```

### ROCmFP4 stack (optional, earlier runs)

Same box, measured 2026-07-09 on `docker-compose.rocmfp4.yml` (custom fork, Vulkan0, `-ub 1024`, MTP on, f16 KV, single slot in that run). Not re-run in the 2026-07-29 stock suite above. Detail and MTP A/Bs: [docs/qwen3.6-35b-a3b-mtp-rocmfp4.md](docs/qwen3.6-35b-a3b-mtp-rocmfp4.md), [docs/qwen3.6-27b-mtp-rocmfp4.md](docs/qwen3.6-27b-mtp-rocmfp4.md).

**Served prefill/decode by depth** (best-of-3, `/completion`, 128 forced tokens):

| Model | Quant | Prefill 2k / 8k / 16k / 32k (t/s) | Decode MTP 2k / 8k / 16k / 32k (t/s) |
|---|---|---|---|
| Qwen3.6-35B-A3B | ROCmFP4 | 714 / 865 / **810** / **707** | 119 / 114 / **105** / **101** |
| Qwen3.6-27B | ROCmFP4 | 217 / 227 / **212** / - | 39.0 / 39.3 / **39.4** / - |
| Qwen3.6-27B-OBLITERATED | ROCmFP4 | 213 / 221 / - / - | 37.0 / 39.2 / - / - |

**llama-bench** pure batch (no MTP, `-ub 1024`): 35B pp2048 **1195** Vulkan / **1411** ROCm, tg128 71.1 / 63.4; dense 27B pp2048 ~294 Vulkan / ~342 ROCm, tg128 ~13.8 / ~13.5.

MTP on Vulkan costs ~15% prefill and roughly doubles decode vs no-MTP on the 35B (69.6 → 119 t/s at 2k). Dense 27B decode is content-dependent (~23-39 t/s); without MTP base tg is ~14 t/s.

## Layout

```text
docker-compose.yml                 default stock llama.cpp Vulkan service
compose/models/*.yml               per-model packages (MTP, KV, sampling)
docker-compose.vulkan.yml          compatibility entry for the default service
docker-compose.rocmfp4.yml         optional ROCmFP4 + MTP fork (slow first build)
docker-compose.laguna-rocmfpx.yml  Laguna ROCmFPX Runtime V2 service
.env.example                       model packages, ROCmFP4 section, ports, GIDs
scripts/gpu_mem.py                 amdgpu VRAM vs GTT counters
scripts/verify-gtt.sh              wait for /health, assert model is in GTT
scripts/bench_server.py            served prefill/decode by context depth
scripts/bench_full.py              one-load full suite (prefill table + 10x decode avgs)
scripts/run_model_benches.sh       stock packages in order (no ROCmFP4)
scripts/check_context_config.py    ctx x parallel arithmetic + /slots check
tools/Dockerfile.rocmfp4           ROCmFP4 fork build
tools/Dockerfile.laguna-rocmfpx    pinned Laguna ROCmFPX Runtime V2 build
docs/                              per-model notes
docs/bench-results/                measured JSON from this rig
tests/                             compose invariants, packages, scripts
```

```bash
uvx --with pyyaml pytest tests/ -q
```

## Credits

ROCmFP4 packaging and measurement sit on work by [charlie12345](https://github.com/charlie12345/rocmfp4-llama), [plunderstruck](https://huggingface.co/plunderstruck), [wendell / Level1Techs](https://forum.level1techs.com/t/n5-max-proxmox-strix-halo-with-docker-rocm-fp4-and-mtp-ultimate-setup-guide/251182), [kyuz0](https://github.com/kyuz0/amd-strix-halo-toolboxes), and [TheRock](https://github.com/ROCm/TheRock). Heretic/abliteration builds: [llmfan46](https://huggingface.co/llmfan46), [SevenOfNine](https://huggingface.co/SevenOfNine). Laguna: [poolside](https://poolside.ai) / [vcruz305](https://huggingface.co/vcruz305) imatrix GGUFs.

## License

[MIT](LICENSE) for the build glue. Models keep their own licenses (Gemma, Qwen, OpenMDW, etc.).
