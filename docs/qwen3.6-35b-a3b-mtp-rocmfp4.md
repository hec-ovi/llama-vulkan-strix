# Qwen3.6-35B-A3B-MTP-ROCmFP4 on Strix Halo

Benchmarks and run notes for [`plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF`](https://huggingface.co/plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF) served through the ROCmFP4 stack in this repo (`docker-compose.rocmfp4.yml`). Measured 2026-07-09 on the Ubuntu 26.04 + TheRock 7.13 image.

## The model

A 35B-parameter Mixture-of-Experts (about 3B active per token, hence "A3B") in the community ROCmFP4 format. The file served here is the STRIX hybrid, `Qwen3.6-35B-A3B-MTP-ROCmFP4-STRIX-embF16-headQ6.gguf` (18.5 GB): f16 token embeddings and a Q6_K output head over a `q4_0_rocmfp4` body, with a built-in MTP (multi-token-prediction) head for self-speculative decoding. The repo also carries `mmproj-F32.gguf` (a Qwen3-VL vision projector, 1.7 GB) and `chat_template.jinja`.

These `q4_0_rocmfp4` tensors do not load in upstream llama.cpp. They need the [`charlie12345/rocmfp4-llama`](https://github.com/charlie12345/rocmfp4-llama) fork (branch `mtp-rocmfp4-strix`), which this repo builds via `tools/Dockerfile.rocmfp4`.

## Test rig

| | |
|---|---|
| GPU | AMD Radeon 8060S (Strix Halo, RADV `STRIX_HALO`), 120832 MiB visible as GTT |
| Host | AMD Ryzen AI Max+ 395, 128 GB unified RAM |
| Image | `ubuntu:26.04` + TheRock ROCm `7.13.0a20260515`, mesa 26.0.3 RADV, glslc 2026.1 |
| Build | `charlie12345/rocmfp4-llama` @ `mtp-rocmfp4-strix`, `gfx1151`, HIP + Vulkan backends |
| Vulkan caps | `matrix cores: KHR_coopmat`, `int dot: 1` (the 24.04-based image had `int dot: 0`) |
| Server flags | `--ctx-size 32768 -ctk f16 -ctv f16 -fa on -b 2048 -ub 1024 --no-mmap --parallel 1`, MTP `--spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-p-split 0.10` |

## llama-bench (pure batch throughput, no MTP)

`llama-bench -ngl 999 -fa 1 -ub 1024 -r 3`, both compiled backends, idle box:

| Device | pp2048 | pp4096 | tg128 |
|---|--:|--:|--:|
| Vulkan0 (RADV) | 1195 | 1077 | 71.1 |
| ROCm0 (HIP) | 1271 | 1162 | 63.4 |

Micro-batch sweep, pp2048 only: Vulkan peaks at `-ub 1024` (256 → 896, 512 → 1066, 1024 → 1195, 2048 → 902); ROCm keeps climbing to `-ub 2048` (512 → 1185, 2048 → 1411). `GGML_VK_PREFER_HOST_MEMORY=1` (the GTT guarantee this repo ships) costs about 4% prefill and nothing on decode; it stays on.

For scale: the same llama-bench on the previous 24.04/ROCm-7.2.1 image gave pp2048 972 and tg128 68.3 on Vulkan. The prefill gain is the 26.04 shader toolchain (integer-dot variants now compile) plus mesa 26.0.3.

## Server results by context depth (the real thing)

Fresh prompts against `/completion` (`cache_prompt=false`, 128 forced output tokens, best of 3), through the shipped compose stack: Vulkan0, `-ub 1024`, MTP on. These numbers include everything real use includes; note prefill is lower than llama-bench because with MTP the prompt is also processed through the draft context (about 15% toll, measured below).

| Context depth | Prompt tokens | Prefill (t/s) | Decode (t/s, MTP) |
|--------------:|--------------:|--------------:|------------------:|
| 2k  |  1780 | 714 | 119 |
| 8k  |  7159 | 865 | 114 |
| 16k | 14347 | 810 | 105 |
| 32k | 28807 | 707 | 101 |

## MTP and backend trade-offs (measured)

Same server, same flags, one variable at a time, at 2k / 8k depth:

| Config | Prefill 2k / 8k | Decode 2k / 8k |
|---|--:|--:|
| Vulkan0 + MTP (shipped) | 714 / 865 | 119 / 114 |
| Vulkan0, no MTP | 834 / 1042 | 69.6 / 66.4 |
| ROCm0 + MTP | 955 / 904 | 66.5 / 88.0 (erratic) |

MTP costs ~15% prefill and buys ~70% decode on Vulkan; that trade is why it ships on. ROCm0 prefills faster but its MTP decode is both slower and unstable run-to-run (53 to 88 t/s swings), so Vulkan0 stays the default, which matches the model card's advice.

## Advertised vs measured

| Metric | Source claim | Measured (this rig) |
|---|---|---|
| Decode | 78-90 t/s (model card, Vulkan/Strix Halo); 104.4 short / 89.3 sustained (fork docs, ROCm) | 101-119 t/s (MTP, Vulkan, 2k-32k) |
| Prefill | 802-859 t/s (L1T guide, Strix Halo rows, pp512 ROCm) | 1195 (Vulkan) / 1411 (ROCm, ub2048) pp2048 |

The "~4355 t/s prefill" figure floating around for this model is **not a Strix Halo number**. In the [Level1Techs thread](https://forum.level1techs.com/t/n5-max-proxmox-strix-halo-with-docker-rocm-fp4-and-mtp-ultimate-setup-guide/251182) it comes from post #7, measured on a discrete Radeon R9700 ("VRAM at 2750MHz ... power limit at 330W", device `Vulkan1`). The guide's own Strix Halo table (post #1) reports 802-859 t/s prefill for this model class, which this stack now exceeds.

## Decode depends on MTP acceptance

MTP self-speculation drafts tokens with the model's own head and verifies them against the target model. Its speedup scales with how predictable the text is: the numbers above use mixed technical prose; highly novel text lands lower. Treat decode as a band, roughly 100 to 120 t/s at these depths, not a fixed number. Sampling temperature adds run-to-run noise on top.

## Memory

With the model loaded through the compose stack, GTT (unified RAM) held 20.7 GB while the VRAM carve-out stayed idle at 656 MiB. This is the whole point of the Strix Halo setup; verify it yourself with:

```bash
LLM_PORT=8081 scripts/verify-gtt.sh --min-gtt-mib 18000
```

## Reproduce

Bring the stack up (see the [ROCmFP4 section](../README.md#rocmfp4--mtp-separate-stack) of the README). llama-bench ships inside the image:

```bash
docker run --rm --device /dev/dri --device /dev/kfd \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  --ipc host --security-opt seccomp=unconfined \
  -v "$MODELS_DIR":/models:ro --entrypoint /app/llama-bench llama-rocmfp4-strix:latest \
  -m /models/Qwen3.6-35B-A3B-MTP-ROCmFP4/Qwen3.6-35B-A3B-MTP-ROCmFP4-STRIX-embF16-headQ6.gguf \
  -dev Vulkan0 -ngl 999 -fa 1 -ub 1024 -p 2048 -n 128 -r 3
```

Server-level timings come from the native `/completion` endpoint:

```bash
# prefill: process a long prompt, generate one token
curl -s localhost:8081/completion -H 'content-type: application/json' \
  -d '{"prompt":"<~2000 token prompt>","n_predict":1,"cache_prompt":false}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["timings"]["prompt_per_second"])'

# decode: force 128 generated tokens so the rate is real
curl -s localhost:8081/completion -H 'content-type: application/json' \
  -d '{"prompt":"Explain unified memory on an APU.","n_predict":128,"ignore_eos":true,"cache_prompt":false}' \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["timings"]["predicted_per_second"])'
```

## Caveats

- One APU, idle during the run. An actively decoding neighbour will drag these down; take your own numbers when your box is quiet.
- Decode figures use MTP, on by default in `docker-compose.rocmfp4.yml`. Without `--spec-type draft-mtp` decode drops to ~70 t/s (measured above).
- `--ctx-size` is 32768 by default; the model trains to 262144. Raising `ROCMFP4_CTX` costs KV memory and, at depth, some decode speed.
