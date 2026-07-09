# Qwen3.6-35B-A3B-MTP-ROCmFP4 on Strix Halo

Benchmarks and run notes for [`plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF`](https://huggingface.co/plunderstruck/Qwen3.6-35B-A3B-MTP-ROCmFP4-GGUF) served through the ROCmFP4 stack in this repo (`docker-compose.rocmfp4.yml`). Measured 2026-07-09.

## The model

A 35B-parameter Mixture-of-Experts (about 3B active per token, hence "A3B") in the community ROCmFP4 format. The file served here is the STRIX hybrid, `Qwen3.6-35B-A3B-MTP-ROCmFP4-STRIX-embF16-headQ6.gguf` (18.5 GB): f16 token embeddings and a Q6_K output head over a `q4_0_rocmfp4` body, with a built-in MTP (multi-token-prediction) head for self-speculative decoding. The repo also carries `mmproj-F32.gguf` (a Qwen3-VL vision projector, 1.7 GB) and `chat_template.jinja`.

These `q4_0_rocmfp4` tensors do not load in upstream llama.cpp. They need the [`charlie12345/rocmfp4-llama`](https://github.com/charlie12345/rocmfp4-llama) fork (branch `mtp-rocmfp4-strix`), which this repo builds via `tools/Dockerfile.rocmfp4`.

## Test rig

| | |
|---|---|
| GPU | AMD Radeon 8060S (Strix Halo, RADV `GFX1151`), 120832 MiB visible as GTT |
| Host | AMD Ryzen AI Max+, 128 GB unified RAM |
| Compute backend | Vulkan (`-dev Vulkan0`), Mesa RADV. ROCm 7.2.1 is present only for the HIP init the binary does at startup; the math runs on Vulkan. |
| Build | `charlie12345/rocmfp4-llama` @ `mtp-rocmfp4-strix`, base `rocm/dev-ubuntu-24.04:7.2.1-complete`, target `gfx1151` |
| Server flags | `--ctx-size 32768 -ctk f16 -ctv f16 -fa on -b 2048 -ub 256 --no-mmap --parallel 1`, MTP `--spec-type draft-mtp --spec-draft-n-max 5 --spec-draft-p-split 0.10` |

## Results by context depth

Prefill is prompt processing (best of 3 fresh runs). Decode is MTP-accelerated generation of 128 tokens (best of 4). Numbers come from llama.cpp's own `/completion` timings, so prefill time is excluded from the decode figure and vice versa. This run had the GPU to itself, with no other containers running. An earlier run with an idle gemma server alongside (sampled every 0.5s, busy 0% of 745 samples) gave the same prefill and decode within MTP noise, so contention was not a factor either way.

| Context depth | Prompt tokens | Prefill (t/s) | Decode (t/s, MTP) |
|--------------:|--------------:|--------------:|------------------:|
| 2k  |  2019 | 551 | 110 |
| 8k  |  8151 | 581 | 110 |
| 16k | 16257 | 562 | 100 |
| 32k | 31755 | 520 |  93 |

Prefill holds around 520 to 580 t/s and eases off at 32k as attention grows. Decode falls from ~110 t/s at 2k to ~93 t/s at 32k as the KV cache deepens.

## Advertised vs measured

| Metric | Advertised | Measured (this rig) |
|---|---|---|
| Prefill | ~4000 t/s | 520 to 580 t/s |
| Decode | ~50 t/s | 90 to 110 t/s |

Decode beats the claim comfortably. Prefill does not come close, and it is not a batching problem: sweeping the micro-batch shows the ceiling is real.

| `-ub` | Peak prefill (t/s) |
|------:|-------------------:|
| 256 (shipped) | ~560 |
| 2048 | ~790 |
| 4096 | ~570 to 710 (worse) |

`-ub 2048` buys about 40% on prefill for more compute-buffer memory, and past that it regresses. Even at the sweet spot the Vulkan MoE prefill runs at roughly 10% of the iGPU's fp16 peak, so ~4000 t/s is not reachable here at any batch size. That figure most likely comes from a different model, quant, or measurement.

## Decode depends on MTP acceptance

MTP self-speculation drafts tokens with the model's own head and verifies them against the target model. Its speedup scales with how predictable the text is: highly predictable continuations reached ~110 t/s, novel technical prose landed at ~90 t/s (best of 4, low context). Treat decode as a range, roughly 90 to 110 t/s in normal use, not a fixed number. Sampling temperature adds run-to-run noise on top.

## Memory

With the model loaded, GTT (unified RAM) held about 37.9 GB (18.5 GB of weights plus the MTP draft context and the f16 KV cache) while the VRAM carve-out stayed idle at ~800 MiB. This is the whole point of the Strix Halo setup; verify it yourself with:

```bash
LLM_PORT=8081 scripts/verify-gtt.sh --min-gtt-mib 18000
```

## Reproduce

Bring the stack up (see the [ROCmFP4 section](../README.md#rocmfp4--mtp-separate-stack) of the README), then hit the native `/completion` endpoint, which returns per-request timings:

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

- One APU. This run had the box to itself, and a prior run alongside an idle gemma server matched it, so contention was not the story here. Still, an actively decoding neighbour will drag these down; take your own numbers when your box is quiet.
- Decode figures use MTP, which is on by default in `docker-compose.rocmfp4.yml`. Without `--spec-type draft-mtp` the base decode rate is lower.
- `--ctx-size` is 32768 by default; the model trains to 262144. Raising `ROCMFP4_CTX` costs KV memory and, at these depths, some decode speed.
