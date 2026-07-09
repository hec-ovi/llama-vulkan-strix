# Qwen3.6-27B-MTP-ROCmFP4 (and OBLITERATED) on Strix Halo

Benchmarks and run notes for the two 27B dense ROCmFP4 builds served through this repo's ROCmFP4 stack, measured 2026-07-09 on the same rig and image as the [35B-A3B page](qwen3.6-35b-a3b-mtp-rocmfp4.md) (Ubuntu 26.04 + TheRock 7.13 image, RADV `STRIX_HALO`, `-ub 1024`, MTP on, idle box):

- [`plunderstruck/Qwen3.6-27B-MTP-ROCmFP4-GGUF`](https://huggingface.co/plunderstruck/Qwen3.6-27B-MTP-ROCmFP4-GGUF), file `Qwen3.6-27B-MTP-ROCmFP4-STRIX-imatrix-embF16-headQ6.gguf` (16.9 GB)
- [`plunderstruck/Qwen3.6-27B-OBLITERATED-MTP-ROCmFP4-GGUF`](https://huggingface.co/plunderstruck/Qwen3.6-27B-OBLITERATED-MTP-ROCmFP4-GGUF), the uncensored [OBLITERATUS](https://huggingface.co/OBLITERATUS) base with a grafted MTP head, file `Qwen3.6-27B-OBLITERATED-MTP-ROCmFP4-STRIX-embF16-imatrix-headQ6.gguf` (16.9 GB)

Both are dense 27B models: every token pays for all 27B parameters, so they run an order of magnitude slower than the 35B-A3B MoE (3B active). Serve them by pointing `ROCMFP4_MODEL` / `ROCMFP4_CHAT_TEMPLATE` / `ROCMFP4_ALIAS` in `.env` at the model's folder and re-running `docker compose -f docker-compose.rocmfp4.yml up -d`.

## llama-bench (pure batch throughput, no MTP)

`llama-bench -ngl 999 -fa 1 -ub 1024 -r 3`:

| Model | Device | pp2048 | pp4096 | tg128 |
|---|---|--:|--:|--:|
| 27B | Vulkan0 | 294 | 287 | 13.8 |
| 27B | ROCm0 | 342 | 322 | 13.5 |
| 27B OBLITERATED | Vulkan0 | 295 | 288 | 13.7 |
| 27B OBLITERATED | ROCm0 | 335 | 325 | 13.5 |

The two builds are the same speed within noise, as expected (same architecture, same tensor layout). For reference, the Level1Techs Strix Halo guide reports 332 pp512 / 14.0 tg128 for a 27B ROCmFP4 build on the ROCm backend, which matches.

## Server results by context depth (MTP on, Vulkan0)

Fresh prompts against `/completion` (`cache_prompt=false`, 128 forced output tokens, best of 3). Prefill through the server is lower than llama-bench because the prompt is also processed through the MTP draft context.

| Model | Context depth | Prompt tokens | Prefill (t/s) | Decode (t/s, MTP) |
|---|--:|--:|--:|--:|
| 27B | 2k | 1780 | 217 | 39.0 |
| 27B | 8k | 7159 | 227 | 39.3 |
| 27B | 16k | 14347 | 212 | 39.4 |
| 27B OBLITERATED | 2k | 1780 | 213 | 37.0 |
| 27B OBLITERATED | 8k | 7159 | 221 | 39.2 |

## Decode is a band, not a number

MTP acceptance decides everything for a dense model: across reps the same config swung between 23 and 39 t/s depending on how predictable the continuation was. That band brackets the published claims: the model card says ~33 t/s at short context, the fork docs say 33.6 short / 28.0 sustained, and the Level1Techs guide says 23-36 depending on prompt. Without MTP the base decode is ~13.7 t/s (llama-bench above), so speculation is worth roughly 2-3x here, more than on the MoE.

## Memory

Each 27B build holds ~17 GB of weights in GTT plus KV; same GTT-not-VRAM behaviour as the 35B (`LLM_PORT=8081 scripts/verify-gtt.sh --min-gtt-mib 16000`).
