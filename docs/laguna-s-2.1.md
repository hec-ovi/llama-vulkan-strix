# laguna-s-2.1 (Q4_K_M) on Strix Halo

Benchmarks and run notes for poolside's Laguna S 2.1, the standard model of the base Vulkan stack (`docker-compose.yml`, stock `ghcr.io/ggml-org/llama.cpp:server-vulkan` image, no fork, no MTP). Measured 2026-07-22.

## The model

[poolside's Laguna S 2.1](https://huggingface.co/poolside/Laguna-S-2.1): a 118B-total / 8B-active Mixture-of-Experts, 256 routed experts with top-10 routing plus a shared expert, 48 layers of which 12 are global attention and 36 sliding-window (512-token window, split rope bases). GQA with 8 KV heads and 128-wide K/V, YaRN-scaled context of 262,144 tokens in this GGUF (32x over the native 8,192). The file served here is `laguna-s-2.1-Q4_K_M.gguf`, 75 GB, imatrix-calibrated Q4_K_M.

## Test rig

| | |
|---|---|
| GPU | AMD Radeon 8060S (Strix Halo, RADV `STRIX_HALO`), 118784 MiB visible as GTT |
| Host | AMD Ryzen AI Max+ 395, 128 GB unified RAM |
| Image | `ghcr.io/ggml-org/llama.cpp:server-vulkan` (pulled 2026-07-22) |
| Server flags | `--ctx-size 655360 --parallel 5 --no-kv-unified --cache-ram 0`, f16 KV, image defaults otherwise |
| Context layout | five fixed slots, 131,072 tokens per request |

## Server results by context depth

Fresh prompts against `/completion` (`cache_prompt=false`), 128 forced output tokens, best of 3, one request at a time, via `scripts/bench_server.py`:

| Context depth | Prompt tokens | Prefill (t/s) | Decode (t/s) |
|--------------:|--------------:|--------------:|-------------:|
| 2k  |  2199 | 293 | 22.7 |
| 8k  |  8975 | 311 | 22.0 |
| 16k | 18141 | 275 | 21.1 |
| 32k | 36936 | 196 | 19.5 |

Prefill peaks at 8k depth (batch ramp), then falls off with attention cost; decode only drops ~14% from 2k to 32k, which the sliding-window layers deserve credit for. No speculative decoding on this stack, so decode is a narrow band, not the content-dependent spread the MTP models show.

## Memory

Loaded through the compose stack with all five slots allocated: 103,471 MiB (101 GiB) in GTT, of which 70 GiB is weights and roughly 31 GiB is KV cache plus compute buffers, about 6 GiB per 131k slot (the 12 global-attention layers at f16; the 36 sliding-window layers cost almost nothing). The VRAM carve-out stayed idle at 936 MiB. That leaves ~15 GiB of GTT headroom, so this is close to the ceiling for a five-slot layout on the 128 GB box. Verify on your box:

```bash
scripts/verify-gtt.sh --min-gtt-mib 70000
```

## Reproduce

```bash
docker compose up -d
curl -fsS http://localhost:${LLM_PORT:-8080}/slots | python3 scripts/check_context_config.py .env --slots-json -
python3 scripts/bench_server.py --url http://localhost:${LLM_PORT:-8080}
```

## Caveats

- One APU, idle box, one request in flight. Five slots share the same GPU; concurrent decode divides these numbers.
- The 32k row actually measured ~37k prompt tokens (the filler prompt overshoots its target); the label is the nominal depth.
- Sampling defaults come from the GGUF (temp 0.7, top-p 0.9); timings are sampler-independent since generation length is forced.
