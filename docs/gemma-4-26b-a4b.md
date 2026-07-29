# Gemma 4 26B-A4B It Abliterated (Q5_K_M) on Strix Halo

Run notes for [SevenOfNine/Gemma-4-26B-A4B-It-Abliterated-GGUF](https://huggingface.co/SevenOfNine/Gemma-4-26B-A4B-It-Abliterated-GGUF) on the default stock Vulkan stack.

## Package

```dotenv
COMPOSE_FILE=docker-compose.yml:compose/models/gemma-4-26b-a4b.yml
LLM_MODEL=gemma-4-26b-a4b-abliterated/Gemma-4-26B-A4B-It-Abliterated-Q5_K_M.gguf
LLM_CHAT_TEMPLATE=gemma-4-26b-a4b-abliterated/chat_template.jinja
LLM_ALIAS=gemma-4-26b-a4b-abliterated-q5
LLM_CTX_PER_SLOT=131072
LLM_PARALLEL=5
LLM_CTX_TOTAL=655360
```

```bash
HF_TOKEN="$HF_TOKEN" hf download \
  SevenOfNine/Gemma-4-26B-A4B-It-Abliterated-GGUF \
  Gemma-4-26B-A4B-It-Abliterated-Q5_K_M.gguf \
  --local-dir "$MODELS_DIR/gemma-4-26b-a4b-abliterated"
docker compose up -d
```

## Quality vs original

| Metric | Abliterated | Original |
|---|---|---|
| KL divergence | 0.0845 | 0 |
| Refusals (hard set) | 18 / 100 | 100 / 100 |

Source: [abliteration model card](https://huggingface.co/SevenOfNine/Gemma-4-26B-A4B-It-Abliterated). Separate KL for Q5_K_M vs the abliterated BF16 was not published.

## Context

Medium Gemma 4 (26B A4B / 31B) is **256k** native context, not 128k (128k is E2B/E4B). Package default is five slots at 131072 with q8 KV.

## Speed on this rig

Not measured yet. Run:

```bash
python3 scripts/bench_server.py --url http://localhost:${LLM_PORT:-8080}
```

Publisher figure (different hardware): 34.5 t/s decode on RTX 4080 Super Q5_K_M with `-cmoe`.
