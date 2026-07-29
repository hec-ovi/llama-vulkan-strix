#!/usr/bin/env bash
# Full end-to-end served benches, one model load at a time.
# Order: dense 27B, Laguna 8B-active, 35B MoE, Gemma. ROCmFP4 is NOT here.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p docs/bench-results
SUMMARY=docs/bench-results/run.log

wait_health() {
  local name=$1 max=${2:-360}
  local i=0
  while (( i < max )); do
    if curl -fsS --max-time 5 http://127.0.0.1:8080/health >/dev/null 2>&1; then
      echo "healthy $name $(date -Is)" | tee -a "$SUMMARY"
      return 0
    fi
    if ! docker compose ps --status running -q 2>/dev/null | grep -q .; then
      echo "container died during load: $name" | tee -a "$SUMMARY"
      docker compose logs --tail 80 | tee -a "docs/bench-results/${name}.log"
      return 1
    fi
    i=$((i + 1))
    sleep 10
    if (( i % 6 == 0 )); then echo "waiting $name ${i}0s..." | tee -a "$SUMMARY"; fi
  done
  echo "health timeout: $name" | tee -a "$SUMMARY"
  docker compose logs --tail 80 | tee -a "docs/bench-results/${name}.log"
  return 1
}

run_model() {
  local name=$1 pkg=$2 model=$3 template=$4 alias=$5
  local out="docs/bench-results/${name}.json"
  local log="docs/bench-results/${name}.log"
  : >"$log"
  echo "======== $(date -Is) START $name ========" | tee -a "$SUMMARY"

  export COMPOSE_FILE="docker-compose.yml:compose/models/${pkg}"
  export MODELS_DIR="${MODELS_DIR:-/home/hec/models/gguf}"
  export LLM_MODEL="$model"
  export LLM_CHAT_TEMPLATE="$template"
  export LLM_ALIAS="$alias"
  export LLM_PORT=8080 LLM_NGL=99
  export LLM_CTX_PER_SLOT=131072 LLM_PARALLEL=5 LLM_CTX_TOTAL=655360
  export RENDER_GID="${RENDER_GID:-990}" VIDEO_GID="${VIDEO_GID:-44}"

  docker compose down --remove-orphans >>"$log" 2>&1 || true
  docker compose config -q >>"$log" 2>&1
  docker compose up -d >>"$log" 2>&1
  wait_health "$name" || return 1

  # warm-up
  curl -fsS --max-time 300 http://127.0.0.1:8080/completion \
    -H 'content-type: application/json' \
    -d '{"prompt":"hi","n_predict":8,"cache_prompt":false,"ignore_eos":true}' \
    >/dev/null 2>&1 || true

  # full suite on this one load: prefill table + 10x decode avgs
  set +e
  python3 scripts/bench_full.py \
    --url http://127.0.0.1:8080 \
    --name "$name" \
    --depths 64,16384,32768 \
    --prefill-runs 3 \
    --decode-avg-runs 10 \
    --decode-tokens 128 \
    --timeout 1800 \
    >"$out" 2> >(tee -a "$log" >&2)
  local rc=$?
  set -e
  echo "bench exit $rc $name" | tee -a "$SUMMARY"
  if [[ -s $out ]]; then
    python3 - <<PY | tee -a "$SUMMARY"
import json
p=json.load(open("$out"))
print("  prefill:")
for r in p.get("prefill_table",[]):
    print(f"    {r['label']}: prefill={r['prefill_best']:.0f} decode={r['decode_best']:.1f} (n={r['prompt_n']})")
da=p.get("decode_avg",{})
np=da.get("no_prefill",{})
c=da.get("cached_32k",{})
if np:
    print(f"  decode avg no-prefill: mean={np.get('decode_mean',0):.1f} median={np.get('decode_median',0):.1f}")
if c:
    print(f"  decode avg 32k-cache: mean={c.get('decode_mean',0):.1f} median={c.get('decode_median',0):.1f} cache_hit_mean={c.get('cache_hit_mean',0):.0f}")
PY
  fi
  docker compose down --remove-orphans >>"$log" 2>&1 || true
  return $rc
}

# --- stock Vulkan packages only (ROCmFP4 intentionally omitted) ---
# 1) dense 27B (largest cost per token)
run_model qwen3.6-27b-heretic-q8 qwen3.6-27b-heretic.yml \
  Qwen3.6-27B-heretic-v2/Qwen3.6-27B-uncensored-heretic-v2-Native-MTP-Preserved-Q8_0.gguf \
  Qwen3.6-27B-heretic-v2/chat_template.jinja \
  qwen3.6-27b-heretic-v2-q8 || echo "FAILED 27b" | tee -a "$SUMMARY"

# 2) Laguna 8B active
run_model laguna-s-2.1-iq4xs laguna-s-2.1.yml \
  laguna-s-2.1/Laguna-S-2.1-IQ4_XS-00001-of-00002.gguf \
  laguna-s-2.1/chat_template.jinja \
  laguna-s21-iq4xs || echo "FAILED laguna" | tee -a "$SUMMARY"

# 3) 35B MoE heretic Q8
run_model qwen3.6-35b-heretic-q8 qwen3.6-35b-heretic.yml \
  Qwen3.6-35B-A3B-heretic/Qwen3.6-35B-A3B-uncensored-heretic-Q8_0.gguf \
  Qwen3.6-35B-A3B-heretic/chat_template.jinja \
  qwen3.6-35b-a3b-heretic-q8 || echo "FAILED 35b" | tee -a "$SUMMARY"

# 4) Gemma 4 26B A4B Q5
run_model gemma-4-26b-a4b-q5 gemma-4-26b-a4b.yml \
  gemma-4-26b-a4b-abliterated/Gemma-4-26B-A4B-It-Abliterated-Q5_K_M.gguf \
  gemma-4-26b-a4b-abliterated/chat_template.jinja \
  gemma-4-26b-a4b-abliterated-q5 || echo "FAILED gemma" | tee -a "$SUMMARY"

echo "======== $(date -Is) ALL STOCK MODELS DONE (ROCmFP4 not run) ========" | tee -a "$SUMMARY"
ls -la docs/bench-results/*.json
