#!/usr/bin/env bash
# Prove the loaded model lives in GTT (unified system RAM), not the small VRAM
# carve-out. Run on the HOST after `docker compose up -d`, once the model has
# loaded. Reads the kernel's amdgpu sysfs counters (the source of truth on Strix
# Halo, where rocm-smi can misreport against the tiny VRAM pool).
#
#   scripts/verify-gtt.sh                         # defaults: gtt>=1024, vram<=1024 MiB
#   scripts/verify-gtt.sh --min-gtt-mib 14000     # expect ~14 GB model in GTT
#
# Any extra args are passed through to gpu_mem.py --verify.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${LLM_PORT:-8080}"

echo "[verify-gtt] waiting for llama.cpp /health on :${PORT} ..."
until curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; do
  sleep 2
done
echo "[verify-gtt] server healthy; reading amdgpu memory counters"
echo

python3 "${HERE}/gpu_mem.py" --verify "$@"
