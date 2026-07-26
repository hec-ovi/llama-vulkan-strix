#!/bin/sh
set -eu

expected_sha256="${EXPECTED_MODEL_SHA256:-ea1d854a72c47ec8e72c16ea91b8ff3cd5e1620b834df175f683c86f27dc26d6}"
server_bin="${LLAMA_SERVER_BIN:-/app/llama-server}"
model_path=
next_is_model=0

for arg in "$@"; do
    if [ "$next_is_model" = 1 ]; then
        model_path=$arg
        next_is_model=0
        continue
    fi

    case "$arg" in
        --model)
            next_is_model=1
            ;;
        --model=*)
            model_path=${arg#--model=}
            ;;
    esac
done

if [ -z "$model_path" ] || [ ! -r "$model_path" ]; then
    echo "Laguna model is not readable: ${model_path:-<missing --model>}" >&2
    exit 2
fi

actual_sha256=$(sha256sum "$model_path" | awk '{print $1}')
if [ "$actual_sha256" != "$expected_sha256" ]; then
    echo "Laguna model SHA-256 mismatch: $model_path" >&2
    echo "expected: $expected_sha256" >&2
    echo "actual:   $actual_sha256" >&2
    exit 3
fi

exec "$server_bin" "$@"
