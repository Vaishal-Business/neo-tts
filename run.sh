#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [ ! -e "$ROOT/libonnxruntime.so.1" ]; then
    ln -s libonnxruntime.so "$ROOT/libonnxruntime.so.1"
fi

export LD_LIBRARY_PATH="$ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$ROOT/pocket-tts" --server --port "${PORT:-8080}" \
  --precision int8 \
  --models-dir "$ROOT/models" \
  --voices-dir "$ROOT/voices" "$@"
