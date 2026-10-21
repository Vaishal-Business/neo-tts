#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec ./pocket-tts --server --port "${PORT:-8080}" \
  --precision int8 --models-dir "$PWD/models" \
  --voices-dir "$PWD/voices" --tokenizer "$PWD/models/tokenizer.model"
