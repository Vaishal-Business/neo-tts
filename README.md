# Pocket TTS — compiled INT8 runtime

This repository contains a compiled x86-64 Pocket TTS runtime and its INT8 ONNX models.

## Run in a compatible Linux x86-64 environment

```bash
chmod +x run.sh
./run.sh
```

The HTTP server defaults to port 8080. Set `PORT` to change it.

The compiled server advertises `/health`, `/tts`, and `/v1/audio/speech`.
This package does not include a separate assistant LLM, web-search service,
Cloudflare tunnel, or the previous `/api/health` and `/api/speak` wrapper.
The included voice cache is `myvoice_clean`; the original custom voice source
file is not included.
