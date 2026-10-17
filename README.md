# neo-tts

Llama-3.2-1B-Instruct (Q4_K_M) + PocketTTS streaming server — Colab CPU build.

## Restore on a fresh runtime
1. Mount Drive:
   ```python
   from google.colab import drive; drive.mount('/content/drive')
   ```
2. Keep `voice.safetensors` in `/content/drive/MyDrive/neo-tts/`,
   or let the bootstrap cell auto-fetch it from this repo.
3. Run the Colab cell (`deploy_neo_tts.py` if present).

Last updated: 2026-10-17 12:50:25
