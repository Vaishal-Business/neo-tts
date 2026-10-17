import os
import threading
from pathlib import Path

import numpy as np
import torch
from pocket_tts import TTSModel

VOICE_FILE = Path(__file__).with_name("my_robot_voice.safetensors")

_model = None
_voice = None
_model_lock = threading.Lock()


def is_loaded():
    return _model is not None and _voice is not None


def get_engine():
    global _model, _voice

    if is_loaded():
        return _model, _voice

    with _model_lock:
        if is_loaded():
            return _model, _voice

        if not VOICE_FILE.is_file():
            raise FileNotFoundError(
                f"Voice file is missing: {VOICE_FILE}"
            )

        # Be considerate of the limited CPU allocation.
        torch.set_num_threads(min(2, torch.get_num_threads()))

        print("Loading Pocket TTS model...", flush=True)
        model = TTSModel.load_model()
        voice = model.get_state_for_audio_prompt(str(VOICE_FILE))

        _model = model
        _voice = voice

        print(
            f"Pocket TTS ready. Sample rate: {model.sample_rate}",
            flush=True,
        )

    return _model, _voice


def sample_rate():
    if is_loaded():
        return int(_model.sample_rate)
    return 24000


def stream_float32(text, stop_event=None):
    """Yield raw little-endian float32 mono audio chunks."""
    model, voice = get_engine()

    for chunk in model.generate_audio_stream(
        voice,
        text,
        copy_state=True,
        stop=stop_event,
    ):
        if stop_event is not None and stop_event.is_set():
            break

        audio = (
            chunk.detach()
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        yield audio.astype("<f4", copy=False).tobytes()
