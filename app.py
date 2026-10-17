import io
import wave

import numpy as np
import streamlit as st

import tts_engine
import stream_api

st.set_page_config(
    page_title="Pocket TTS Robot API",
    page_icon="🔊",
    layout="centered",
)

st.title("Pocket TTS — Robot Voice API")
st.write(
    "This app experiments with a direct HTTP streaming endpoint "
    "for a Raspberry Pi robot."
)

routes_ready = stream_api.register_routes()

if routes_ready:
    st.success("Experimental API routes registered in this process.")
else:
    st.warning(
        "The API route hook was not found. Check the logs and "
        "test /api/health after deployment."
    )

st.code(
    "GET  /api/health\n"
    "POST /api/speak\n\n"
    'JSON body: {"text": "Hello Sir"}\n'
    "Authorization: Bearer YOUR_API_KEY",
    language="text",
)

st.caption(
    "The custom REST route uses an unofficial Streamlit/Tornado "
    "workaround. Hosting infrastructure may not forward these routes."
)

if st.button("Load model and voice", type="primary"):
    with st.spinner("Loading Pocket TTS and voice state..."):
        try:
            model, voice = tts_engine.get_engine()
            st.success(
                f"Loaded. Sample rate: {model.sample_rate} Hz."
            )
        except Exception as exc:
            st.error(f"Model load failed: {exc}")

st.write(
    "Model loaded:",
    "yes" if tts_engine.is_loaded() else "not yet",
)

text = st.text_area(
    "Test speech",
    value="Hello Sir. Your robot voice server is ready.",
    max_chars=1000,
)

if st.button("Generate browser preview"):
    if not text.strip():
        st.warning("Enter some text first.")
    elif not stream_api.ACTIVE_GENERATION.acquire(blocking=False):
        st.warning("Another generation is running. Try again shortly.")
    else:
        try:
            with st.spinner("Generating speech..."):
                audio_parts = list(tts_engine.stream_float32(text.strip()))
                raw = b"".join(audio_parts)
                samples = np.frombuffer(raw, dtype="<f4")

                pcm = (
                    np.clip(samples, -1.0, 1.0) * 32767
                ).astype("<i2")

                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(tts_engine.sample_rate())
                    wav.writeframes(pcm.tobytes())

                st.audio(buffer.getvalue(), format="audio/wav")
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
        finally:
            stream_api.ACTIVE_GENERATION.release()

st.info(
    "Configure API_KEY under your deployed app's Settings → Secrets "
    "before sending requests from the Raspberry Pi."
)
