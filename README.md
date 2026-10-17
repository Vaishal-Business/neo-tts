# Pocket TTS Robot API

Pocket TTS streaming experiment for a Raspberry Pi client.

## Files

- `app.py`: Streamlit status and browser-preview interface.
- `stream_api.py`: experimental REST routes on Streamlit's internal Tornado server.
- `tts_engine.py`: loads Pocket TTS and the custom voice once per process.
- `my_robot_voice.safetensors`: private voice-conditioning state.
- `pi_client.py`: streams raw audio bytes into the Pi audio output.

## Deploy to Streamlit Community Cloud

1. Keep this repository private.
2. Open https://share.streamlit.io/ and create an app from this repository.
3. Select `app.py` as the main file.
4. Under app Settings -> Secrets, add:

   ```toml
   API_KEY = "PASTE_THE_GENERATED_API_KEY_HERE"
   ```

5. Deploy and open the app once to wake it up and initialise it.
6. Test `https://YOUR-APP-URL/api/health`.

## Raspberry Pi

Install client dependencies:

```bash
python -m pip install requests numpy sounddevice
```

Configure the URL and API key in the Pi terminal:

```bash
export POCKET_TTS_URL="https://YOUR-APP-URL"
export POCKET_TTS_API_KEY="YOUR_API_KEY"
```

Test:

```bash
python pi_client.py --health
python pi_client.py --text "Hello Sir, I am online."
```

## Streaming protocol

`POST /api/speak` with JSON:

```json
{"text":"Hello Sir"}
```

Header:

```
Authorization: Bearer YOUR_API_KEY
```

The response is a stream of raw mono `float32le` PCM audio.
Read `X-Audio-Sample-Rate` from the response headers.

## Important limitations

The route injection uses a community workaround against Streamlit internals.
It is not an officially supported REST API feature, and a platform proxy or
future Streamlit upgrade may break it. If `/api/health` does not work on
Community Cloud, this approach cannot be relied on there.

Community Cloud has constrained CPU/RAM and sleeps after inactivity.
Cold starts can be slow. One generation at a time is allowed by this app.
