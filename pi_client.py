import os
import argparse
import requests
import numpy as np
import sounddevice as sd

SERVER_URL = os.environ.get(
    "POCKET_TTS_URL",
    "https://YOUR-APP-NAME.streamlit.app",
).rstrip("/")

API_KEY = os.environ.get(
    "POCKET_TTS_API_KEY",
    "PASTE_API_KEY_HERE",
)


def check_health():
    response = requests.get(
        SERVER_URL + "/api/health",
        timeout=30,
    )
    print("HTTP:", response.status_code)
    print(response.text)
    response.raise_for_status()


def speak(text):
    if "YOUR-APP-NAME" in SERVER_URL:
        raise ValueError("Set POCKET_TTS_URL to your deployed app URL.")
    if API_KEY == "PASTE_API_KEY_HERE":
        raise ValueError("Set POCKET_TTS_API_KEY to your API key.")

    with requests.post(
        SERVER_URL + "/api/speak",
        json={"text": text},
        headers={"Authorization": f"Bearer {API_KEY}"},
        stream=True,
        timeout=(30, None),
    ) as response:
        if not response.ok:
            print("Server error:", response.status_code, response.text[:1000])
            response.raise_for_status()

        sample_rate = int(
            response.headers.get("X-Audio-Sample-Rate", "24000")
        )

        print(
            f"Streaming {sample_rate} Hz audio from the server..."
        )

        # The HTTP chunks may split a float32 sample at any byte,
        # so preserve incomplete bytes between network reads.
        leftover = b""

        with sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
        ) as output:
            for piece in response.iter_content(chunk_size=4096):
                if not piece:
                    continue

                data = leftover + piece
                complete_bytes = len(data) - (len(data) % 4)

                if complete_bytes:
                    samples = np.frombuffer(
                        data[:complete_bytes],
                        dtype="<f4",
                    )
                    output.write(samples.reshape(-1, 1))

                leftover = data[complete_bytes:]

        print("Playback complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check the server without generating audio.",
    )
    parser.add_argument(
        "--text",
        type=str,
        help="Text to speak. If omitted, interactive mode is used.",
    )
    args = parser.parse_args()

    if args.health:
        check_health()
    elif args.text:
        speak(args.text)
    else:
        while True:
            text = input("\nText for the robot (or 'exit'): ").strip()
            if text.lower() in ("exit", "quit"):
                break
            if text:
                try:
                    speak(text)
                except Exception as exc:
                    print("Request failed:", exc)
