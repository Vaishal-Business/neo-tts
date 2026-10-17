import asyncio
import gc
import hmac
import json
import os
import queue
import threading

from tornado.web import Application, RequestHandler
from tornado.routing import Rule, PathMatches

import tts_engine

# Allow one active generation at a time on the free CPU instance.
ACTIVE_GENERATION = threading.Lock()


def configured_api_key():
    try:
        import streamlit as st
        key = st.secrets.get("API_KEY", "")
        if key:
            return str(key)
    except Exception:
        pass

    return os.environ.get("API_KEY", "")


def send_json(handler, status, data):
    handler.set_status(status)
    handler.set_header("Content-Type", "application/json")
    handler.finish(json.dumps(data))


class HealthHandler(RequestHandler):
    def get(self):
        self.set_header("Cache-Control", "no-store")
        self.set_header("Content-Type", "application/json")
        self.finish(json.dumps({
            "status": "ok",
            "model_loaded": tts_engine.is_loaded(),
            "sample_rate": tts_engine.sample_rate(),
            "api": "pocket-tts-stream",
        }))


class SpeakHandler(RequestHandler):
    async def post(self):
        expected = configured_api_key()
        authorization = self.request.headers.get("Authorization", "")
        provided = authorization.removeprefix("Bearer ").strip()

        if not expected:
            send_json(self, 503, {
                "error": "API_KEY is not configured in Streamlit Secrets."
            })
            return

        if not hmac.compare_digest(provided, expected):
            send_json(self, 401, {"error": "Invalid API key."})
            return

        try:
            payload = json.loads(self.request.body.decode("utf-8"))
            text = payload.get("text", "")
        except Exception:
            send_json(self, 400, {"error": "Expected JSON with a text field."})
            return

        if not isinstance(text, str):
            send_json(self, 400, {"error": "text must be a string."})
            return

        text = text.strip()
        if not text:
            send_json(self, 400, {"error": "Text cannot be empty."})
            return

        if len(text) > 1000:
            send_json(self, 413, {"error": "Text limit is 1000 characters."})
            return

        if not ACTIVE_GENERATION.acquire(blocking=False):
            send_json(self, 429, {
                "error": "Another speech request is already running. Retry shortly."
            })
            return

        chunks = queue.Queue(maxsize=4)
        stop_event = threading.Event()
        sentinel = object()

        def put_item(item):
            while not stop_event.is_set():
                try:
                    chunks.put(item, timeout=0.25)
                    return True
                except queue.Full:
                    continue
            return False

        def produce():
            try:
                for audio_bytes in tts_engine.stream_float32(text, stop_event):
                    if not put_item(("audio", audio_bytes)):
                        break
            except Exception as exc:
                print(f"TTS streaming error: {exc!r}", flush=True)
                put_item(("error", str(exc)))
            finally:
                put_item(("done", sentinel))
                ACTIVE_GENERATION.release()

        self.set_header("Content-Type", "application/octet-stream")
        self.set_header("Cache-Control", "no-cache, no-transform")
        self.set_header("X-Accel-Buffering", "no")
        self.set_header("X-Audio-Format", "float32le")
        self.set_header("X-Audio-Channels", "1")
        self.set_header("X-Audio-Sample-Rate", "24000")

        worker = threading.Thread(target=produce, daemon=True)

        try:
            worker.start()
            await self.flush()

            while True:
                kind, item = await asyncio.to_thread(chunks.get)

                if kind == "audio":
                    self.write(item)
                    await self.flush()

                elif kind == "error":
                    print(
                        "Streaming ended because of an error:",
                        item,
                        flush=True,
                    )
                    break

                elif kind == "done":
                    break

        except Exception as exc:
            print(f"HTTP streaming error: {exc!r}", flush=True)

        finally:
            stop_event.set()
            try:
                self.finish()
            except Exception:
                pass


def register_routes():
    """Experimental: inject routes into Streamlit's internal Tornado app."""
    installed = 0

    try:
        candidates = [
            obj for obj in gc.get_referrers(Application)
            if obj.__class__ is Application
            and hasattr(obj, "wildcard_router")
        ]

        for app in candidates:
            if getattr(app, "_pocket_tts_api_installed", False):
                continue

            rules = app.wildcard_router.rules
            rules.insert(
                0,
                Rule(PathMatches(r"/api/health/?"), HealthHandler),
            )
            rules.insert(
                0,
                Rule(PathMatches(r"/api/speak/?"), SpeakHandler),
            )

            app._pocket_tts_api_installed = True
            installed += 1

    except Exception as exc:
        print(f"Could not register experimental API routes: {exc!r}")

    return installed > 0
