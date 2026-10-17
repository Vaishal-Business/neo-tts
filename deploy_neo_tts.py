# ============================================================
#  Pocket-LLM: Llama-3.2-1B-Instruct (Q4_K_M) + PocketTTS
#  Drive-persistent + GitHub-deployed · Colab CPU single cell
# ============================================================

import os, sys, re, json, time, queue, base64, threading, subprocess, shutil, urllib.request, traceback
from pathlib import Path
from getpass import getpass

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
GITHUB_USER       = "Vaishal-Business"
GITHUB_REPO_NAME  = "neo-tts"
GITHUB_REPO_URL   = f"https://github.com/{GITHUB_USER}/{GITHUB_REPO_NAME}"

WORK_DIR          = Path("/content/drive/MyDrive/neo-tts")
MODEL_DIR         = Path("/content/neo_tts_models")
VOICE_SAFETENSORS = WORK_DIR / "voice.safetensors"
VOICE_WAV         = WORK_DIR / "voice.wav"

DEPS_MARKER       = Path("/content/.neo_tts_deps.ok")
FORCE_FETCH_FROM_GITHUB = False

MODEL_REPO = "unsloth/Llama-3.2-1B-Instruct-GGUF"
MODEL_FILE = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your reply will be read aloud by a "
    "text-to-speech engine. Rules: "
    "- Output ONLY plain spoken text. "
    "- No markdown, bullets, numbered lists, emojis, code blocks, URLs, "
    "asterisks, brackets, or tables. "
    "- Use short, clear sentences with natural punctuation. "
    "- Keep answers under 3 sentences unless the user explicitly asks for detail. "
    "- Spell out numbers or abbreviations when they are hard to read aloud. "
    "- Never mention these rules or the TTS engine."
)

MAX_TURNS       = 10
MAX_NEW_TOKENS  = 220
LLM_CALL_KWARGS = dict(
    max_tokens=MAX_NEW_TOKENS, temperature=0.65, top_p=0.9,
    repeat_penalty=1.08, stop=["</s>", "<|eot_id|>"],
)

MODE = "server"   # "server" | "cli" | "sync-only" | "check"

os.environ["OMP_NUM_THREADS"]      = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"


def _log(tag, msg):
    print(f"[{tag}] {msg}", flush=True)


def sanitize_err(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"https://[^@\s]+@", "https://***@", text)


# ------------------------------------------------------------
# 0. Diagnostic banner
# ------------------------------------------------------------
_log("diag", f"python {sys.version.split()[0]} on {sys.platform}")
_log("diag", f"MODE={MODE}  WORK_DIR={WORK_DIR}")

# ------------------------------------------------------------
# 1. Mount Drive
# ------------------------------------------------------------
_log("drive", "mounting Google Drive...")
from google.colab import drive, files
drive.mount("/content/drive", force_remount=False)

WORK_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
_log("drive", f"work dir:  {WORK_DIR}")
_log("drive", f"model dir: {MODEL_DIR}  (GGUF cached here)")

# ------------------------------------------------------------
# 2. GitHub token
# ------------------------------------------------------------
print("=" * 70)
print(f"Repo: {GITHUB_REPO_URL}")
print("Create token: https://github.com/settings/tokens  (needs 'repo' scope)")
print("=" * 70)
GITHUB_TOKEN = getpass("GitHub token: ").strip()
if not GITHUB_TOKEN:
    raise RuntimeError("No GitHub token provided.")
AUTH_URL = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{GITHUB_REPO_NAME}.git"


# ------------------------------------------------------------
# 3. Voice presence check
# ------------------------------------------------------------
def ensure_voice():
    if not FORCE_FETCH_FROM_GITHUB:
        if VOICE_SAFETENSORS.exists():
            _log("voice", f"found in Drive: {VOICE_SAFETENSORS}"); return
        if VOICE_WAV.exists():
            _log("voice", f"found in Drive: {VOICE_WAV} (will export on load)"); return

    _log("voice", "not in Drive — trying GitHub restore...")
    tmp = Path("/content/_voice_restore")
    if tmp.exists(): shutil.rmtree(tmp)
    r = subprocess.run(["git", "clone", "--depth", "1", AUTH_URL, str(tmp)],
                       capture_output=True, text=True, timeout=180)
    if r.returncode == 0:
        for f in tmp.iterdir():
            if f.is_file() and f.suffix.lower() in (".safetensors", ".wav"):
                shutil.copy2(f, WORK_DIR / f.name)
                _log("voice", f"restored: {f.name}")
        shutil.rmtree(tmp, ignore_errors=True)
        if VOICE_SAFETENSORS.exists() or VOICE_WAV.exists(): return
    else:
        _log("voice", f"GitHub fetch failed: {sanitize_err(r.stderr)[:200]}")

    print("=" * 70)
    print("No voice file. Upload voice.safetensors (preferred) or voice.wav")
    print("=" * 70)
    uploaded = files.upload()
    if not uploaded: raise RuntimeError("No voice file uploaded.")
    for fname, data in uploaded.items():
        dest = WORK_DIR / fname
        dest.write_bytes(data)
        _log("voice", f"saved {dest} ({len(data)//1024} KB)")
        suf = Path(fname).suffix.lower()
        if suf == ".safetensors" and not VOICE_SAFETENSORS.exists():
            shutil.copy2(dest, VOICE_SAFETENSORS)
        elif suf == ".wav" and not VOICE_WAV.exists():
            shutil.copy2(dest, VOICE_WAV)
    if not (VOICE_SAFETENSORS.exists() or VOICE_WAV.exists()):
        raise RuntimeError("Uploaded file is not .safetensors or .wav")


ensure_voice()

# ------------------------------------------------------------
# 4. Clean-push to GitHub
# ------------------------------------------------------------
def sync_to_github(commit_msg: str = "Deploy neo-tts") -> bool:
    _log("sync", f"clean-pushing to {GITHUB_REPO_URL} ...")
    tmp = Path("/content/_neo_tts_sync")
    if tmp.exists(): shutil.rmtree(tmp)

    r = subprocess.run(["git", "clone", AUTH_URL, str(tmp)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        err = r.stderr.lower()
        if "empty repository" in err or "could not read" in err or "not found" in err:
            tmp.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init"], cwd=str(tmp), check=True)
            subprocess.run(["git", "remote", "add", "origin", AUTH_URL],
                           cwd=str(tmp), check=True)
            _log("sync", "repo was empty — initialised new git dir")
        else:
            _log("sync", f"clone failed: {sanitize_err(r.stderr)[:300]}")
            return False

    wiped = 0
    for item in tmp.iterdir():
        if item.name == ".git": continue
        if item.is_dir(): shutil.rmtree(item)
        else:             item.unlink()
        wiped += 1
    _log("sync", f"wiped {wiped} existing file(s) from repo")

    copied = 0
    for item in WORK_DIR.iterdir():
        dst = tmp / item.name
        if item.is_dir(): shutil.copytree(item, dst)
        else:             shutil.copy2(item, dst)
        copied += 1
    _log("sync", f"copied {copied} file(s) from Drive")

    try:
        import IPython
        ip = IPython.get_ipython()
        src = None
        for cand in (ip.user_ns.get("In", []),
                     getattr(ip.history_manager, "input_hist_raw", [])):
            if cand and isinstance(cand, list) and cand:
                last = cand[-1]
                if last and "GITHUB_REPO_NAME" in last and "neo-tts" in last:
                    src = last; break
        if src:
            (tmp / "deploy_neo_tts.py").write_text(src)
            _log("sync", f"saved cell source ({len(src)} chars)")
    except Exception as e:
        _log("sync", f"source capture skipped: {e}")

    readme = (
        "# neo-tts\n\n"
        "Llama-3.2-1B-Instruct (Q4_K_M) + PocketTTS streaming server — Colab CPU build.\n\n"
        "## Restore on a fresh runtime\n"
        "1. Mount Drive:\n"
        "   ```python\n"
        "   from google.colab import drive; drive.mount('/content/drive')\n"
        "   ```\n"
        "2. Keep `voice.safetensors` in `/content/drive/MyDrive/neo-tts/`,\n"
        "   or let the bootstrap cell auto-fetch it from this repo.\n"
        "3. Run the Colab cell (`deploy_neo_tts.py` if present).\n\n"
        f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    )
    (tmp / "README.md").write_text(readme)

    for cmd in (["git", "config", "user.email", "colab@neo-tts.local"],
                ["git", "config", "user.name",  "Colab Deployer"],
                ["git", "add", "-A"]):
        subprocess.run(cmd, cwd=str(tmp), check=True)

    r = subprocess.run(["git", "commit", "-m", commit_msg],
                       cwd=str(tmp), capture_output=True, text=True)
    _log("sync", (r.stdout.strip() or r.stderr.strip() or "committed")[:150])

    pushed, last_err = False, ""
    for branch in ("main", "master"):
        r = subprocess.run(["git", "push", "--force", "origin", f"HEAD:{branch}"],
                           cwd=str(tmp), capture_output=True, text=True)
        if r.returncode == 0:
            _log("sync", f"✓ pushed to '{branch}' → {GITHUB_REPO_URL}")
            pushed = True; break
        last_err = r.stderr

    if not pushed:
        _log("sync", f"✗ push failed: {sanitize_err(last_err)[:300]}")

    shutil.rmtree(tmp, ignore_errors=True)
    return pushed


try:
    sync_to_github("Backup voice + deploy neo-tts")
except Exception as e:
    _log("sync", f"initial sync skipped: {e}")


# ============================================================
# Rest of the pipeline
# ============================================================
if MODE != "sync-only":

    # --------------------------------------------------------
    # 5. Dependency install (robust)
    # --------------------------------------------------------
    def _try_imports():
        status = {}
        for name in ("llama_cpp", "pocket_tts", "fastapi", "uvicorn", "nest_asyncio"):
            try:
                __import__(name); status[name] = True
            except Exception as e:
                status[name] = f"{type(e).__name__}: {e}"
        return status


    def deps_ok() -> bool:
        return all(v is True for v in _try_imports().values())


    if MODE == "check":
        _log("check", "import status:")
        for k, v in _try_imports().items():
            print(f"    {k:14s} → {v}")
        _log("check", "done")
        raise SystemExit(0)


    if DEPS_MARKER.exists() and deps_ok():
        _log("setup", "dependencies already installed — skipping.")
    else:
        _log("setup", "installing system packages...")
        subprocess.run(["apt-get", "update", "-qq"], check=False)
        subprocess.run(["apt-get", "install", "-y", "-qq",
                        "libopenblas-dev", "cmake", "build-essential"], check=False)

        _log("setup", "building llama-cpp-python with OpenBLAS (~5-10 min first run)...")
        env = os.environ.copy()
        env["CMAKE_ARGS"] = "-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS"
        built = False
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "--no-binary", "llama-cpp-python",
                            "--no-cache-dir", "llama-cpp-python"],
                           env=env, check=True)
            built = True
        except subprocess.CalledProcessError:
            _log("setup", "source build failed → trying prebuilt wheel")
        if not built:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                                "llama-cpp-python"], check=True)
            except subprocess.CalledProcessError as e:
                _log("setup", f"llama-cpp-python install failed: {e}")
                _log("setup", "→ try running the cell once more; if it persists, "
                              "!pip install --upgrade pip setuptools wheel and retry")

        _log("setup", "installing PocketTTS + server deps...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                            "pocket-tts", "fastapi", "uvicorn", "nest_asyncio",
                            "--extra-index-url", "https://download.pytorch.org/whl/cpu"],
                           check=True)
        except subprocess.CalledProcessError as e:
            _log("setup", f"pocket-tts install FAILED: {e}")
            _log("setup", "→ `pocket-tts` may not exist on PyPI. Check the package "
                          "name; if the project ships a git URL, install it manually.")

        if not deps_ok():
            _log("setup", "one or more imports still failing:")
            for k, v in _try_imports().items():
                if v is not True:
                    print(f"    {k:14s} → {v}")
            raise RuntimeError("Dependency check failed after install. See [setup] lines above.")

        DEPS_MARKER.write_text("ok")

    # --------------------------------------------------------
    # 6. Imports
    # --------------------------------------------------------
    try:
        import numpy as np
        import torch
        from llama_cpp import Llama
    except Exception as e:
        raise RuntimeError(f"core import failed: {e}") from e

    _log("diag", f"torch {torch.__version__}")

    # pocket_tts is imported flexibly so a minor API rename doesn't kill the cell
    TTSModel = None
    export_model_state = None
    pocket_tts_err = None
    try:
        from pocket_tts import TTSModel as _TTSModel  # type: ignore
        TTSModel = _TTSModel
    except Exception as e:
        pocket_tts_err = e

    try:
        from pocket_tts import export_model_state as _export  # type: ignore
        export_model_state = _export
    except Exception:
        pass

    if TTSModel is None:
        _log("warn", f"pocket_tts unavailable: {pocket_tts_err}")
        _log("warn", "continuing in TEXT-ONLY mode (no audio)")

    from huggingface_hub import hf_hub_download
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse
    from pydantic import BaseModel
    import uvicorn
    import nest_asyncio

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # --------------------------------------------------------
    # 7. Global state
    # --------------------------------------------------------
    HISTORY   = [{"role": "system", "content": SYSTEM_PROMPT}]
    TURN_LOCK = threading.Lock()
    TTS_LOCK  = threading.Lock()

    llm         = None
    tts_model   = None
    voice_state = None
    SAMPLE_RATE = 24000

    # --------------------------------------------------------
    # 8. Helpers
    # --------------------------------------------------------
    def trim_history(history, max_turns=MAX_TURNS):
        while len(history) - 1 > max_turns * 2:
            del history[1:3]


    def split_sentences(buffer: str, force: bool = False):
        sentences = []
        while True:
            m = re.search(r"[.!?]", buffer)
            if not m: break
            end = m.end()
            s = buffer[:end].strip()
            if s: sentences.append(s)
            buffer = buffer[end:].lstrip()

        if len(buffer) > 180:
            idx = max(buffer.rfind(", "), buffer.rfind(" "))
            if idx > 80:
                sentences.append(buffer[: idx + 1].strip())
                buffer = buffer[idx + 1:].lstrip()

        if force and buffer.strip():
            sentences.append(buffer.strip()); buffer = ""
        return sentences, buffer


    def clean_for_tts(text: str) -> str:
        text = re.sub(r"```.*?```", " ", text, flags=re.S)
        text = re.sub(r"`[^`]*`", " ", text)
        text = re.sub(r"https?://\S+", " ", text)
        text = re.sub(r"\[([^\]]*?)\]\([^)]*?\)", r"\1", text)
        text = re.sub(r"(?m)^\s*(?:\d+[.)]|[-•])\s+", "", text)
        text = re.sub(r"[_#>*`{}\[\]|]", " ", text)
        text = re.sub(r"[\U0001F000-\U0001FAFF\u2600-\u27BF]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\s+([,.!?;:])", r"\1", text)
        return text


    def pcm_bytes(chunk):
        arr = chunk.detach().cpu().numpy()
        if arr.ndim > 1: arr = arr.reshape(-1)
        arr = np.clip(arr, -1.0, 1.0)
        return (arr * 32767.0).astype(np.int16).tobytes()


    def safe_put(q, item, stop_event, timeout=0.5):
        while not stop_event.is_set():
            try:
                q.put(item, timeout=timeout); return True
            except queue.Full:
                continue
        return False

    # --------------------------------------------------------
    # 9. Load engine (with kwarg fallbacks)
    # --------------------------------------------------------
    def _load_llama(model_path):
        attempts = [
            dict(model_path=model_path, n_ctx=8192, n_threads=2, n_threads_batch=2,
                 n_batch=512, n_gpu_layers=0, flash_attn=True, verbose=False,
                 use_mlock=False, seed=42),
            dict(model_path=model_path, n_ctx=8192, n_threads=2, n_threads_batch=2,
                 n_batch=512, n_gpu_layers=0, flash_attn=False, verbose=False,
                 use_mlock=False, seed=42),
            dict(model_path=model_path, n_ctx=8192, n_threads=2, verbose=False),
            dict(model_path=model_path),
        ]
        last = None
        for i, kw in enumerate(attempts, 1):
            try:
                _log("load", f"Llama attempt {i}/{len(attempts)} (kw={sorted(kw.keys())})")
                return Llama(**kw)
            except Exception as e:
                last = e
                _log("load", f"  failed: {type(e).__name__}: {e}")
        raise RuntimeError(f"all Llama init attempts failed. Last: {last}")


    def _load_tts():
        if TTSModel is None: return None, None, SAMPLE_RATE
        try:
            m = TTSModel.load_model(language="english", quantize=True)
            _log("load", "TTS: quantize=True")
        except Exception as e:
            _log("load", f"quantize=True failed ({e}); retrying fp32")
            m = TTSModel.load_model(language="english", quantize=False)
        sr = int(getattr(m, "sample_rate", 24000) or 24000)
        return m, m, sr


    def _load_voice(m):
        try:
            if VOICE_SAFETENSORS.exists():
                _log("load", f"voice ← {VOICE_SAFETENSORS.name}")
                return m.get_state_for_audio_prompt(str(VOICE_SAFETENSORS))
            if VOICE_WAV.exists():
                _log("load", f"voice ← {VOICE_WAV.name} (exporting…)")
                st = m.get_state_for_audio_prompt(str(VOICE_WAV))
                if export_model_state is not None:
                    try: export_model_state(st, str(VOICE_SAFETENSORS))
                    except Exception as e: _log("load", f"  export failed: {e}")
                return st
            _log("load", "voice ← builtin 'alba'")
            return m.get_state_for_audio_prompt("alba")
        except Exception as e:
            _log("load", f"voice load failed ({e}); falling back to 'alba'")
            return m.get_state_for_audio_prompt("alba")


    def load_engine():
        global llm, tts_model, voice_state, SAMPLE_RATE

        _log("load", "downloading GGUF (cached after first run)...")
        model_path = hf_hub_download(
            repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=str(MODEL_DIR),
        )
        _log("load", f"model: {model_path}")

        llm = _load_llama(model_path)

        tts_model, _, SAMPLE_RATE = _load_tts()
        if tts_model is not None:
            _log("load", f"TTS sample rate: {SAMPLE_RATE} Hz")
            voice_state = _load_voice(tts_model)
        else:
            voice_state = None
        _log("load", "ready.")

    # --------------------------------------------------------
    # 10. Producer / consumer
    # --------------------------------------------------------
    def llm_producer(user_text, event_q, tts_q, stop_event, metrics):
        t0 = time.time()
        first_token_time = None
        token_count = 0
        parts, buf = [], ""
        try:
            with TURN_LOCK:
                HISTORY.append({"role": "user", "content": user_text})
                for chunk in llm.create_chat_completion(
                    messages=HISTORY, stream=True, **LLM_CALL_KWARGS
                ):
                    if stop_event.is_set(): break
                    delta = chunk["choices"][0].get("delta", {}).get("content", "")
                    if not delta: continue
                    if first_token_time is None:
                        first_token_time = time.time()
                        metrics["ttft_ms"] = (first_token_time - t0) * 1000.0
                    token_count += 1
                    parts.append(delta)
                    safe_put(event_q, {"type": "text", "text": delta}, stop_event)
                    buf += delta
                    ss, buf = split_sentences(buf)
                    for s in ss:
                        c = clean_for_tts(s)
                        if c: safe_put(tts_q, c, stop_event)
                if buf.strip():
                    c = clean_for_tts(buf)
                    if c: safe_put(tts_q, c, stop_event)
                HISTORY.append({"role": "assistant", "content": "".join(parts).strip()})
                trim_history(HISTORY)
        except Exception as e:
            _log("LLM ERROR", str(e)); traceback.print_exc()
            metrics["llm_error"] = str(e)
            if HISTORY and HISTORY[-1]["role"] == "user":
                HISTORY.pop()
        finally:
            metrics["llm_time_s"] = time.time() - t0
            metrics["tokens"]     = token_count
            metrics["tok_per_s"]  = token_count / max(metrics["llm_time_s"], 1e-6)
            try: tts_q.put(None, timeout=2)
            except queue.Full: pass
            safe_put(event_q, {"type": "metrics", "metrics": dict(metrics)}, stop_event)
            safe_put(event_q, {"type": "llm_done"}, stop_event)


    def tts_consumer(tts_q, event_q, stop_event, metrics):
        t0 = time.time()
        first_audio_time = None
        total_samples = 0
        try:
            if tts_model is None or voice_state is None:
                while True:
                    try: s = tts_q.get(timeout=60)
                    except queue.Empty: break
                    if s is None: break
            else:
                with TTS_LOCK:
                    while not stop_event.is_set():
                        try: sentence = tts_q.get(timeout=120)
                        except queue.Empty: break
                        if sentence is None or stop_event.is_set(): break
                        try:
                            stream = tts_model.generate_audio_stream(
                                voice_state, sentence, copy_state=True)
                        except TypeError:
                            stream = tts_model.generate_audio_stream(voice_state, sentence)
                        for chunk in stream:
                            if stop_event.is_set(): break
                            pcm = pcm_bytes(chunk)
                            if not pcm: continue
                            if first_audio_time is None:
                                first_audio_time = time.time()
                                metrics["first_audio_ms"] = (first_audio_time - t0) * 1000.0
                            total_samples += len(pcm) // 2
                            safe_put(event_q,
                                     {"type": "audio",
                                      "pcm": base64.b64encode(pcm).decode("ascii")},
                                     stop_event)
        except Exception as e:
            _log("TTS ERROR", str(e)); traceback.print_exc()
            metrics["tts_error"] = str(e)
        finally:
            metrics["audio_s"]    = total_samples / max(SAMPLE_RATE, 1)
            metrics["tts_time_s"] = time.time() - t0
            safe_put(event_q, {"type": "metrics", "metrics": dict(metrics)}, stop_event)
            safe_put(event_q, {"type": "tts_done"}, stop_event)


    def event_stream(event_q, stop_event):
        done = 0
        try:
            while done < 2:
                try: item = event_q.get(timeout=120)
                except queue.Empty: break
                if item is None: break
                if item.get("type") in ("llm_done", "tts_done"):
                    done += 1; continue
                yield json.dumps(item, ensure_ascii=False) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        finally:
            stop_event.set()

    # --------------------------------------------------------
    # 11. FastAPI
    # --------------------------------------------------------
    app = FastAPI()


    class ChatRequest(BaseModel):
        text: str


    @app.get("/")
    def index():
        return HTMLResponse(UI_HTML)


    @app.get("/health")
    def health():
        return {
            "ok": True,
            "model_loaded": llm is not None,
            "tts_loaded":   tts_model is not None,
            "voice": (VOICE_SAFETENSORS.name if VOICE_SAFETENSORS.exists()
                      else VOICE_WAV.name if VOICE_WAV.exists() else "alba"),
        }


    @app.post("/reset")
    def reset():
        with TURN_LOCK:
            del HISTORY[1:]
        return {"ok": True}


    @app.post("/chat")
    def chat(req: ChatRequest):
        text = (req.text or "").strip()
        if not text: raise HTTPException(400, "empty text")
        if len(text) > 1000: raise HTTPException(400, "text too long")
        event_q, tts_q = queue.Queue(maxsize=256), queue.Queue()
        stop_event, metrics = threading.Event(), {}
        threading.Thread(target=llm_producer,
                         args=(text, event_q, tts_q, stop_event, metrics),
                         daemon=True).start()
        threading.Thread(target=tts_consumer,
                         args=(tts_q, event_q, stop_event, metrics),
                         daemon=True).start()
        return StreamingResponse(
            event_stream(event_q, stop_event),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache",
                     "X-Accel-Buffering": "no",
                     "Connection": "keep-alive"},
        )


    UI_HTML = r"""<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>neo-tts</title>
<style>
:root{color-scheme:dark}
body{margin:0;font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#0b0f14;color:#e6edf3}
.wrap{max-width:860px;margin:0 auto;padding:18px}
h1{font-size:20px;margin:0 0 4px}
.sub{color:#8b949e;font-size:12px;margin-bottom:14px}
#log{border:1px solid #21262d;border-radius:12px;padding:12px;height:52vh;overflow:auto;background:#010409}
.msg{margin:10px 0;padding:9px 12px;border-radius:10px;line-height:1.45;white-space:pre-wrap;word-break:break-word}
.user{background:#1f6feb22;border:1px solid #1f6feb55}
.bot{background:#23863622;border:1px solid #23863655}
.meta{color:#8b949e;font-size:11px;margin-top:4px}
.row{display:flex;gap:8px;margin-top:12px}
input{flex:1;padding:12px;border-radius:10px;border:1px solid #30363d;background:#0d1117;color:#e6edf3;font-size:15px;outline:none}
button{padding:12px 16px;border-radius:10px;border:1px solid #30363d;background:#238636;color:#fff;font-weight:600;cursor:pointer}
button.secondary{background:#21262d}
button:disabled{opacity:.5;cursor:not-allowed}
</style></head><body>
<div class="wrap">
<h1>neo-tts</h1>
<div class="sub">Llama-3.2-1B-Instruct Q4_K_M + PocketTTS · streaming text + 24 kHz PCM</div>
<div id="log"></div>
<div class="row">
  <input id="input" placeholder="Type a message and press Enter..." autocomplete="off">
  <button id="send" onclick="sendMessage()">Send</button>
  <button class="secondary" onclick="resetChat()">Reset</button>
</div>
<div id="status" class="meta">Ready.</div>
</div>
<script>
const logEl=document.getElementById('log'),inputEl=document.getElementById('input'),
      sendEl=document.getElementById('send'),statusEl=document.getElementById('status');
let ctx=null,nextTime=0;
function ensureAudio(){if(!ctx)ctx=new(window.AudioContext||window.webkitAudioContext)();return ctx}
function schedulePcm(b64){
  const ac=ensureAudio(),bin=atob(b64),bytes=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++)bytes[i]=bin.charCodeAt(i);
  const n=Math.floor(bytes.length/2);if(!n)return;
  const i16=new Int16Array(bytes.buffer,0,n),f32=new Float32Array(n);
  for(let i=0;i<n;i++)f32[i]=i16[i]/32768;
  const buf=ac.createBuffer(1,n,24000);buf.copyToChannel(f32,0);
  const src=ac.createBufferSource();src.buffer=buf;src.connect(ac.destination);
  const t=Math.max(ac.currentTime+0.04,nextTime);src.start(t);nextTime=t+buf.duration;
}
function addBubble(role,text){
  const d=document.createElement('div');d.className='msg '+role;d.textContent=text;
  logEl.appendChild(d);logEl.scrollTop=logEl.scrollHeight;return d;
}
function setMeta(el,m){
  if(!m)return;const p=[];
  if(m.ttft_ms!=null)p.push(`TTFT ${Math.round(m.ttft_ms)}ms`);
  if(m.llm_time_s!=null)p.push(`LLM ${m.llm_time_s.toFixed(2)}s`);
  if(m.tok_per_s!=null)p.push(`${m.tok_per_s.toFixed(1)} tok/s`);
  if(m.first_audio_ms!=null)p.push(`first audio ${Math.round(m.first_audio_ms)}ms`);
  if(m.audio_s!=null)p.push(`audio ${m.audio_s.toFixed(2)}s`);
  if(p.length)el.textContent=p.join(' · ');
}
async function sendMessage(){
  const text=inputEl.value.trim();if(!text)return;
  inputEl.value='';sendEl.disabled=true;statusEl.textContent='Generating...';
  await ensureAudio().resume();
  addBubble('user',text);
  const bot=addBubble('bot','');
  const textEl=document.createElement('span');bot.textContent='';bot.appendChild(textEl);
  const meta=document.createElement('div');meta.className='meta';bot.appendChild(meta);
  let full='';
  try{
    const resp=await fetch('/chat',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    if(!resp.ok)throw new Error(await resp.text());
    const rd=resp.body.getReader(),dec=new TextDecoder();let lineBuf='';
    while(true){
      const {done,value}=await rd.read();if(done)break;
      lineBuf+=dec.decode(value,{stream:true});let idx;
      while((idx=lineBuf.indexOf('\n'))>=0){
        const line=lineBuf.slice(0,idx).trim();lineBuf=lineBuf.slice(idx+1);
        if(!line)continue;const ev=JSON.parse(line);
        if(ev.type==='text'){full+=ev.text;textEl.textContent=full;logEl.scrollTop=logEl.scrollHeight;}
        else if(ev.type==='audio'){schedulePcm(ev.pcm);}
        else if(ev.type==='metrics'){setMeta(meta,ev.metrics);}
        else if(ev.type==='done'){statusEl.textContent='Ready.';}
      }
    }
  }catch(e){bot.textContent='Error: '+e.message;statusEl.textContent='Error.';}
  finally{sendEl.disabled=false;inputEl.focus();}
}
async function resetChat(){
  await fetch('/reset',{method:'POST'});logEl.innerHTML='';nextTime=0;
  statusEl.textContent='History reset.';
}
inputEl.addEventListener('keydown',e=>{if(e.key==='Enter')sendMessage();});
</script></body></html>
"""


    def start_cloudflare_tunnel(port: int = 8000):
        cf = Path("/content/cloudflared")
        if not cf.exists():
            _log("tunnel", "downloading cloudflared...")
            url = ("https://github.com/cloudflare/cloudflared/releases/latest/"
                   "download/cloudflared-linux-amd64")
            urllib.request.urlretrieve(url, str(cf)); os.chmod(cf, 0o755)
        log_path = Path("/content/cloudflared.log")
        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            [str(cf), "tunnel", "--url", f"http://127.0.0.1:{port}",
             "--no-autoupdate", "--protocol", "http2"],
            stdout=log_file, stderr=subprocess.STDOUT,
        )
        _log("tunnel", "waiting for public URL...")
        for _ in range(60):
            time.sleep(1)
            txt = log_path.read_text(errors="ignore") if log_path.exists() else ""
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", txt)
            if m: return m.group(0), proc
        return None, proc


    def start_server():
        _log("server", "starting uvicorn...")
        nest_asyncio.apply()
        cfg = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
        server = uvicorn.Server(cfg)
        threading.Thread(target=server.run, daemon=True).start()

        for _ in range(30):
            try:
                urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        _log("server", "local: http://127.0.0.1:8000")

        url, cf_proc = start_cloudflare_tunnel(8000)
        if url:
            print("\n" + "=" * 70)
            print(f" PUBLIC URL: {url}")
            print(" No API key required (private tunnel).")
            print("=" * 70 + "\n")
        else:
            _log("tunnel", "failed to obtain URL; see /content/cloudflared.log")

        try:
            while True: time.sleep(10)
        except KeyboardInterrupt:
            _log("server", "shutting down...")
            try: cf_proc.terminate()
            except Exception: pass

    # --------------------------------------------------------
    # 12. CLI mode
    # --------------------------------------------------------
    def cli_loop():
        print("\nneo-tts CLI. Type 'exit' or 'quit' to stop.\n")
        while True:
            try: user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye."); break
            if user_text.lower() in {"exit", "quit"}:
                print("Bye."); break
            if not user_text: continue

            event_q, tts_q, audio_q = queue.Queue(256), queue.Queue(), queue.Queue()
            stop_event, metrics = threading.Event(), {}
            collected, wav = [], WORK_DIR / f"turn_{int(time.time())}.wav"

            def player():
                live, stream = False, None
                try:
                    import sounddevice as sd
                    stream = sd.RawOutputStream(samplerate=SAMPLE_RATE,
                                                channels=1, dtype="int16")
                    stream.start(); live = True
                except Exception:
                    live = False
                while True:
                    c = audio_q.get()
                    if c is None: break
                    if live: stream.write(c)
                    else:    collected.append(c)
                if live:
                    stream.stop(); stream.close()

            th = threading.Thread(target=player, daemon=True); th.start()
            threading.Thread(target=llm_producer,
                             args=(user_text, event_q, tts_q, stop_event, metrics),
                             daemon=True).start()
            threading.Thread(target=tts_consumer,
                             args=(tts_q, event_q, stop_event, metrics),
                             daemon=True).start()

            print("Bot: ", end="", flush=True)
            done = 0
            while done < 2:
                try: item = event_q.get(timeout=120)
                except queue.Empty: break
                if item is None: break
                t = item.get("type")
                if t == "text":   print(item["text"], end="", flush=True)
                elif t == "audio": audio_q.put(base64.b64decode(item["pcm"]))
                elif t == "metrics": metrics.update(item.get("metrics", {}))
                elif t in ("llm_done", "tts_done"): done += 1
            print()
            audio_q.put(None); th.join(timeout=30)

            if collected:
                pcm = b"".join(collected)
                samples = np.frombuffer(pcm, dtype=np.int16)
                try:
                    import scipy.io.wavfile as wavfile
                    wavfile.write(str(wav), SAMPLE_RATE, samples)
                    _log("audio", f"saved {wav}")
                    try:
                        from IPython.display import Audio, display
                        display(Audio(str(wav)))
                    except Exception:
                        pass
                except Exception as e:
                    _log("audio", f"could not save WAV: {e}")

            ttft  = metrics.get("ttft_ms"); llm_s = metrics.get("llm_time_s")
            tps   = metrics.get("tok_per_s"); fa = metrics.get("first_audio_ms")
            a_s   = metrics.get("audio_s"); t_s = metrics.get("tts_time_s")
            print("[latency] ", end="")
            print(f"TTFT={ttft:.0f}ms | " if ttft is not None else "TTFT=n/a | ", end="")
            if llm_s is not None: print(f"LLM={llm_s:.2f}s | {tps:.1f} tok/s | ", end="")
            if fa is not None:    print(f"first_audio={fa:.0f}ms | ", end="")
            if a_s is not None:   print(f"audio={a_s:.2f}s | ", end="")
            if t_s is not None:   print(f"TTS={t_s:.2f}s", end="")
            print("\n")

    # --------------------------------------------------------
    # 13. Entry point
    # --------------------------------------------------------
    print("=" * 70)
    print(" neo-tts · Llama-3.2-1B Q4_K_M + PocketTTS")
    print(f" MODE = {MODE}  WORK_DIR = {WORK_DIR}")
    print("=" * 70)

    load_engine()

    if MODE.lower() == "cli":
        cli_loop()
    else:
        start_server()

    # Runs after Ctrl-C (start_server blocks until then)
    try:
        voice_label = (VOICE_SAFETENSORS.name if VOICE_SAFETENSORS.exists()
                       else VOICE_WAV.name if VOICE_WAV.exists() else "alba")
        sync_to_github(f"Deploy neo-tts (voice={voice_label})")
    except Exception as e:
        _log("sync", f"final sync skipped: {e}")