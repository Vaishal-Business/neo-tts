#!/usr/bin/env python3
"""
neo-tts v2 deploy script. Runs as __main__ (python or %run).

How audio stays gap-free on 2 vCPUs (the core idea of v2)
---------------------------------------------------------
TTS here runs slower than real time (wall seconds per audio second, "w", is
about 1.3-1.6 for PocketTTS). Any "start playing, hope the buffer holds"
strategy underruns. But underrun is predictable. After the LLM finishes, the
full text is known, so the server can estimate how long every remaining
sentence will take to synthesise (wall) and how much audio it will produce.
Playback may start the moment, for EVERY remaining sentence j:

    audio_buffered + sum(audio of sentences before j)  >=  safety * sum(wall of sentences up to j) + lead

That is the earliest start that is still provably gap-free under the
estimate. With w=1.43 this starts audio after ~30% of the synthesis instead
of 100% (pure smooth mode), and with a fast backend (Kokoro, w<1) it starts
after the first sentence. The safety factor self-tunes: the client reports
late chunks (underruns) and the server raises safety; clean turns lower it.
smooth mode (one AudioBuffer, zero risk) remains available per request.
"""
import os, sys, re, io, json, time, glob, base64, shutil, threading, subprocess
import importlib.util, urllib.request, urllib.error
from pathlib import Path

# Learned the hard way: progress bars leave Jupyter cells in "partial line" state.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Match the 2 physical vCPUs; more threads = context switching.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "2")

def _env(n, d):
    return os.environ.get(n, d)

# ----- config (model is config, not a constant) -----
PORT = int(_env("NEO_PORT", "8000"))
MODEL_REPO = _env("NEO_MODEL_REPO", "unsloth/Llama-3.2-1B-Instruct-GGUF")
MODEL_FILE = _env("NEO_MODEL_FILE", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
# Size gate: a truncated GGUF (interrupted copy / Drive sync) makes the model
# emit immediate EOS = "0 tok/s". For Qwen2.5-0.5B Q4_K_M (~400 MB) set NEO_MODEL_MIN_MB=350.
MODEL_MIN_BYTES = int(float(_env("NEO_MODEL_MIN_MB", "700")) * 1e6)
AUDIO_MODE = _env("NEO_AUDIO_MODE", "adaptive").lower()
TTS_BACKEND = _env("NEO_TTS_BACKEND", "pocket").lower()
KOKORO_VOICE = _env("NEO_KOKORO_VOICE", "af_heart")
MAX_TURNS = 10            # rolling window of user/assistant pairs
MAX_TOKENS = int(_env("NEO_MAX_TOKENS", "160"))
PAD_S = 0.10              # silence between sentences
LEAD_S = float(_env("NEO_LEAD_S", "0.8"))   # extra cushion for tunnel jitter
SAFETY0, SAFETY_MIN, SAFETY_MAX = 1.3, 1.15, 2.5
MAX_SENT_CHARS = 220

SYSTEM_PROMPT = (
    "You are Neo, a voice assistant. Your replies are spoken aloud. "
    "Use plain spoken English only. Never use markdown, bullet points, numbered lists, "
    "headings, asterisks, emojis, code blocks, or URLs. Write numbers and symbols as words. "
    "Keep answers to two to four short sentences unless the person asks for more detail."
)

_T0 = time.time()
def log(msg):
    print(f"[{time.time() - _T0:6.1f}s] {msg}", flush=True)

try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()

# ===== PATHS / DRIVE =====
def resolve_state_dir():
    if not os.path.isdir("/content/drive/MyDrive"):
        try:
            from google.colab import drive  # guarded: same script must run in a local venv
            drive.mount("/content/drive")
        except ImportError:
            log("not on Colab; using local cache dir instead of Drive")
        except Exception as e:
            log(f"Drive mount failed: {e!r}; using local cache dir")
    base = Path("/content/drive/MyDrive") if os.path.isdir("/content/drive/MyDrive") else Path.home() / ".neo_tts_drive"
    d = base / "neo-tts"
    d.mkdir(parents=True, exist_ok=True)
    return d

STATE_DIR = resolve_state_dir()
MODELS_DRIVE = STATE_DIR / "models"
WHEELS_DRIVE = STATE_DIR / "wheels"
LOCAL_MODELS = Path("/content/neo_tts_models") if os.path.isdir("/content") else Path("/tmp/neo_tts_models")
URL_FILE = Path("/content/.neo_tts_url.txt") if os.path.isdir("/content") else Path("/tmp/.neo_tts_url.txt")
for _d in (MODELS_DRIVE, WHEELS_DRIVE, LOCAL_MODELS):
    _d.mkdir(parents=True, exist_ok=True)

def atomic_copy(src, dst):
    """Copy to .part, verify size, then rename. Root-cause fix for truncated GGUFs."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copyfile(src, tmp)
    if os.path.getsize(tmp) != os.path.getsize(src):
        raise IOError(f"short copy {src} -> {dst}")
    os.replace(tmp, dst)

def download(url, dst, min_bytes=0):
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(4 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
    if (total and got != total) or got < min_bytes:
        raise IOError(f"incomplete download {url}: {got}/{total} bytes")
    os.replace(tmp, dst)

def pip(*args):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--disable-pip-version-check", *args],
                          check=False, capture_output=True, text=True)

def run_with_heartbeat(cmd, env, label, every=30):
    lp = Path("/tmp") / (label.replace(" ", "_") + ".log")
    with open(lp, "w") as lf:
        p = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        t = time.time()
        while p.poll() is None:
            time.sleep(2)
            if time.time() - t >= every:
                log(f"{label}: still running ({int(time.time() - _T0)}s elapsed)")
                t = time.time()
    return p.returncode, lp.read_text(errors="ignore")[-1500:]

# ===== LLAMA-CPP (cached wheel first, source build fallback) =====
def llama_ok():
    # Subprocess check: an illegal-instruction crash must not kill the kernel.
    r = subprocess.run([sys.executable, "-c", "import llama_cpp"], check=False, capture_output=True, text=True)
    return r.returncode == 0

def ensure_llama():
    if llama_ok():
        log("llama_cpp already importable")
        return
    for d in (WHEELS_DRIVE, HERE / "wheels"):
        if d.is_dir() and glob.glob(str(d / "llama_cpp_python-*.whl")):
            r = pip("--no-index", "--no-deps", "--find-links", str(d), "llama-cpp-python")
            pip("diskcache", "jinja2", "numpy", "typing-extensions")
            if r.returncode == 0 and llama_ok():
                log(f"installed llama_cpp from cached wheel in {d}")
                return
            log(f"cached wheel in {d} unusable: {r.stderr[-300:]}")
    log("no usable cached wheel; building from source (5-10 min, first boot only)")
    subprocess.run(["apt-get", "install", "-y", "-qq", "libopenblas-dev", "pkg-config"],
                   check=False, capture_output=True)
    # DEVIATION from the old design: GGML_NATIVE=OFF + explicit AVX2/FMA/F16C.
    # Colab hands out different CPU models; a -march=native wheel cached on Drive
    # can SIGILL on a different host. Explicit flags give the same AVX2 kernels
    # (no SSE fallback) and the wheel stays portable. Both GGML_* and legacy
    # LLAMA_* BLAS names are passed because llama.cpp renamed them.
    cmake = ("-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS "
             "-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON")
    env = dict(os.environ, CMAKE_ARGS=cmake, FORCE_CMAKE="1", CMAKE_BUILD_PARALLEL_LEVEL="2")
    wh = Path("/tmp/neo_wheelhouse")
    shutil.rmtree(wh, ignore_errors=True)
    wh.mkdir(parents=True)
    rc, tail = run_with_heartbeat([sys.executable, "-m", "pip", "wheel", "--no-cache-dir", "--no-deps",
                                   "-w", str(wh), "llama-cpp-python"], env, "llama build")
    wheels = glob.glob(str(wh / "llama_cpp_python-*.whl"))
    if rc != 0 or not wheels:
        raise RuntimeError(f"llama-cpp-python build failed:\n{tail}")
    pip("--no-index", "--no-deps", "--find-links", str(wh), "llama-cpp-python")
    pip("diskcache", "jinja2", "numpy", "typing-extensions")
    if not llama_ok():
        raise RuntimeError("llama_cpp built but not importable")
    for w in wheels:
        shutil.copy2(w, WHEELS_DRIVE / Path(w).name)
        if os.path.getsize(w) < 95e6:  # GitHub hard limit is 100 MB per file
            try:
                (HERE / "wheels").mkdir(exist_ok=True)
                shutil.copy2(w, HERE / "wheels" / Path(w).name)
            except OSError as e:
                log(f"could not copy wheel into repo dir: {e!r}")
        else:
            log("wheel >95 MB: cached on Drive only (too big for plain git)")
    log("wheel cached to Drive (next boot installs in ~10 s)")

def ensure_py_deps():
    need = []
    for mod, pkg in (("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("numpy", "numpy"), ("pocket_tts", "pocket-tts")):
        if importlib.util.find_spec(mod) is None:
            need.append(pkg)
    if TTS_BACKEND == "kokoro" and importlib.util.find_spec("kokoro_onnx") is None:
        need.append("kokoro-onnx")
    if need:
        log(f"pip installing: {' '.join(need)}")
        r = pip(*need)
        if r.returncode != 0:
            raise RuntimeError(f"pip failed: {r.stderr[-500:]}")

def ensure_cloudflared():
    cf = shutil.which("cloudflared")
    if cf:
        return cf
    cached = STATE_DIR / "bin" / "cloudflared"
    dst = Path("/usr/local/bin/cloudflared")
    try:
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            download("https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                     cached, 10_000_000)
        shutil.copyfile(cached, dst)
        os.chmod(dst, 0o755)
        return str(dst)
    except Exception as e:
        raise RuntimeError(f"cloudflared unavailable: {e!r}")

# ===== GGUF RESOLUTION (priority order matters) =====
def good(p):
    return p.is_file() and p.stat().st_size >= MODEL_MIN_BYTES

def resolve_gguf():
    local = LOCAL_MODELS / MODEL_FILE
    # Exact filename only: a stale GGUF of a different model must never be picked up.
    for p in (local, HERE / "models" / MODEL_FILE):          # 1. session-local / LFS clone
        if good(p):
            log(f"GGUF: session-local {p}")
            return p
        if p.is_file():
            log(f"GGUF {p} is too small ({p.stat().st_size} B): truncated, ignoring")
    d = MODELS_DRIVE / MODEL_FILE                              # 2. Drive cache (fast path)
    if good(d):
        log("GGUF: copying from Drive cache")
        atomic_copy(d, local)
        return local
    log("GGUF: downloading from HuggingFace")                 # 3. HF, then cache to Drive
    download(f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}", local, MODEL_MIN_BYTES)
    if not good(local):
        raise RuntimeError("downloaded GGUF failed size check")
    try:
        atomic_copy(local, d)
        log("GGUF cached to Drive")
    except Exception as e:
        log(f"could not cache GGUF to Drive: {e!r}")
    return local

# ===== TTS BACKENDS (pluggable) =====
import numpy as np

class TTSBackend:
    name = "base"
    sample_rate = 24000
    def load(self): raise NotImplementedError
    def synth(self, text):  # -> float32 mono ndarray
        raise NotImplementedError

def find_voice():
    for p in (HERE / "voice.safetensors", Path("/content/neo-tts/voice.safetensors"), STATE_DIR / "voice.safetensors"):
        if p.is_file() and p.stat().st_size > 1000:
            return p
    return None

class PocketBackend(TTSBackend):
    name = "pocket-tts"
    def load(self):
        import torch
        torch.set_num_threads(2)
        from pocket_tts import TTSModel
        try:
            self.model = TTSModel.load_model(quantize=True)   # int8
        except TypeError as e:
            log(f"load_model(quantize=True) unsupported ({e}); loading fp32")
            self.model = TTSModel.load_model()
        vp = find_voice()
        if vp is None:
            log("WARNING: voice.safetensors not found; using the built-in default voice")
            self.voice = self.model.get_state_for_audio_prompt("alba")
        else:
            log(f"voice: {vp}")
            self.voice = self.model.get_state_for_audio_prompt(str(vp))
        self.sample_rate = int(self.model.sample_rate)
    def synth(self, text):
        a = self.model.generate_audio(self.voice, text)
        return np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, dtype=np.float32).reshape(-1)

class KokoroBackend(TTSBackend):
    name = "kokoro-82m-int8"
    def load(self):
        from kokoro_onnx import Kokoro
        base = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
        files = {"kokoro-v1.0.int8.onnx": 60_000_000, "voices-v1.0.bin": 10_000_000}
        for fn, mn in files.items():
            p = MODELS_DRIVE / fn
            if not (p.is_file() and p.stat().st_size >= mn):
                log(f"downloading {fn}")
                download(base + fn, p, mn)
        self.k = Kokoro(str(MODELS_DRIVE / "kokoro-v1.0.int8.onnx"), str(MODELS_DRIVE / "voices-v1.0.bin"))
        self.sample_rate = 24000
    def synth(self, text):
        s, sr = self.k.create(text, voice=KOKORO_VOICE, speed=1.0, lang="en-us")
        self.sample_rate = int(sr)
        return np.asarray(s, dtype=np.float32).reshape(-1)

def make_backend():
    if TTS_BACKEND == "kokoro":
        try:
            b = KokoroBackend(); b.load(); return b
        except Exception as e:
            log(f"kokoro failed ({e!r}); falling back to pocket-tts")
    b = PocketBackend(); b.load(); return b

# ===== TEXT HELPERS =====
_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_EMOJI = re.compile("[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F]")

def clean_for_tts(t):
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"https?://\S+", "link", t)
    t = _EMOJI.sub("", t)
    t = re.sub(r"[*_#`>~|]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def split_sentences(t):
    out = []
    for s in _SPLIT.split(t):
        s = s.strip()
        while len(s) > MAX_SENT_CHARS:
            cut = max(s.rfind(", ", 0, MAX_SENT_CHARS), s.rfind(" ", 0, MAX_SENT_CHARS))
            if cut < 40:
                cut = MAX_SENT_CHARS
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if not s:
            continue
        # tiny fragments ("Ok.") cost a full synth call and sound choppy: merge back
        if out and len(s) < 14:
            out[-1] += " " + s
        else:
            out.append(s)
    return [s for s in out if re.search(r"[A-Za-z0-9]", s)]

def pcm_b64(x):
    x = np.clip(np.nan_to_num(x), -1.0, 1.0)
    return base64.b64encode((x * 32767).astype("<i2").tobytes()).decode()

# ===== PACE ESTIMATOR (the gap-free start gate) =====
class Pace:
    def __init__(self):
        self.spc = 0.065      # audio seconds per character
        self.wpc = 0.09       # wall seconds of synthesis per character (includes per-call overhead)
        self.safety = SAFETY0
        self.lock = threading.Lock()
    def seed(self, chars, audio_s, wall_s):
        with self.lock:
            self.spc = min(max(audio_s / chars, 0.03), 0.12)
            self.wpc = min(max(wall_s / chars, 0.005), 0.4)
    def gate(self, buffered, rem_chars, speech_done, chars_done, wall_done):
        """True if starting playback now cannot underrun, under the estimate."""
        with self.lock:
            spc, wpc, sf = self.spc, self.wpc, self.safety
        if chars_done > 0:
            lam = min(1.0, chars_done / 60.0) * 0.8   # trust this turn's own measurements more as they accrue
            spc = (1 - lam) * spc + lam * (speech_done / chars_done)
            wpc = (1 - lam) * wpc + lam * (wall_done / chars_done)
        t, avail = 0.0, buffered
        for c in rem_chars:
            t += sf * wpc * c            # wall time until this sentence's audio arrives
            if avail < t + LEAD_S:       # playback would have run dry before it arrives
                return False
            avail += spc * c + PAD_S
        return avail >= LEAD_S or not rem_chars
    def update(self, chars, speech, wall):
        if chars < 20 or speech <= 0:
            return
        with self.lock:
            self.spc = 0.6 * self.spc + 0.4 * (speech / chars)
            self.wpc = 0.6 * self.wpc + 0.4 * (wall / chars)
    def feedback(self, late):
        with self.lock:
            if late > 0:
                self.safety = min(SAFETY_MAX, self.safety + 0.2)
            else:
                self.safety = max(SAFETY_MIN, self.safety - 0.03)
            return self.safety

# ===== GLOBAL STATE =====
LLM = None
TTS = None
HISTORY = [{"role": "system", "content": SYSTEM_PROMPT}]
PACE = Pace()
TURN_LOCK = threading.Lock()   # LLM and TTS share 2 vCPUs: never run them concurrently
CANCEL = threading.Event()
SERVER = None
_FRESH = False

def reset_history():
    HISTORY[:] = [{"role": "system", "content": SYSTEM_PROMPT}]

def load_llm(path):
    from llama_cpp import Llama
    kw = dict(model_path=str(path), n_ctx=4096, n_threads=2, n_threads_batch=2,
              n_batch=256, n_ubatch=128, flash_attn=False,   # flash_attn is a GPU/Metal flag; never enable on CPU
              use_mmap=True, use_mlock=True, verbose=False)
    try:
        return Llama(**kw)
    except Exception as e:
        log(f"Llama() failed with mlock ({e!r}); retrying without use_mlock")
        kw["use_mlock"] = False
        return Llama(**kw)

def boot_models():
    global LLM, TTS
    # pip work is serialised in one thread (concurrent pip corrupts site-packages);
    # GGUF + cloudflared fetch run in parallel with it to hit the 60 s target.
    res = {}
    def t_pip():
        try:
            ensure_llama(); ensure_py_deps(); res["pip"] = True
        except Exception as e:
            res["pip"] = e
    def t_assets():
        try:
            res["gguf"] = resolve_gguf(); res["cf"] = ensure_cloudflared()
        except Exception as e:
            res["assets"] = e
    th = [threading.Thread(target=t_pip), threading.Thread(target=t_assets)]
    for t in th: t.start()
    for t in th: t.join()
    for k in ("pip", "assets"):
        if isinstance(res.get(k), Exception):
            raise res[k]
    log("loading LLM")
    LLM = load_llm(res["gguf"])
    log(f"loading TTS backend: {TTS_BACKEND}")
    TTS = make_backend()
    # prewarm LLM: 1-token completion, then reset history (first real TTFT drops 3-4 s)
    LLM.create_chat_completion(messages=[HISTORY[0], {"role": "user", "content": "hi"}], max_tokens=1)
    reset_history()
    # prewarm TTS and seed the pace estimator with a real measurement
    probe = "This is a short test of the voice engine."
    TTS.synth("Hello.")
    t = time.perf_counter(); a = TTS.synth(probe); wall = time.perf_counter() - t
    PACE.seed(len(probe), len(a) / TTS.sample_rate, wall)
    log(f"TTS prewarm: {wall:.2f}s wall for {len(a)/TTS.sample_rate:.2f}s audio (wall/audio = {wall/(len(a)/TTS.sample_rate):.2f}; >1 means slower than real time)")

# ===== THE TURN PIPELINE =====
def turn_events(user_text, mode, tts_on, max_tokens):
    t_req = time.perf_counter()
    with TURN_LOCK:
        CANCEL.clear()
        while len(HISTORY) - 1 >= MAX_TURNS * 2:   # rolling window; system prompt stays at index 0
            del HISTORY[1:3]
        HISTORY.append({"role": "user", "content": user_text})
        parts, committed, ntok = [], False, 0
        t_first = t_llm_end = None
        stream = None
        try:
            yield {"type": "meta", "sample_rate": TTS.sample_rate, "audio_mode": mode, "backend": TTS.name,
                   "pid": os.getpid()}
            # Pass the full growing history: llama.cpp reuses the KV prefix, past turns are never re-tokenised.
            stream = LLM.create_chat_completion(messages=HISTORY, stream=True, max_tokens=max_tokens,
                                                temperature=0.6, top_p=0.9, repeat_penalty=1.1)
            for ch in stream:
                if CANCEL.is_set():
                    break
                d = ch["choices"][0]["delta"].get("content")
                if d:
                    if t_first is None:
                        t_first = time.perf_counter()
                    parts.append(d); ntok += 1
                    yield {"type": "text", "delta": d}
            t_llm_end = time.perf_counter()
            text = "".join(parts).strip()
            if not text:
                # Empty generation: the finally block pops the user message. Keeping an
                # assistant "" pair makes Llama's template emit instant EOS forever ("0 tok/s" bug).
                yield {"type": "error", "message": "empty generation (rolled back, try again)"}
                return
            HISTORY.append({"role": "assistant", "content": text}); committed = True

            tts_wall = speech = 0.0
            go_s = None
            n_late_est = 0
            sr = TTS.sample_rate
            if tts_on and not CANCEL.is_set():
                sents = split_sentences(clean_for_tts(text))
                n = len(sents)
                chars = [len(s) for s in sents]
                pcms, buffered, chars_done, went = [], 0.0, 0, False
                for i, s in enumerate(sents):
                    if CANCEL.is_set():
                        break
                    yield {"type": "status", "stage": "synth", "i": i, "n": n}   # also keeps the tunnel warm
                    try:
                        t = time.perf_counter(); pcm = TTS.synth(s); dt = time.perf_counter() - t
                    except Exception as e:
                        print(f"[tts] sentence {i} failed: {e!r}", flush=True)
                        yield {"type": "error", "message": f"tts failed on sentence {i}: {e!r}"}
                        continue
                    sr = TTS.sample_rate
                    tts_wall += dt; speech += len(pcm) / sr; chars_done += len(s)
                    pad = PAD_S if i < n - 1 else 0.0
                    pcm = np.concatenate([pcm, np.zeros(int(pad * sr), np.float32)])
                    if mode == "smooth":
                        pcms.append(pcm)
                        continue
                    buffered += len(pcm) / sr
                    yield {"type": "sentence_audio", "i": i, "n": n, "sr": sr, "pcm": pcm_b64(pcm)}
                    if not went:
                        if mode == "stream" or i == n - 1:
                            ok = True
                        else:
                            ok = PACE.gate(buffered, chars[i + 1:], speech, chars_done, tts_wall)
                        if ok:
                            went = True
                            go_s = time.perf_counter() - t_req
                            yield {"type": "go", "after_sentence": i, "buffered_s": round(buffered, 2)}
                if mode == "smooth" and pcms:
                    full = np.concatenate(pcms)
                    go_s = time.perf_counter() - t_req
                    yield {"type": "audio_full", "sr": sr, "dur": len(full) / sr, "pcm": pcm_b64(full)}
                PACE.update(chars_done, speech, tts_wall)
            t_end = time.perf_counter()
            llm_s = (t_llm_end - t_req)
            yield {"type": "metrics",
                   "ttft_s": round((t_first - t_req), 2) if t_first else None,
                   "tok_s": round(ntok / max(t_llm_end - t_first, 1e-6), 1) if t_first else None,
                   "tokens": ntok, "llm_s": round(llm_s, 2),
                   "tts_s": round(tts_wall, 2), "audio_s": round(speech, 2),
                   "tts_wall_per_audio": round(tts_wall / speech, 2) if speech else None,
                   "audio_start_s": round(go_s, 2) if go_s else None,
                   "total_s": round(t_end - t_req, 2), "mode": mode, "backend": TTS.name,
                   "safety": round(PACE.safety, 2), "history_len": len(HISTORY)}
            yield {"type": "done"}
        finally:
            if stream is not None and hasattr(stream, "close"):
                try: stream.close()
                except Exception as e: print(f"[llm] stream close: {e!r}", flush=True)
            if not committed:
                # client disconnect / cancel / error: never leave a dangling user message
                partial = "".join(parts).strip()
                if partial:
                    HISTORY.append({"role": "assistant", "content": partial})
                elif HISTORY and HISTORY[-1]["role"] == "user":
                    HISTORY.pop()

# ===== WEB APP =====
_UI_B64 = "PCFkb2N0eXBlIGh0bWw+CjxodG1sIGxhbmc9ImVuIj48aGVhZD48bWV0YSBjaGFyc2V0PSJ1dGYtOCI+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsaW5pdGlhbC1zY2FsZT0xLHZpZXdwb3J0LWZpdD1jb3ZlciI+Cjx0aXRsZT5uZW8tdHRzPC90aXRsZT4KPHN0eWxlPgo6cm9vdHstLWJnOiMxMjE1MWE7LS1wYW5lbDojMWExZTI1Oy0tZmc6I2U5ZWJmMDstLW11dDojOGU5N2E4Oy0tbGluZTojMmEzMDNiOy0tYWNjOiM3ZmQ2YzI7LS1tZTojMjMyYTM2fQpAbWVkaWEgKHByZWZlcnMtY29sb3Itc2NoZW1lOmxpZ2h0KXs6cm9vdHstLWJnOiNmNGY1Zjc7LS1wYW5lbDojZmZmOy0tZmc6IzE2MWEyMTstLW11dDojNWY2ODc5Oy0tbGluZTojZGRlMWU5Oy0tYWNjOiMwZjdmNmE7LS1tZTojZThlY2YzfX0KKntib3gtc2l6aW5nOmJvcmRlci1ib3h9Cmh0bWwsYm9keXtoZWlnaHQ6MTAwJTttYXJnaW46MH0KYm9keXtiYWNrZ3JvdW5kOnZhcigtLWJnKTtjb2xvcjp2YXIoLS1mZyk7Zm9udDoxNnB4LzEuNSBzeXN0ZW0tdWksLWFwcGxlLXN5c3RlbSwiU2Vnb2UgVUkiLHNhbnMtc2VyaWY7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjttYXgtd2lkdGg6NzYwcHg7bWFyZ2luOjAgYXV0bztwYWRkaW5nOmVudihzYWZlLWFyZWEtaW5zZXQtdG9wKSAxMnB4IGVudihzYWZlLWFyZWEtaW5zZXQtYm90dG9tKX0KaGVhZGVye2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEwcHg7cGFkZGluZzoxNHB4IDRweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1saW5lKX0KaDF7Zm9udC1zaXplOjE3cHg7bWFyZ2luOjA7Zm9udC13ZWlnaHQ6NjAwO2xldHRlci1zcGFjaW5nOi4ycHh9CiNjaGlwe2NvbG9yOnZhcigtLW11dCk7Zm9udC1zaXplOjEzcHg7ZmxleDoxfQpzZWxlY3QsYnV0dG9uLGlucHV0e2ZvbnQ6aW5oZXJpdDtjb2xvcjp2YXIoLS1mZyl9CnNlbGVjdHtiYWNrZ3JvdW5kOnZhcigtLXBhbmVsKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpO2JvcmRlci1yYWRpdXM6OHB4O3BhZGRpbmc6NnB4IDhweH0KI2xvZ3tmbGV4OjE7b3ZlcmZsb3cteTphdXRvO3BhZGRpbmc6MTZweCAycHg7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjtnYXA6MTJweH0KLm17bWF4LXdpZHRoOjg4JTtwYWRkaW5nOjEwcHggMTRweDtib3JkZXItcmFkaXVzOjE0cHg7d2hpdGUtc3BhY2U6cHJlLXdyYXA7d29yZC13cmFwOmJyZWFrLXdvcmR9Ci51e2FsaWduLXNlbGY6ZmxleC1lbmQ7YmFja2dyb3VuZDp2YXIoLS1tZSl9Ci5he2FsaWduLXNlbGY6ZmxleC1zdGFydDtiYWNrZ3JvdW5kOnZhcigtLXBhbmVsKTtib3JkZXI6MXB4IHNvbGlkIHZhcigtLWxpbmUpfQoubWV0YXtmb250LXNpemU6MTJweDtjb2xvcjp2YXIoLS1tdXQpO21hcmdpbi10b3A6NnB4fQojc3RhdHVze21pbi1oZWlnaHQ6MjBweDtmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1hY2MpO3BhZGRpbmc6MCA0cHh9CmZvcm17ZGlzcGxheTpmbGV4O2dhcDo4cHg7cGFkZGluZzoxMHB4IDAgMTRweH0KaW5wdXR7ZmxleDoxO2JhY2tncm91bmQ6dmFyKC0tcGFuZWwpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czoxMnB4O3BhZGRpbmc6MTJweCAxNHB4fQppbnB1dDpmb2N1cy12aXNpYmxlLGJ1dHRvbjpmb2N1cy12aXNpYmxlLHNlbGVjdDpmb2N1cy12aXNpYmxle291dGxpbmU6MnB4IHNvbGlkIHZhcigtLWFjYyk7b3V0bGluZS1vZmZzZXQ6MnB4fQpidXR0b257YmFja2dyb3VuZDp2YXIoLS1hY2MpO2NvbG9yOiMwYjEyMTA7Ym9yZGVyOjA7Ym9yZGVyLXJhZGl1czoxMnB4O3BhZGRpbmc6MCAxOHB4O2ZvbnQtd2VpZ2h0OjYwMDtjdXJzb3I6cG9pbnRlcn0KYnV0dG9uLmdob3N0e2JhY2tncm91bmQ6dHJhbnNwYXJlbnQ7Y29sb3I6dmFyKC0tZmcpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSl9CmJ1dHRvbjpkaXNhYmxlZHtvcGFjaXR5Oi41O2N1cnNvcjpkZWZhdWx0fQo8L3N0eWxlPjwvaGVhZD4KPGJvZHk+CjxoZWFkZXI+PGgxPm5lby10dHM8L2gxPjxzcGFuIGlkPSJjaGlwIj5jb25uZWN0aW5nPC9zcGFuPgo8c2VsZWN0IGlkPSJtb2RlIiB0aXRsZT0iQXVkaW8gbW9kZSI+PG9wdGlvbiB2YWx1ZT0iYWRhcHRpdmUiPkFkYXB0aXZlPC9vcHRpb24+PG9wdGlvbiB2YWx1ZT0ic21vb3RoIj5TbW9vdGg8L29wdGlvbj48b3B0aW9uIHZhbHVlPSJzdHJlYW0iPlN0cmVhbTwvb3B0aW9uPjwvc2VsZWN0Pgo8YnV0dG9uIGNsYXNzPSJnaG9zdCIgaWQ9InJlc2V0IiB0eXBlPSJidXR0b24iPk5ldyBjaGF0PC9idXR0b24+PC9oZWFkZXI+CjxkaXYgaWQ9ImxvZyIgYXJpYS1saXZlPSJwb2xpdGUiPjwvZGl2Pgo8ZGl2IGlkPSJzdGF0dXMiPjwvZGl2Pgo8Zm9ybSBpZD0iZiI+PGlucHV0IGlkPSJxIiBhdXRvY29tcGxldGU9Im9mZiIgcGxhY2Vob2xkZXI9IlR5cGUgYSBtZXNzYWdlIiBhcmlhLWxhYmVsPSJNZXNzYWdlIj48YnV0dG9uIGlkPSJzZW5kIj5TZW5kPC9idXR0b24+PGJ1dHRvbiBjbGFzcz0iZ2hvc3QiIGlkPSJzdG9wIiB0eXBlPSJidXR0b24iIGRpc2FibGVkPlN0b3A8L2J1dHRvbj48L2Zvcm0+CjxzY3JpcHQ+CiJ1c2Ugc3RyaWN0IjsKY29uc3QgJCA9IHMgPT4gZG9jdW1lbnQucXVlcnlTZWxlY3RvcihzKTsKY29uc3QgQSA9IHtjdHg6IG51bGwsIHNyOiAyNDAwMCwgaGVhZDogMCwgc3RhcnRlZDogZmFsc2UsIHF1ZXVlOiBbXSwgbGF0ZTogMCwgc291cmNlczogW119OwpsZXQgY3RybCA9IG51bGw7CgpmdW5jdGlvbiBlbnN1cmVDdHgoc3IpIHsKICBpZiAoQS5jdHggJiYgQS5jdHguc2FtcGxlUmF0ZSA9PT0gc3IpIHJldHVybjsKICBpZiAoQS5jdHgpIEEuY3R4LmNsb3NlKCk7CiAgQS5zciA9IHNyOwogIC8vIEZpeGVkIDI0IGtIeiBjb250ZXh0OiB0aGUgZGVmYXVsdCA0OCBrSHogY29udGV4dCByZXNhbXBsZXMgb3VyIGJ1ZmZlcnMgYW5kIGNyYWNrbGVzLgogIEEuY3R4ID0gbmV3ICh3aW5kb3cuQXVkaW9Db250ZXh0IHx8IHdpbmRvdy53ZWJraXRBdWRpb0NvbnRleHQpKHtzYW1wbGVSYXRlOiBzciwgbGF0ZW5jeUhpbnQ6ICJwbGF5YmFjayJ9KTsKfQpmdW5jdGlvbiByZXNldEF1ZGlvKCkgewogIGZvciAoY29uc3QgcyBvZiBBLnNvdXJjZXMpIHsgdHJ5IHsgcy5zdG9wKCk7IH0gY2F0Y2ggKGUpIHsgY29uc29sZS53YXJuKCJzdG9wIiwgZSk7IH0gfQogIEEuc291cmNlcyA9IFtdOyBBLmhlYWQgPSAwOyBBLnN0YXJ0ZWQgPSBmYWxzZTsgQS5xdWV1ZSA9IFtdOyBBLmxhdGUgPSAwOwp9CmZ1bmN0aW9uIHBjbVRvRjMyKGI2NCkgewogIGNvbnN0IGJpbiA9IGF0b2IoYjY0KSwgbiA9IGJpbi5sZW5ndGggPj4gMSwgb3V0ID0gbmV3IEZsb2F0MzJBcnJheShuKTsKICBmb3IgKGxldCBpID0gMDsgaSA8IG47IGkrKykgewogICAgbGV0IHYgPSBiaW4uY2hhckNvZGVBdCgyICogaSkgfCAoYmluLmNoYXJDb2RlQXQoMiAqIGkgKyAxKSA8PCA4KTsKICAgIGlmICh2ICYgMHg4MDAwKSB2IC09IDB4MTAwMDA7CiAgICBvdXRbaV0gPSB2IC8gMzI3Njg7CiAgfQogIHJldHVybiBvdXQ7Cn0KZnVuY3Rpb24gZmFkZShmLCBpbk1zLCBvdXRNcykgewogIGNvbnN0IG5pID0gTWF0aC5taW4oZi5sZW5ndGggPj4gMSwgTWF0aC5yb3VuZChBLnNyICogaW5NcyAvIDEwMDApKSwgbm8gPSBNYXRoLm1pbihmLmxlbmd0aCA+PiAxLCBNYXRoLnJvdW5kKEEuc3IgKiBvdXRNcyAvIDEwMDApKTsKICBmb3IgKGxldCBpID0gMDsgaSA8IG5pOyBpKyspIGZbaV0gKj0gaSAvIG5pOwogIGZvciAobGV0IGkgPSAwOyBpIDwgbm87IGkrKykgZltmLmxlbmd0aCAtIDEgLSBpXSAqPSBpIC8gbm87CiAgcmV0dXJuIGY7Cn0KZnVuY3Rpb24gc2NoZWR1bGUoZikgewogIGNvbnN0IGJ1ZiA9IEEuY3R4LmNyZWF0ZUJ1ZmZlcigxLCBmLmxlbmd0aCwgQS5zcik7CiAgYnVmLmNvcHlUb0NoYW5uZWwoZiwgMCk7CiAgY29uc3Qgc3JjID0gQS5jdHguY3JlYXRlQnVmZmVyU291cmNlKCk7CiAgc3JjLmJ1ZmZlciA9IGJ1Zjsgc3JjLmNvbm5lY3QoQS5jdHguZGVzdGluYXRpb24pOwogIGNvbnN0IG5vdyA9IEEuY3R4LmN1cnJlbnRUaW1lOwogIGlmIChBLmhlYWQgPCBub3cgKyAwLjAxKSB7ICAgICAgICAgICAgLy8gY2h1bmsgYXJyaXZlZCBhZnRlciB0aGUgcXVldWUgcmFuIGRyeTogYW4gYXVkaWJsZSBnYXAKICAgIGlmIChBLnN0YXJ0ZWQgJiYgQS5oZWFkID4gMCkgQS5sYXRlKys7CiAgICBBLmhlYWQgPSBub3cgKyAwLjA2OwogIH0KICBzcmMuc3RhcnQoQS5oZWFkKTsgICAgICAgICAgICAgICAgICAgIC8vIHNhbXBsZS1hY2N1cmF0ZTogaGVhZCBhZHZhbmNlcyBieSBleGFjdCBidWZmZXIgZHVyYXRpb24KICBBLmhlYWQgKz0gYnVmLmR1cmF0aW9uOwogIEEuc291cmNlcy5wdXNoKHNyYyk7Cn0KZnVuY3Rpb24gZ28oKSB7CiAgaWYgKEEuc3RhcnRlZCkgcmV0dXJuOwogIEEuc3RhcnRlZCA9IHRydWU7IEEuaGVhZCA9IEEuY3R4LmN1cnJlbnRUaW1lICsgMC4wODsKICBmb3IgKGNvbnN0IGYgb2YgQS5xdWV1ZSkgc2NoZWR1bGUoZik7CiAgQS5xdWV1ZSA9IFtdOwp9CgpmdW5jdGlvbiBidWJibGUoY2xzLCB0ZXh0KSB7CiAgY29uc3QgZCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoImRpdiIpOyBkLmNsYXNzTmFtZSA9ICJtICIgKyBjbHM7IGQudGV4dENvbnRlbnQgPSB0ZXh0OwogICQoIiNsb2ciKS5hcHBlbmRDaGlsZChkKTsgJCgiI2xvZyIpLnNjcm9sbFRvcCA9IDFlOTsgcmV0dXJuIGQ7Cn0KCmFzeW5jIGZ1bmN0aW9uIHNlbmQodGV4dCkgewogIGVuc3VyZUN0eChBLnNyKTsgYXdhaXQgQS5jdHgucmVzdW1lKCk7IHJlc2V0QXVkaW8oKTsKICBidWJibGUoInUiLCB0ZXh0KTsKICBjb25zdCBvdXQgPSBidWJibGUoImEiLCAiIik7IGNvbnN0IG1ldGEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCJkaXYiKTsgbWV0YS5jbGFzc05hbWUgPSAibWV0YSI7IG91dC5hcHBlbmRDaGlsZChtZXRhKTsKICBjb25zdCBib2R5ID0gZG9jdW1lbnQuY3JlYXRlVGV4dE5vZGUoIiIpOyBvdXQuaW5zZXJ0QmVmb3JlKGJvZHksIG1ldGEpOwogIGN0cmwgPSBuZXcgQWJvcnRDb250cm9sbGVyKCk7CiAgJCgiI3NlbmQiKS5kaXNhYmxlZCA9IHRydWU7ICQoIiNzdG9wIikuZGlzYWJsZWQgPSBmYWxzZTsKICB0cnkgewogICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2goIi9jaGF0Iiwge21ldGhvZDogIlBPU1QiLCBoZWFkZXJzOiB7IkNvbnRlbnQtVHlwZSI6ICJhcHBsaWNhdGlvbi9qc29uIn0sCiAgICAgIGJvZHk6IEpTT04uc3RyaW5naWZ5KHttZXNzYWdlOiB0ZXh0LCBtb2RlOiAkKCIjbW9kZSIpLnZhbHVlfSksIHNpZ25hbDogY3RybC5zaWduYWx9KTsKICAgIGNvbnN0IHJkID0gcmVzLmJvZHkuZ2V0UmVhZGVyKCksIGRlYyA9IG5ldyBUZXh0RGVjb2RlcigpOyBsZXQgYnVmID0gIiI7CiAgICBmb3IgKDs7KSB7CiAgICAgIGNvbnN0IHt2YWx1ZSwgZG9uZX0gPSBhd2FpdCByZC5yZWFkKCk7IGlmIChkb25lKSBicmVhazsKICAgICAgYnVmICs9IGRlYy5kZWNvZGUodmFsdWUsIHtzdHJlYW06IHRydWV9KTsKICAgICAgbGV0IGk7CiAgICAgIHdoaWxlICgoaSA9IGJ1Zi5pbmRleE9mKCJcbiIpKSA+PSAwKSB7CiAgICAgICAgY29uc3QgbGluZSA9IGJ1Zi5zbGljZSgwLCBpKTsgYnVmID0gYnVmLnNsaWNlKGkgKyAxKTsKICAgICAgICBpZiAobGluZS50cmltKCkpIGhhbmRsZShKU09OLnBhcnNlKGxpbmUpLCBib2R5LCBtZXRhKTsKICAgICAgfQogICAgfQogIH0gY2F0Y2ggKGUpIHsKICAgIGlmIChlLm5hbWUgIT09ICJBYm9ydEVycm9yIikgeyBtZXRhLnRleHRDb250ZW50ID0gIkNvbm5lY3Rpb24gcHJvYmxlbTogIiArIGUubWVzc2FnZTsgY29uc29sZS5lcnJvcihlKTsgfQogIH0KICBpZiAoIUEuc3RhcnRlZCAmJiBBLnF1ZXVlLmxlbmd0aCkgZ28oKTsgICAvLyBmYWxsYmFjazogbmV2ZXIgbGVhdmUgYXVkaW8gdW5wbGF5ZWQKICBmZXRjaCgiL2ZlZWRiYWNrIiwge21ldGhvZDogIlBPU1QiLCBoZWFkZXJzOiB7IkNvbnRlbnQtVHlwZSI6ICJhcHBsaWNhdGlvbi9qc29uIn0sIGJvZHk6IEpTT04uc3RyaW5naWZ5KHtsYXRlOiBBLmxhdGV9KX0pCiAgICAuY2F0Y2goZSA9PiBjb25zb2xlLndhcm4oImZlZWRiYWNrIiwgZSkpOwogICQoIiNzdGF0dXMiKS50ZXh0Q29udGVudCA9ICIiOyAkKCIjc2VuZCIpLmRpc2FibGVkID0gZmFsc2U7ICQoIiNzdG9wIikuZGlzYWJsZWQgPSB0cnVlOyAkKCIjcSIpLmZvY3VzKCk7Cn0KCmZ1bmN0aW9uIGhhbmRsZShldiwgYm9keSwgbWV0YSkgewogIHN3aXRjaCAoZXYudHlwZSkgewogICAgY2FzZSAibWV0YSI6CiAgICAgIGlmIChldi5zYW1wbGVfcmF0ZSAhPT0gQS5zcikgeyBlbnN1cmVDdHgoZXYuc2FtcGxlX3JhdGUpOyBBLmN0eC5yZXN1bWUoKTsgfQogICAgICAkKCIjY2hpcCIpLnRleHRDb250ZW50ID0gZXYuYmFja2VuZDsgYnJlYWs7CiAgICBjYXNlICJ0ZXh0IjogYm9keS5ub2RlVmFsdWUgKz0gZXYuZGVsdGE7ICQoIiNsb2ciKS5zY3JvbGxUb3AgPSAxZTk7IGJyZWFrOwogICAgY2FzZSAic3RhdHVzIjogJCgiI3N0YXR1cyIpLnRleHRDb250ZW50ID0gIlByZXBhcmluZyB2b2ljZSwgc2VudGVuY2UgIiArIChldi5pICsgMSkgKyAiIG9mICIgKyBldi5uOyBicmVhazsKICAgIGNhc2UgInNlbnRlbmNlX2F1ZGlvIjogewogICAgICBjb25zdCBmID0gZmFkZShwY21Ub0YzMihldi5wY20pLCAzLCAzKTsKICAgICAgaWYgKEEuc3RhcnRlZCkgc2NoZWR1bGUoZik7IGVsc2UgQS5xdWV1ZS5wdXNoKGYpOwogICAgICBicmVhazsgfQogICAgY2FzZSAiZ28iOiBnbygpOyAkKCIjc3RhdHVzIikudGV4dENvbnRlbnQgPSAiU3BlYWtpbmciOyBicmVhazsKICAgIGNhc2UgImF1ZGlvX2Z1bGwiOiB7CiAgICAgIGNvbnN0IGYgPSBmYWRlKHBjbVRvRjMyKGV2LnBjbSksIDUsIDE1KTsKICAgICAgQS5xdWV1ZS5wdXNoKGYpOyBnbygpOyAkKCIjc3RhdHVzIikudGV4dENvbnRlbnQgPSAiU3BlYWtpbmciOyBicmVhazsgfQogICAgY2FzZSAiZXJyb3IiOiBtZXRhLnRleHRDb250ZW50ID0gZXYubWVzc2FnZTsgYnJlYWs7CiAgICBjYXNlICJtZXRyaWNzIjogewogICAgICBjb25zdCBtID0gZXY7CiAgICAgIG1ldGEudGV4dENvbnRlbnQgPSAobS50b2tfcyA/PyAiPyIpICsgIiB0b2svcywgZmlyc3QgdGV4dCAiICsgKG0udHRmdF9zID8/ICI/IikgKyAiIHMsIHZvaWNlIHN0YXJ0cyAiICsgKG0uYXVkaW9fc3RhcnRfcyA/PyAibi9hIikgKyAiIHMsIHZvaWNlIGJ1aWx0IGF0ICIgKyAobS50dHNfd2FsbF9wZXJfYXVkaW8gPz8gIm4vYSIpICsgInggYXVkaW8gbGVuZ3RoLCAiICsgbS5tb2RlOwogICAgICBicmVhazsgfQogICAgY2FzZSAiZG9uZSI6IGJyZWFrOwogIH0KfQoKJCgiI2YiKS5hZGRFdmVudExpc3RlbmVyKCJzdWJtaXQiLCBlID0+IHsgZS5wcmV2ZW50RGVmYXVsdCgpOyBjb25zdCB0ID0gJCgiI3EiKS52YWx1ZS50cmltKCk7IGlmICghdCkgcmV0dXJuOyAkKCIjcSIpLnZhbHVlID0gIiI7IHNlbmQodCk7IH0pOwokKCIjc3RvcCIpLmFkZEV2ZW50TGlzdGVuZXIoImNsaWNrIiwgKCkgPT4geyBmZXRjaCgiL2NhbmNlbCIsIHttZXRob2Q6ICJQT1NUIn0pOyBpZiAoY3RybCkgY3RybC5hYm9ydCgpOyByZXNldEF1ZGlvKCk7IH0pOwokKCIjcmVzZXQiKS5hZGRFdmVudExpc3RlbmVyKCJjbGljayIsIGFzeW5jICgpID0+IHsgYXdhaXQgZmV0Y2goIi9yZXNldCIsIHttZXRob2Q6ICJQT1NUIn0pOyAkKCIjbG9nIikuaW5uZXJIVE1MID0gIiI7IHJlc2V0QXVkaW8oKTsgfSk7CmZldGNoKCIvY29uZmlnIikudGhlbihyID0+IHIuanNvbigpKS50aGVuKGMgPT4geyBBLnNyID0gYy5zYW1wbGVfcmF0ZTsgJCgiI21vZGUiKS52YWx1ZSA9IGMuYXVkaW9fbW9kZTsgJCgiI2NoaXAiKS50ZXh0Q29udGVudCA9IGMuYmFja2VuZCArICIgLyAiICsgYy5tb2RlbDsgfSkKICAuY2F0Y2goZSA9PiB7ICQoIiNjaGlwIikudGV4dENvbnRlbnQgPSAic2VydmVyIG5vdCByZWFjaGFibGUiOyBjb25zb2xlLmVycm9yKGUpOyB9KTsKPC9zY3JpcHQ+PC9ib2R5PjwvaHRtbD4K"
def load_ui():
    try:
        return (HERE / "_ui.html").read_text(encoding="utf-8")
    except FileNotFoundError:   # fresh clone missing the file / not pushed
        return base64.b64decode(_UI_B64).decode("utf-8")

LOG_CONFIG = {"version": 1, "disable_existing_loggers": False,
              "handlers": {"null": {"class": "logging.NullHandler"}},
              "loggers": {n: {"handlers": ["null"], "level": "CRITICAL", "propagate": False}
                          for n in ("uvicorn", "uvicorn.error", "uvicorn.access")}}

def build_app():
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    def index():
        return load_ui()

    @app.get("/health")
    def health():
        return {"ok": LLM is not None and TTS is not None, "backend": TTS.name if TTS else None}

    @app.get("/version")
    def version():
        return {"pid": os.getpid(), "history_len": len(HISTORY), "history_roles": [m["role"] for m in HISTORY]}

    @app.get("/config")
    def config():
        return {"audio_mode": AUDIO_MODE, "sample_rate": TTS.sample_rate, "backend": TTS.name,
                "model": MODEL_FILE, "safety": PACE.safety}

    @app.post("/chat")
    async def chat(req: Request):
        b = await req.json()
        msg = (b.get("message") or "").strip()
        if not msg:
            return JSONResponse({"error": "empty message"}, status_code=400)
        mode = b.get("mode") if b.get("mode") in ("adaptive", "smooth", "stream") else AUDIO_MODE
        mt = int(b.get("max_tokens") or MAX_TOKENS)
        def gen():
            for ev in turn_events(msg, mode, bool(b.get("tts", True)), mt):
                yield json.dumps(ev, separators=(",", ":")) + "\n"
        return StreamingResponse(gen(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/cancel")
    def cancel():
        CANCEL.set(); return {"ok": True}

    @app.post("/reset")
    def reset():
        reset_history(); return {"ok": True, "history_len": len(HISTORY)}

    @app.post("/feedback")
    async def feedback(req: Request):
        b = await req.json()
        return {"safety": PACE.feedback(int(b.get("late", 0)))}

    return app

# ===== TUNNEL / LIFECYCLE =====
def http_json(path, timeout=3):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())

def alive():
    try:
        return bool(http_json("/health").get("ok"))
    except Exception:
        return False

def start_tunnel(cf):
    pat = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    lp = Path("/tmp/cloudflared.log")
    for attempt in (1, 2):
        subprocess.run(["pkill", "-f", "cloudflared"], check=False)   # orphans from prior sessions
        time.sleep(0.5)
        lf = open(lp, "w")
        proc = subprocess.Popen([cf, "tunnel", "--url", f"http://127.0.0.1:{PORT}", "--no-autoupdate",
                                 "--protocol", "http2"], stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True)   # detaches: survives the cell returning
        end = time.time() + 60
        while time.time() < end:
            m = pat.search(lp.read_text(errors="ignore"))
            if m:
                return m.group(0)
            if proc.poll() is not None:
                break
            time.sleep(0.5)
        log(f"tunnel attempt {attempt} failed: {lp.read_text(errors='ignore')[-300:]}")
    return None

def start_server():
    """Idempotent. Returns after printing the URL; server (daemon thread) and tunnel (detached) keep running."""
    global SERVER, _FRESH
    if alive():
        if URL_FILE.exists():
            url = URL_FILE.read_text().strip()
            log(f"server already running: {url}")
            return url
        url = start_tunnel(ensure_cloudflared())
        if url: URL_FILE.write_text(url)
        return url
    _FRESH = True
    boot_models()
    import uvicorn
    cfg = uvicorn.Config(build_app(), host="127.0.0.1", port=PORT, log_config=LOG_CONFIG,
                         access_log=False, log_level="critical")
    SERVER = uvicorn.Server(cfg)
    threading.Thread(target=SERVER.run, daemon=True, name="neo-uvicorn").start()
    for _ in range(100):
        if alive(): break
        time.sleep(0.2)
    else:
        raise RuntimeError("server did not become healthy")
    log("server healthy; opening tunnel")
    url = start_tunnel(ensure_cloudflared())
    if not url:
        raise RuntimeError("cloudflared did not produce a URL (see /tmp/cloudflared.log)")
    URL_FILE.write_text(url)
    return url

def stop_server():
    global SERVER
    if SERVER is not None:
        SERVER.should_exit = True
    subprocess.run(["pkill", "-f", "cloudflared"], check=False)
    try: URL_FILE.unlink()
    except FileNotFoundError: pass
    log("server stopping; port frees in a moment")

# ===== SELF TEST =====
def post_chat(msg, **kw):
    body = json.dumps(dict(message=msg, **kw)).encode()
    rq = urllib.request.Request(f"http://127.0.0.1:{PORT}/chat", data=body, headers={"Content-Type": "application/json"})
    evs = []
    with urllib.request.urlopen(rq, timeout=240) as r:
        for line in r:
            if line.strip():
                evs.append(json.loads(line))
    return evs

def selftest():
    try:
        for q in ("Say hello in five words.", "Name one color.", "What is two plus two?"):
            post_chat(q, tts=False, max_tokens=16, mode="smooth")
        v = http_json("/version")
        print(f"history_len after 3 turns: {v['history_len']} (expected 7) -> {'PASS' if v['history_len'] == 7 else 'FAIL'}", flush=True)
        print("/health:", http_json("/health"), flush=True)
        ev = post_chat("Say good morning.", tts=True, max_tokens=24, mode="smooth")
        ty = [e["type"] for e in ev]
        print(f"smooth round trip: text={ty.count('text')} audio_full={ty.count('audio_full')} -> "
              f"{'PASS' if 'text' in ty and 'audio_full' in ty else 'FAIL'}", flush=True)
        ev = post_chat("Count to three.", tts=True, max_tokens=24, mode="adaptive")
        ty = [e["type"] for e in ev]
        m = [e for e in ev if e["type"] == "metrics"]
        print(f"adaptive round trip: sentence_audio={ty.count('sentence_audio')} go={ty.count('go')} -> "
              f"{'PASS' if 'go' in ty and 'sentence_audio' in ty else 'FAIL'}", flush=True)
        if m: print("sample metrics:", m[0], flush=True)
    except Exception as e:
        print(f"SELFTEST ERROR: {e!r}", flush=True)
    finally:
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{PORT}/reset", data=b"{}", method="POST"), timeout=5)
        except Exception as e:
            print(f"reset after selftest failed: {e!r}", flush=True)

if __name__ == "__main__":
    try:
        url = start_server()
        print("\n" + "=" * 64 + f"\n  neo-tts is live: {url}\n  audio mode: {AUDIO_MODE}   tts backend: {TTS_BACKEND}"
              "\n  stop with stop_server()\n" + "=" * 64, flush=True)
        if _FRESH and os.environ.get("NEO_SELFTEST", "1") == "1":
            selftest()
        print("SCRIPT COMPLETE", flush=True)
    finally:
        sys.stdout.flush()
