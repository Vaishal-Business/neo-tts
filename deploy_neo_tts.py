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
import os, sys, re, io, json, time, glob, base64, shutil, threading, subprocess, queue, collections, concurrent.futures
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
TTS_BACKEND = _env("NEO_TTS_BACKEND", "pocket").lower()   # pocket (your voice, auto-tuned) | auto | kokoro
KOKORO_VOICE = _env("NEO_KOKORO_VOICE", "af_heart")
MAX_TURNS = 10            # rolling window of user/assistant pairs
MAX_TOKENS = int(_env("NEO_MAX_TOKENS", "90"))
PAD_S = 0.10              # silence between sentences
LEAD_S = float(_env("NEO_LEAD_S", "0.8"))   # extra cushion for tunnel jitter
SAFETY0, SAFETY_MIN, SAFETY_MAX = 1.3, 1.15, 2.5
MAX_SENT_CHARS = 220

SYSTEM_PROMPT = (
    "You are Neo, a voice assistant. Your replies are spoken aloud. "
    "Reply in one or two short sentences, under thirty words in total, unless the person explicitly asks for more. "
    "Use plain spoken English only. Never use markdown, lists, headings, asterisks, emojis, code, or URLs. "
    "Write numbers and symbols as words."
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
    if TTS_BACKEND in ("kokoro", "auto") and importlib.util.find_spec("kokoro_onnx") is None:
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
    parallel = 1            # how many sentences can be synthesised at once
    @property
    def executor(self):
        if not hasattr(self, "_ex"):
            self._ex = concurrent.futures.ThreadPoolExecutor(max_workers=self.parallel)
        return self._ex
    def load(self): raise NotImplementedError
    def synth(self, text):  # -> float32 mono ndarray
        raise NotImplementedError
    def close(self): pass

def find_voice():
    for p in (HERE / "voice.safetensors", Path("/content/neo-tts/voice.safetensors"), STATE_DIR / "voice.safetensors"):
        if p.is_file() and p.stat().st_size > 1000:
            return p
    return None

def _to_np(a):
    return np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, dtype=np.float32).reshape(-1)

class PocketBackend(TTSBackend):
    """One in-process model. quant=int8 or fp32, threads=torch intra-op threads."""
    def __init__(self, quant=True, threads=2):
        self.quant, self.threads = quant, threads
        self.name = f"pocket-tts ({'int8' if quant else 'fp32'}, {threads} threads)"
    def load(self):
        import torch
        torch.set_num_threads(self.threads)
        from pocket_tts import TTSModel
        if self.quant:
            try:
                self.model = TTSModel.load_model(quantize=True)
            except TypeError as e:
                log(f"load_model(quantize=True) unsupported ({e}); loading fp32")
                self.model = TTSModel.load_model()
        else:
            self.model = TTSModel.load_model()
        vp = find_voice()
        if vp is None:
            log("WARNING: voice.safetensors not found; using the built-in default voice")
            self.voice = self.model.get_state_for_audio_prompt("alba")
        else:
            self.voice = self.model.get_state_for_audio_prompt(str(vp))
        self.sample_rate = int(self.model.sample_rate)
    def synth(self, text):
        import torch
        torch.set_num_threads(self.threads)
        return _to_np(self.model.generate_audio(self.voice, text))
    def close(self):
        self.model = None

# Worker for PocketPoolBackend. Separate PROCESSES, not threads: one autoregressive stream scales
# poorly past 1 thread (tiny matrices, sync overhead), so two single-thread processes working on
# two sentences give close to 2x aggregate throughput on 2 vCPUs. A subprocess (not multiprocessing)
# is used because spawn cannot import functions defined in a notebook's __main__.
_POCKET_WORKER_SRC = r"""
import sys, json
proto = sys.stdout.buffer
sys.stdout = sys.stderr          # library prints must never corrupt the binary protocol
import numpy as np, torch
torch.set_num_threads(1)
voice, quant = sys.argv[1], sys.argv[2] == "1"
from pocket_tts import TTSModel
try:
    model = TTSModel.load_model(quantize=True) if quant else TTSModel.load_model()
except TypeError:
    model = TTSModel.load_model()
state = model.get_state_for_audio_prompt(voice)
proto.write((json.dumps({"ready": True, "sr": int(model.sample_rate)}) + "\n").encode()); proto.flush()
for line in sys.stdin:
    try:
        a = model.generate_audio(state, json.loads(line)["text"])
        a = np.asarray(a.detach().cpu().numpy() if hasattr(a, "detach") else a, dtype=np.float32).reshape(-1)
        proto.write((json.dumps({"n": int(a.size)}) + "\n").encode()); proto.write(a.tobytes())
    except Exception as e:
        proto.write((json.dumps({"error": repr(e)}) + "\n").encode())
    proto.flush()
"""

class PocketPoolBackend(TTSBackend):
    def __init__(self, n=2, quant=True):
        self.n, self.quant, self.parallel = n, quant, n
        self.name = f"pocket-tts ({'int8' if quant else 'fp32'}, {n} procs x 1 thread)"
    def load(self):
        wp = Path("/tmp/neo_pocket_worker.py")
        wp.write_text(_POCKET_WORKER_SRC)
        vp = find_voice()
        if vp is None:
            log("WARNING: voice.safetensors not found; using the built-in default voice")
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        self.procs, self.free = [], queue.Queue()
        for k in range(self.n):   # start all first so the models load in parallel
            err = open(f"/tmp/neo_pocket_worker_{k}.log", "w")
            p = subprocess.Popen([sys.executable, str(wp), str(vp) if vp else "alba", "1" if self.quant else "0"],
                                 stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env)
            self.procs.append(p)
        for k, p in enumerate(self.procs):
            line = p.stdout.readline()
            if not line:
                tail = Path(f"/tmp/neo_pocket_worker_{k}.log").read_text(errors="ignore")[-600:]
                raise RuntimeError(f"pocket worker {k} died on startup: {tail}")
            self.sample_rate = int(json.loads(line)["sr"])
            self.free.put(p)
    def synth(self, text):
        p = self.free.get()
        try:
            p.stdin.write((json.dumps({"text": text}) + "\n").encode()); p.stdin.flush()
            hdr = p.stdout.readline()
            if not hdr:
                raise RuntimeError("pocket worker died mid-synthesis")
            h = json.loads(hdr)
            if "error" in h:
                raise RuntimeError(h["error"])
            need = h["n"] * 4
            buf = p.stdout.read(need)
            if len(buf) != need:
                raise RuntimeError("short read from pocket worker")
            return np.frombuffer(buf, dtype=np.float32).copy()
        finally:
            self.free.put(p)
    def close(self):
        for p in getattr(self, "procs", []):
            try:
                p.kill()
            except OSError as e:
                log(f"worker kill: {e!r}")

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
    out, carry = [], ""
    for s in _SPLIT.split(t):
        s = (carry + " " + s).strip() if carry else s.strip()
        carry = ""
        while len(s) > MAX_SENT_CHARS:
            cut = max(s.rfind(", ", 0, MAX_SENT_CHARS), s.rfind(" ", 0, MAX_SENT_CHARS))
            if cut < 40:
                cut = MAX_SENT_CHARS
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if not s:
            continue
        # tiny fragments ("Ok.") cost a full synth call and sound choppy: merge into a neighbour
        if len(s) < 14:
            if out:
                out[-1] += " " + s
            else:
                carry = s
        else:
            out.append(s)
    if carry:
        out.append(carry)
    return [s for s in out if re.search(r"[A-Za-z0-9]", s)]

def pcm_b64(x):
    x = np.clip(np.nan_to_num(x), -1.0, 1.0)
    return base64.b64encode((x * 32767).astype("<i2").tobytes()).decode()

# ===== PACE ESTIMATOR (the gap-free start gate) =====
class Pace:
    def __init__(self):
        self.spc = 0.065      # audio seconds per character
        self.wpc = 0.09       # wall seconds of synthesis per character (measured under real contention)
        self.lcps = 18.0      # characters per second the LLM writes
        self.reply = 120.0    # typical reply length in characters (EMA)
        self.safety = SAFETY0
        self.lock = threading.Lock()
    def seed(self, chars, audio_s, wall_s):
        with self.lock:
            self.spc = min(max(audio_s / chars, 0.03), 0.12)
            self.wpc = min(max(wall_s / chars, 0.005), 0.4)
    def gate(self, buffered, rem_chars, speech_done, chars_done, wall_done, future_chars=0):
        """True if starting playback now cannot underrun, under the estimate.
        rem_chars: sentences already written but not yet synthesised.
        future_chars: text the LLM has not written yet, modelled as 90-char chunks that arrive at LLM speed."""
        with self.lock:
            spc, wpc, sf, lcps = self.spc, self.wpc, self.safety, max(self.lcps, 5.0)
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
        text_t, left = 0.0, future_chars
        while left > 0:
            c = min(left, 90); left -= c
            text_t += sf * c / lcps      # LLM has to write the chunk first
            t = max(t, text_t) + sf * wpc * c
            if avail < t + LEAD_S:
                return False
            avail += spc * c + PAD_S
        return avail >= LEAD_S
    def update(self, chars, speech, wall):
        if chars < 20 or speech <= 0:
            return
        with self.lock:
            self.spc = 0.6 * self.spc + 0.4 * (speech / chars)
            self.wpc = 0.6 * self.wpc + 0.4 * (wall / chars)
    def update_llm(self, chars, secs):
        if chars < 20 or secs <= 0:
            return
        with self.lock:
            self.lcps = 0.6 * self.lcps + 0.4 * (chars / secs)
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
TTS_SERIAL = False
FILLERS = []

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
    # prewarm LLM: 1-token completion, then reset history (first real TTFT drops 3-4 s)
    LLM.create_chat_completion(messages=[HISTORY[0], {"role": "user", "content": "hi"}], max_tokens=1)
    reset_history()
    log("selecting TTS backend")
    TTS, ratio = select_tts()
    finalize_tts(ratio)
    try:
        FILLERS[:] = build_fillers()
    except Exception as e:
        log(f"fillers unavailable: {e!r}")

def _measure(b):
    """Returns (chars, audio_seconds, wall_seconds_per_stream). With P parallel workers, wall is divided by P:
    it is the aggregate throughput that decides whether playback can keep up."""
    probe = "This is a short test of the voice engine."
    P = max(1, b.parallel)
    for f in [b.executor.submit(b.synth, "Hello.") for _ in range(P)]:   # first call pays one-off graph cost
        f.result()
    t = time.perf_counter()
    outs = [f.result() for f in [b.executor.submit(b.synth, probe) for _ in range(P)]]
    wall = (time.perf_counter() - t) / P
    return len(probe), len(outs[0]) / b.sample_rate, wall

def _cpu_key():
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                return l.split(":", 1)[1].strip()
    except OSError as e:
        log(f"cpuinfo unreadable: {e!r}")
    return "unknown"

def tune_pocket():
    """Find the fastest way to run YOUR voice on THIS CPU. Candidates differ in int8 vs fp32 (int8 can be
    slower at batch 1) and in-process threads vs two single-thread worker processes. First boot tries them
    (about a minute); the winner is cached on Drive per CPU model, later boots load only the winner."""
    cands = {"pool2_int8": lambda: PocketPoolBackend(2, True), "inproc_int8": lambda: PocketBackend(True, 2),
             "pool2_fp32": lambda: PocketPoolBackend(2, False), "inproc_fp32": lambda: PocketBackend(False, 2)}
    tf, key, cached = STATE_DIR / "pocket_tuning.json", _cpu_key(), None
    try:
        cached = json.loads(tf.read_text())
    except FileNotFoundError:
        pass
    except ValueError as e:
        log(f"tuning cache unreadable: {e!r}")
    if cached and cached.get("cpu") == key and cached.get("best") in cands:
        b = cands[cached["best"]](); b.load()
        log(f"pocket-tts variant from cache: {b.name}")
        return b, _measure(b)
    log("first boot on this CPU: benchmarking pocket-tts variants (cached afterwards)")
    best, results = None, {}
    for name, mk in cands.items():
        try:
            b = mk(); b.load(); m = _measure(b)
        except Exception as e:
            log(f"variant {name} failed: {e!r}")
            continue
        ratio = m[2] / m[1]; results[name] = round(ratio, 2)
        log(f"  {b.name}: wall/audio = {ratio:.2f}")
        if best is None or ratio < best[0]:
            if best: best[2].close()
            best = (ratio, name, b, m)
        else:
            b.close()
        if ratio <= 0.5:
            break
    if best is None:
        raise RuntimeError("no pocket-tts variant could be loaded")
    try:
        tf.write_text(json.dumps({"cpu": key, "best": best[1], "ratios": results}))
    except OSError as e:
        log(f"could not write tuning cache: {e!r}")
    return best[2], best[3]

def finalize_tts(ratio):
    """Adapt the whole pipeline to how fast the voice really is."""
    global TTS_SERIAL, SYSTEM_PROMPT, MAX_TOKENS
    # Default: speak WHILE the LLM streams (overlap). The adaptive gate decides when it is safe to start;
    # Live mode starts at the first chunk. NEO_SERIAL=1 forces the old "wait for the full text first"
    # behaviour, which is exact but silent until the LLM finishes.
    TTS_SERIAL = os.environ.get("NEO_SERIAL", "0") == "1" and ratio > 0.55
    if ratio > 1.0:
        # Slower than real time: audio length is the cost, so keep replies to one short sentence.
        SYSTEM_PROMPT = SYSTEM_PROMPT.replace("one or two short sentences, under thirty words in total",
                                              "one short sentence, under fifteen words")
        MAX_TOKENS = min(MAX_TOKENS, 48)
        reset_history()
    log(f"pipeline: serial={TTS_SERIAL}, max_tokens={MAX_TOKENS}, parallel voice workers={TTS.parallel}")

FILLER_TEXTS = ["Okay.", "Sure thing.", "Let me see."]

def build_fillers():
    """Short acknowledgements in YOUR voice, rendered once and cached on Drive. The browser plays one the
    instant you send, so sound starts at 0 ms even though the real answer takes seconds to prepare."""
    import hashlib
    vp = find_voice()
    h = hashlib.sha1(vp.read_bytes() if vp else b"default"); h.update(TTS.name.encode())
    cache = STATE_DIR / f"fillers_{h.hexdigest()[:12]}.json"
    try:
        return json.loads(cache.read_text())
    except FileNotFoundError:
        pass
    except ValueError as e:
        log(f"filler cache unreadable: {e!r}")
    outl = []
    for f in [TTS.executor.submit(TTS.synth, t) for t in FILLER_TEXTS]:
        pcm = f.result()
        outl.append({"sr": TTS.sample_rate, "dur": len(pcm) / TTS.sample_rate, "pcm": pcm_b64(pcm)})
    try:
        cache.write_text(json.dumps(outl))
    except OSError as e:
        log(f"could not cache fillers: {e!r}")
    return outl

def select_tts():
    """pocket (default): your custom voice, auto-tuned for speed. kokoro: forced Kokoro.
    auto: pocket, but fall back to Kokoro if even the tuned pocket is slower than ~0.9x real time."""
    if TTS_BACKEND == "kokoro":
        b = make_backend()
        c, a, w = _measure(b); PACE.seed(c, a, w)
        log(f"TTS {b.name}: wall/audio = {w/a:.2f}")
        return b, w / a
    b, (c, a, w) = tune_pocket()
    ratio = w / a
    log(f"TTS {b.name}: aggregate wall/audio = {ratio:.2f} (<1.0 = faster than real time)")
    if TTS_BACKEND == "auto" and ratio > 0.9:
        log("pocket-tts still too slow to stream gap-free next to the LLM; trying Kokoro")
        try:
            k = KokoroBackend(); k.load()
            kc, ka, kw = _measure(k)
            if kw / ka < ratio:
                log(f"USING {k.name} ({kw/ka:.2f}); voice is Kokoro '{KOKORO_VOICE}', not your custom voice")
                b.close(); PACE.seed(kc, ka, kw)
                return k, kw / ka
            log("Kokoro was not faster; keeping pocket-tts")
        except Exception as e:
            log(f"Kokoro unavailable ({e!r}); keeping pocket-tts")
    PACE.seed(c, a, w)
    return b, ratio

# ===== THE TURN PIPELINE (LLM -> sentences -> TTS -> client, all overlapped) =====
_BOUND = re.compile(r"[.!?][\"')\]]*\s")

def _emit_ready(buf, sq, st):
    """Move completed sentences out of the streaming text buffer into the TTS queue.
    The FIRST chunk of a turn may be cut at a comma (28+ chars) so speech can start while the LLM is still writing."""
    def put(txt):
        for s in split_sentences(clean_for_tts(txt)):
            sq.put(s); st["sent"] += 1
    pos = 0
    while True:
        m = _BOUND.search(buf, pos)
        if not m:
            break
        if len(buf[:m.end()].strip()) >= 14:    # too-short fragments ("Ok.") wait for the next boundary
            put(buf[:m.end()]); buf = buf[m.end():]; pos = 0
        else:
            pos = m.end()
    if st["sent"] == 0 and len(buf) >= 28:
        cut = buf.rfind(", ")
        if cut >= 20:
            put(buf[:cut + 1]); buf = buf[cut + 1:]
    if len(buf) > 160:                          # long run without punctuation: cut at a comma
        cut = buf.rfind(", ", 0, 160)
        if cut > 40:
            put(buf[:cut + 1]); buf = buf[cut + 1:]
    return buf

def turn_events(user_text, mode, tts_on, max_tokens):
    """Three cooperating threads per turn: LLM producer -> TTS worker -> this generator (to the client).
    Sentences are synthesised while the LLM is still writing. Playback is released by PACE.gate, which
    accounts for text the LLM has not written yet. Everything funnels through one queue so the NDJSON
    stream stays ordered."""
    t_req = time.perf_counter()
    with TURN_LOCK:
        CANCEL.clear()
        while len(HISTORY) - 1 >= MAX_TURNS * 2:   # rolling window; system prompt stays at index 0
            del HISTORY[1:3]
        HISTORY.append({"role": "user", "content": user_text})
        out, sq = queue.Queue(), queue.Queue()
        parts = []
        st = {"committed": False, "clean_end": False, "llm_done": False, "chars": 0, "ntok": 0,
              "t_first": None, "t_llm_end": None, "tts_wall": 0.0, "speech": 0.0, "chars_done": 0,
              "go_s": None, "stream": None, "sent": 0}

        def llm_worker():
            buf = ""
            try:
                # Full growing history: llama.cpp reuses the KV prefix, past turns are never re-tokenised.
                st["stream"] = LLM.create_chat_completion(messages=HISTORY, stream=True, max_tokens=max_tokens,
                                                          temperature=0.6, top_p=0.9, repeat_penalty=1.1)
                for ch in st["stream"]:
                    if CANCEL.is_set():
                        break
                    d = ch["choices"][0]["delta"].get("content")
                    if not d:
                        continue
                    if st["t_first"] is None:
                        st["t_first"] = time.perf_counter()
                    parts.append(d); st["ntok"] += 1; st["chars"] += len(d)
                    out.put({"type": "text", "delta": d})
                    if tts_on:
                        buf = _emit_ready(buf + d, sq, st)
                st["t_llm_end"] = time.perf_counter()
                text = "".join(parts).strip()
                if text:
                    HISTORY.append({"role": "assistant", "content": text}); st["committed"] = True
                if tts_on and not CANCEL.is_set():
                    for s in split_sentences(clean_for_tts(buf)):
                        sq.put(s)
            except Exception as e:
                print(f"[llm] failed: {e!r}", flush=True)
                out.put({"type": "error", "message": f"llm failed: {e!r}"})
            finally:
                st["llm_done"] = True        # set AFTER the tail flush so the gate never misses queued text
                sq.put(None)
                if not tts_on:
                    out.put(("_end",))

        def tts_worker():
            pcms, buffered, went, sr, last_k = [], 0.0, False, TTS.sample_rate, -1
            P = max(1, TTS.parallel)
            inflight, eof, idx = collections.deque(), False, 0
            def timed(text):
                t = time.perf_counter(); p = TTS.synth(text); return p, time.perf_counter() - t
            try:
                while not CANCEL.is_set():
                    if TTS_SERIAL and not st["llm_done"]:
                        time.sleep(0.05)       # slow voice: let the LLM have both cores; text is then fully known
                        continue
                    while not eof and len(inflight) < P:     # keep every voice worker busy
                        try:
                            item = sq.get(timeout=0.05) if not inflight else sq.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            eof = True
                            break
                        inflight.append((idx, item, TTS.executor.submit(timed, item))); idx += 1
                    if not inflight:
                        if eof:
                            break
                        continue
                    k, text, fut = inflight.popleft()
                    out.put({"type": "status", "stage": "synth", "i": k})   # also keeps the tunnel warm
                    try:
                        pcm, dt = fut.result()
                    except Exception as e:
                        print(f"[tts] sentence {k} failed: {e!r}", flush=True)
                        out.put({"type": "error", "message": f"tts failed on sentence {k}: {e!r}"})
                        continue
                    sr = TTS.sample_rate
                    # dt/P: with P workers busy at once, aggregate wall per audio second is dt/P
                    st["tts_wall"] += dt / P; st["speech"] += len(pcm) / sr; st["chars_done"] += len(text)
                    pcm = np.concatenate([pcm, np.zeros(int(PAD_S * sr), np.float32)])
                    last_k = k
                    if mode == "smooth":
                        pcms.append(pcm)
                        continue
                    buffered += len(pcm) / sr
                    out.put({"type": "sentence_audio", "i": k, "sr": sr, "pcm": pcm_b64(pcm)})
                    if not went:
                        if mode == "stream":
                            ok = True
                        else:
                            with sq.mutex:
                                pending = [len(t) for _, t, _ in inflight] + [len(x) for x in list(sq.queue) if x]
                            # text the LLM has not written yet: expect a typical reply length, at least one more sentence
                            fut_chars = 0 if st["llm_done"] else max(50, int(PACE.reply * 1.25) - st["chars"])
                            ok = PACE.gate(buffered, pending, st["speech"], st["chars_done"], st["tts_wall"], fut_chars)
                        if ok:
                            went = True
                            st["go_s"] = time.perf_counter() - t_req
                            out.put({"type": "go", "after_sentence": k, "buffered_s": round(buffered, 2)})
                if mode == "smooth" and pcms and not CANCEL.is_set():
                    full = np.concatenate(pcms)
                    st["go_s"] = time.perf_counter() - t_req
                    out.put({"type": "audio_full", "sr": sr, "dur": len(full) / sr, "pcm": pcm_b64(full)})
                elif mode != "smooth" and not went and buffered > 0:
                    st["go_s"] = time.perf_counter() - t_req
                    out.put({"type": "go", "after_sentence": last_k, "buffered_s": round(buffered, 2)})
            except Exception as e:
                print(f"[tts] worker failed: {e!r}", flush=True)
                out.put({"type": "error", "message": f"tts worker failed: {e!r}"})
            finally:
                out.put(("_end",))

        threads = [threading.Thread(target=llm_worker, daemon=True, name="neo-llm")]
        if tts_on:
            threads.append(threading.Thread(target=tts_worker, daemon=True, name="neo-tts"))
        try:
            yield {"type": "meta", "sample_rate": TTS.sample_rate, "audio_mode": mode, "backend": TTS.name,
                   "pid": os.getpid()}
            for th in threads:
                th.start()
            while True:
                try:
                    ev = out.get(timeout=0.25)
                except queue.Empty:
                    if not any(th.is_alive() for th in threads) and out.empty():
                        break
                    continue
                if isinstance(ev, tuple):       # ("_end",)
                    break
                yield ev
            text = "".join(parts).strip()
            if not text:
                # Empty generation: the finally block pops the user message. Keeping an
                # assistant "" pair makes Llama's template emit instant EOS forever ("0 tok/s" bug).
                yield {"type": "error", "message": "empty generation (rolled back, try again)"}
                return
            st["clean_end"] = True
            t_end = time.perf_counter()
            if st["chars_done"]:
                PACE.update(st["chars_done"], st["speech"], st["tts_wall"])
            if st["t_first"] and st["t_llm_end"] and st["t_llm_end"] > st["t_first"]:
                PACE.update_llm(st["chars"], st["t_llm_end"] - st["t_first"])
            PACE.reply = 0.6 * PACE.reply + 0.4 * st["chars"]
            tf, te = st["t_first"], st["t_llm_end"]
            yield {"type": "metrics",
                   "ttft_s": round(tf - t_req, 2) if tf else None,
                   "tok_s": round(st["ntok"] / max(te - tf, 1e-6), 1) if tf and te else None,
                   "tokens": st["ntok"], "llm_s": round((te - t_req), 2) if te else None,
                   "tts_s": round(st["tts_wall"], 2), "audio_s": round(st["speech"], 2),
                   "tts_wall_per_audio": round(st["tts_wall"] / st["speech"], 2) if st["speech"] else None,
                   "audio_start_s": round(st["go_s"], 2) if st["go_s"] else None,
                   "total_s": round(t_end - t_req, 2), "mode": mode, "backend": TTS.name,
                   "safety": round(PACE.safety, 2), "history_len": len(HISTORY)}
            yield {"type": "done"}
        finally:
            if not st["clean_end"]:
                CANCEL.set()                    # client left or error: stop both workers before releasing the lock
            for th in threads:
                if th.is_alive():
                    th.join(timeout=40)
            s = st["stream"]
            if s is not None and hasattr(s, "close"):
                try: s.close()
                except Exception as e: print(f"[llm] stream close: {e!r}", flush=True)
            if not st["committed"]:
                partial = "".join(parts).strip()
                if partial:
                    HISTORY.append({"role": "assistant", "content": partial})
                elif HISTORY and HISTORY[-1]["role"] == "user":
                    HISTORY.pop()

# ===== WEB APP =====
_UI_B64 = "PCFkb2N0eXBlIGh0bWw+CjxodG1sIGxhbmc9ImVuIj48aGVhZD48bWV0YSBjaGFyc2V0PSJ1dGYtOCI+CjxtZXRhIG5hbWU9InZpZXdwb3J0IiBjb250ZW50PSJ3aWR0aD1kZXZpY2Utd2lkdGgsaW5pdGlhbC1zY2FsZT0xLHZpZXdwb3J0LWZpdD1jb3ZlciI+Cjx0aXRsZT5uZW8tdHRzPC90aXRsZT4KPHN0eWxlPgo6cm9vdHstLWJnOiMxMjE1MWE7LS1wYW5lbDojMWExZTI1Oy0tZmc6I2U5ZWJmMDstLW11dDojOGU5N2E4Oy0tbGluZTojMmEzMDNiOy0tYWNjOiM3ZmQ2YzI7LS1tZTojMjMyYTM2fQpAbWVkaWEgKHByZWZlcnMtY29sb3Itc2NoZW1lOmxpZ2h0KXs6cm9vdHstLWJnOiNmNGY1Zjc7LS1wYW5lbDojZmZmOy0tZmc6IzE2MWEyMTstLW11dDojNWY2ODc5Oy0tbGluZTojZGRlMWU5Oy0tYWNjOiMwZjdmNmE7LS1tZTojZThlY2YzfX0KKntib3gtc2l6aW5nOmJvcmRlci1ib3h9Cmh0bWwsYm9keXtoZWlnaHQ6MTAwJTttYXJnaW46MH0KYm9keXtiYWNrZ3JvdW5kOnZhcigtLWJnKTtjb2xvcjp2YXIoLS1mZyk7Zm9udDoxNnB4LzEuNSBzeXN0ZW0tdWksLWFwcGxlLXN5c3RlbSwiU2Vnb2UgVUkiLHNhbnMtc2VyaWY7ZGlzcGxheTpmbGV4O2ZsZXgtZGlyZWN0aW9uOmNvbHVtbjttYXgtd2lkdGg6NzYwcHg7bWFyZ2luOjAgYXV0bztwYWRkaW5nOmVudihzYWZlLWFyZWEtaW5zZXQtdG9wKSAxMnB4IGVudihzYWZlLWFyZWEtaW5zZXQtYm90dG9tKX0KaGVhZGVye2Rpc3BsYXk6ZmxleDthbGlnbi1pdGVtczpjZW50ZXI7Z2FwOjEwcHg7cGFkZGluZzoxNHB4IDRweDtib3JkZXItYm90dG9tOjFweCBzb2xpZCB2YXIoLS1saW5lKX0KaDF7Zm9udC1zaXplOjE3cHg7bWFyZ2luOjA7Zm9udC13ZWlnaHQ6NjAwO2xldHRlci1zcGFjaW5nOi4ycHh9CiNjaGlwe2NvbG9yOnZhcigtLW11dCk7Zm9udC1zaXplOjEzcHg7ZmxleDoxfQpzZWxlY3QsYnV0dG9uLGlucHV0e2ZvbnQ6aW5oZXJpdDtjb2xvcjp2YXIoLS1mZyl9CmxhYmVsLmNoa3tmb250LXNpemU6MTNweDtjb2xvcjp2YXIoLS1tdXQpO2Rpc3BsYXk6ZmxleDtnYXA6NHB4O2FsaWduLWl0ZW1zOmNlbnRlcn0Kc2VsZWN0e2JhY2tncm91bmQ6dmFyKC0tcGFuZWwpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSk7Ym9yZGVyLXJhZGl1czo4cHg7cGFkZGluZzo2cHggOHB4fQojbG9ne2ZsZXg6MTtvdmVyZmxvdy15OmF1dG87cGFkZGluZzoxNnB4IDJweDtkaXNwbGF5OmZsZXg7ZmxleC1kaXJlY3Rpb246Y29sdW1uO2dhcDoxMnB4fQoubXttYXgtd2lkdGg6ODglO3BhZGRpbmc6MTBweCAxNHB4O2JvcmRlci1yYWRpdXM6MTRweDt3aGl0ZS1zcGFjZTpwcmUtd3JhcDt3b3JkLXdyYXA6YnJlYWstd29yZH0KLnV7YWxpZ24tc2VsZjpmbGV4LWVuZDtiYWNrZ3JvdW5kOnZhcigtLW1lKX0KLmF7YWxpZ24tc2VsZjpmbGV4LXN0YXJ0O2JhY2tncm91bmQ6dmFyKC0tcGFuZWwpO2JvcmRlcjoxcHggc29saWQgdmFyKC0tbGluZSl9Ci5tZXRhe2ZvbnQtc2l6ZToxMnB4O2NvbG9yOnZhcigtLW11dCk7bWFyZ2luLXRvcDo2cHh9CiNzdGF0dXN7bWluLWhlaWdodDoyMHB4O2ZvbnQtc2l6ZToxM3B4O2NvbG9yOnZhcigtLWFjYyk7cGFkZGluZzowIDRweH0KZm9ybXtkaXNwbGF5OmZsZXg7Z2FwOjhweDtwYWRkaW5nOjEwcHggMCAxNHB4fQppbnB1dHtmbGV4OjE7YmFja2dyb3VuZDp2YXIoLS1wYW5lbCk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKTtib3JkZXItcmFkaXVzOjEycHg7cGFkZGluZzoxMnB4IDE0cHh9CmlucHV0OmZvY3VzLXZpc2libGUsYnV0dG9uOmZvY3VzLXZpc2libGUsc2VsZWN0OmZvY3VzLXZpc2libGV7b3V0bGluZToycHggc29saWQgdmFyKC0tYWNjKTtvdXRsaW5lLW9mZnNldDoycHh9CmJ1dHRvbntiYWNrZ3JvdW5kOnZhcigtLWFjYyk7Y29sb3I6IzBiMTIxMDtib3JkZXI6MDtib3JkZXItcmFkaXVzOjEycHg7cGFkZGluZzowIDE4cHg7Zm9udC13ZWlnaHQ6NjAwO2N1cnNvcjpwb2ludGVyfQpidXR0b24uZ2hvc3R7YmFja2dyb3VuZDp0cmFuc3BhcmVudDtjb2xvcjp2YXIoLS1mZyk7Ym9yZGVyOjFweCBzb2xpZCB2YXIoLS1saW5lKX0KYnV0dG9uOmRpc2FibGVke29wYWNpdHk6LjU7Y3Vyc29yOmRlZmF1bHR9Cjwvc3R5bGU+PC9oZWFkPgo8Ym9keT4KPGhlYWRlcj48aDE+bmVvLXR0czwvaDE+PHNwYW4gaWQ9ImNoaXAiPmNvbm5lY3Rpbmc8L3NwYW4+CjxzZWxlY3QgaWQ9Im1vZGUiIHRpdGxlPSJBdWRpbyBtb2RlIj48b3B0aW9uIHZhbHVlPSJhZGFwdGl2ZSI+QWRhcHRpdmUgKHNhZmUgc3RhcnQpPC9vcHRpb24+PG9wdGlvbiB2YWx1ZT0ic3RyZWFtIj5MaXZlIChzcGVhayBhdCBvbmNlKTwvb3B0aW9uPjxvcHRpb24gdmFsdWU9InNtb290aCI+U21vb3RoIChhZnRlciBmdWxsIHJlcGx5KTwvb3B0aW9uPjwvc2VsZWN0Pgo8bGFiZWwgY2xhc3M9ImNoayI+PGlucHV0IHR5cGU9ImNoZWNrYm94IiBpZD0iZmlsbCIgY2hlY2tlZD4gUXVpY2sgYWNrPC9sYWJlbD48YnV0dG9uIGNsYXNzPSJnaG9zdCIgaWQ9InJlc2V0IiB0eXBlPSJidXR0b24iPk5ldyBjaGF0PC9idXR0b24+PC9oZWFkZXI+CjxkaXYgaWQ9ImxvZyIgYXJpYS1saXZlPSJwb2xpdGUiPjwvZGl2Pgo8ZGl2IGlkPSJzdGF0dXMiPjwvZGl2Pgo8Zm9ybSBpZD0iZiI+PGlucHV0IGlkPSJxIiBhdXRvY29tcGxldGU9Im9mZiIgcGxhY2Vob2xkZXI9IlR5cGUgYSBtZXNzYWdlIiBhcmlhLWxhYmVsPSJNZXNzYWdlIj48YnV0dG9uIGlkPSJzZW5kIj5TZW5kPC9idXR0b24+PGJ1dHRvbiBjbGFzcz0iZ2hvc3QiIGlkPSJzdG9wIiB0eXBlPSJidXR0b24iIGRpc2FibGVkPlN0b3A8L2J1dHRvbj48L2Zvcm0+CjxzY3JpcHQ+CiJ1c2Ugc3RyaWN0IjsKY29uc3QgJCA9IHMgPT4gZG9jdW1lbnQucXVlcnlTZWxlY3RvcihzKTsKY29uc3QgQSA9IHtjdHg6IG51bGwsIHNyOiAyNDAwMCwgaGVhZDogMCwgc3RhcnRlZDogZmFsc2UsIHF1ZXVlOiBbXSwgbGF0ZTogMCwgc291cmNlczogW119OwpsZXQgY3RybCA9IG51bGwsIGZpbGxlcnMgPSBbXSwgZmkgPSAwOwpmZXRjaCgiL2ZpbGxlcnMiKS50aGVuKHIgPT4gci5qc29uKCkpLnRoZW4oZiA9PiB7IGZpbGxlcnMgPSBmOyB9KS5jYXRjaChlID0+IGNvbnNvbGUud2FybigiZmlsbGVycyIsIGUpKTsKZnVuY3Rpb24gcGxheUZpbGxlcigpIHsKICBpZiAoISQoIiNmaWxsIikuY2hlY2tlZCB8fCAhZmlsbGVycy5sZW5ndGgpIHJldHVybjsKICBjb25zdCBmID0gZmlsbGVyc1tmaSsrICUgZmlsbGVycy5sZW5ndGhdOwogIGlmIChmLnNyICE9PSBBLnNyKSByZXR1cm47CiAgQS5oZWFkID0gQS5jdHguY3VycmVudFRpbWUgKyAwLjAzOyAgICAvLyBwbGF5cyBhdCBvbmNlLCBiZWZvcmUgdGhlIHNlcnZlciBoYXMgcHJvZHVjZWQgYW55dGhpbmcKICBzY2hlZHVsZShmYWRlKHBjbVRvRjMyKGYucGNtKSwgMywgMTApKTsKfQoKZnVuY3Rpb24gZW5zdXJlQ3R4KHNyKSB7CiAgaWYgKEEuY3R4ICYmIEEuY3R4LnNhbXBsZVJhdGUgPT09IHNyKSByZXR1cm47CiAgaWYgKEEuY3R4KSBBLmN0eC5jbG9zZSgpOwogIEEuc3IgPSBzcjsKICAvLyBGaXhlZCAyNCBrSHogY29udGV4dDogdGhlIGRlZmF1bHQgNDgga0h6IGNvbnRleHQgcmVzYW1wbGVzIG91ciBidWZmZXJzIGFuZCBjcmFja2xlcy4KICBBLmN0eCA9IG5ldyAod2luZG93LkF1ZGlvQ29udGV4dCB8fCB3aW5kb3cud2Via2l0QXVkaW9Db250ZXh0KSh7c2FtcGxlUmF0ZTogc3IsIGxhdGVuY3lIaW50OiAicGxheWJhY2sifSk7Cn0KZnVuY3Rpb24gcmVzZXRBdWRpbygpIHsKICBmb3IgKGNvbnN0IHMgb2YgQS5zb3VyY2VzKSB7IHRyeSB7IHMuc3RvcCgpOyB9IGNhdGNoIChlKSB7IGNvbnNvbGUud2Fybigic3RvcCIsIGUpOyB9IH0KICBBLnNvdXJjZXMgPSBbXTsgQS5oZWFkID0gMDsgQS5zdGFydGVkID0gZmFsc2U7IEEucXVldWUgPSBbXTsgQS5sYXRlID0gMDsKfQpmdW5jdGlvbiBwY21Ub0YzMihiNjQpIHsKICBjb25zdCBiaW4gPSBhdG9iKGI2NCksIG4gPSBiaW4ubGVuZ3RoID4+IDEsIG91dCA9IG5ldyBGbG9hdDMyQXJyYXkobik7CiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuOyBpKyspIHsKICAgIGxldCB2ID0gYmluLmNoYXJDb2RlQXQoMiAqIGkpIHwgKGJpbi5jaGFyQ29kZUF0KDIgKiBpICsgMSkgPDwgOCk7CiAgICBpZiAodiAmIDB4ODAwMCkgdiAtPSAweDEwMDAwOwogICAgb3V0W2ldID0gdiAvIDMyNzY4OwogIH0KICByZXR1cm4gb3V0Owp9CmZ1bmN0aW9uIGZhZGUoZiwgaW5Ncywgb3V0TXMpIHsKICBjb25zdCBuaSA9IE1hdGgubWluKGYubGVuZ3RoID4+IDEsIE1hdGgucm91bmQoQS5zciAqIGluTXMgLyAxMDAwKSksIG5vID0gTWF0aC5taW4oZi5sZW5ndGggPj4gMSwgTWF0aC5yb3VuZChBLnNyICogb3V0TXMgLyAxMDAwKSk7CiAgZm9yIChsZXQgaSA9IDA7IGkgPCBuaTsgaSsrKSBmW2ldICo9IGkgLyBuaTsKICBmb3IgKGxldCBpID0gMDsgaSA8IG5vOyBpKyspIGZbZi5sZW5ndGggLSAxIC0gaV0gKj0gaSAvIG5vOwogIHJldHVybiBmOwp9CmZ1bmN0aW9uIHNjaGVkdWxlKGYpIHsKICBjb25zdCBidWYgPSBBLmN0eC5jcmVhdGVCdWZmZXIoMSwgZi5sZW5ndGgsIEEuc3IpOwogIGJ1Zi5jb3B5VG9DaGFubmVsKGYsIDApOwogIGNvbnN0IHNyYyA9IEEuY3R4LmNyZWF0ZUJ1ZmZlclNvdXJjZSgpOwogIHNyYy5idWZmZXIgPSBidWY7IHNyYy5jb25uZWN0KEEuY3R4LmRlc3RpbmF0aW9uKTsKICBjb25zdCBub3cgPSBBLmN0eC5jdXJyZW50VGltZTsKICBpZiAoQS5oZWFkIDwgbm93ICsgMC4wMSkgeyAgICAgICAgICAgIC8vIGNodW5rIGFycml2ZWQgYWZ0ZXIgdGhlIHF1ZXVlIHJhbiBkcnk6IGFuIGF1ZGlibGUgZ2FwCiAgICBpZiAoQS5zdGFydGVkICYmIEEuaGVhZCA+IDApIEEubGF0ZSsrOwogICAgQS5oZWFkID0gbm93ICsgMC4wNjsKICB9CiAgc3JjLnN0YXJ0KEEuaGVhZCk7ICAgICAgICAgICAgICAgICAgICAvLyBzYW1wbGUtYWNjdXJhdGU6IGhlYWQgYWR2YW5jZXMgYnkgZXhhY3QgYnVmZmVyIGR1cmF0aW9uCiAgQS5oZWFkICs9IGJ1Zi5kdXJhdGlvbjsKICBBLnNvdXJjZXMucHVzaChzcmMpOwp9CmZ1bmN0aW9uIGdvKCkgewogIGlmIChBLnN0YXJ0ZWQpIHJldHVybjsKICBBLnN0YXJ0ZWQgPSB0cnVlOyBBLmhlYWQgPSBNYXRoLm1heChBLmhlYWQsIEEuY3R4LmN1cnJlbnRUaW1lICsgMC4wOCk7CiAgZm9yIChjb25zdCBmIG9mIEEucXVldWUpIHNjaGVkdWxlKGYpOwogIEEucXVldWUgPSBbXTsKfQoKZnVuY3Rpb24gYnViYmxlKGNscywgdGV4dCkgewogIGNvbnN0IGQgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCJkaXYiKTsgZC5jbGFzc05hbWUgPSAibSAiICsgY2xzOyBkLnRleHRDb250ZW50ID0gdGV4dDsKICAkKCIjbG9nIikuYXBwZW5kQ2hpbGQoZCk7ICQoIiNsb2ciKS5zY3JvbGxUb3AgPSAxZTk7IHJldHVybiBkOwp9Cgphc3luYyBmdW5jdGlvbiBzZW5kKHRleHQpIHsKICBlbnN1cmVDdHgoQS5zcik7IGF3YWl0IEEuY3R4LnJlc3VtZSgpOyByZXNldEF1ZGlvKCk7IHBsYXlGaWxsZXIoKTsKICBidWJibGUoInUiLCB0ZXh0KTsKICBjb25zdCBvdXQgPSBidWJibGUoImEiLCAiIik7IGNvbnN0IG1ldGEgPSBkb2N1bWVudC5jcmVhdGVFbGVtZW50KCJkaXYiKTsgbWV0YS5jbGFzc05hbWUgPSAibWV0YSI7IG91dC5hcHBlbmRDaGlsZChtZXRhKTsKICBjb25zdCBib2R5ID0gZG9jdW1lbnQuY3JlYXRlVGV4dE5vZGUoIiIpOyBvdXQuaW5zZXJ0QmVmb3JlKGJvZHksIG1ldGEpOwogIGN0cmwgPSBuZXcgQWJvcnRDb250cm9sbGVyKCk7CiAgJCgiI3NlbmQiKS5kaXNhYmxlZCA9IHRydWU7ICQoIiNzdG9wIikuZGlzYWJsZWQgPSBmYWxzZTsKICB0cnkgewogICAgY29uc3QgcmVzID0gYXdhaXQgZmV0Y2goIi9jaGF0Iiwge21ldGhvZDogIlBPU1QiLCBoZWFkZXJzOiB7IkNvbnRlbnQtVHlwZSI6ICJhcHBsaWNhdGlvbi9qc29uIn0sCiAgICAgIGJvZHk6IEpTT04uc3RyaW5naWZ5KHttZXNzYWdlOiB0ZXh0LCBtb2RlOiAkKCIjbW9kZSIpLnZhbHVlfSksIHNpZ25hbDogY3RybC5zaWduYWx9KTsKICAgIGNvbnN0IHJkID0gcmVzLmJvZHkuZ2V0UmVhZGVyKCksIGRlYyA9IG5ldyBUZXh0RGVjb2RlcigpOyBsZXQgYnVmID0gIiI7CiAgICBmb3IgKDs7KSB7CiAgICAgIGNvbnN0IHt2YWx1ZSwgZG9uZX0gPSBhd2FpdCByZC5yZWFkKCk7IGlmIChkb25lKSBicmVhazsKICAgICAgYnVmICs9IGRlYy5kZWNvZGUodmFsdWUsIHtzdHJlYW06IHRydWV9KTsKICAgICAgbGV0IGk7CiAgICAgIHdoaWxlICgoaSA9IGJ1Zi5pbmRleE9mKCJcbiIpKSA+PSAwKSB7CiAgICAgICAgY29uc3QgbGluZSA9IGJ1Zi5zbGljZSgwLCBpKTsgYnVmID0gYnVmLnNsaWNlKGkgKyAxKTsKICAgICAgICBpZiAobGluZS50cmltKCkpIGhhbmRsZShKU09OLnBhcnNlKGxpbmUpLCBib2R5LCBtZXRhKTsKICAgICAgfQogICAgfQogIH0gY2F0Y2ggKGUpIHsKICAgIGlmIChlLm5hbWUgIT09ICJBYm9ydEVycm9yIikgeyBtZXRhLnRleHRDb250ZW50ID0gIkNvbm5lY3Rpb24gcHJvYmxlbTogIiArIGUubWVzc2FnZTsgY29uc29sZS5lcnJvcihlKTsgfQogIH0KICBpZiAoIUEuc3RhcnRlZCAmJiBBLnF1ZXVlLmxlbmd0aCkgZ28oKTsgICAvLyBmYWxsYmFjazogbmV2ZXIgbGVhdmUgYXVkaW8gdW5wbGF5ZWQKICBmZXRjaCgiL2ZlZWRiYWNrIiwge21ldGhvZDogIlBPU1QiLCBoZWFkZXJzOiB7IkNvbnRlbnQtVHlwZSI6ICJhcHBsaWNhdGlvbi9qc29uIn0sIGJvZHk6IEpTT04uc3RyaW5naWZ5KHtsYXRlOiBBLmxhdGV9KX0pCiAgICAuY2F0Y2goZSA9PiBjb25zb2xlLndhcm4oImZlZWRiYWNrIiwgZSkpOwogICQoIiNzdGF0dXMiKS50ZXh0Q29udGVudCA9ICIiOyAkKCIjc2VuZCIpLmRpc2FibGVkID0gZmFsc2U7ICQoIiNzdG9wIikuZGlzYWJsZWQgPSB0cnVlOyAkKCIjcSIpLmZvY3VzKCk7Cn0KCmZ1bmN0aW9uIGhhbmRsZShldiwgYm9keSwgbWV0YSkgewogIHN3aXRjaCAoZXYudHlwZSkgewogICAgY2FzZSAibWV0YSI6CiAgICAgIGlmIChldi5zYW1wbGVfcmF0ZSAhPT0gQS5zcikgeyBlbnN1cmVDdHgoZXYuc2FtcGxlX3JhdGUpOyBBLmN0eC5yZXN1bWUoKTsgfQogICAgICAkKCIjY2hpcCIpLnRleHRDb250ZW50ID0gZXYuYmFja2VuZDsgYnJlYWs7CiAgICBjYXNlICJ0ZXh0IjogYm9keS5ub2RlVmFsdWUgKz0gZXYuZGVsdGE7ICQoIiNsb2ciKS5zY3JvbGxUb3AgPSAxZTk7IGJyZWFrOwogICAgY2FzZSAic3RhdHVzIjogJCgiI3N0YXR1cyIpLnRleHRDb250ZW50ID0gIlByZXBhcmluZyB2b2ljZSwgc2VudGVuY2UgIiArIChldi5pICsgMSk7IGJyZWFrOwogICAgY2FzZSAic2VudGVuY2VfYXVkaW8iOiB7CiAgICAgIGNvbnN0IGYgPSBmYWRlKHBjbVRvRjMyKGV2LnBjbSksIDMsIDMpOwogICAgICBpZiAoQS5zdGFydGVkKSBzY2hlZHVsZShmKTsgZWxzZSBBLnF1ZXVlLnB1c2goZik7CiAgICAgIGJyZWFrOyB9CiAgICBjYXNlICJnbyI6IGdvKCk7ICQoIiNzdGF0dXMiKS50ZXh0Q29udGVudCA9ICJTcGVha2luZyI7IGJyZWFrOwogICAgY2FzZSAiYXVkaW9fZnVsbCI6IHsKICAgICAgY29uc3QgZiA9IGZhZGUocGNtVG9GMzIoZXYucGNtKSwgNSwgMTUpOwogICAgICBBLnF1ZXVlLnB1c2goZik7IGdvKCk7ICQoIiNzdGF0dXMiKS50ZXh0Q29udGVudCA9ICJTcGVha2luZyI7IGJyZWFrOyB9CiAgICBjYXNlICJlcnJvciI6IG1ldGEudGV4dENvbnRlbnQgPSBldi5tZXNzYWdlOyBicmVhazsKICAgIGNhc2UgIm1ldHJpY3MiOiB7CiAgICAgIGNvbnN0IG0gPSBldjsKICAgICAgbWV0YS50ZXh0Q29udGVudCA9IChtLnRva19zID8/ICI/IikgKyAiIHRvay9zLCBmaXJzdCB0ZXh0ICIgKyAobS50dGZ0X3MgPz8gIj8iKSArICIgcywgdm9pY2Ugc3RhcnRzICIgKyAobS5hdWRpb19zdGFydF9zID8/ICJuL2EiKSArICIgcywgdm9pY2UgYnVpbHQgYXQgIiArIChtLnR0c193YWxsX3Blcl9hdWRpbyA/PyAibi9hIikgKyAieCBhdWRpbyBsZW5ndGgsICIgKyBtLm1vZGU7CiAgICAgIGJyZWFrOyB9CiAgICBjYXNlICJkb25lIjogYnJlYWs7CiAgfQp9CgokKCIjZiIpLmFkZEV2ZW50TGlzdGVuZXIoInN1Ym1pdCIsIGUgPT4geyBlLnByZXZlbnREZWZhdWx0KCk7IGNvbnN0IHQgPSAkKCIjcSIpLnZhbHVlLnRyaW0oKTsgaWYgKCF0KSByZXR1cm47ICQoIiNxIikudmFsdWUgPSAiIjsgc2VuZCh0KTsgfSk7CiQoIiNzdG9wIikuYWRkRXZlbnRMaXN0ZW5lcigiY2xpY2siLCAoKSA9PiB7IGZldGNoKCIvY2FuY2VsIiwge21ldGhvZDogIlBPU1QifSk7IGlmIChjdHJsKSBjdHJsLmFib3J0KCk7IHJlc2V0QXVkaW8oKTsgfSk7CiQoIiNyZXNldCIpLmFkZEV2ZW50TGlzdGVuZXIoImNsaWNrIiwgYXN5bmMgKCkgPT4geyBhd2FpdCBmZXRjaCgiL3Jlc2V0Iiwge21ldGhvZDogIlBPU1QifSk7ICQoIiNsb2ciKS5pbm5lckhUTUwgPSAiIjsgcmVzZXRBdWRpbygpOyB9KTsKZmV0Y2goIi9jb25maWciKS50aGVuKHIgPT4gci5qc29uKCkpLnRoZW4oYyA9PiB7IEEuc3IgPSBjLnNhbXBsZV9yYXRlOyAkKCIjbW9kZSIpLnZhbHVlID0gYy5hdWRpb19tb2RlOyAkKCIjY2hpcCIpLnRleHRDb250ZW50ID0gYy5iYWNrZW5kICsgIiAvICIgKyBjLm1vZGVsOyB9KQogIC5jYXRjaChlID0+IHsgJCgiI2NoaXAiKS50ZXh0Q29udGVudCA9ICJzZXJ2ZXIgbm90IHJlYWNoYWJsZSI7IGNvbnNvbGUuZXJyb3IoZSk7IH0pOwo8L3NjcmlwdD48L2JvZHk+PC9odG1sPgo="
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

    @app.get("/fillers")
    def fillers():
        return FILLERS

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
    import uvicorn, socket
    global PORT
    # A previous run in this same kernel may still hold the port (its daemon thread
    # survives re-runs and %run gives each run fresh globals). Never fight for it.
    for cand in range(PORT, PORT + 10):
        sk = socket.socket()
        try:
            sk.bind(("127.0.0.1", cand)); free = True
        except OSError:
            free = False
        finally:
            sk.close()
        if free:
            if cand != PORT:
                log(f"port {PORT} is busy (stale server from an earlier run); using {cand}")
            PORT = cand
            break
    else:
        raise RuntimeError("no free port in range; restart the runtime")
    cfg = uvicorn.Config(build_app(), host="127.0.0.1", port=PORT, log_config=LOG_CONFIG,
                         access_log=False, log_level="critical")
    SERVER = uvicorn.Server(cfg)
    errs = []
    def _run():
        try:
            SERVER.run()
        except BaseException as e:   # SystemExit on bind failure is otherwise invisible (loggers are silenced)
            errs.append(repr(e)); print(f"[uvicorn] crashed: {e!r}", flush=True)
    th = threading.Thread(target=_run, daemon=True, name="neo-uvicorn")
    th.start()
    for _ in range(150):
        if alive(): break
        if not th.is_alive():
            raise RuntimeError(f"uvicorn thread exited before serving: {errs or 'no exception (port bind failed?)'}")
        time.sleep(0.2)
    else:
        raise RuntimeError("server not healthy after 30 s")
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
        print("\n" + "=" * 64 + f"\n  neo-tts is live: {url}\n  audio mode: {AUDIO_MODE}   tts backend: {TTS.name if TTS else TTS_BACKEND}"
              "\n  stop with stop_server()\n" + "=" * 64, flush=True)
        if _FRESH and os.environ.get("NEO_SELFTEST", "1") == "1":
            selftest()
        print("SCRIPT COMPLETE", flush=True)
    finally:
        sys.stdout.flush()
