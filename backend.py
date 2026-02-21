"""KittenTTS Studio — Flask API Server"""

import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import time
import threading
from datetime import datetime
from queue import Queue

import numpy as np
import soundfile as sf
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "audio")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
os.makedirs(AUDIO_DIR, exist_ok=True)

MODELS = {
    "mini": {
        "name": "Kitten TTS Mini",
        "params": "80M",
        "size": "80MB",
        "repo": "KittenML/kitten-tts-mini-0.8",
    },
    "micro": {
        "name": "Kitten TTS Micro",
        "params": "40M",
        "size": "41MB",
        "repo": "KittenML/kitten-tts-micro-0.8",
    },
    "nano": {
        "name": "Kitten TTS Nano",
        "params": "15M",
        "size": "56MB",
        "repo": "KittenML/kitten-tts-nano-0.8",
    },
    "nano-int8": {
        "name": "Kitten TTS Nano INT8",
        "params": "15M",
        "size": "19MB",
        "repo": "KittenML/kitten-tts-nano-0.8-int8",
    },
    "nano-fp32": {
        "name": "Kitten TTS Nano FP32",
        "params": "15M",
        "size": "~56MB",
        "repo": "KittenML/kitten-tts-nano-0.8-fp32",
    },
}

VOICES = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]

# Cache of loaded KittenTTS model instances: {model_id: KittenTTS}
loaded_models = {}
model_lock = threading.Lock()

# Alignment model (stable-ts / Whisper) — optional feature
alignment_model = None
alignment_lock = threading.Lock()
alignment_available = None  # None = not checked yet, True/False after first check
alignment_tasks = {}        # {basename: threading.Thread}
alignment_tasks_lock = threading.Lock()

# Enhancement model (LavaSR) — optional feature
enhance_model = None
enhance_lock = threading.Lock()
enhance_available = None  # None = not checked yet, True/False after first check
enhance_tasks = {}        # {basename: threading.Thread}
enhance_tasks_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_filename(prompt: str) -> str:
    excerpt = re.sub(r"[^a-zA-Z0-9]+", "-", prompt[:30].lower()).strip("-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{excerpt}_{timestamp}"


def find_available_port(start: int = 5000) -> int:
    port = start
    while port < 65535:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    return start


def is_model_cached(repo_id: str) -> bool:
    """Check if config.json is already in the HF cache (quick proxy check)."""
    result = try_to_load_from_cache(repo_id, "config.json")
    return result is not None and not isinstance(result, str) is False


def get_model_files(repo_id: str) -> list[str]:
    """Download config.json (usually cached) and return [config, model_file, voices]."""
    config_path = try_to_load_from_cache(repo_id, "config.json")
    if config_path and isinstance(config_path, str):
        with open(config_path, "r") as f:
            cfg = json.load(f)
        return ["config.json", cfg["model_file"], cfg["voices"]]
    return ["config.json"]


def load_model(model_id: str):
    """Load (or return cached) KittenTTS model instance."""
    if model_id in loaded_models:
        return loaded_models[model_id]

    repo = MODELS[model_id]["repo"]
    from kittentts import KittenTTS

    with model_lock:
        if model_id not in loaded_models:
            loaded_models[model_id] = KittenTTS(repo)
    return loaded_models[model_id]


# ---------------------------------------------------------------------------
# SSE Progress tqdm
# ---------------------------------------------------------------------------


class SSEProgressCapture(tqdm):
    """Custom tqdm that pushes progress events to a Queue for SSE streaming."""

    progress_queue: Queue | None = None

    def __init__(self, *args, **kwargs):
        self._sse_queue = kwargs.pop("sse_queue", None) or getattr(
            SSEProgressCapture, "progress_queue", None
        )
        kwargs.pop("name", None)
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        super().update(n)
        if self._sse_queue is None:
            return
        total = self.total or 0
        downloaded = self.n or 0
        progress = int((downloaded / total) * 100) if total else 0
        speed = self.format_dict.get("rate", 0) or 0

        if speed >= 1_000_000:
            speed_str = f"{speed / 1_000_000:.1f}MB/s"
        elif speed >= 1_000:
            speed_str = f"{speed / 1_000:.1f}KB/s"
        else:
            speed_str = f"{speed:.0f}B/s"

        if total >= 1_000_000:
            size_str = f"{total / 1_000_000:.2f}MB"
        elif total >= 1_000:
            size_str = f"{total / 1_000:.1f}KB"
        else:
            size_str = f"{total}B"

        event = {
            "phase": "downloading",
            "file": self.desc or "unknown",
            "progress": progress,
            "downloaded_mb": round(downloaded / 1_000_000, 2),
            "total_mb": round(total / 1_000_000, 2),
            "size": size_str,
            "speed": speed_str,
        }
        self._sse_queue.put(event)


# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder=None)
CORS(app)


# --- Serve frontend ---
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


# --- Health ---
@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "port": request.host.split(":")[-1], "ffmpeg": _find_ffmpeg() is not None, "alignment": _check_alignment_available(), "enhance": _check_enhance_available()})


# --- Models ---
@app.route("/api/models")
def models():
    out = []
    for mid, m in MODELS.items():
        out.append({"id": mid, **m})
    return jsonify(out)


# --- Voices ---
@app.route("/api/voices")
def voices():
    return jsonify(VOICES)


# --- Model status ---
@app.route("/api/model-status/<model_id>")
def model_status(model_id):
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model"}), 404
    repo = MODELS[model_id]["repo"]
    cached = try_to_load_from_cache(repo, "config.json")
    is_cached = cached is not None and isinstance(cached, str)

    files = []
    if is_cached:
        files = get_model_files(repo)
    return jsonify({"model_id": model_id, "cached": is_cached, "files": files})


# --- Download model with SSE progress ---
@app.route("/api/download-model/<model_id>")
def download_model(model_id):
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model"}), 404

    repo = MODELS[model_id]["repo"]

    def _download_file(repo, filename, result):
        """Run hf_hub_download in a thread, storing result or exception."""
        try:
            path = hf_hub_download(
                repo_id=repo, filename=filename, tqdm_class=SSEProgressCapture
            )
            result["path"] = path
        except Exception as e:
            result["error"] = e

    def _stream_download(repo, filename, q):
        """Start download in thread, yield SSE events as they arrive."""
        result = {}
        t = threading.Thread(target=_download_file, args=(repo, filename, result))
        t.start()
        while t.is_alive():
            t.join(timeout=0.15)
            while not q.empty():
                yield f"data: {json.dumps(q.get())}\n\n"
        # Drain remaining events
        while not q.empty():
            yield f"data: {json.dumps(q.get())}\n\n"
        if "error" in result:
            raise result["error"]

    def stream():
        q = Queue()
        yield f"data: {json.dumps({'phase': 'checking', 'model': model_id})}\n\n"

        try:
            SSEProgressCapture.progress_queue = q

            # Step 1: download config.json
            for event in _stream_download(repo, "config.json", q):
                yield event

            config_path = try_to_load_from_cache(repo, "config.json")
            if not isinstance(config_path, str):
                raise RuntimeError("Failed to download config.json")

            # Step 2: read config to get model_file and voices filenames
            with open(config_path, "r") as f:
                cfg = json.load(f)
            model_file = cfg["model_file"]
            voices_file = cfg["voices"]

            # Step 3: download model ONNX
            for event in _stream_download(repo, model_file, q):
                yield event

            # Step 4: download voices
            for event in _stream_download(repo, voices_file, q):
                yield event

            SSEProgressCapture.progress_queue = None

            # Step 5: load model into memory
            yield f"data: {json.dumps({'phase': 'loading', 'message': 'Loading model...'})}\n\n"
            load_model(model_id)

            yield f"data: {json.dumps({'phase': 'ready', 'message': 'Model ready'})}\n\n"

        except Exception as e:
            SSEProgressCapture.progress_queue = None
            yield f"data: {json.dumps({'phase': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# --- Generate audio ---
@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    model_id = data.get("model", "mini")
    voice = data.get("voice", "Jasper")
    prompt = data.get("prompt", "")
    speed = float(data.get("speed", 1.0))
    speed = max(0.5, min(2.0, speed))  # clamp to 0.5–2.0

    if not prompt.strip():
        return jsonify({"error": "Prompt is required"}), 400
    if model_id not in MODELS:
        return jsonify({"error": "Unknown model"}), 404
    if voice not in VOICES:
        return jsonify({"error": f"Unknown voice. Choose from: {VOICES}"}), 400

    m = load_model(model_id)

    words = len(prompt.split())
    approx_tokens = int(words * 1.3)

    start = time.perf_counter()
    try:
        audio = m.generate(prompt, voice=voice, speed=speed)
    except Exception as e:
        print(f"[generate] TTS inference failed: {e}")
        return jsonify({"error": f"Generation failed: {e}"}), 500
    end = time.perf_counter()

    duration_generated = len(audio) / 24000
    inference_time = end - start
    rtf = inference_time / duration_generated

    basename = generate_filename(prompt)
    wav_name = f"{basename}.wav"
    json_name = f"{basename}.json"

    sf.write(os.path.join(AUDIO_DIR, wav_name), audio, 24000)

    metadata = {
        "filename": wav_name,
        "prompt": prompt.strip(),
        "model": MODELS[model_id]["repo"],
        "model_id": model_id,
        "voice": voice,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "inference_time": round(inference_time, 3),
        "rtf": round(rtf, 4),
        "duration_seconds": round(duration_generated, 2),
        "sample_rate": 24000,
        "speed": speed,
        "words": words,
        "approx_tokens": approx_tokens,
    }
    with open(os.path.join(AUDIO_DIR, json_name), "w") as f:
        json.dump(metadata, f, indent=2)

    # Kick off background alignment and enhancement
    _start_alignment(basename)
    metadata["alignment_status"] = "pending" if _check_alignment_available() else "unavailable"
    _start_enhancement(basename)
    metadata["enhance_status"] = "pending" if _check_enhance_available() else "unavailable"

    return jsonify(metadata)


# --- List audio files ---
@app.route("/api/audio")
def list_audio():
    files = []
    if not os.path.exists(AUDIO_DIR):
        return jsonify(files)
    for fname in sorted(os.listdir(AUDIO_DIR), reverse=True):
        if fname.endswith(".json"):
            with open(os.path.join(AUDIO_DIR, fname), "r") as f:
                files.append(json.load(f))
    return jsonify(files)


# --- Delete audio file ---
@app.route("/api/audio/<filename>", methods=["DELETE"])
def delete_audio(filename):
    wav_path = os.path.join(AUDIO_DIR, filename)
    json_path = wav_path.rsplit(".", 1)[0] + ".json"
    deleted = False
    for p in [wav_path, json_path]:
        if os.path.exists(p):
            os.remove(p)
            deleted = True
    if deleted:
        return jsonify({"status": "deleted", "filename": filename})
    return jsonify({"error": "File not found"}), 404


# --- Word alignment for karaoke ---
@app.route("/api/audio/<filename>/alignment")
def get_alignment(filename):
    """Return alignment data, triggering retroactive alignment for old files."""
    if not filename.endswith(".wav"):
        return jsonify({"error": "Expected .wav filename"}), 400

    basename = filename.rsplit(".", 1)[0]
    json_path = os.path.join(AUDIO_DIR, basename + ".json")

    if not os.path.exists(json_path):
        return jsonify({"error": "Metadata not found"}), 404

    if not _check_alignment_available():
        return jsonify({"status": "unavailable"})

    with open(json_path, "r") as f:
        metadata = json.load(f)

    status = metadata.get("alignment_status")

    if status == "ready":
        return jsonify({
            "status": "ready",
            "word_alignment": metadata.get("word_alignment", []),
        })

    if status == "failed":
        # Retry — previous failure may have been due to a fixable issue
        _start_alignment(basename)
        return jsonify({"status": "aligning"})

    if status == "aligning":
        # Check if thread is actually still running (may have crashed)
        with alignment_tasks_lock:
            if basename not in alignment_tasks:
                _start_alignment(basename)
        return jsonify({"status": "aligning"})

    # No alignment attempted yet (old file) — trigger retroactive alignment
    _start_alignment(basename)
    return jsonify({"status": "aligning"})


# --- Audio enhancement status ---
@app.route("/api/audio/<filename>/enhance-status")
def get_enhance_status(filename):
    """Return enhancement status, triggering retroactive enhancement for old files."""
    if not filename.endswith(".wav"):
        return jsonify({"error": "Expected .wav filename"}), 400

    basename = filename.rsplit(".", 1)[0]
    json_path = os.path.join(AUDIO_DIR, basename + ".json")

    if not os.path.exists(json_path):
        return jsonify({"error": "Metadata not found"}), 404

    if not _check_enhance_available():
        return jsonify({"status": "unavailable"})

    with open(json_path, "r") as f:
        metadata = json.load(f)

    status = metadata.get("enhance_status")

    if status == "ready" and metadata.get("enhanced_filename"):
        enhanced_path = os.path.join(AUDIO_DIR, metadata["enhanced_filename"])
        if os.path.exists(enhanced_path):
            return jsonify({
                "status": "ready",
                "enhanced_filename": metadata["enhanced_filename"],
            })
        # File missing — re-enhance
        _start_enhancement(basename)
        return jsonify({"status": "enhancing"})

    if status == "failed":
        _start_enhancement(basename)
        return jsonify({"status": "enhancing"})

    if status == "enhancing":
        with enhance_tasks_lock:
            if basename not in enhance_tasks:
                _start_enhancement(basename)
        return jsonify({"status": "enhancing"})

    # No enhancement attempted yet (old file) — trigger retroactive
    _start_enhancement(basename)
    return jsonify({"status": "enhancing"})


# --- Locate ffmpeg helper ---
def _find_ffmpeg():
    bin_dir = os.path.join(os.path.dirname(__file__), "bin")
    local = os.path.join(bin_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    return local if os.path.isfile(local) else shutil.which("ffmpeg")


ALIGNMENT_VERSION = 2  # Bump when alignment logic changes to invalidate cached data

# --- Alignment helpers (stable-ts) ---

def _check_alignment_available():
    """Check if stable-ts is importable. Cached after first call."""
    global alignment_available
    if alignment_available is not None:
        return alignment_available
    try:
        import stable_whisper  # noqa: F401
        alignment_available = True
    except ImportError:
        alignment_available = False
    return alignment_available


def _load_alignment_model():
    """Load (or return cached) stable-ts Whisper tiny.en model."""
    global alignment_model
    if alignment_model is not None:
        return alignment_model
    import stable_whisper
    with alignment_lock:
        if alignment_model is None:
            alignment_model = stable_whisper.load_model("tiny.en")
    return alignment_model


def _run_alignment(wav_path, prompt_text):
    """Run forced alignment. Returns list of {word, begin, end} or None."""
    try:
        model = _load_alignment_model()
        # Load audio as numpy array to avoid stable-ts needing ffmpeg in PATH
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        # Whisper expects 16kHz — resample if needed (KittenTTS outputs 24kHz)
        if sr != 16000:
            target_len = int(len(audio) * 16000 / sr)
            audio = np.interp(
                np.linspace(0, len(audio), target_len, endpoint=False),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        result = model.align(audio, prompt_text, language="en", fast_mode=True)
        alignment = []
        for w in result.all_words():
            word_text = w.word.strip()
            if word_text:
                alignment.append({
                    "word": word_text,
                    "begin": round(w.start, 3),
                    "end": round(w.end, 3),
                })
        return alignment if alignment else None
    except Exception as e:
        print(f"[alignment] Error aligning {wav_path}: {e}")
        return None


def _audio_hash(wav_path):
    """Compute SHA-256 hash of audio file for cache validation."""
    h = hashlib.sha256()
    with open(wav_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _background_align(basename):
    """Run alignment in background thread, update metadata JSON when done."""
    json_path = os.path.join(AUDIO_DIR, basename + ".json")
    wav_path = os.path.join(AUDIO_DIR, basename + ".wav")

    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        return

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Check if alignment is cached, audio hasn't changed, and version matches
        current_hash = _audio_hash(wav_path)
        if (metadata.get("alignment_status") == "ready"
                and metadata.get("audio_hash") == current_hash
                and metadata.get("alignment_version") == ALIGNMENT_VERSION
                and metadata.get("word_alignment")):
            return

        metadata["alignment_status"] = "aligning"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        prompt_text = metadata.get("prompt", "")
        if not prompt_text.strip():
            metadata["alignment_status"] = "failed"
            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)
            return

        alignment = _run_alignment(wav_path, prompt_text)

        if alignment:
            metadata["alignment_status"] = "ready"
            metadata["word_alignment"] = alignment
            metadata["audio_hash"] = current_hash
            metadata["alignment_version"] = ALIGNMENT_VERSION
        else:
            metadata["alignment_status"] = "failed"

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"[alignment] Background align failed for {basename}: {e}")
    finally:
        with alignment_tasks_lock:
            alignment_tasks.pop(basename, None)


def _start_alignment(basename):
    """Spawn alignment thread if not already running for this file."""
    if not _check_alignment_available():
        return
    with alignment_tasks_lock:
        if basename in alignment_tasks:
            return
        t = threading.Thread(target=_background_align, args=(basename,), daemon=True)
        alignment_tasks[basename] = t
        t.start()


# --- Enhancement helpers (LavaSR) ---

def _check_enhance_available():
    """Check if LavaSR is importable. Cached after first call."""
    global enhance_available
    if enhance_available is not None:
        return enhance_available
    try:
        from LavaSR.model import LavaEnhance  # noqa: F401
        enhance_available = True
    except ImportError:
        enhance_available = False
    return enhance_available


def _load_enhance_model():
    """Load (or return cached) LavaSR enhancement model."""
    global enhance_model
    if enhance_model is not None:
        return enhance_model
    from LavaSR.model import LavaEnhance
    with enhance_lock:
        if enhance_model is None:
            enhance_model = LavaEnhance("YatharthS/LavaSR", "cpu")
    return enhance_model


def _run_enhance(wav_path):
    """Enhance audio file. Returns enhanced filename or None."""
    try:
        model = _load_enhance_model()
        audio, sr = model.load_audio(wav_path)
        enhanced = model.enhance(audio)
        enhanced_np = enhanced.cpu().numpy().squeeze()

        basename = os.path.splitext(os.path.basename(wav_path))[0]
        enhanced_name = f"{basename}_enhanced.wav"
        enhanced_path = os.path.join(AUDIO_DIR, enhanced_name)
        sf.write(enhanced_path, enhanced_np, 48000)
        return enhanced_name
    except Exception as e:
        print(f"[enhance] Error enhancing {wav_path}: {e}")
        return None


def _background_enhance(basename):
    """Run enhancement in background thread, update metadata JSON when done."""
    json_path = os.path.join(AUDIO_DIR, basename + ".json")
    wav_path = os.path.join(AUDIO_DIR, basename + ".wav")

    if not os.path.exists(json_path) or not os.path.exists(wav_path):
        return

    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)

        # Skip if already enhanced and file exists
        if (metadata.get("enhance_status") == "ready"
                and metadata.get("enhanced_filename")
                and os.path.exists(os.path.join(AUDIO_DIR, metadata["enhanced_filename"]))):
            return

        metadata["enhance_status"] = "enhancing"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        enhanced_name = _run_enhance(wav_path)

        if enhanced_name:
            metadata["enhance_status"] = "ready"
            metadata["enhanced_filename"] = enhanced_name
        else:
            metadata["enhance_status"] = "failed"

        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

    except Exception as e:
        print(f"[enhance] Background enhance failed for {basename}: {e}")
    finally:
        with enhance_tasks_lock:
            enhance_tasks.pop(basename, None)


def _start_enhancement(basename):
    """Spawn enhancement thread if not already running for this file."""
    if not _check_enhance_available():
        return
    with enhance_tasks_lock:
        if basename in enhance_tasks:
            return
        t = threading.Thread(target=_background_enhance, args=(basename,), daemon=True)
        enhance_tasks[basename] = t
        t.start()


# --- Serve cached MP3 ---
@app.route("/api/audio/<filename>/mp3")
def serve_mp3(filename):
    if not filename.endswith(".wav"):
        return jsonify({"error": "Only .wav files can be converted"}), 400
    mp3_name = filename.rsplit(".", 1)[0] + ".mp3"
    mp3_path = os.path.join(AUDIO_DIR, mp3_name)
    if not os.path.exists(mp3_path):
        return jsonify({"error": "MP3 not found — convert first"}), 404
    return send_from_directory(AUDIO_DIR, mp3_name, as_attachment=True)


# --- Convert WAV to MP3 with SSE progress ---
@app.route("/api/audio/<filename>/mp3-convert")
def convert_to_mp3(filename):
    if not filename.endswith(".wav"):
        return jsonify({"error": "Only .wav files can be converted"}), 400
    wav_path = os.path.join(AUDIO_DIR, filename)
    if not os.path.exists(wav_path):
        return jsonify({"error": "File not found"}), 404

    mp3_name = filename.rsplit(".", 1)[0] + ".mp3"
    mp3_path = os.path.join(AUDIO_DIR, mp3_name)

    # Already converted — instant done
    if os.path.exists(mp3_path):
        def _done():
            yield f"data: {json.dumps({'phase': 'done', 'progress': 100})}\n\n"
        return Response(
            _done(), mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return jsonify({"error": "ffmpeg not found. Place ffmpeg in bin/ or install it system-wide."}), 501

    # Get total duration for progress calculation
    total_duration = 0.0
    json_path = wav_path.rsplit(".", 1)[0] + ".json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            total_duration = json.load(f).get("duration_seconds", 0.0)
    if total_duration <= 0:
        try:
            info = sf.info(wav_path)
            total_duration = info.duration
        except Exception:
            pass

    def stream():
        yield f"data: {json.dumps({'phase': 'converting', 'progress': 0})}\n\n"

        proc = subprocess.Popen(
            [ffmpeg, "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2",
             "-progress", "pipe:1", "-nostats", "-y", mp3_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )

        last_pct = 0
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("out_time_us="):
                try:
                    us = int(line.split("=", 1)[1])
                    if total_duration > 0:
                        pct = min(99, int((us / 1_000_000) / total_duration * 100))
                        if pct > last_pct:
                            last_pct = pct
                            yield f"data: {json.dumps({'phase': 'converting', 'progress': pct})}\n\n"
                except (ValueError, ZeroDivisionError):
                    pass
            elif line == "progress=end":
                break

        proc.wait(timeout=30)

        if proc.returncode == 0:
            yield f"data: {json.dumps({'phase': 'done', 'progress': 100})}\n\n"
        else:
            err = proc.stderr.read()[:200] if proc.stderr else "Unknown error"
            yield f"data: {json.dumps({'phase': 'error', 'message': err})}\n\n"

    return Response(
        stream(), mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )


# --- Serve audio files ---
@app.route("/audio/<path:filename>")
def serve_audio(filename):
    return send_from_directory(AUDIO_DIR, filename)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KittenTTS Studio Backend")
    parser.add_argument("--port", type=int, default=0, help="Port to listen on")
    args = parser.parse_args()

    port = args.port if args.port else find_available_port(5000)
    print(f"\n>>> KittenTTS Studio running on http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
