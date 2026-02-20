"""KittenTTS Studio — Flask API Server"""

import argparse
import json
import os
import re
import socket
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
    return jsonify({"status": "ok", "port": request.host.split(":")[-1]})


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
    audio = m.generate(prompt, voice=voice)
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
        "words": words,
        "approx_tokens": approx_tokens,
    }
    with open(os.path.join(AUDIO_DIR, json_name), "w") as f:
        json.dump(metadata, f, indent=2)

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
