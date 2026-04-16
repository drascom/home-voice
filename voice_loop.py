#!/usr/bin/env python3
"""Voice Loop — a minimal on-device voice agent. macOS Apple Silicon + Linux Nvidia GPU.

Supports multiple backends for LLM. faster-whisper for STT. Piper or XTTS-v2 for TTS.

Usage:
    uv run voice_loop.py                                  # OS defaults (Piper TTS)
    uv run voice_loop.py --lang tr --tts-engine xtts      # Turkish with XTTS-v2 (streaming + voice interrupt)
    uv run voice_loop.py --tts-engine xtts --speaker-wav ref.wav  # XTTS-v2 with voice cloning
    uv run voice_loop.py --backend api --base-url http://localhost:11434/v1 --model gemma3:4b
"""

import argparse
import asyncio
import os
import platform
import queue
import select
import sys
import tempfile
import termios
import time as _time
import tty
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import shutil
import urllib.request

import numpy as np
import sounddevice as sd
sd.default.latency = 'high'
import torch

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # 32ms at 16kHz (required by Silero VAD)
MAX_HISTORY = 10
CHIME_SR = 24000
_DIR = Path(__file__).parent
IS_MAC = platform.system() == "Darwin"
IS_LINUX = platform.system() == "Linux"


def _download(url, dest, label=None):
    """Download a file with a progress bar."""
    label = label or os.path.basename(dest)
    cols = shutil.get_terminal_size().columns
    bar_width = max(20, cols - 45)

    def _reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            filled = bar_width * downloaded // total_size
            bar = "=" * filled + "-" * (bar_width - filled)
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r  {label}: [{bar}] {pct}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        else:
            mb_done = downloaded / (1024 * 1024)
            print(f"\r  {label}: {mb_done:.1f} MB downloaded", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    print(flush=True)  # newline after progress bar


def load_system_prompt(include_memory: bool = False) -> str:
    names = ("SOUL.md", "MEMORY.md") if include_memory else ("SOUL.md",)
    parts = [(_DIR / n).read_text().strip() for n in names if (_DIR / n).exists()]
    return "\n\n".join(p for p in parts if p)


def _fade_tone(freq, dur, amp=0.6):
    n = int(dur * CHIME_SR)
    t = np.linspace(0, dur, n, dtype=np.float32)
    env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / (n - 1)))
    return amp * np.sin(2 * np.pi * freq * t) * env

def _silence(dur):
    return np.zeros(int(dur * CHIME_SR), dtype=np.float32)

def make_chime(duration=30.0, tick_every=1.5):
    head = np.concatenate([_fade_tone(880, 0.09), _silence(0.03), _fade_tone(1320, 0.10)])
    tick = _fade_tone(550, 0.04, amp=0.18)
    total = int(duration * CHIME_SR)
    buf = np.zeros(total, dtype=np.float32)
    buf[:len(head)] = head
    step = int(tick_every * CHIME_SR)
    for pos in range(len(head), total, step):
        end = min(pos + len(tick), total)
        buf[pos:end] = tick[:end - pos]
    return buf


def save_wav(audio, sr=SAMPLE_RATE):
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((audio * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
    return path


def load_smart_turn():
    import onnxruntime as ort
    from transformers import WhisperFeatureExtractor
    model_path = os.path.join(tempfile.gettempdir(), "smart_turn_v3", "smart_turn_v3.2_cpu.onnx")
    if not os.path.exists(model_path):
        print("Downloading Smart Turn v3.2 model...", flush=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        _download(
            "https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx",
            model_path, "smart-turn-v3.2")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

    def predict(audio_float32: np.ndarray) -> float:
        max_samples = 8 * SAMPLE_RATE
        audio_float32 = audio_float32[-max_samples:]
        features = extractor(
            audio_float32, sampling_rate=SAMPLE_RATE, max_length=max_samples,
            padding="max_length", return_attention_mask=False, return_tensors="np",
        )
        return float(session.run(None, {"input_features": features.input_features.astype(np.float32)})[0].flatten()[0])
    return predict

def _vad_prob(vad, chunk):
    p = vad(torch.from_numpy(chunk), SAMPLE_RATE)
    return p.item() if hasattr(p, "item") else p

def _get_ref_segment(tts_concat, pos, length):
    if pos >= len(tts_concat):
        return np.zeros(length, dtype=np.float32)
    seg = tts_concat[pos:pos + length]
    return np.concatenate([seg, np.zeros(length - len(seg), dtype=np.float32)]) if len(seg) < length else seg


# ---------------------------------------------------------------------------
# LLM Backends
# ---------------------------------------------------------------------------

def load_llm_mlx(model_id):
    from mlx_vlm import load, generate as mlx_generate
    model, processor = load(model_id)

    def generate(messages, max_tokens=200, temperature=0.7, **kwargs):
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        r = mlx_generate(model, processor, prompt, max_tokens=max_tokens,
                         temperature=temperature, repetition_penalty=1.2, verbose=False, **kwargs)
        return r.text if hasattr(r, "text") else str(r)
    return generate


def load_llm_local(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading {model_id} via transformers (CUDA)...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)

    def generate(messages, max_tokens=200, temperature=0.7, **kwargs):
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature,
                                 do_sample=temperature > 0, repetition_penalty=1.2)
        return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return generate


def load_llm_api(model_id, base_url, api_key):
    import openai
    client = openai.OpenAI(base_url=base_url, api_key=api_key)

    def generate(messages, max_tokens=200, temperature=0.7, **kwargs):
        resp = client.chat.completions.create(
            model=model_id, messages=messages,
            max_tokens=max_tokens, temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    return generate


# ---------------------------------------------------------------------------
# STT — faster-whisper
# ---------------------------------------------------------------------------

def load_stt_whisper(lang="en", model_size="medium"):
    from faster_whisper import WhisperModel
    device = "cuda" if not IS_MAC else "cpu"
    compute = "float16" if device == "cuda" else "int8"
    print(f"Loading faster-whisper {model_size} ({device}/{compute})...", flush=True)

    # faster-whisper suppresses download progress by default (disabled_tqdm).
    # Pre-download with huggingface_hub so the user sees a progress bar.
    repo_id = f"Systran/faster-whisper-{model_size}"
    try:
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(repo_id)
    except Exception:
        # Fall back to letting WhisperModel handle it
        model_path = model_size

    wm = WhisperModel(model_path, device=device, compute_type=compute)

    def transcribe(audio):
        segments, _ = wm.transcribe(audio, language=lang, beam_size=5)
        return " ".join(s.text for s in segments).strip()
    return transcribe


# ---------------------------------------------------------------------------
# TTS — Piper
# ---------------------------------------------------------------------------

# Piper model download URL patterns per language
_PIPER_MODELS = {
    "tr_TR-dfki-medium": "tr/tr_TR/dfki/medium",
    "en_US-lessac-medium": "en/en_US/lessac/medium",
    "en_GB-alba-medium": "en/en_GB/alba/medium",
    "es_ES-sharvard-medium": "es/es_ES/sharvard/medium",
    "fr_FR-siwis-medium": "fr/fr_FR/siwis/medium",
    "de_DE-thorsten-medium": "de/de_DE/thorsten/medium",
}

def load_tts_piper(piper_model="en_US-lessac-medium"):
    from piper import PiperVoice

    cache_dir = os.path.join(tempfile.gettempdir(), "piper_tts")
    os.makedirs(cache_dir, exist_ok=True)
    onnx_path = os.path.join(cache_dir, f"{piper_model}.onnx")
    json_path = os.path.join(cache_dir, f"{piper_model}.onnx.json")

    if not os.path.exists(onnx_path):
        url_path = _PIPER_MODELS.get(piper_model)
        if url_path is None:
            print(f"Error: unknown Piper model '{piper_model}'. Known: {', '.join(_PIPER_MODELS)}", file=sys.stderr)
            sys.exit(1)
        base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/{url_path}"
        print(f"Downloading Piper model {piper_model}...", flush=True)
        _download(f"{base}/{piper_model}.onnx", onnx_path, f"{piper_model}.onnx")
        _download(f"{base}/{piper_model}.onnx.json", json_path, f"{piper_model}.json")

    piper_voice = PiperVoice.load(onnx_path, json_path)

    def speak(text):
        import io
        wav_io = io.BytesIO()
        with wave.open(wav_io, "wb") as wf:
            piper_voice.synthesize(text, wf)
        wav_io.seek(0)
        with wave.open(wav_io, "rb") as wf:
            sr = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767
        sd.play(audio, sr); sd.wait()

    return speak


# Default Piper models per language code
_LANG_TO_PIPER = {
    "tr": "tr_TR-dfki-medium",
    "en": "en_US-lessac-medium",
    "es": "es_ES-sharvard-medium",
    "fr": "fr_FR-siwis-medium",
    "de": "de_DE-thorsten-medium",
}


# ---------------------------------------------------------------------------
# TTS — Coqui XTTS-v2 (streaming, Turkish, voice cloning)
# ---------------------------------------------------------------------------

XTTS_SR = 24000  # XTTS-v2 output sample rate

def load_tts_xtts(lang="en", speaker_wav=None):
    from TTS.api import TTS as CoquiTTS

    print("Loading XTTS-v2 (first run downloads ~1.8GB)...", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = CoquiTTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

    # Get the underlying model for streaming access
    model = tts.synthesizer.tts_model

    # Compute speaker conditioning once
    if speaker_wav:
        print(f"  Using speaker reference: {speaker_wav}", flush=True)
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[speaker_wav])
    else:
        # Use a built-in default speaker
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[])

    def speak(text):
        """Blocking full synthesis (used for greeting)."""
        wav = tts.tts(text=text, language=lang, speaker_wav=speaker_wav)
        audio = np.array(wav, dtype=np.float32)
        sd.play(audio, XTTS_SR); sd.wait()

    def create_stream(text):
        """Generator yielding (chunk_np_float32, sample_rate) tuples."""
        chunks = model.inference_stream(
            text, lang, gpt_cond_latent, speaker_embedding,
            stream_chunk_size=20, enable_text_splitting=True,
        )
        for chunk in chunks:
            chunk_np = chunk.cpu().numpy().squeeze().astype(np.float32)
            yield chunk_np, XTTS_SR

    return speak, create_stream


# ---------------------------------------------------------------------------
# Resolve API key
# ---------------------------------------------------------------------------

def _resolve_api_key(cli_key, base_url):
    if cli_key:
        return cli_key
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    if base_url and any(h in base_url for h in ("localhost", "127.0.0.1")):
        return "ollama"
    print("Error: --api-key or OPENAI_API_KEY required for remote --base-url", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os_label = "macOS" if IS_MAC else "Linux"
    default_backend = "mlx" if IS_MAC else "local"
    default_model = "mlx-community/gemma-4-E4B-it-4bit" if IS_MAC else "google/gemma-4-E2B-it"

    ap = argparse.ArgumentParser(description=f"Voice Loop — a minimal on-device voice agent ({os_label})")
    B = argparse.BooleanOptionalAction

    # LLM
    ap.add_argument("--backend", choices=["mlx", "local", "api"], default=default_backend,
                    help=f"LLM backend (default: {default_backend})")
    ap.add_argument("--model", default=default_model, help="Model ID")
    ap.add_argument("--base-url", default="http://localhost:11434/v1",
                    help="OpenAI-compatible API base URL (for --backend api)")
    ap.add_argument("--api-key", default=None, help="API key (or set OPENAI_API_KEY)")

    # Language
    ap.add_argument("--lang", default="en", help="Language for both STT and TTS (default: en)")

    # STT
    ap.add_argument("--stt-model", default="medium", help="Whisper model size (default: medium)")

    # TTS
    ap.add_argument("--tts", action=B, default=True, help="Enable/disable TTS")
    ap.add_argument("--tts-engine", choices=["piper", "xtts"], default="piper",
                    help="TTS engine: piper (CPU, lightweight) or xtts (GPU, streaming, voice cloning)")
    ap.add_argument("--piper-model", default=None,
                    help="Piper model name (default: auto from --lang)")
    ap.add_argument("--speaker-wav", default=None,
                    help="Reference WAV for XTTS voice cloning (optional)")

    # General
    ap.add_argument("--smart-turn", action=B, default=True, help="Smart Turn v3 endpoint detection")
    ap.add_argument("--aec", action=B, default=True, help="WebRTC AEC3 voice interrupt")
    ap.add_argument("--chime", action=B, default=True, help="Chime + ticks while generating")
    ap.add_argument("--memory", action="store_true", help="Read/write MEMORY.md")
    ap.add_argument("--silence-ms", type=int, default=700)
    ap.add_argument("--record", nargs="?", const="", metavar="FILE",
                    help="Record mic to WAV for debugging")

    args = ap.parse_args()

    # --- Platform guards ---
    if args.backend == "mlx" and not IS_MAC:
        print("Error: --backend mlx is only supported on macOS", file=sys.stderr); sys.exit(1)
    if args.backend == "local" and IS_MAC:
        print("Error: --backend local is only supported on Linux (CUDA)", file=sys.stderr); sys.exit(1)

    # --- Resolve Piper model from language ---
    if args.piper_model is None:
        args.piper_model = _LANG_TO_PIPER.get(args.lang, "en_US-lessac-medium")

    if args.record == "":
        tmp_dir = _DIR / "tmp"
        tmp_dir.mkdir(exist_ok=True)
        args.record = str(tmp_dir / f"recording-{_time.strftime('%Y%m%d-%H%M%S')}.wav")
    silence_limit = max(1, int(args.silence_ms / (CHUNK_SAMPLES / SAMPLE_RATE * 1000)))

    # --- Banner ---
    gpu_label = "Apple Silicon" if IS_MAC else "Nvidia GPU"
    lang_tag = f"  [{args.lang.upper()}]" if args.lang and args.lang != "en" else ""
    print(f"\nVoice Loop — {os_label} / {gpu_label}{lang_tag}")
    if args.backend == "api":
        print(f"  backend : api  ({args.base_url} / {args.model})")
    else:
        print(f"  backend : {args.backend}  ({args.model})")
    print(f"  stt     : whisper-{args.stt_model} ({args.lang})")
    if args.tts:
        if args.tts_engine == "xtts":
            clone_tag = f" clone:{args.speaker_wav}" if args.speaker_wav else ""
            print(f"  tts     : xtts-v2 ({args.lang}{clone_tag})")
        else:
            print(f"  tts     : piper ({args.piper_model})")
    else:
        print("  tts     : off")
    print()

    # --- Load VAD ---
    print("Loading Silero VAD...", flush=True)
    from silero_vad import load_silero_vad
    vad = load_silero_vad(onnx=True)

    # --- Load STT ---
    transcribe = load_stt_whisper(args.lang, args.stt_model)

    # --- Load LLM ---
    if args.backend == "mlx":
        print(f"Loading {args.model} (first run downloads ~3GB)...", flush=True)
        llm_generate = load_llm_mlx(args.model)
    elif args.backend == "local":
        llm_generate = load_llm_local(args.model)
    else:
        api_key = _resolve_api_key(args.api_key, args.base_url)
        print(f"Using OpenAI-compatible API at {args.base_url}...", flush=True)
        llm_generate = load_llm_api(args.model, args.base_url, api_key)

    # --- Load Smart Turn ---
    smart_turn = load_smart_turn() if args.smart_turn else None

    # --- Load TTS ---
    tts_speak = None
    tts_create_stream = None
    if args.tts:
        if args.tts_engine == "xtts":
            tts_speak, tts_create_stream = load_tts_xtts(args.lang, args.speaker_wav)
        else:
            print(f"Loading Piper TTS ({args.piper_model})...", flush=True)
            tts_speak = load_tts_piper(args.piper_model)

    # --- AEC ---
    make_aec_processor = None
    if args.aec:
        from livekit.rtc import AudioFrame
        from livekit.rtc.apm import AudioProcessingModule
        WF = 160  # 10ms @ 16kHz
        def _to_i16(x):
            s = (x * 32767).clip(-32768, 32767).astype(np.int16)
            return np.pad(s, (0, max(0, WF - len(s)))) if len(s) < WF else s
        def _frame(b):
            return AudioFrame(b.tobytes(), sample_rate=SAMPLE_RATE, num_channels=1, samples_per_channel=WF)
        def make_aec_processor():
            apm = AudioProcessingModule(echo_cancellation=True, noise_suppression=True)
            def process(mic, ref):
                cleaned = np.zeros_like(mic)
                for i in range(0, len(mic), WF):
                    mic_f = _frame(_to_i16(mic[i:i+WF]))
                    apm.process_reverse_stream(_frame(_to_i16(ref[i:i+WF])))
                    apm.process_stream(mic_f)
                    cleaned[i:i+WF] = (np.frombuffer(bytes(mic_f.data), dtype=np.int16).astype(np.float32) / 32767)[:len(mic[i:i+WF])]
                return cleaned
            return process
        print("  AEC: WebRTC AEC3 (LiveKit APM)")

    executor = ThreadPoolExecutor(max_workers=1)
    chime_sound = make_chime() if args.chime else None
    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    record_buf: list[np.ndarray] | None = [] if args.record else None

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        chunk = indata[:, 0].copy()
        if record_buf is not None:
            record_buf.append(chunk)
        audio_q.put(chunk)

    def drain_audio_q():
        while not audio_q.empty():
            audio_q.get_nowait()

    _mem_path = _DIR / "MEMORY.md"

    def _read_memory():
        return _mem_path.read_text() if _mem_path.exists() else "# Memory\n"

    def _run_memory(prompt, max_tokens, temperature, label):
        try:
            return llm_generate(
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=temperature,
            ).strip()
        except Exception as e:
            print(f"  [{label} failed: {e}]", file=sys.stderr)
            return None

    def update_memory(heard, response):
        result = _run_memory(
            f"Current memory:\n{_read_memory()}\n\n"
            f"User said: {heard}\n\n"
            "Did the user state a new durable fact about themselves? "
            "If yes, output one short fact per line starting with '- '. "
            "If no, output ONLY: NONE. Do not invent facts.",
            max_tokens=60, temperature=0.2, label="memory update",
        )
        if result and "NONE" not in result.upper():
            lines = [l for l in result.splitlines() if l.strip().startswith("-")]
            if lines:
                with open(_mem_path, "a") as f:
                    f.write("\n" + "\n".join(lines) + "\n")
                print(f"  [memory +{len(lines)}]", flush=True)

    def consolidate_memory():
        if not _mem_path.exists():
            return
        result = _run_memory(
            f"Here is a memory file about a user:\n\n{_read_memory()}\n\n"
            "Rewrite it: merge duplicates, remove transient/session-specific "
            "items (questions asked, topics discussed, tests), keep only "
            "durable facts (identity, preferences, relationships, location, "
            "ongoing projects). Output the cleaned file, starting with '# Memory' "
            "followed by bullets starting with '- '. No explanation.",
            max_tokens=300, temperature=0.2, label="memory consolidation",
        )
        if result and result.startswith("# Memory"):
            _mem_path.write_text(result + "\n")
            print("  [memory consolidated]", flush=True)

    def _sys_messages():
        sp = load_system_prompt(include_memory=args.memory)
        return [{"role": "system", "content": sp}] if sp else []

    chime_started_at = [0.0]

    def _wait_for_chime_gap():
        if chime_sound is None or chime_started_at[0] == 0:
            return
        CHIME_HEAD = 0.22
        TICK_DUR = 0.04
        TICK_EVERY = 1.5
        t = _time.monotonic() - chime_started_at[0]
        if t < CHIME_HEAD:
            _time.sleep(CHIME_HEAD - t)
            return
        phase = (t - CHIME_HEAD) % TICK_EVERY
        if phase < TICK_DUR:
            _time.sleep(TICK_DUR - phase + 0.005)

    def play_tts(response):
        """Play TTS — streaming with barge-in (XTTS) or blocking (Piper)."""
        if tts_create_stream is None:
            # Blocking TTS (Piper)
            if chime_sound is not None:
                _wait_for_chime_gap()
                sd.stop()
            tts_speak(response)
            drain_audio_q()
            vad.reset_states()
            return False

        # Streaming TTS (XTTS-v2) with AEC voice interrupt
        drain_audio_q()
        out_stream, interrupted = None, False
        tts_16k_buf: list[np.ndarray] = []
        state = {"play_start": None, "consec_speech": 0, "mic_pos": 0}
        aec_process = make_aec_processor() if make_aec_processor else None

        def check_barge_in():
            if not (aec_process and state["play_start"] and tts_16k_buf):
                return False
            if _time.monotonic() - state["play_start"] < 0.5:
                return False
            tts_concat = np.concatenate(tts_16k_buf)
            while not audio_q.empty():
                mic_chunk = audio_q.get_nowait()
                if len(mic_chunk) < CHUNK_SAMPLES:
                    continue
                ref = _get_ref_segment(tts_concat, state["mic_pos"], len(mic_chunk))
                state["mic_pos"] += len(mic_chunk)
                cleaned = aec_process(mic_chunk, ref)
                if _vad_prob(vad, cleaned.astype(np.float32)) > 0.8:
                    state["consec_speech"] += 1
                    if state["consec_speech"] >= 5:
                        return True
                else:
                    state["consec_speech"] = 0
            return False

        try:
            for chunk_samples, sr in tts_create_stream(response):
                if out_stream is None:
                    if chime_sound is not None:
                        _wait_for_chime_gap()
                        sd.stop()
                    out_stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
                    out_stream.start()
                    drain_audio_q(); vad.reset_states()
                    state["play_start"] = _time.monotonic()
                # Collect resampled chunks for AEC reference
                if aec_process is not None:
                    if sr == SAMPLE_RATE:
                        tts_16k_buf.append(chunk_samples.astype(np.float32))
                    else:
                        idx = np.arange(0, len(chunk_samples), sr / SAMPLE_RATE)
                        tts_16k_buf.append(np.interp(idx, np.arange(len(chunk_samples)), chunk_samples).astype(np.float32))
                data = chunk_samples.reshape(-1, 1)
                for i in range(0, len(data), 4096):
                    if select.select([sys.stdin], [], [], 0)[0]:
                        sys.stdin.read(1); interrupted = True
                    elif check_barge_in():
                        interrupted = True; print("  [voice interrupt]", flush=True)
                    if interrupted:
                        break
                    out_stream.write(data[i:i+4096])
                if interrupted:
                    break
        finally:
            if out_stream:
                out_stream.stop(); out_stream.close()

        if interrupted and state["consec_speech"] < 3:
            print("  [interrupted]")
        drain_audio_q()
        vad.reset_states()
        return interrupted

    def process_utterance(audio, history):
        print(f" ({len(audio) / SAMPLE_RATE:.1f}s)")
        if chime_sound is not None:
            print("  *chime*", flush=True)
            sd.play(chime_sound, CHIME_SR)
            chime_started_at[0] = _time.monotonic()
        try:
            messages = _sys_messages()
            for h in history[-MAX_HISTORY:]:
                messages += [{"role": "user", "content": h["user"]},
                             {"role": "assistant", "content": h["assistant"]}]
            heard = transcribe(audio)
            print(f"  [{heard}]")
            messages.append({"role": "user", "content": heard})
            response = llm_generate(messages)
            print(f"\n> {response}\n", flush=True)
            if tts_speak and response:
                play_tts(response)
            elif chime_sound is not None:
                _wait_for_chime_gap()
                sd.stop()
            history.append({"user": heard, "assistant": response})
            if len(history) > MAX_HISTORY:
                history.pop(0)
            if args.memory:
                update_memory(heard, response)
                if len(history) % 5 == 0:
                    consolidate_memory()
        except Exception as e:
            print(f"\nError: {e}\n", file=sys.stderr)

    history, buf = [], []
    speaking, silent_chunks = False, 0

    old_term = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    tts_label = args.tts_engine if args.tts else "off"
    print(f"Listening (stt: whisper-{args.stt_model}, tts: {tts_label}, "
          f"silence: {args.silence_ms}ms, smart-turn: {args.smart_turn})")
    if args.tts and args.tts_engine == "xtts" and args.aec:
        tts_hint = " Speak or press any key to interrupt TTS."
    elif args.tts:
        tts_hint = " Press any key to interrupt TTS."
    else:
        tts_hint = ""
    print(f"Speak into your microphone. Ctrl+C to quit.{tts_hint}\n", flush=True)

    greeting = llm_generate(_sys_messages() + [
        {"role": "user", "content": (
            "Greet the user as Voice Loop in one short sentence. "
            "If my name is in memory, use it and ask how you can help. "
            "Otherwise, ask for my name."
        )},
    ], max_tokens=60)
    print(f"> {greeting}\n", flush=True)
    if tts_speak:
        tts_speak(greeting)

    with sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, dtype="float32",
        blocksize=CHUNK_SAMPLES, callback=callback,
    ):
        try:
            while True:
                chunk = audio_q.get()
                if len(chunk) < CHUNK_SAMPLES:
                    continue
                speech_prob = _vad_prob(vad, chunk)
                if speech_prob > 0.5:
                    if not speaking:
                        speaking = True
                        print("[listening...]", end="", flush=True)
                    silent_chunks = 0
                    buf.append(chunk)
                elif speaking:
                    silent_chunks += 1
                    buf.append(chunk)
                    if silent_chunks < silence_limit:
                        continue
                    if smart_turn and buf:
                        prob = smart_turn(np.concatenate(buf))
                        print(f" [turn prob: {prob:.2f}]", end="", flush=True)
                        if prob < 0.5:
                            silent_chunks = 0
                            continue
                    process_utterance(np.concatenate(buf), history)
                    buf.clear()
                    speaking, silent_chunks = False, 0
                    vad.reset_states()

        except KeyboardInterrupt:
            print("\nBye!")
            executor.shutdown(wait=False)
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_term)
            if args.record and record_buf:
                full = np.concatenate(record_buf)
                with wave.open(args.record, "wb") as wf:
                    wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
                    wf.writeframes((full * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
                print(f"Recorded {len(full) / SAMPLE_RATE:.1f}s to {args.record}", flush=True)


if __name__ == "__main__":
    main()
