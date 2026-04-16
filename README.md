# Voice Loop

A minimal on-device voice agent loop. macOS Apple Silicon + Linux Nvidia GPU.

> Need a custom voice model or production voice agent? See [Trelis Voice AI Services](https://trelis.com/voice-ai-services/).

## Features

- **Smart turn detection** — Silero VAD + pipecat's Smart Turn v3, so the agent waits when you pause mid-sentence
- **Voice interruption** — speak over the agent; WebRTC AEC3 cancels echo from speakers so your voice cuts through (streaming TTS engines only)
- **Editable persona** — `SOUL.md` controls the agent's style, live-reloaded each turn
- **Optional long-term memory** — enable with `--memory`; the agent learns durable facts about you in `MEMORY.md` and consolidates every 5 turns
- **Cross-platform** — macOS (MLX/Metal) and Linux (CUDA) with auto-detected defaults
- **Multi-language** — Turkish, English, Spanish, French, German, and 90+ languages via faster-whisper + Piper/XTTS

## Stack

| Component | Engine | Notes |
|-----------|--------|-------|
| **STT** | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | 99+ languages, CUDA on Linux, CPU on macOS |
| **LLM** | MLX / transformers / OpenAI API | `--backend mlx\|local\|api` |
| **TTS** | [Piper](https://github.com/rhasspy/piper) or [XTTS-v2](https://huggingface.co/coqui/XTTS-v2) | `--tts-engine piper\|xtts` |
| **VAD** | [Silero VAD](https://github.com/snakers4/silero-vad) + [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) | Turn detection |
| **AEC** | [WebRTC AEC3](https://github.com/livekit/python-sdks) (LiveKit APM) | Voice interruption |

## Setup

**macOS:**
```bash
brew install portaudio espeak-ng
git clone https://github.com/TrelisResearch/voice-loop.git
cd voice-loop
uv sync
```

**Linux (Nvidia GPU):**
```bash
sudo apt install portaudio19-dev espeak-ng libcudnn9-cuda-12
git clone https://github.com/TrelisResearch/voice-loop.git
cd voice-loop
uv sync
```

First run downloads models: faster-whisper medium (~1.5GB), LLM (~3GB), TTS model (Piper ~30MB or XTTS ~1.8GB).

## Usage

### Quick start

```bash
# macOS — defaults (MLX Gemma 4, whisper STT, Piper TTS, English)
uv run voice_loop.py

# Linux — defaults (transformers Gemma on CUDA, whisper STT, Piper TTS, English)
uv run voice_loop.py
```

### Language (`--lang`)

Single flag sets language for both STT and TTS.

```bash
# English (default)
uv run voice_loop.py --lang en

# Turkish
uv run voice_loop.py --lang tr

# Spanish
uv run voice_loop.py --lang es
```

### LLM backend (`--backend`)

```bash
# macOS — MLX Gemma 4 on Metal (default on macOS)
uv run voice_loop.py --backend mlx

# Linux — transformers + CUDA (default on Linux)
uv run voice_loop.py --backend local

# Both — OpenAI-compatible API (Ollama, OpenAI, etc.)
uv run voice_loop.py --backend api --base-url http://localhost:11434/v1 --model gemma3:4b

# OpenAI API
uv run voice_loop.py --backend api --base-url https://api.openai.com/v1 --model gpt-4o-mini

# Custom model
uv run voice_loop.py --backend local --model google/gemma-4-E2B-it
```

### STT options (`--stt-model`)

```bash
# Larger whisper model for better accuracy
uv run voice_loop.py --stt-model large-v3

# Smaller model for faster transcription
uv run voice_loop.py --stt-model tiny
```

### TTS engine (`--tts-engine`)

```bash
# Piper — lightweight, CPU, no streaming (default)
uv run voice_loop.py --tts-engine piper

# Piper — specific model
uv run voice_loop.py --tts-engine piper --piper-model en_GB-alba-medium

# XTTS-v2 — GPU, streaming, voice interrupt, higher quality
uv run voice_loop.py --tts-engine xtts

# XTTS-v2 — voice cloning from a reference WAV
uv run voice_loop.py --tts-engine xtts --speaker-wav my_voice.wav

# Disable TTS (text-only output)
uv run voice_loop.py --no-tts
```

### Full Turkish setup

```bash
# Turkish with Piper (lightweight, CPU)
uv run voice_loop.py --lang tr

# Turkish with XTTS-v2 (streaming + voice interrupt)
uv run voice_loop.py --lang tr --tts-engine xtts

# Turkish with XTTS-v2 + voice cloning
uv run voice_loop.py --lang tr --tts-engine xtts --speaker-wav my_voice.wav

# Turkish with Ollama backend
uv run voice_loop.py --lang tr --tts-engine xtts \
  --backend api --base-url http://localhost:11434/v1 --model gemma3:4b
```

### Other options

```bash
# Enable persistent memory
uv run voice_loop.py --memory

# Disable voice interrupt (keypress only)
uv run voice_loop.py --no-aec

# Disable smart turn detection
uv run voice_loop.py --no-smart-turn

# Disable chime sounds
uv run voice_loop.py --no-chime

# Custom silence timeout (ms)
uv run voice_loop.py --silence-ms 500

# Record mic to WAV for debugging
uv run voice_loop.py --record
```

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `--lang` | `en` | Language for both STT and TTS (e.g. `tr`, `es`, `fr`) |
| `--backend` | `mlx` (macOS) / `local` (Linux) | LLM backend: `mlx`, `local`, or `api` |
| `--model` | auto per backend | HuggingFace model ID or Ollama model name |
| `--base-url` | `http://localhost:11434/v1` | OpenAI-compatible API URL (for `--backend api`) |
| `--api-key` | env `OPENAI_API_KEY` | API key (auto `"ollama"` for localhost) |
| `--stt-model` | `medium` | Whisper model size: `tiny`, `small`, `medium`, `large-v3` |
| `--tts` / `--no-tts` | on | Enable/disable TTS |
| `--tts-engine` | `piper` | TTS engine: `piper` (CPU) or `xtts` (GPU, streaming) |
| `--piper-model` | auto from `--lang` | Piper ONNX model name |
| `--speaker-wav` | none | Reference WAV for XTTS voice cloning |
| `--smart-turn` / `--no-smart-turn` | on | Smart Turn v3 endpoint detection |
| `--aec` / `--no-aec` | on | WebRTC AEC3 voice interrupt |
| `--chime` / `--no-chime` | on | Chime + ticks while generating |
| `--memory` | off | Read/write `MEMORY.md` |
| `--silence-ms` | `700` | Silence timeout before processing |
| `--record` | off | Record mic to WAV file |

## TTS Engine Comparison

| | Piper | XTTS-v2 |
|---|---|---|
| **Quality** | Good | High (expressive, natural) |
| **Streaming** | No (blocking) | Yes (~200ms first chunk) |
| **Voice interrupt** | Keypress only | Voice + keypress (AEC) |
| **Voice cloning** | No | Yes (`--speaker-wav`) |
| **GPU required** | No (CPU/ONNX) | Yes (CUDA recommended) |
| **Model size** | ~30MB | ~1.8GB |
| **Languages** | Model-dependent | 17 languages |
| **Turkish** | Yes | Yes |

## Architecture

```
Mic (16kHz) --> Silero VAD --> Smart Turn --> faster-whisper --> LLM --> TTS --> Speakers
                                                                 ^               |
                                                     SOUL.md + MEMORY.md         |
                                                                                 v
Mic during TTS --> WebRTC AEC3 (LiveKit APM) --> Silero VAD --> voice interrupt <-+
                   (streaming TTS only: XTTS-v2)
```

## How it works

1. **Mic capture** via sounddevice (16kHz mono)
2. **Silero VAD** detects speech vs silence
3. **Smart Turn** confirms end-of-turn on silence
4. **faster-whisper** transcribes audio to text (CUDA on Linux, CPU on macOS)
5. **LLM** responds using `SOUL.md` (+ `MEMORY.md` if `--memory`) as system prompt
6. **TTS** synthesizes speech — Piper (blocking) or XTTS-v2 (streaming with voice interrupt)
7. **WebRTC AEC3** cleans mic during streaming TTS playback for voice interrupt

## Persona & Memory

- `SOUL.md` — persona / style (always loaded, live-reloaded each turn)
- `MEMORY.md` — long-term facts. Only read/written when `--memory` is passed. The agent extracts new durable facts after each turn and consolidates every 5 turns.

Both files are re-read at the start of every turn, so edits take effect immediately.

## Credits

Built with:
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — STT
- [Piper](https://github.com/rhasspy/piper) — TTS (lightweight)
- [Coqui XTTS-v2](https://huggingface.co/coqui/XTTS-v2) — TTS (streaming, voice cloning)
- [Silero VAD](https://github.com/snakers4/silero-vad) — voice activity detection
- [Smart Turn v3](https://github.com/pipecat-ai/smart-turn) — end-of-turn detection
- [LiveKit APM](https://github.com/livekit/python-sdks) — WebRTC AEC3
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) — MLX multimodal inference (macOS)
- [Gemma 4](https://huggingface.co/google/gemma-4-E4B-it) — LLM

## License

Apache 2.0.
