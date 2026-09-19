# LiveTrans

Real-time desktop audio transcription, romanization, and translation overlay for Windows and Linux.

Captures system audio output (WASAPI loopback on Windows, PulseAudio/PipeWire monitor on Linux), streams it into Faster-Whisper for live transcription, identifies speakers using CAM++ diarization, romanizes CJK text (Romaji, Pinyin, Hangul), and optionally translates into English or French via a local LLM endpoint (TranslateGemma / llama.cpp).

---

## Features

- **Audio Loopback Capture**: Captures any playing audio directly via system loopback (WASAPI on Windows, PulseAudio/PipeWire monitor on Linux) without virtual audio cables.
- **Streaming STT**: Low-latency live speech recognition powered by Faster-Whisper.
- **Speaker Diarization**: Online speaker identification and turn-splitting using CAM++ ONNX embeddings.
- **Phonetic Romanization**: Real-time phonetic transliteration for Japanese (Romaji), Chinese (Pinyin), and Korean (Revised Romanization).
- **Smart Translation Bypass**: Automatically skips LLM translation when the detected language matches your target or is in your skip list (`en`, `fr` by default) to save GPU memory and eliminate latency.
- **Context-Aware Translation**: Translates non-skipped speech using TranslateGemma-4B or any OpenAI-compatible completions server with multi-turn history.
- **Desktop Overlay**: Semi-transparent, resizable, movable, and always-on-top HUD with audio visualizer and pause/clear buttons.
- **Conversational Pace Modes**: `Auto` (adapts dynamically to speech tempo), `Fast`, `Medium`, or `Accurate`.

---

## Requirements

- **OS**: Windows 10 / 11 or Linux (Ubuntu, Debian, Fedora, Arch, etc. with PulseAudio or PipeWire)
- **Python**: 3.10 – 3.12
- **GPU**: NVIDIA GPU with CUDA recommended (Whisper `base` runs alongside TranslateGemma-4B within 6GB VRAM)
- **Optional**: `llama.cpp` server running TranslateGemma-4B (or another translation model)
- **Linux system dependencies** (if not already installed):
  ```bash
  # Debian / Ubuntu
  sudo apt install libpulse0 pulseaudio-utils
  # Arch Linux
  sudo pacman -S libpulse
  ```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/LiveTrans.git
   cd LiveTrans
   ```

2. **Create and activate a virtual environment:**
   - **Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   > **Note (CUDA Support)**: If you want GPU acceleration for Whisper and PyTorch, install the matching CUDA wheel:
   > ```bash
   > pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   > ```
   >
   > **Linux Wayland Note**: If your desktop compositor restricts transparent frameless windows, launch with X11 backend:
   > ```bash
   > QT_QPA_PLATFORM=xcb python main.py
   > ```

---

## Quick Start

### 1. (Optional) Launch Translation Server
If you want live translations for foreign speech (e.g. Japanese/Chinese to English or French), start your local `llama.cpp` server:

```bash
llama-server -m translategemma-4b.gguf --port 8080 -ngl 99
```

### 2. Start LiveTrans
```bash
python main.py
```

By default:
- Listens to default audio output device.
- Uses Whisper `tiny` on CUDA (switch to `base` or `small` with `--model`).
- Input language: `auto` (can be changed on the fly in the UI).
- Target translation language: `en` (English).
- Skips translation for English (`en`) and French (`fr`).

---

## CLI Options

```text
usage: main.py [-h] [--speaker SPEAKER] [--sample-rate SAMPLE_RATE]
               [--buffer-frames BUFFER_FRAMES] [--model MODEL] [--device DEVICE]
               [--compute-type COMPUTE_TYPE] [--beam-size BEAM_SIZE]
               [--language LANGUAGE] [--translate-url TRANSLATE_URL]
               [--translate-model TRANSLATE_MODEL] [--target-lang {en,fr}]
               [--skip-langs SKIP_LANGS] [--http-client {httpx,requests}]
               [--history-turns HISTORY_TURNS] [--width WIDTH] [--height HEIGHT]
               [--opacity OPACITY] [--no-top] [--pace {auto,fast,medium,accurate}]
               [--no-diarization]
```

### Common Commands

- **Run Whisper `base` with French translation target:**
  ```bash
  python main.py --model base --target-lang fr
  ```

- **Lock input language to Japanese:**
  ```bash
  python main.py --model base --language ja --target-lang en
  ```

- **Disable speaker diarization (saves CPU resources):**
  ```bash
  python main.py --no-diarization
  ```

- **Translate everything (disable skip bypass):**
  ```bash
  python main.py --skip-langs ""
  ```

---

## UI Overview

- **Audio Device Dropdown**: Select which audio output device to monitor.
- **🎙 In (Input Language)**: Select `Auto` or lock to a language (`zh`, `ko`, `ja`, `en`, `fr`, `es`, `de`, `it`, `pt`, `ru`, `ar`).
- **🌐 Out (Translation Target)**: Toggle translation output between English (`en`) and French (`fr`).
- **Pace Selector**: Toggle between `🔄 Auto`, `⚡ Fast`, `⚖ Med`, and `🎯 Acc`.
- **⏸ Pause**: Toggle audio capture and transcription.
- **📌 Pin**: Toggle always-on-top window behavior.
- **🗑 Clear**: Clear historical subtitle cards and reset dialogue context.
- **✕ Close**: Exit application cleanly.

---

## Tests & Code Quality

```bash
# Run test suite
pytest

# Code style and typing checks
pylint src main.py tests/test_livetrans.py
mypy src main.py tests/test_livetrans.py
```

---

## License

MIT
