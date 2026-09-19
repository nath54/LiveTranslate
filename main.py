"""
LiveTrans application launcher and command line interface.
"""

# Import Modules
import sys
import argparse

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from src.ui import LiveTransOverlay
from src.stt import WhisperStreamer
from src.config import AppConfig
from src.audio import AudioLoopbackCapture
from src.romanizer import TextRomanizer
from src.translation import GemmaTranslator


def build_argument_parser() -> argparse.ArgumentParser:
    """
    Constructs the CLI argument parser with configurable options for all subsystems.

    Returns:
        argparse.ArgumentParser: Configured parser instance.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="LiveTrans: Audio Output Streamer, STT, Romanization & Translation Overlay",
    )

    # Audio parameters
    parser.add_argument(
        "--speaker",
        type=str,
        default=None,
        help="Target speaker device name for loopback capture",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Audio loopback capture sample rate in Hz (default: 48000)",
    )
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=1024,
        help="Audio buffer slice size in frames (default: 1024)",
    )

    # Whisper STT parameters
    parser.add_argument(
        "--model",
        type=str,
        default="tiny",
        help="Whisper model size (tiny, base, small, medium, large-v3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Inference compute device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="default",
        help="Quantization precision type (default, float16, int8)",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=3,
        help="Beam size for transcription search (default: 3)",
    )

    # Translation parameters
    parser.add_argument(
        "--translate-url",
        type=str,
        default="http://127.0.0.1:8080/v1/chat/completions",
        help="TranslateGemma-4B OpenAI-compatible endpoint URL",
    )
    parser.add_argument(
        "--translate-model",
        type=str,
        default="translategemma4b",
        help="Translation LLM model identifier (default: translategemma4b)",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="en",
        choices=["en", "fr"],
        help="Target translation language code (default: en)",
    )
    parser.add_argument(
        "--http-client",
        type=str,
        default="httpx",
        choices=["httpx", "requests"],
        help="Preferred HTTP library for translation requests (httpx or requests)",
    )
    parser.add_argument(
        "--history-turns",
        type=int,
        default=6,
        help="Number of previous dialogue turns passed for context (default: 6)",
    )

    # UI parameters
    parser.add_argument(
        "--width",
        type=int,
        default=580,
        help="Initial overlay window width in pixels (default: 580)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=360,
        help="Initial overlay window height in pixels (default: 360)",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.82,
        help="Window background opacity between 0.1 and 1.0 (default: 0.82)",
    )
    parser.add_argument(
        "--no-top",
        action="store_true",
        help="Disable always-on-top window behavior",
    )

    return parser


def parse_arguments_to_config() -> AppConfig:
    """
    Parses command line arguments and maps them into an AppConfig instance.

    Returns:
        AppConfig: Fully initialized application configuration.
    """

    parser: argparse.ArgumentParser = build_argument_parser()
    args: argparse.Namespace = parser.parse_args()

    # Instantiate config
    config: AppConfig = AppConfig()

    # Populate Audio config
    config.audio.speaker_name = args.speaker
    config.audio.sample_rate = args.sample_rate
    config.audio.buffer_frames = args.buffer_frames

    # Populate STT config
    config.stt.model_size = args.model
    config.stt.device = args.device
    config.stt.compute_type = args.compute_type
    config.stt.beam_size = args.beam_size

    # Populate Translation config
    config.translation.server_url = args.translate_url
    config.translation.model_name = args.translate_model
    config.translation.target_language = args.target_lang
    config.translation.http_client = args.http_client
    config.translation.history_max_turns = args.history_turns

    # Populate UI config
    config.ui.window_width = args.width
    config.ui.window_height = args.height
    config.ui.window_opacity = args.opacity
    config.ui.always_on_top = not args.no_top

    return config


def main() -> int:
    """
    Launches the LiveTrans application services and Qt event loop.

    Returns:
        int: Exit status code.
    """

    # Parse arguments
    config: AppConfig = parse_arguments_to_config()

    # Initialize PySide6 application
    app: QApplication = QApplication(sys.argv)
    app.setApplicationName("LiveTrans")

    # Configure positive base font to prevent point size warnings
    base_font: QFont = QFont("sans-serif", 10)
    app.setFont(base_font)

    # Instantiate translation and romanization services
    romanizer: TextRomanizer = TextRomanizer()
    translator: GemmaTranslator = GemmaTranslator(config=config.translation)

    # Instantiate audio capture monitor
    audio_monitor: AudioLoopbackCapture = AudioLoopbackCapture(
        sample_rate=config.audio.sample_rate,
        target_sample_rate=config.audio.target_sample_rate,
        buffer_frames=config.audio.buffer_frames,
    )

    # Instantiate overlay widget
    overlay: LiveTransOverlay = LiveTransOverlay(
        config=config,
        audio_monitor=audio_monitor,
        romanizer=romanizer,
        translator=translator,
    )

    # Instantiate Whisper streaming engine
    whisper_engine: WhisperStreamer = WhisperStreamer(
        config=config.stt,
        on_partial_text=overlay.on_partial_transcription,
        on_final_text=overlay.on_final_transcription,
        on_vad_state=overlay.on_vad_state_changed,
    )

    # Route captured audio chunks directly to Whisper engine
    audio_monitor.on_audio_chunk = whisper_engine.feed_audio

    # Start services
    whisper_engine.start()
    audio_monitor.start()

    # Show floating overlay
    overlay.show()

    # Run Qt main loop
    exit_code: int = app.exec()

    # Clean shutdown on application exit
    audio_monitor.stop()
    whisper_engine.stop()
    translator.close()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
