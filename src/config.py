"""
Configuration module for LiveTrans application settings.
"""

# Import Modules
from dataclasses import field, dataclass


@dataclass
class AudioConfig:
    """
    Configuration parameters for loopback audio capture and preprocessing.

    Attributes:
        sample_rate (int): Native loopback capture sampling rate in Hz.
        target_sample_rate (int): Resampled rate in Hz required by Whisper.
        buffer_frames (int): Number of audio frames read per capture loop.
        speaker_name (str | None): Target speaker device name or None for default.
    """

    sample_rate: int = 48000
    target_sample_rate: int = 16000
    buffer_frames: int = 2048
    speaker_name: str | None = None


@dataclass
class STTConfig:
    """
    Configuration parameters for Whisper speech-to-text recognition.

    Attributes:
        model_size (str): Whisper model identifier (e.g. tiny, base, small, large-v3).
        device (str): Computation device (cuda or cpu).
        compute_type (str): Quantization or precision type (e.g. float16, int8, default).
        beam_size (int): Beam search width for transcription.
        vad_sensitivity (float): Voice activity detection sensitivity threshold.
        post_speech_silence (float): Duration in seconds of silence before finalizing sentence.
        early_transcription_on_silence (int): Silence threshold in ms to begin early decoding.
        split_punctuation (str): Punctuation splitting preset for ongoing speech.
        max_sentence_duration (float): Maximum continuous speech seconds before forcing a split.
    """

    model_size: str = "tiny"
    device: str = "cuda"
    compute_type: str = "default"
    beam_size: int = 3
    vad_sensitivity: float = 0.4
    post_speech_silence: float = 0.5
    early_transcription_on_silence: int = 150
    split_punctuation: str = "sentence"
    max_sentence_duration: float = 12.0


@dataclass
class TranslationConfig:
    """
    Configuration parameters for the LLM translation service.

    Attributes:
        server_url (str): OpenAI-compatible chat completions endpoint URL.
        model_name (str): LLM model identifier (translategemma4b).
        target_language (str): Target output language code ('en' or 'fr').
        http_client (str): Preferred HTTP engine ('httpx' or 'requests').
        history_max_turns (int): Number of recent conversation turns to retain for context.
        request_timeout (float): HTTP request timeout in seconds.
        temperature (float): Generation temperature for faithful translation.
    """

    server_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    model_name: str = "translategemma4b"
    target_language: str = "en"
    http_client: str = "httpx"
    history_max_turns: int = 6
    request_timeout: float = 10.0
    temperature: float = 0.1


@dataclass
class UIConfig:
    """
    Configuration parameters for the floating PySide6 overlay window.

    Attributes:
        window_width (int): Initial width of the overlay widget in pixels.
        window_height (int): Initial height of the overlay widget in pixels.
        window_opacity (float): Background translucency level between 0.1 and 1.0.
        always_on_top (bool): Whether window stays pinned above other applications.
        font_size (int): Base font size in points for transcribed and translated text.
    """

    window_width: int = 580
    window_height: int = 360
    window_opacity: float = 0.82
    always_on_top: bool = True
    font_size: int = 13


@dataclass
class AppConfig:
    """
    Aggregated master configuration for the LiveTrans application.

    Attributes:
        audio (AudioConfig): Audio capture configuration instance.
        stt (STTConfig): Speech-to-text configuration instance.
        translation (TranslationConfig): Translation service configuration instance.
        ui (UIConfig): User interface configuration instance.
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    ui: UIConfig = field(default_factory=UIConfig)
