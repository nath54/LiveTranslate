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
        language (str): Whisper audio input language code or empty string for auto-detection.
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
    language: str = ""


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
        skip_languages (list[str]): Language codes to bypass LLM translation for.
    """

    server_url: str = "http://127.0.0.1:8080/v1/chat/completions"
    model_name: str = "translategemma4b"
    target_language: str = "en"
    http_client: str = "httpx"
    history_max_turns: int = 6
    request_timeout: float = 10.0
    temperature: float = 0.1
    skip_languages: list[str] = field(default_factory=lambda: ["en", "fr"])


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

    window_width: int = 660
    window_height: int = 360
    window_opacity: float = 0.82
    always_on_top: bool = True
    font_size: int = 13


@dataclass
class PacePreset:
    """
    Tuned parameters corresponding to a specific conversational pace preset.

    Attributes:
        post_speech_silence (float): Duration in seconds of silence before finalizing sentence.
        max_sentence_duration (float): Maximum continuous speech seconds before forcing a split.
        similarity_threshold (float): Cosine similarity threshold for speaker clustering.
        beam_size (int): Whisper decoding beam width.
    """

    post_speech_silence: float = 0.50
    max_sentence_duration: float = 12.0
    similarity_threshold: float = 0.55
    beam_size: int = 3


PACE_PRESETS: dict[str, PacePreset] = {
    "fast": PacePreset(
        post_speech_silence=0.30,
        max_sentence_duration=7.0,
        similarity_threshold=0.50,
        beam_size=2,
    ),
    "medium": PacePreset(
        post_speech_silence=0.50,
        max_sentence_duration=12.0,
        similarity_threshold=0.55,
        beam_size=3,
    ),
    "accurate": PacePreset(
        post_speech_silence=0.75,
        max_sentence_duration=18.0,
        similarity_threshold=0.62,
        beam_size=5,
    ),
}


@dataclass
class DiarizationConfig:
    """
    Configuration parameters for CAM++ speaker diarization.

    Attributes:
        enabled (bool): Whether speaker identification and turn-splitting are active.
        model_repo (str): HuggingFace repository identifier for CAM++ ONNX model.
        model_filename (str): Model file name within the repository.
        similarity_threshold (float): Minimum cosine similarity to match an existing speaker.
        min_speech_duration (float): Minimum speech duration in seconds to compute embeddings.
        max_speakers (int): Maximum number of distinct speaker centroids to track.
    """

    enabled: bool = True
    model_repo: str = "welcomyou/campplus-3dspeaker-200k-onnx"
    model_filename: str = "campplus_cn_en_common_200k.onnx"
    similarity_threshold: float = 0.55
    min_speech_duration: float = 0.4
    max_speakers: int = 4


@dataclass
class AppConfig:
    """
    Aggregated master configuration for the LiveTrans application.

    Attributes:
        audio (AudioConfig): Audio capture configuration instance.
        stt (STTConfig): Speech-to-text configuration instance.
        translation (TranslationConfig): Translation service configuration instance.
        ui (UIConfig): User interface configuration instance.
        diarization (DiarizationConfig): Speaker diarization configuration instance.
        pace_mode (str): Active conversational pace mode ('auto', 'fast', 'medium', 'accurate').
    """

    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    pace_mode: str = "auto"
