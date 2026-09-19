"""
Unit and integration tests for LiveTrans audio, romanization, and translation services.
"""

# Import Modules
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.translation import GemmaTranslator
from src.romanizer import TextRomanizer
from src.config import AppConfig, STTConfig, TranslationConfig
from src.audio import AudioResampler
from src.stt import WhisperStreamer


def test_audio_resampler() -> None:
    """
    Tests stereo to mono downmixing and polyphase resampling from 48kHz to 16kHz.
    """

    resampler: AudioResampler = AudioResampler(source_rate=48000, target_rate=16000)

    # 1. Test stereo to mono conversion
    stereo_chunk: NDArray[np.float32] = np.ones((4800, 2), dtype=np.float32)
    mono_chunk: NDArray[np.float32] = resampler.to_mono(stereo_chunk)

    assert mono_chunk.ndim == 1, "Mono output must be 1-dimensional"
    assert mono_chunk.shape[0] == 4800, "Mono length must match frame count"
    assert float(mono_chunk[0]) == 1.0, "Averaged value must match"

    # 2. Test resampling factors
    assert resampler.up_factor == 1, "Up factor for 48000->16000 must be 1"
    assert resampler.down_factor == 3, "Down factor for 48000->16000 must be 3"

    # 3. Test full process pipeline
    processed: NDArray[np.float32] = resampler.process(stereo_chunk)
    assert processed.ndim == 1, "Processed output must be 1D"
    assert processed.shape[0] == 1600, "4800 frames at 48kHz must resample to 1600 frames at 16kHz"


def test_text_romanizer_korean() -> None:
    """
    Tests Korean Hangul detection and Revised Romanization.
    """

    romanizer: TextRomanizer = TextRomanizer()
    korean_text: str = "안녕하세요"

    # Verify script detection
    assert romanizer.has_hangul(korean_text), "Must detect Korean Hangul"

    # Verify romanization output
    romanized: str = romanizer.romanize(korean_text, language="ko")
    assert "annyeonghase" in romanized.lower(), f"Unexpected romanization: {romanized}"


def test_text_romanizer_japanese() -> None:
    """
    Tests Japanese Kana detection and Hepburn Romaji transliteration.
    """

    romanizer: TextRomanizer = TextRomanizer()
    japanese_text: str = "こんにちは"

    # Verify script detection
    assert romanizer.has_kana(japanese_text), "Must detect Japanese Kana"

    # Verify romanization output
    romanized: str = romanizer.romanize(japanese_text, language="ja")
    assert "konnichi" in romanized.lower(), f"Unexpected romaji: {romanized}"


def test_text_romanizer_chinese() -> None:
    """
    Tests Chinese Hanzi detection and Pinyin transliteration with tones.
    """

    romanizer: TextRomanizer = TextRomanizer()
    chinese_text: str = "你好世界"

    # Verify script detection
    assert romanizer.has_hanzi(chinese_text), "Must detect Chinese Hanzi"

    # Verify romanization output
    romanized: str = romanizer.romanize(chinese_text, language="zh")
    assert "nǐ" in romanized and "hǎo" in romanized, f"Unexpected pinyin: {romanized}"


def test_gemma_translator_history_and_payload() -> None:
    """
    Tests translation client conversation context buffer and OpenAI-compatible payload formatting.
    """

    config: TranslationConfig = TranslationConfig(
        server_url="http://127.0.0.1:8080/v1/chat/completions",
        model_name="translategemma4b",
        target_language="fr",
        history_max_turns=3,
    )
    translator: GemmaTranslator = GemmaTranslator(config=config)

    # 1. Add conversation history turns
    translator.add_turn(
        source_text="안녕하세요",
        source_language="ko",
        translated_text="Bonjour",
        target_language="fr",
    )
    translator.add_turn(
        source_text="어디 가세요?",
        source_language="ko",
        translated_text="Où allez-vous ?",
        target_language="fr",
    )

    # Verify history size
    assert len(translator.history) == 2, "History must hold 2 entries"

    # Verify history summary string
    history_summary: str = translator.get_history_summary()
    assert "Bonjour" in history_summary, "History summary must contain previous translations"
    assert "Où allez-vous" in history_summary, "History summary must contain previous translations"

    # Verify payload construction
    payload: dict[str, Any] = translator._build_payload(  # pylint: disable=protected-access
        text="집에 가요",
        source_lang="ko",
        target_lang="fr",
    )

    assert payload["model"] == "translategemma4b", "Model name must be translategemma4b"
    # 1 system + 4 history messages (2 pairs) + 1 current user message = 6
    assert len(payload["messages"]) == 6, (
        "Payload must contain system, 2 history pairs, and current prompt"
    )
    assert "<|im_start|>" in payload["stop"], "Payload must specify stop tokens"

    current_msg: str = str(payload["messages"][-1]["content"])
    assert "집에 가요" in current_msg, "Current text must be present in final user message"

    # Clean up client
    translator.close()


def test_gemma_translator_graceful_offline() -> None:
    """
    Verifies that translation failure returns a clean error badge rather than raising exceptions.
    """

    config: TranslationConfig = TranslationConfig(
        server_url="http://127.0.0.1:59999/v1/chat/completions",
        model_name="translategemma4b",
        request_timeout=0.5,
    )
    translator: GemmaTranslator = GemmaTranslator(config=config)

    # Translate with dead port
    result: str = translator.translate("Hello world", source_lang="en", target_lang="fr")
    err_prefix: str = "[Translation Server Offline"
    assert err_prefix in result, f"Expected offline status message, got: {result}"

    translator.close()


def test_gemma_translator_clean_response() -> None:
    """
    Tests sanitization of model prompt tokens and wrapping quote delimiters.
    """

    dirty_output: str = '  "Mon Dieu, ma voix est tellement rauque, non ?"<|im_start|>  '
    cleaned: str = GemmaTranslator._clean_response(dirty_output)  # pylint: disable=protected-access
    expected: str = "Mon Dieu, ma voix est tellement rauque, non ?"
    assert cleaned == expected, f"Failed cleaning: expected '{expected}', got '{cleaned}'"


def test_app_config_defaults() -> None:
    """
    Verifies default application configuration values.
    """

    app_config: AppConfig = AppConfig()

    assert app_config.audio.sample_rate == 48000, "Audio rate default should be 48000"
    assert app_config.audio.target_sample_rate == 16000, "Whisper target rate must be 16000"
    assert app_config.stt.model_size == "tiny", "Default STT model size should be tiny"
    assert app_config.translation.model_name == "translategemma4b", "LLM must be translategemma4b"
    assert app_config.translation.target_language == "en", "Default target lang must be en"
    assert app_config.ui.window_opacity == 0.82, "Default UI opacity should be 0.82"


def test_whisper_streamer_pause_and_state() -> None:
    """
    Verifies WhisperStreamer pause toggle and language resolution.
    """

    config: STTConfig = STTConfig(
        model_size="tiny",
        device="cpu",
        compute_type="default",
    )
    received_partial: list[str] = []
    received_final: list[str] = []

    streamer: WhisperStreamer = WhisperStreamer(
        config=config,
        on_partial_text=lambda t, l: received_partial.append(t),
        on_final_text=lambda t, l: received_final.append(t),
    )

    # Test initial state
    assert not streamer.get_detected_language() or streamer.get_detected_language() == "unknown"

    # Test pause flag
    streamer.set_paused(True)
    assert streamer._is_paused, "Pause flag must be True"  # pylint: disable=protected-access

    streamer.set_paused(False)
    assert not streamer._is_paused, "Pause flag must be False"  # pylint: disable=protected-access
