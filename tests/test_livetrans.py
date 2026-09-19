"""
Unit and integration tests for LiveTrans audio, romanization, and translation services.
"""

# Import Modules
from unittest.mock import patch, MagicMock
from typing import Any
import argparse
import time

from numpy.typing import NDArray
from PySide6.QtWidgets import QApplication
import numpy as np
import RealtimeSTT.core.realtime as r_realtime

from main import build_argument_parser
from src.ui import LiveTransOverlay
from src.stt import WhisperStreamer
from src.audio import AudioResampler
from src.romanizer import TextRomanizer
from src.diarization import SpeakerDiarizer
from src.translation import GemmaTranslator
from src.ui.device_selector import ControlHeaderWidget
from src.config import AppConfig, STTConfig, PACE_PRESETS, DiarizationConfig, TranslationConfig


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
    Tests sanitization of model prompt tokens, file separators, and wrapping quote delimiters.
    """

    dirty_output: str = (
        '  "Mon Dieu, ma voix est tellement rauque, non ?"<|file_separator|><|im_start|>  '
    )
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
    assert app_config.stt.post_speech_silence == 0.5, "Default silence should be 0.5s"
    assert app_config.stt.max_sentence_duration == 12.0, "Default max duration should be 12.0s"
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


def test_whisper_streamer_punctuation_and_duration_cutoff() -> None:
    """
    Verifies that CJK punctuation marks are registered and speech duration triggers recorder cut.
    """

    # 1. Verify CJK punctuation presets and helper functions
    assert "。" in r_realtime._SUPPORTED_PUNCTUATION_SPLIT_MARKS  # pylint: disable=protected-access
    assert "？" in r_realtime._SUPPORTED_PUNCTUATION_SPLIT_MARKS  # pylint: disable=protected-access
    assert "！" in r_realtime._SUPPORTED_PUNCTUATION_SPLIT_MARKS  # pylint: disable=protected-access
    assert "。" in r_realtime._PUNCTUATION_SPLIT_MARK_PRESETS["sentence"]  # pylint: disable=protected-access

    # Test CJK normalized words extraction
    cjk_words: list[str] = r_realtime._normalized_words("我们先从说起吧。")  # pylint: disable=protected-access
    assert len(cjk_words) >= 6, "Must extract CJK character tokens"

    # 2. Verify max sentence duration triggers recorder.stop() when clause boundary is present
    config: STTConfig = STTConfig(
        max_sentence_duration=3.0,
    )
    streamer: WhisperStreamer = WhisperStreamer(config=config)

    # Attach mock recorder
    mock_recorder: MagicMock = MagicMock()
    streamer._recorder = mock_recorder  # pylint: disable=protected-access

    # Simulate ongoing speech started 4 seconds ago (> 3.0s) with clause boundary
    streamer._speech_start_time = time.time() - 4.0  # pylint: disable=protected-access

    # Trigger interim update with comma clause mark
    streamer._handle_realtime_update("こんにちは、元気？")  # pylint: disable=protected-access

    # Verify stop() was invoked to finalize long utterance
    mock_recorder.stop.assert_called_once()
    assert streamer.get_detected_language() == "ja"


def test_pace_presets_configuration() -> None:
    """
    Verifies pace presets exist and hot-reloading switches STT and diarizer thresholds.
    """

    # 1. Verify preset map completeness
    assert "fast" in PACE_PRESETS
    assert "medium" in PACE_PRESETS
    assert "accurate" in PACE_PRESETS

    fast_p = PACE_PRESETS["fast"]
    med_p = PACE_PRESETS["medium"]
    acc_p = PACE_PRESETS["accurate"]

    assert fast_p.post_speech_silence == 0.30
    assert med_p.post_speech_silence == 0.50
    assert acc_p.post_speech_silence == 0.75

    # 2. Test streamer set_pace_mode hot-reload
    stt_cfg: STTConfig = STTConfig()
    diar_cfg: DiarizationConfig = DiarizationConfig()
    diarizer: SpeakerDiarizer = SpeakerDiarizer(config=diar_cfg)

    streamer: WhisperStreamer = WhisperStreamer(
        config=stt_cfg,
        diarizer=diarizer,
        initial_pace_mode="auto",
    )

    # Attach mock recorder
    mock_rec: MagicMock = MagicMock()
    streamer._recorder = mock_rec  # pylint: disable=protected-access

    # Switch to fast mode
    streamer.set_pace_mode("fast")
    assert streamer.config.post_speech_silence == 0.30
    assert streamer.config.max_sentence_duration == 7.0
    assert diarizer._active_threshold == 0.50  # pylint: disable=protected-access
    assert mock_rec.post_speech_silence_duration == 0.30

    # Switch to accurate mode
    streamer.set_pace_mode("accurate")
    assert streamer.config.post_speech_silence == 0.75
    assert streamer.config.max_sentence_duration == 18.0
    assert diarizer._active_threshold == 0.62  # pylint: disable=protected-access


def test_speaker_diarizer_clustering_and_split() -> None:
    """
    Verifies SpeakerDiarizer clustering, centroid updates, and dialogue turn splitting.
    """

    config: DiarizationConfig = DiarizationConfig(
        similarity_threshold=0.55,
        min_speech_duration=0.1,
    )
    diarizer: SpeakerDiarizer = SpeakerDiarizer(config=config)

    # Create dummy embeddings
    emb1: NDArray[np.float32] = np.zeros(192, dtype=np.float32)
    emb1[0] = 1.0  # Unit vector pointing along axis 0

    emb2: NDArray[np.float32] = np.zeros(192, dtype=np.float32)
    emb2[1] = 1.0  # Orthogonal vector (sim = 0.0)

    # Mock extract_embedding to return synthetic vectors
    current_emb: list[NDArray[np.float32]] = [emb1]

    def mock_extract(
        audio_slice: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> NDArray[np.float32]:
        _ = (audio_slice, sample_rate)
        return current_emb[0]

    diarizer.extract_embedding = mock_extract  # type: ignore[method-assign]

    dummy_audio: NDArray[np.float32] = np.zeros(16000, dtype=np.float32)

    # First turn registers Speaker 1
    spk_a, sim_a = diarizer.identify_or_register(dummy_audio)
    assert spk_a == "Speaker 1"
    assert sim_a == 1.0

    # Same speaker returns Speaker 1 with high similarity
    spk_same, sim_same = diarizer.identify_or_register(dummy_audio)
    assert spk_same == "Speaker 1"
    assert sim_same >= 0.99

    # Different speaker registers Speaker 2
    current_emb[0] = emb2
    spk_b, sim_b = diarizer.identify_or_register(dummy_audio)
    assert spk_b == "Speaker 2"
    assert sim_b < 0.55

    # Test dialogue split
    words: list[dict[str, Any]] = [
        {"word": "吃了没？", "start": 0.0, "end": 0.5},
        {"word": "吃了！", "start": 0.6, "end": 1.0},
    ]
    split_audio: NDArray[np.float32] = np.zeros(16000, dtype=np.float32)

    # Mock detect_speaker_change to indicate different speakers
    diarizer.detect_speaker_change = (  # type: ignore[method-assign]
        lambda audio_slice_a, audio_slice_b, sample_rate=16000: (True, 0.3)
    )

    turns = diarizer.split_dialogue(
        text="吃了没？吃了！",
        words_meta=words,
        full_audio=split_audio,
    )
    assert len(turns) == 2, "Must split into two dialogue turns"
    assert turns[0][0] == "吃了没？"
    assert turns[1][0] == "吃了！"


def test_adaptive_pace_tracking() -> None:
    """
    Verifies that real-time utterance statistics trigger automatic pace mode transitions.
    """

    stt_cfg: STTConfig = STTConfig()
    streamer: WhisperStreamer = WhisperStreamer(
        config=stt_cfg,
        initial_pace_mode="auto",
    )

    # Mock recorder
    mock_rec: MagicMock = MagicMock()
    streamer._recorder = mock_rec  # pylint: disable=protected-access

    # Simulate rapid chatter (short pauses, short utterances, high character rate)
    for _ in range(4):
        streamer._last_sentence_end = time.time() - 0.25  # pylint: disable=protected-access
        streamer._update_adaptive_pace(  # pylint: disable=protected-access
            duration_sec=1.5,
            num_chars=12,
        )

    # Must have adapted to fast mode
    assert streamer._active_preset == "fast"  # pylint: disable=protected-access
    assert streamer.config.post_speech_silence == 0.30

    # Simulate slow, deliberate speech (long pauses, long clauses)
    for _ in range(4):
        streamer._last_sentence_end = time.time() - 2.0  # pylint: disable=protected-access
        streamer._update_adaptive_pace(  # pylint: disable=protected-access
            duration_sec=8.0,
            num_chars=20,
        )

    # Must have adapted to accurate mode
    assert streamer._active_preset == "accurate"  # pylint: disable=protected-access
    assert streamer.config.post_speech_silence == 0.75


def test_cli_skip_languages_parsing() -> None:
    """
    Verifies that CLI argument parser correctly parses skip languages options.
    """

    parser: argparse.ArgumentParser = build_argument_parser()

    # 1. Test default value
    args_default: argparse.Namespace = parser.parse_args([])
    assert args_default.skip_langs == "en,fr"

    # 2. Test custom comma-separated list
    args_custom: argparse.Namespace = parser.parse_args(["--skip-langs", "en,fr,es,de"])
    assert args_custom.skip_langs == "en,fr,es,de"

    # 3. Test empty string override
    args_empty: argparse.Namespace = parser.parse_args(["--skip-langs", ""])
    assert args_empty.skip_langs == ""


def test_overlay_translation_bypass() -> None:
    """
    Verifies that spoken sentences in skip_languages or target_language bypass the LLM translator.
    """

    # Ensure QApplication exists for widget instantiation
    _ = QApplication.instance() or QApplication([])

    config: AppConfig = AppConfig()
    config.translation.skip_languages = ["en", "fr"]
    config.translation.target_language = "en"

    mock_audio: MagicMock = MagicMock()
    romanizer: TextRomanizer = TextRomanizer()
    mock_translator: MagicMock = MagicMock()

    overlay: LiveTransOverlay = LiveTransOverlay(
        config=config,
        audio_monitor=mock_audio,
        romanizer=romanizer,
        translator=mock_translator,
    )

    # 1. English is in skip_languages and matches target_language -> must bypass
    overlay.on_final_transcription(text="Hello world", language="en")
    mock_translator.translate.assert_not_called()

    # 2. French is in skip_languages -> must bypass
    overlay.on_final_transcription(text="Bonjour tout le monde", language="fr")
    mock_translator.translate.assert_not_called()

    # 3. Japanese is not in skip_languages -> must trigger background translation thread
    with patch("threading.Thread") as mock_thread:
        overlay.on_final_transcription(text="こんにちは", language="ja")
        mock_thread.assert_called_once()

    overlay.close()


def test_whisper_streamer_dynamic_language() -> None:
    """
    Verifies that setting the audio input language updates config and language resolution.
    """

    stt_cfg: STTConfig = STTConfig(language="")
    streamer: WhisperStreamer = WhisperStreamer(config=stt_cfg)

    # 1. Default empty (auto)
    assert streamer.config.language == ""

    # 2. Set to Japanese
    streamer.set_language("ja")
    assert streamer.config.language == "ja"
    assert streamer._resolve_language("Any text") == "ja"  # pylint: disable=protected-access

    # 3. Set to French
    streamer.set_language("FR")
    assert streamer.config.language == "fr"
    assert streamer._resolve_language("Bonjour") == "fr"  # pylint: disable=protected-access

    # 4. Reset to auto
    streamer.set_language("auto")
    assert streamer.config.language == ""


def test_control_header_language_selectors() -> None:
    """
    Verifies ControlHeaderWidget audio input and translation output dropdowns and signals.
    """

    _ = QApplication.instance() or QApplication([])
    header: ControlHeaderWidget = ControlHeaderWidget()

    # 1. Verify input language dropdown population
    input_items: list[str] = [
        str(header.combo_input_lang.itemData(i)) for i in range(header.combo_input_lang.count())
    ]
    assert "auto" in input_items
    assert "en" in input_items
    assert "fr" in input_items
    assert "zh" in input_items
    assert "ja" in input_items
    assert "ko" in input_items

    # 2. Verify target output language dropdown population
    output_items: list[str] = [
        str(header.combo_language.itemData(i)) for i in range(header.combo_language.count())
    ]
    assert "en" in output_items
    assert "fr" in output_items

    # 3. Test programmatic setters and signal emissions
    received_input: list[str] = []
    received_output: list[str] = []
    header.input_language_changed.connect(received_input.append)
    header.language_changed.connect(received_output.append)

    header.set_input_language("ja")
    assert header.combo_input_lang.currentData() == "ja"

    header.set_target_language("fr")
    assert header.combo_language.currentData() == "fr"


def test_cli_language_options_parsing() -> None:
    """
    Verifies CLI parsing for input language and updated default window width.
    """

    parser: argparse.ArgumentParser = build_argument_parser()

    # 1. Default language is auto and width is 660
    args_default: argparse.Namespace = parser.parse_args([])
    assert args_default.language == "auto"
    assert args_default.width == 660

    # 2. Custom language option
    args_custom: argparse.Namespace = parser.parse_args(["--language", "ko", "--width", "720"])
    assert args_custom.language == "ko"
    assert args_custom.width == 720
