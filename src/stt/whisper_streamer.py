"""
Real-time streaming speech-to-text transcriber using Whisper with VAD callbacks.
"""

# Import Modules
from collections.abc import Callable
import threading
from typing import Any

import os
import re
import time
import logging

import numpy as np
from numpy.typing import NDArray
from RealtimeSTT import AudioToTextRecorder
import huggingface_hub.constants as hf_constants
import RealtimeSTT.core.realtime as r_realtime
from RealtimeSTT.transcription_engines.faster_whisper_engine import FasterWhisperEngine

from src.config import STTConfig, PACE_PRESETS
from src.diarization.speaker_diarizer import SpeakerDiarizer


def _dummy_close(self: Any) -> None:
    """
    Dummy close implementation to safely avoid RealtimeSTT shutdown bug.

    Args:
        self (Any): FasterWhisperEngine instance.
    """

    _ = self


def _cjk_normalized_words(text: str) -> list[str]:
    """
    Extracts alphanumeric words and CJK characters to enable punctuation splitting.

    Args:
        text (str): Raw transcription text to normalize.

    Returns:
        list[str]: Extracted words or CJK characters.
    """

    return re.findall(
        r"[a-z0-9]+|[\u4e00-\u9fff]|[\u3040-\u30ff]|[\uac00-\ud7af]+",
        (text or "").casefold(),
    )


def _cjk_last_strong_punctuation_index(text: str, before_index: int) -> int:
    """
    Finds the index of the latest strong punctuation mark including CJK full-width marks.

    Args:
        text (str): Search text.
        before_index (int): Upper bound index.

    Returns:
        int: Index of last strong punctuation mark, or -1.
    """

    indices: list[int] = [
        text.rfind(".", 0, before_index),
        text.rfind("?", 0, before_index),
        text.rfind("!", 0, before_index),
        text.rfind("。", 0, before_index),
        text.rfind("？", 0, before_index),
        text.rfind("！", 0, before_index),
    ]

    return max(indices)


# Patch missing close method in upstream FasterWhisperEngine
if not hasattr(FasterWhisperEngine, "close"):
    FasterWhisperEngine.close = _dummy_close

# Patch RealtimeSTT punctuation presets to include full-width CJK punctuation marks
# pylint: disable=protected-access
if hasattr(r_realtime, "_SUPPORTED_PUNCTUATION_SPLIT_MARKS"):
    r_realtime._SUPPORTED_PUNCTUATION_SPLIT_MARKS.update({"。", "？", "！", "，"})
if hasattr(r_realtime, "_PUNCTUATION_SPLIT_MARK_PRESETS"):
    r_realtime._PUNCTUATION_SPLIT_MARK_PRESETS["sentence"] = (
        ".", "?", "!", "。", "？", "！",
    )
    r_realtime._PUNCTUATION_SPLIT_MARK_PRESETS["all"] = (
        ".", "?", "!", ",", "...", "—", "–", "-", "。", "？", "！", "，",
    )
if hasattr(r_realtime, "_normalized_words"):
    r_realtime._normalized_words = _cjk_normalized_words
if hasattr(r_realtime, "_last_strong_punctuation_index"):
    r_realtime._last_strong_punctuation_index = _cjk_last_strong_punctuation_index
# pylint: enable=protected-access

# Suppress verbose RealtimeSTT debug logs that fail to encode on Windows cp1252 consoles
logging.getLogger("realtimestt").setLevel(logging.WARNING)


class WhisperStreamer:
    """
    Manages real-time transcription using Faster-Whisper via RealtimeSTT without translation.

    Attributes:
        config (STTConfig): Speech-to-text configuration parameters.
        on_partial_text (Callable[[str, str], None] | None): Callback for live updates.
        on_final_text (Callable[..., None] | None): Callback for finalized phrases.
        on_vad_state (Callable[[bool], None] | None): Callback for VAD speech activity state.
        diarizer (SpeakerDiarizer | None): Optional speaker diarization module.
        initial_pace_mode (str): Conversational pace preset ('auto', 'fast', 'medium', 'accurate').
    """

    def __init__(
        self,
        config: STTConfig,
        on_partial_text: Callable[[str, str], None] | None = None,
        on_final_text: Callable[..., None] | None = None,
        on_vad_state: Callable[[bool], None] | None = None,
        diarizer: SpeakerDiarizer | None = None,
        initial_pace_mode: str = "auto",
    ) -> None:
        """
        Initializes the streaming Whisper transcriber.

        Args:
            config (STTConfig): STT configuration model.
            on_partial_text (Callable[[str, str], None] | None): Partial update callback.
            on_final_text (Callable[..., None] | None): Final stabilized callback.
            on_vad_state (Callable[[bool], None] | None): Voice activity state callback.
            diarizer (SpeakerDiarizer | None): Optional speaker diarizer instance.
            initial_pace_mode (str): Initial conversational pace mode.
        """

        # Configuration and callbacks
        self.config: STTConfig = config
        self.on_partial_text: Callable[[str, str], None] | None = on_partial_text
        self.on_final_text: Callable[..., None] | None = on_final_text
        self.on_vad_state: Callable[[bool], None] | None = on_vad_state
        self.diarizer: SpeakerDiarizer | None = diarizer

        # State tracking
        self._is_active: bool = False
        self._is_paused: bool = False
        self._speech_start_time: float = 0.0
        self._current_language: str = "unknown"
        self._recorder: Any = None
        self._worker_thread: threading.Thread | None = None
        self._lock: threading.Lock = threading.Lock()
        self._pace_mode: str = initial_pace_mode
        self._active_preset: str = "medium"
        self._recent_stats: list[dict[str, float]] = []
        self._last_sentence_end: float = 0.0

    def start(self) -> None:
        """
        Instantiates and begins listening with the streaming Whisper engine.
        """

        # Avoid redundant initialization
        with self._lock:
            if self._is_active:
                return

            self._is_active = True
            self._is_paused = False

        # Disable HuggingFace symlinks on Windows to avoid WinError 1314 privilege requirement
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        hf_constants.HF_HUB_DISABLE_SYMLINKS = True
        hf_constants.HF_HUB_DISABLE_SYMLINKS_WARNING = True

        # Initialize the underlying RealtimeSTT engine
        # Whisper performs native transcription only (never translation)
        self._recorder = AudioToTextRecorder(
            model=self.config.model_size,
            language=self.config.language if self.config.language else "",
            device=self.config.device,
            compute_type=self.config.compute_type,
            beam_size=self.config.beam_size,
            beam_size_realtime=2,
            silero_sensitivity=self.config.vad_sensitivity,
            webrtc_sensitivity=2,
            post_speech_silence_duration=self.config.post_speech_silence,
            min_length_of_recording=0.4,
            use_microphone=False,
            spinner=False,
            enable_realtime_transcription=True,
            early_transcription_on_silence=self.config.early_transcription_on_silence,
            realtime_punctuation_split_marks=self.config.split_punctuation,
            final_transcription_word_timestamps=True,
            initial_prompt="",
            no_log_file=True,
            use_extended_logging=False,
            on_vad_start=self._handle_vad_start,
            on_vad_stop=self._handle_vad_stop,
            on_realtime_transcription_update=self._handle_realtime_update,
            on_realtime_transcription_stabilized=self._handle_stabilized_update,
        )

        # Launch continuous transcription worker thread
        self._worker_thread = threading.Thread(
            target=self._transcription_worker,
            daemon=True,
        )
        self._worker_thread.start()

        # Apply initial pace mode and load diarizer model
        if self._pace_mode in PACE_PRESETS:
            self._apply_preset(self._pace_mode)

        if self.diarizer is not None:
            threading.Thread(
                target=self.diarizer.load_model,
                daemon=True,
            ).start()

    def stop(self) -> None:
        """
        Shuts down the Whisper recognition worker and frees model memory.
        """

        # Update active state flag
        with self._lock:
            if not self._is_active:
                return

            self._is_active = False

        # Abort pending recording to unblock worker thread
        if self._recorder is not None:
            try:
                self._recorder.abort()
            except Exception:
                pass

        # Wait for worker thread to terminate
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        # Terminate recorder safely
        if self._recorder is not None:
            try:
                self._recorder.shutdown()
            except Exception:
                pass
            finally:
                self._recorder = None

    def set_paused(self, is_paused: bool) -> None:
        """
        Sets whether incoming audio should be dropped or processed.

        Args:
            is_paused (bool): True to pause recognition, False to resume.
        """

        self._is_paused = is_paused

    def set_pace_mode(self, pace_mode: str) -> None:
        """
        Sets the conversational pace preset or enables adaptive auto mode.

        Args:
            pace_mode (str): 'auto', 'fast', 'medium', or 'accurate'.
        """

        with self._lock:
            self._pace_mode = pace_mode

        if pace_mode in PACE_PRESETS:
            self._apply_preset(pace_mode)

    def set_language(self, language: str) -> None:
        """
        Updates the speech-to-text audio input language specification dynamically.

        Args:
            language (str): Language code (e.g. 'auto', 'zh', 'ja', 'en', 'fr') or empty for auto.
        """

        clean_lang: str = "" if language.lower() in ("auto", "") else language.lower()
        self.config.language = clean_lang
        if self._recorder is not None:
            self._recorder.language = clean_lang

    def _apply_preset(self, preset_name: str) -> None:
        """
        Applies a tuned parameter preset to Whisper VAD and Diarizer.

        Args:
            preset_name (str): 'fast', 'medium', or 'accurate'.
        """

        preset = PACE_PRESETS.get(preset_name)
        if preset is None:
            return

        self._active_preset = preset_name
        self.config.post_speech_silence = preset.post_speech_silence
        self.config.max_sentence_duration = preset.max_sentence_duration

        if self._recorder is not None:
            setattr(
                self._recorder,
                "post_speech_silence_duration",
                preset.post_speech_silence,
            )

        if self.diarizer is not None:
            self.diarizer.set_similarity_threshold(preset.similarity_threshold)

    def _update_adaptive_pace(
        self,
        duration_sec: float,
        num_chars: int,
    ) -> None:
        """
        Computes rolling speech metrics and dynamically tunes pace preset in auto mode.

        Args:
            duration_sec (float): Utterance speech duration in seconds.
            num_chars (int): Number of characters in the utterance.
        """

        now: float = time.time()
        pause_sec: float = (
            now - self._last_sentence_end if self._last_sentence_end > 0.0 else 0.50
        )
        self._last_sentence_end = now

        # Compute speech rate
        safe_dur: float = max(0.5, duration_sec)
        rate: float = float(num_chars) / safe_dur

        # Record recent statistics
        self._recent_stats.append({
            "pause": pause_sec,
            "duration": duration_sec,
            "rate": rate,
        })
        if len(self._recent_stats) > 6:
            self._recent_stats.pop(0)

        # Only adapt if auto mode is selected and we have at least 2 samples
        if self._pace_mode != "auto" or len(self._recent_stats) < 2:
            return

        avg_pause: float = float(
            np.mean([s["pause"] for s in self._recent_stats])
        )
        avg_dur: float = float(
            np.mean([s["duration"] for s in self._recent_stats])
        )
        avg_rate: float = float(
            np.mean([s["rate"] for s in self._recent_stats])
        )

        # Classify dynamic pace
        target_preset: str = "medium"
        if avg_rate > 6.5 or (avg_pause < 0.40 and avg_dur < 2.8):
            target_preset = "fast"
        elif avg_pause > 1.20 or avg_dur > 6.5:
            target_preset = "accurate"

        if target_preset != self._active_preset:
            self._apply_preset(target_preset)

    def _is_valid_speech(self, text: str) -> bool:
        """
        Validates whether text contains actual spoken linguistic words or characters.

        Args:
            text (str): Transcribed text candidate.

        Returns:
            bool: True if text contains alphanumeric or native Asian characters.
        """

        word_pattern: str = r"[a-zA-Z0-9\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
        return bool(re.search(word_pattern, text))

    def _dispatch_final_text(
        self,
        text: str,
        lang: str,
        speaker: str,
    ) -> None:
        """
        Dispatches finalized sentence to listener callback.

        Args:
            text (str): Final transcribed phrase.
            lang (str): Spoken language code.
            speaker (str): Identified speaker label.
        """

        cleaned: str = text.strip()
        if self.on_final_text is None or not self._is_valid_speech(cleaned):
            return

        try:
            self.on_final_text(cleaned, lang, speaker)
        except TypeError:
            self.on_final_text(cleaned, lang)

    def feed_audio(self, audio_chunk: NDArray[np.float32]) -> None:
        """
        Feeds a 16kHz mono audio slice into the recognition engine.

        Args:
            audio_chunk (NDArray[np.float32]): Resampled 1D audio array.
        """

        # Discard audio if engine is inactive, paused, or chunk is empty
        if (
            not self._is_active
            or self._is_paused
            or self._recorder is None
            or audio_chunk.size == 0
        ):
            return

        # Feed the audio buffer
        try:
            self._recorder.feed_audio(
                audio_chunk,
                original_sample_rate=16000,
            )
        except Exception:
            pass

    def get_detected_language(self) -> str:
        """
        Returns the most recently detected ISO language code.

        Returns:
            str: Detected language code (e.g. 'ko', 'ja', 'zh', 'en', 'fr').
        """

        return self._current_language

    def _resolve_language(self, text: str = "") -> str:
        """
        Inspects text characters and recorder attributes to obtain detected language.

        Args:
            text (str): Optional transcribed text to analyze for native script.

        Returns:
            str: Resolved language code or 'auto'.
        """

        # Forced language specification takes precedence if configured
        if self.config.language:
            return self.config.language

        # Script-based verification for Korean and Japanese
        for char in text:
            char_code: int = ord(char)
            if (
                0xAC00 <= char_code <= 0xD7A3
                or 0x1100 <= char_code <= 0x11FF
                or 0x3130 <= char_code <= 0x318F
            ):
                return "ko"
            if 0x3040 <= char_code <= 0x309F or 0x30A0 <= char_code <= 0x30FF:
                return "ja"

        # Chinese Hanzi verification
        for char in text:
            char_code = ord(char)
            if 0x4E00 <= char_code <= 0x9FFF:
                return "zh"

        if self._recorder is None:
            return "auto"

        # Check detected realtime language
        detected: str = getattr(self._recorder, "detected_realtime_language", "")
        if not detected:
            detected = getattr(self._recorder, "detected_language", "")

        return detected if detected else "auto"

    def _is_streamer_active(self) -> bool:
        """
        Thread-safe check for active streaming status.

        Returns:
            bool: True if streamer is running.
        """

        with self._lock:
            return self._is_active

    def _transcription_worker(self) -> None:
        """
        Continuously calls recorder.text() to drive the RealtimeSTT loop and finalize sentences.
        """

        while self._is_streamer_active():
            try:
                if self._recorder is None:
                    time.sleep(0.05)
                    continue

                # text() blocks until voice activity ends and returns the complete sentence
                sentence: str = self._recorder.text()
                if not self._is_streamer_active():
                    break

                self._process_finalized_sentence(sentence)

            except Exception:
                time.sleep(0.05)

    def _process_finalized_sentence(self, sentence: str) -> None:
        """
        Processes a finalized transcription phrase, computes pace metrics and speaker turns.

        Args:
            sentence (str): Finalized speech string returned by RealtimeSTT.
        """

        cleaned: str = sentence.strip() if sentence else ""
        self._speech_start_time = 0.0
        if not cleaned or self._is_paused or not self._is_valid_speech(cleaned):
            return

        lang: str = self._resolve_language(cleaned)
        self._current_language = lang

        # Extract recorded audio samples from RealtimeSTT recorder
        audio_bytes: Any = getattr(
            self._recorder, "last_transcription_bytes", None
        )
        audio_np: NDArray[np.float32] = np.array([], dtype=np.float32)
        if audio_bytes is not None and len(audio_bytes) > 0:
            audio_np = (
                np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

        # Update adaptive pace metrics
        utt_dur: float = (
            len(audio_np) / 16000.0 if audio_np.size > 0 else 2.0
        )
        self._update_adaptive_pace(
            duration_sec=utt_dur,
            num_chars=len(cleaned),
        )

        # Identify speakers and split dialogue turns
        if self.diarizer is not None and audio_np.size > 0:
            meta: Any = getattr(
                self._recorder, "last_transcription_metadata", None
            )
            words: list[dict[str, Any]] = (
                meta.get("words", []) if isinstance(meta, dict) else []
            )
            turns: list[tuple[str, str]] = self.diarizer.split_dialogue(
                text=cleaned,
                words_meta=words,
                full_audio=audio_np,
                sample_rate=16000,
            )
            for turn_text, spk in turns:
                cleaned_turn: str = turn_text.strip()
                if cleaned_turn:
                    self._dispatch_final_text(
                        text=cleaned_turn,
                        lang=lang,
                        speaker=spk,
                    )
        else:
            self._dispatch_final_text(
                text=cleaned,
                lang=lang,
                speaker="Speaker 1",
            )

    def _handle_vad_start(self) -> None:
        """
        Dispatches voice activity start event and marks speech beginning.
        """

        self._speech_start_time = time.time()
        if self.on_vad_state is not None:
            self.on_vad_state(True)

    def _handle_vad_stop(self) -> None:
        """
        Dispatches voice activity stop event and resets speech timer.
        """

        self._speech_start_time = 0.0
        if self.on_vad_state is not None:
            self.on_vad_state(False)

    def _handle_realtime_update(self, text: str) -> None:
        """
        Dispatches in-progress transcription updates and bounds speech accumulation duration.

        Args:
            text (str): Interim transcribed text chunk.
        """

        cleaned_text: str = text.strip()
        if not cleaned_text or self._is_paused or not self._is_valid_speech(cleaned_text):
            return

        # Query detected language
        self._current_language = self._resolve_language(cleaned_text)

        # Emit to partial listener
        if self.on_partial_text is not None:
            self.on_partial_text(cleaned_text, self._current_language)

        # Cut sentence early if speaker talks continuously without pause
        if self._speech_start_time <= 0.0:
            self._speech_start_time = time.time()
        else:
            elapsed_sec: float = time.time() - self._speech_start_time
            if elapsed_sec >= self.config.max_sentence_duration:
                # Require a clause or punctuation boundary to avoid slicing mid-word
                clause_marks: tuple[str, ...] = ("，", ",", "。", ".", "？", "?", "！", "!")
                has_boundary: bool = any(mark in cleaned_text for mark in clause_marks)
                if has_boundary or elapsed_sec >= self.config.max_sentence_duration * 1.5:
                    self._speech_start_time = time.time()
                    if self._recorder is not None:
                        try:
                            self._recorder.stop()
                        except Exception:
                            pass

    def _handle_stabilized_update(self, text: str) -> None:
        """
        Dispatches live stabilized text chunk updates during active speech.

        Args:
            text (str): Stabilized transcribed text chunk.
        """

        cleaned_text: str = text.strip()
        if not cleaned_text or self._is_paused or not self._is_valid_speech(cleaned_text):
            return

        # Refresh language tag
        self._current_language = self._resolve_language(cleaned_text)

        # Emit to partial listener for dynamic preview
        if self.on_partial_text is not None:
            self.on_partial_text(cleaned_text, self._current_language)
