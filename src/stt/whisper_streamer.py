"""
Real-time streaming speech-to-text transcriber using Whisper with VAD callbacks.
"""

# Import Modules
from collections.abc import Callable
import threading
from typing import Any
import time

from RealtimeSTT.transcription_engines.faster_whisper_engine import FasterWhisperEngine
from numpy.typing import NDArray
from RealtimeSTT import AudioToTextRecorder
import numpy as np

from src.config import STTConfig


def _dummy_close(self: Any) -> None:
    """
    Dummy close implementation to safely avoid RealtimeSTT shutdown bug.

    Args:
        self (Any): FasterWhisperEngine instance.
    """

    _ = self


# Patch missing close method in upstream FasterWhisperEngine
if not hasattr(FasterWhisperEngine, "close"):
    FasterWhisperEngine.close = _dummy_close


class WhisperStreamer:
    """
    Manages real-time transcription using Faster-Whisper via RealtimeSTT without translation.

    Attributes:
        config (STTConfig): Speech-to-text configuration parameters.
        on_partial_text (Callable[[str, str], None] | None): Callback for live updates.
        on_final_text (Callable[[str, str], None] | None): Callback for finalized phrases.
        on_vad_state (Callable[[bool], None] | None): Callback for VAD speech activity state.
    """

    def __init__(
        self,
        config: STTConfig,
        on_partial_text: Callable[[str, str], None] | None = None,
        on_final_text: Callable[[str, str], None] | None = None,
        on_vad_state: Callable[[bool], None] | None = None,
    ) -> None:
        """
        Initializes the streaming Whisper transcriber.

        Args:
            config (STTConfig): STT configuration model.
            on_partial_text (Callable[[str, str], None] | None): Partial update callback.
            on_final_text (Callable[[str, str], None] | None): Final stabilized callback.
            on_vad_state (Callable[[bool], None] | None): Voice activity state callback.
        """

        # Configuration and callbacks
        self.config: STTConfig = config
        self.on_partial_text: Callable[[str, str], None] | None = on_partial_text
        self.on_final_text: Callable[[str, str], None] | None = on_final_text
        self.on_vad_state: Callable[[bool], None] | None = on_vad_state

        # State tracking
        self._is_active: bool = False
        self._is_paused: bool = False
        self._current_language: str = "unknown"
        self._recorder: Any = None
        self._worker_thread: threading.Thread | None = None
        self._lock: threading.Lock = threading.Lock()

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

        # Initialize the underlying RealtimeSTT engine
        # Whisper performs native transcription only (never translation)
        self._recorder = AudioToTextRecorder(
            model=self.config.model_size,
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

                cleaned: str = sentence.strip() if sentence else ""
                if cleaned and not self._is_paused:
                    lang: str = self._resolve_language(cleaned)
                    self._current_language = lang
                    if self.on_final_text is not None:
                        self.on_final_text(cleaned, lang)

            except Exception:
                time.sleep(0.05)

    def _handle_vad_start(self) -> None:
        """
        Dispatches voice activity start event.
        """

        if self.on_vad_state is not None:
            self.on_vad_state(True)

    def _handle_vad_stop(self) -> None:
        """
        Dispatches voice activity stop event.
        """

        if self.on_vad_state is not None:
            self.on_vad_state(False)

    def _handle_realtime_update(self, text: str) -> None:
        """
        Dispatches in-progress transcription updates.

        Args:
            text (str): Interim transcribed text chunk.
        """

        cleaned_text: str = text.strip()
        if not cleaned_text or self._is_paused:
            return

        # Query detected language
        self._current_language = self._resolve_language(cleaned_text)

        # Emit to partial listener
        if self.on_partial_text is not None:
            self.on_partial_text(cleaned_text, self._current_language)

    def _handle_stabilized_update(self, text: str) -> None:
        """
        Dispatches live stabilized text chunk updates during active speech.

        Args:
            text (str): Stabilized transcribed text chunk.
        """

        cleaned_text: str = text.strip()
        if not cleaned_text or self._is_paused:
            return

        # Refresh language tag
        self._current_language = self._resolve_language(cleaned_text)

        # Emit to partial listener for dynamic preview
        if self.on_partial_text is not None:
            self.on_partial_text(cleaned_text, self._current_language)
