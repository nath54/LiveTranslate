"""
Movable, resizable, semi-transparent desktop overlay widget with live audio visualizer.
"""

# Import Modules
from typing import Any

import threading

from PySide6.QtCore import QPoint, Qt, QObject, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QWidget,
    QSizeGrip,
    QHBoxLayout,
    QScrollArea,
    QVBoxLayout,
)

from src.config import AppConfig
from src.ui.subtitle_card import SubtitleCard
from src.ui.device_selector import ControlHeaderWidget
from src.ui.visualizer_widget import AudioVisualizerWidget
from src.romanizer.text_romanizer import TextRomanizer
from src.audio.loopback_capture import AudioLoopbackCapture
from src.translation.gemma_translator import GemmaTranslator


class OverlaySignals(QObject):
    """
    Qt signal dispatcher for thread-safe cross-thread GUI updates.

    Attributes:
        partial_speech_received (Signal): Emits (partial_text, language).
        final_speech_received (Signal): Emits (card_id, text, roman, language, speaker).
        translation_ready (Signal): Emits (card_id, translation_text).
        status_updated (Signal): Emits status message string.
        audio_metrics_received (Signal): Emits (rms, peak, spectrum_bars).
        vad_state_changed (Signal): Emits True if speech active, False if silent.
    """

    partial_speech_received: Signal = Signal(str, str)
    final_speech_received: Signal = Signal(int, str, str, str, str)
    translation_ready: Signal = Signal(int, str)
    status_updated: Signal = Signal(str)
    audio_metrics_received: Signal = Signal(float, float, list)
    vad_state_changed: Signal = Signal(bool)


class LiveTransOverlay(QWidget):
    """
    Main semi-transparent, resizable, movable floating desktop overlay window.

    Attributes:
        config (AppConfig): Aggregated application configuration.
        audio_monitor (AudioLoopbackCapture): Audio loopback capture monitor.
        romanizer (TextRomanizer): CJK phonetic romanization engine.
        translator (GemmaTranslator): TranslateGemma-4B LLM client.
    """

    def __init__(
        self,
        config: AppConfig,
        audio_monitor: AudioLoopbackCapture,
        romanizer: TextRomanizer,
        translator: GemmaTranslator,
        parent: Any = None,
    ) -> None:
        """
        Initializes the translucent overlay widget and sets up signals and UI layout.

        Args:
            config (AppConfig): Master settings instance.
            audio_monitor (AudioLoopbackCapture): Audio capture service.
            romanizer (TextRomanizer): Romanization utility.
            translator (GemmaTranslator): Translation service.
            parent (Any): Optional Qt parent widget.
        """

        super().__init__(parent)

        # Core services
        self.config: AppConfig = config
        self.audio_monitor: AudioLoopbackCapture = audio_monitor
        self.romanizer: TextRomanizer = romanizer
        self.translator: GemmaTranslator = translator

        # Window state
        self._drag_active: bool = False
        self._drag_position: QPoint = QPoint()
        self._is_paused: bool = False
        self._next_card_id: int = 1
        self._card_registry: dict[int, SubtitleCard] = {}

        # Signal dispatcher for thread safety
        self.signals: OverlaySignals = OverlaySignals()
        self.signals.partial_speech_received.connect(self._on_partial_speech)
        self.signals.final_speech_received.connect(self._on_final_speech)
        self.signals.translation_ready.connect(self._on_translation_ready)
        self.signals.status_updated.connect(self._on_status_updated)
        self.signals.audio_metrics_received.connect(self._on_audio_metrics)
        self.signals.vad_state_changed.connect(self._on_vad_state)

        # Hook audio monitor metrics
        self.audio_monitor.on_audio_metrics = self._dispatch_audio_metrics

        # Configure window flags and translucent background
        self._setup_window_properties()

        # Build UI layout
        self._build_ui()

        # Populate audio devices
        self._refresh_audio_devices()

    def _setup_window_properties(self) -> None:
        """
        Applies frameless, translucent, and top-level window configuration.
        """

        # Set window flags
        flags: Qt.WindowType = (
            Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setWindowFlags(flags)

        # Enable translucent surface
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        # Initial geometry
        self.resize(self.config.ui.window_width, self.config.ui.window_height)
        self.setMinimumSize(340, 240)

    def _build_ui(self) -> None:
        """
        Constructs the widget visual hierarchy, header toolbar, scroll area, and status footer.
        """

        # Outer container frame to hold rounded translucent styling
        outer_layout: QVBoxLayout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        # Translucent glassmorphic body frame
        self.body_frame: QFrame = QFrame(self)
        self.body_frame.setObjectName("BodyFrame")
        opacity_val: float = self.config.ui.window_opacity
        self.body_frame.setStyleSheet(f"""
            QFrame#BodyFrame {{
                background-color: rgba(20, 22, 32, {opacity_val});
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 10px;
            }}
        """)

        body_layout: QVBoxLayout = QVBoxLayout(self.body_frame)
        body_layout.setContentsMargins(6, 6, 6, 6)
        body_layout.setSpacing(4)
        outer_layout.addWidget(self.body_frame)

        # 1. Header toolbar
        self.header: ControlHeaderWidget = ControlHeaderWidget(self.body_frame)
        self.header.speaker_changed.connect(self._handle_speaker_changed)
        self.header.input_language_changed.connect(self._handle_input_language_changed)
        self.header.language_changed.connect(self._handle_target_language_changed)
        self.header.pace_changed.connect(self._handle_pace_changed)
        self.header.pause_toggled.connect(self._handle_pause_toggled)
        self.header.pin_toggled.connect(self._handle_pin_toggled)
        self.header.clear_requested.connect(self._handle_clear_requested)
        self.header.close_requested.connect(self.close)

        # Synchronize header dropdowns with initial configuration
        self.header.set_input_language(self.config.stt.language)
        self.header.set_target_language(self.config.translation.target_language)
        body_layout.addWidget(self.header)

        # 2. Live Audio Spectrogram & Amplitude Visualizer
        self.visualizer: AudioVisualizerWidget = AudioVisualizerWidget(
            parent=self.body_frame,
            num_bands=24,
        )
        body_layout.addWidget(self.visualizer)

        # 3. Live Audio Level & VAD Status Badge
        info_bar: QWidget = QWidget(self.body_frame)
        info_layout: QHBoxLayout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(4, 1, 4, 1)
        info_layout.setSpacing(6)

        self.vad_badge: QLabel = QLabel("⚪ Silence")
        self.vad_badge.setStyleSheet("""
            QLabel {
                color: #D8DEE9;
                font-size: 11px;
                font-weight: bold;
                background-color: rgba(46, 52, 64, 0.7);
                padding: 2px 6px;
                border-radius: 4px;
            }
        """)
        info_layout.addWidget(self.vad_badge)

        self.audio_level_label: QLabel = QLabel("Vol: 0%")
        self.audio_level_label.setStyleSheet("""
            QLabel {
                color: #88C0D0;
                font-size: 11px;
            }
        """)
        info_layout.addWidget(self.audio_level_label)
        info_layout.addStretch()

        self.device_info_label: QLabel = QLabel("Output Loopback Active")
        self.device_info_label.setStyleSheet("""
            QLabel {
                color: #81A1C1;
                font-size: 10px;
            }
        """)
        info_layout.addWidget(self.device_info_label)
        body_layout.addWidget(info_bar)

        # 4. Scrollable transcript cards area
        self.scroll_area: QScrollArea = QScrollArea(self.body_frame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: rgba(30, 32, 44, 0.5);
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: rgba(255, 255, 255, 0.25);
                border-radius: 3px;
            }
        """)

        self.cards_container: QWidget = QWidget()
        self.cards_container.setStyleSheet("background: transparent;")
        self.cards_layout: QVBoxLayout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(2, 2, 2, 2)
        self.cards_layout.setSpacing(4)
        self.cards_layout.addStretch()

        self.scroll_area.setWidget(self.cards_container)
        body_layout.addWidget(self.scroll_area)

        # 5. Live in-progress speech preview card
        self.live_preview: SubtitleCard = SubtitleCard(
            parent=self.body_frame,
            source_text="Ready. Audio output loopback active...",
            roman_text="",
            translation_text="",
            language="Live",
        )
        body_layout.addWidget(self.live_preview)

        # 6. Status footer with size grip
        footer_widget: QWidget = QWidget(self.body_frame)
        footer_layout: QHBoxLayout = QHBoxLayout(footer_widget)
        footer_layout.setContentsMargins(4, 2, 2, 2)
        footer_layout.setSpacing(6)

        self.status_label: QLabel = QLabel("Status: Listening")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #88C0D0;
                font-size: 10px;
            }
        """)
        footer_layout.addWidget(self.status_label)
        footer_layout.addStretch()

        # Resizing grip in bottom-right corner
        self.size_grip: QSizeGrip = QSizeGrip(footer_widget)
        self.size_grip.setStyleSheet("background: transparent;")
        grip_alignment: Qt.AlignmentFlag = (
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight
        )
        footer_layout.addWidget(self.size_grip, 0, grip_alignment)

        body_layout.addWidget(footer_widget)

    def _refresh_audio_devices(self) -> None:
        """
        Discovers all soundcard speaker output devices and populates header selector.
        """

        speakers: list[Any] = self.audio_monitor.get_all_audio_devices()
        current_name: str = self.audio_monitor.get_current_speaker_name()

        # Find matching speaker
        matched_speaker: Any = None
        for speaker in speakers:
            if getattr(speaker, "name", "") == current_name:
                matched_speaker = speaker
                break

        self.header.populate_speakers(speakers, default_speaker=matched_speaker)
        self.header.set_target_language(self.config.translation.target_language)
        self.header.set_pace_mode(self.config.pace_mode)
        self.device_info_label.setText(f"🔊 {current_name}")

    def mousePressEvent(self, event: Any) -> None:
        """
        Initiates window movement when left-clicking outside interactive controls.

        Args:
            event (Any): QMouseEvent instance.
        """

        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_active = True
            self._drag_position = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        """
        Repositions the window during active mouse drag.

        Args:
            event (Any): QMouseEvent instance.
        """

        if self._drag_active and event.buttons() & Qt.MouseButton.LeftButton:
            new_pos: QPoint = event.globalPosition().toPoint() - self._drag_position
            self.move(new_pos)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        """
        Terminates window drag mode on mouse release.

        Args:
            event (Any): QMouseEvent instance.
        """

        self._drag_active = False
        super().mouseReleaseEvent(event)

    def _dispatch_audio_metrics(
        self,
        rms: float,
        peak: float,
        spectrum_bars: list[float],
    ) -> None:
        """
        Helper method receiving metrics from audio thread and emitting Qt signal.

        Args:
            rms (float): RMS amplitude.
            peak (float): Peak amplitude.
            spectrum_bars (list[float]): Energy per frequency band.
        """

        self.signals.audio_metrics_received.emit(rms, peak, spectrum_bars)

    def _on_audio_metrics(
        self,
        rms: float,
        peak: float,
        spectrum_bars: list[float],
    ) -> None:
        """
        Slot updating audio visualizer and volume meters on the GUI thread.

        Args:
            rms (float): RMS amplitude.
            peak (float): Peak amplitude.
            spectrum_bars (list[float]): Energy per frequency band.
        """

        # Update visualizer canvas
        self.visualizer.update_metrics(rms=rms, peak=peak, spectrum_bars=spectrum_bars)

        # Update text readout
        peak_pct: int = int(peak * 100.0)
        rms_pct: int = int(rms * 100.0)
        self.audio_level_label.setText(f"Peak: {peak_pct}% | RMS: {rms_pct}%")

    def _on_vad_state(self, is_speaking: bool) -> None:
        """
        Slot updating the voice activity detection status badge.

        Args:
            is_speaking (bool): True if speech detected, False otherwise.
        """

        if is_speaking:
            self.vad_badge.setText("🔴 Speech Active")
            self.vad_badge.setStyleSheet("""
                QLabel {
                    color: #ECEFF4;
                    font-size: 11px;
                    font-weight: bold;
                    background-color: rgba(191, 97, 106, 0.85);
                    padding: 2px 6px;
                    border-radius: 4px;
                }
            """)
        else:
            self.vad_badge.setText("⚪ Silence")
            self.vad_badge.setStyleSheet("""
                QLabel {
                    color: #D8DEE9;
                    font-size: 11px;
                    font-weight: bold;
                    background-color: rgba(46, 52, 64, 0.7);
                    padding: 2px 6px;
                    border-radius: 4px;
                }
            """)

    def _handle_speaker_changed(self, speaker: Any) -> None:
        """
        Hot-swaps the audio recording output device.

        Args:
            speaker (Any): Selected SoundCard speaker instance.
        """

        # Perform hot-swap on monitor
        self.audio_monitor.switch_speaker(speaker)
        speaker_name: str = str(getattr(speaker, "name", "Selected Speaker"))
        self.device_info_label.setText(f"🔊 {speaker_name}")
        self.signals.status_updated.emit(f"Switched to: {speaker_name}")

    def _handle_input_language_changed(self, lang_code: str) -> None:
        """
        Handles audio input language updates from header dropdown.

        Args:
            lang_code (str): 'auto' or ISO language code ('zh', 'ja', 'en', etc.).
        """

        clean_code: str = "" if lang_code.lower() in ("auto", "") else lang_code.lower()
        self.config.stt.language = clean_code
        display_code: str = clean_code.upper() if clean_code else "AUTO"
        self.signals.status_updated.emit(f"Input language: {display_code}")

    def _handle_target_language_changed(self, lang_code: str) -> None:
        """
        Updates the target language for upcoming translations.

        Args:
            lang_code (str): 'en' or 'fr'.
        """

        self.config.translation.target_language = lang_code
        self.signals.status_updated.emit(f"Target language set to: {lang_code.upper()}")

    def _handle_pace_changed(self, pace_code: str) -> None:
        """
        Handles conversational pace mode updates from header dropdown.

        Args:
            pace_code (str): 'auto', 'fast', 'medium', or 'accurate'.
        """

        self.config.pace_mode = pace_code
        self.signals.status_updated.emit(f"Pace mode: {pace_code.upper()}")

    def _handle_pause_toggled(self, is_paused: bool) -> None:
        """
        Pauses or resumes transcription pipeline.

        Args:
            is_paused (bool): True if paused, False if active.
        """

        self._is_paused = is_paused
        if is_paused:
            self.signals.status_updated.emit("Paused")
        else:
            self.signals.status_updated.emit("Listening")

    def _handle_pin_toggled(self, is_pinned: bool) -> None:
        """
        Toggles always on top flag for the window.

        Args:
            is_pinned (bool): True to stay on top, False otherwise.
        """

        # Modify top hint
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, is_pinned)
        self.show()

    def _handle_clear_requested(self) -> None:
        """
        Removes all historical transcript cards and resets conversational LLM memory.
        """

        # Clear translator history
        self.translator.clear_history()

        # Remove cards from layout
        for card in self._card_registry.values():
            card.deleteLater()

        self._card_registry.clear()
        self.live_preview.update_content(
            source_text="History cleared. Listening...",
            roman_text="",
            translation_text="",
            language="Live",
        )
        self.signals.status_updated.emit("History cleared")

    def on_vad_state_changed(self, is_speaking: bool) -> None:
        """
        Called by WhisperStreamer when speech begins or pauses.

        Args:
            is_speaking (bool): True if speech detected, False otherwise.
        """

        self.signals.vad_state_changed.emit(is_speaking)

    def on_partial_transcription(self, text: str, language: str) -> None:
        """
        Called by WhisperStreamer on live interim speech.

        Args:
            text (str): Interim text from Whisper.
            language (str): Detected language code.
        """

        if self._is_paused:
            return

        self.signals.partial_speech_received.emit(text, language)

    def on_final_transcription(
        self,
        text: str,
        language: str,
        speaker: str = "",
    ) -> None:
        """
        Called by WhisperStreamer when speech ends and sentence is finalized.

        Args:
            text (str): Finalized source transcription.
            language (str): Detected language code.
            speaker (str): Optional identified speaker label.
        """

        if self._is_paused:
            return

        # Compute romanization immediately
        roman_text: str = self.romanizer.romanize(text=text, language=language)

        # Allocate unique card ID
        card_id: int = self._next_card_id
        self._next_card_id += 1

        # Dispatch final speech to GUI thread
        self.signals.final_speech_received.emit(
            card_id, text, roman_text, language, speaker
        )

        # Check if translation should be skipped
        target_lang: str = self.config.translation.target_language.strip().lower()
        skip_langs: list[str] = [
            lang.strip().lower() for lang in self.config.translation.skip_languages
        ]
        lang_lower: str = language.strip().lower()

        if lang_lower in skip_langs or lang_lower == target_lang:
            return

        # Launch background translation request
        threading.Thread(
            target=self._execute_translation_task,
            args=(card_id, text, language),
            daemon=True,
        ).start()

    def _execute_translation_task(
        self,
        card_id: int,
        text: str,
        language: str,
    ) -> None:
        """
        Performs LLM translation in worker thread and dispatches result to GUI.

        Args:
            card_id (int): Card identifier to update.
            text (str): Spoken sentence.
            language (str): Spoken language code.
        """

        # Request translation from TranslateGemma-4B
        translated: str = self.translator.translate(
            text=text,
            source_lang=language,
            target_lang=self.config.translation.target_language,
        )

        # Dispatch translation back to GUI thread
        self.signals.translation_ready.emit(card_id, translated)

    def _on_partial_speech(self, text: str, language: str) -> None:
        """
        Slot handling partial speech text updates in main GUI thread.

        Args:
            text (str): In-progress text.
            language (str): Detected language.
        """

        # Generate live romanization
        roman: str = self.romanizer.romanize(text=text, language=language)

        # Update preview card
        self.live_preview.update_content(
            source_text=text,
            roman_text=roman,
            translation_text="",
            language=f"{language.upper()} • Streaming",
        )

    def _on_final_speech(
        self,
        card_id: int,
        text: str,
        roman_text: str,
        language: str,
        speaker: str,
    ) -> None:
        """
        Slot adding a finalized subtitle card into the history feed.

        Args:
            card_id (int): Assigned card ID.
            text (str): Spoken text.
            roman_text (str): Romanized text.
            language (str): Language code.
            speaker (str): Identified speaker label.
        """

        # Create new permanent subtitle card
        card: SubtitleCard = SubtitleCard(
            parent=self.cards_container,
            source_text=text,
            roman_text=roman_text,
            translation_text="",
            language=language,
            speaker=speaker,
        )

        # Insert before bottom stretch
        count: int = self.cards_layout.count()
        self.cards_layout.insertWidget(max(0, count - 1), card)
        self._card_registry[card_id] = card

        # Reset preview card
        self.live_preview.update_content(
            source_text="Listening on audio output...",
            roman_text="",
            translation_text="",
            language="Live",
        )

        # Scroll to bottom
        self._scroll_to_bottom()

    def _on_translation_ready(self, card_id: int, translation: str) -> None:
        """
        Slot updating existing subtitle card with completed translation.

        Args:
            card_id (int): Card to update.
            translation (str): Translated output text.
        """

        card: SubtitleCard | None = self._card_registry.get(card_id)
        if card is not None and translation.strip():
            card.translation_label.setText(translation)
            card.translation_label.setVisible(True)

        self._scroll_to_bottom()

    def _on_status_updated(self, status: str) -> None:
        """
        Slot updating the status footer text.

        Args:
            status (str): Status description.
        """

        self.status_label.setText(f"Status: {status}")

    def _scroll_to_bottom(self) -> None:
        """
        Scrolls the scroll area to the latest subtitle card.
        """

        v_bar: Any = self.scroll_area.verticalScrollBar()
        if v_bar is not None:
            v_bar.setValue(v_bar.maximum())
