"""
Header control bar with audio device selector, target language switch, and action buttons.
"""

# Import Modules
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QComboBox,
    QHBoxLayout,
    QPushButton,
)


class ControlHeaderWidget(QWidget):
    """
    Control toolbar containing device dropdown, language toggle, and window operations.

    Attributes:
        speaker_changed (Signal): Emits selected SoundCard speaker object.
        language_changed (Signal): Emits selected target language code ('en' or 'fr').
        pause_toggled (Signal): Emits True when paused, False when listening.
        pin_toggled (Signal): Emits current always-on-top pinned status.
        clear_requested (Signal): Emits when user requests transcript clearing.
        close_requested (Signal): Emits when user clicks close window button.
    """

    speaker_changed: Signal = Signal(object)
    language_changed: Signal = Signal(str)
    pause_toggled: Signal = Signal(bool)
    pin_toggled: Signal = Signal(bool)
    clear_requested: Signal = Signal()
    close_requested: Signal = Signal()

    def __init__(self, parent: Any = None) -> None:
        """
        Initializes the control header toolbar.

        Args:
            parent (Any): Optional parent Qt widget.
        """

        super().__init__(parent)

        # Main layout
        layout: QHBoxLayout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(6)

        # Title and drag grip label
        self.title_label: QLabel = QLabel("LiveTrans")
        self.title_label.setStyleSheet("""
            QLabel {
                color: #D8DEE9;
                font-size: 12px;
                font-weight: bold;
                padding-right: 4px;
            }
        """)
        layout.addWidget(self.title_label)

        # Audio speaker output device dropdown
        self.combo_speaker: QComboBox = QComboBox()
        self.combo_speaker.setToolTip("Select Audio Output to Listen (WASAPI Loopback)")
        self.combo_speaker.setStyleSheet("""
            QComboBox {
                background-color: rgba(46, 52, 64, 0.9);
                color: #E5E9F0;
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 11px;
                min-width: 140px;
                max-width: 200px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2E3440;
                color: #ECEFF4;
                selection-background-color: #4C566A;
            }
        """)
        self.combo_speaker.currentIndexChanged.connect(self._handle_speaker_changed)
        layout.addWidget(self.combo_speaker)

        # Target language dropdown (English / French)
        self.combo_language: QComboBox = QComboBox()
        self.combo_language.setToolTip("Select Target Translation Language")
        self.combo_language.addItem("English (en)", userData="en")
        self.combo_language.addItem("Français (fr)", userData="fr")
        self.combo_language.setStyleSheet("""
            QComboBox {
                background-color: rgba(46, 52, 64, 0.9);
                color: #E5E9F0;
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 4px;
                padding: 3px 6px;
                font-size: 11px;
                max-width: 95px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #2E3440;
                color: #ECEFF4;
                selection-background-color: #4C566A;
            }
        """)
        self.combo_language.currentIndexChanged.connect(self._handle_language_changed)
        layout.addWidget(self.combo_language)

        # Pause / Resume toggle button
        self._is_paused: bool = False
        self.btn_pause: QPushButton = QPushButton("⏸")
        self.btn_pause.setToolTip("Pause / Resume audio listening")
        self._style_button(self.btn_pause, width=26)
        self.btn_pause.clicked.connect(self._toggle_pause_state)
        layout.addWidget(self.btn_pause)

        # Pin / Always-on-top toggle button
        self._is_pinned: bool = True
        self.btn_pin: QPushButton = QPushButton("📌")
        self.btn_pin.setToolTip("Toggle Always on Top")
        self._style_button(self.btn_pin, width=26)
        self.btn_pin.clicked.connect(self._toggle_pin_state)
        layout.addWidget(self.btn_pin)

        # Clear transcript history button
        self.btn_clear: QPushButton = QPushButton("🗑")
        self.btn_clear.setToolTip("Clear transcript history")
        self._style_button(self.btn_clear, width=26)
        self.btn_clear.clicked.connect(self.clear_requested.emit)
        layout.addWidget(self.btn_clear)

        # Close button
        self.btn_close: QPushButton = QPushButton("✕")
        self.btn_close.setToolTip("Close LiveTrans")
        self._style_button(self.btn_close, width=26, is_danger=True)
        self.btn_close.clicked.connect(self.close_requested.emit)
        layout.addWidget(self.btn_close)

    def populate_speakers(self, speakers: list[Any], default_speaker: Any = None) -> None:
        """
        Fills the speaker dropdown with available output devices.

        Args:
            speakers (list[Any]): List of SoundCard speaker instances.
            default_speaker (Any): Initially active speaker.
        """

        self.combo_speaker.blockSignals(True)
        self.combo_speaker.clear()

        # Add each detected speaker
        selected_index: int = 0
        for index, speaker in enumerate(speakers):
            label: str = str(
                getattr(speaker, "display_label", getattr(speaker, "name", f"Device {index}"))
            )
            self.combo_speaker.addItem(label, userData=speaker)
            if default_speaker is not None and getattr(speaker, "name", "") == getattr(
                default_speaker, "name", ""
            ):
                selected_index = index

        self.combo_speaker.setCurrentIndex(selected_index)
        self.combo_speaker.blockSignals(False)

    def set_target_language(self, language_code: str) -> None:
        """
        Sets the active target language in the dropdown.

        Args:
            language_code (str): 'en' or 'fr'.
        """

        self.combo_language.blockSignals(True)
        for idx in range(self.combo_language.count()):
            if self.combo_language.itemData(idx) == language_code:
                self.combo_language.setCurrentIndex(idx)
                break
        self.combo_language.blockSignals(False)

    def _style_button(
        self,
        button: QPushButton,
        width: int = 26,
        is_danger: bool = False,
    ) -> None:
        """
        Applies flat translucent styling to toolbar action buttons.

        Args:
            button (QPushButton): Button instance to style.
            width (int): Fixed button width in pixels.
            is_danger (bool): Highlights button with red accent if True.
        """

        hover_color: str = "#BF616A" if is_danger else "#4C566A"
        button.setFixedSize(width, 24)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: rgba(46, 52, 64, 0.8);
                color: #ECEFF4;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 4px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {hover_color};
            }}
        """)

    def _handle_speaker_changed(self, index: int) -> None:
        """
        Emits signal when user selects a different speaker.

        Args:
            index (int): Dropdown combo index.
        """

        speaker: Any = self.combo_speaker.itemData(index)
        if speaker is not None:
            self.speaker_changed.emit(speaker)

    def _handle_language_changed(self, index: int) -> None:
        """
        Emits signal when user selects a different translation language.

        Args:
            index (int): Dropdown combo index.
        """

        lang_code: str = str(self.combo_language.itemData(index))
        self.language_changed.emit(lang_code)

    def _toggle_pause_state(self) -> None:
        """
        Toggles pause state between listening and halted.
        """

        self._is_paused = not self._is_paused
        self.btn_pause.setText("▶" if self._is_paused else "⏸")
        self.pause_toggled.emit(self._is_paused)

    def _toggle_pin_state(self) -> None:
        """
        Toggles always-on-top pinned status.
        """

        self._is_pinned = not self._is_pinned
        self.btn_pin.setStyleSheet(f"""
            QPushButton {{
                background-color: {"#5E81AC" if self._is_pinned else "rgba(46, 52, 64, 0.8)"};
                color: #ECEFF4;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 4px;
                font-size: 11px;
            }}
        """)
        self.pin_toggled.emit(self._is_pinned)
