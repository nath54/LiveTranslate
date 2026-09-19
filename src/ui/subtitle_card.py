"""
Subtitle display card component showing original transcript, romanization, and translation.
"""

# Import Modules
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QVBoxLayout,
)


class SubtitleCard(QFrame):
    """
    Styled visual card containing the three tiers: Original text, Romanized text, and Translation.

    Attributes:
        source_label (QLabel): Label showing original transcribed characters and language badge.
        roman_label (QLabel): Label displaying phonetic transliteration (Romaji/Pinyin/RR).
        translation_label (QLabel): Label showing English or French translation.
    """

    def __init__(
        self,
        parent: Any = None,
        source_text: str = "",
        roman_text: str = "",
        translation_text: str = "",
        language: str = "auto",
    ) -> None:
        """
        Initializes the subtitle card with three distinct text tiers.

        Args:
            parent (Any): Optional parent Qt widget.
            source_text (str): Spoken original transcription.
            roman_text (str): Romanized phonetic reading.
            translation_text (str): Final translated sentence.
            language (str): Detected ISO language code.
        """

        super().__init__(parent)

        # Apply visual card styling with subtle borders and dark translucent fill
        self.setObjectName("SubtitleCard")
        self.setStyleSheet("""
            QFrame#SubtitleCard {
                background-color: rgba(32, 35, 48, 0.75);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 6px;
                margin-bottom: 4px;
            }
        """)

        # Main layout for card
        layout: QVBoxLayout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(3)

        # 1. Source original transcription label
        self.source_label: QLabel = QLabel()
        self.source_label.setWordWrap(True)
        self.source_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.source_label.setStyleSheet("""
            QLabel {
                color: #ECEFF4;
                font-size: 13px;
                font-weight: 500;
            }
        """)
        layout.addWidget(self.source_label)

        # 2. Romanization label (Romaji / Pinyin / Revised Romanization)
        self.roman_label: QLabel = QLabel()
        self.roman_label.setWordWrap(True)
        self.roman_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.roman_label.setStyleSheet("""
            QLabel {
                color: #88C0D0;
                font-size: 12px;
                font-style: italic;
            }
        """)
        layout.addWidget(self.roman_label)

        # 3. Translation label
        self.translation_label: QLabel = QLabel()
        self.translation_label.setWordWrap(True)
        self.translation_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.translation_label.setStyleSheet("""
            QLabel {
                color: #A3BE8C;
                font-size: 13px;
                font-weight: 600;
            }
        """)
        layout.addWidget(self.translation_label)

        # Populate initial content
        self.update_content(
            source_text=source_text,
            roman_text=roman_text,
            translation_text=translation_text,
            language=language,
        )

    def update_content(
        self,
        source_text: str,
        roman_text: str,
        translation_text: str,
        language: str = "auto",
    ) -> None:
        """
        Updates the displayed text across all three tiers.

        Args:
            source_text (str): Whisper original transcript.
            roman_text (str): Phonetic transliteration.
            translation_text (str): Target translation.
            language (str): Spoken language code.
        """

        # Format language badge
        lang_tag: str = language.upper() if language else "AUTO"
        formatted_source: str = f"[{lang_tag}] {source_text}" if source_text else ""
        self.source_label.setText(formatted_source)

        # Format romanized text
        formatted_roman: str = f"↳ {roman_text}" if roman_text else ""
        self.roman_label.setText(formatted_roman)
        self.roman_label.setVisible(bool(roman_text and roman_text != source_text))

        # Format translated text
        self.translation_label.setText(translation_text)
        self.translation_label.setVisible(bool(translation_text))
