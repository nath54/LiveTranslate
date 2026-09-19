"""
Text romanization module for Korean, Japanese, and Chinese scripts.
"""

# Import Modules
from typing import Any

import re

import pykakasi
import pypinyin
from hangul_romanize import Transliter
from hangul_romanize.rule import academic


class TextRomanizer:
    """
    Transliterates Asian non-Latin scripts (Hangul, Kana/Kanji, Hanzi) into readable Roman letters.

    Attributes:
        _korean_transliter (Transliter): Academic transliteration engine for Korean Hangul.
        _kakasi (pykakasi.kakasi): Hepburn transliteration engine for Japanese.
    """

    def __init__(self) -> None:
        """
        Initializes the Korean, Japanese, and Chinese romanization engines.
        """

        # Korean romanizer using standard academic rules
        self._korean_transliter: Transliter = Transliter(academic)

        # Japanese converter using Kakasi Hepburn style
        self._kakasi: Any = pykakasi.kakasi()

    def has_hangul(self, text: str) -> bool:
        """
        Checks if text contains Korean Hangul characters.

        Args:
            text (str): Source text string to inspect.

        Returns:
            bool: True if Hangul characters exist in text.
        """

        # Hangul syllables and jamo unicode range
        has_korean: bool = bool(re.search(r"[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]", text))

        return has_korean

    def has_kana(self, text: str) -> bool:
        """
        Checks if text contains Japanese Hiragana or Katakana characters.

        Args:
            text (str): Source text string to inspect.

        Returns:
            bool: True if Japanese Kana exists in text.
        """

        # Hiragana and Katakana unicode ranges
        has_japanese_kana: bool = bool(re.search(r"[\u3040-\u309f\u30a0-\u30ff]", text))

        return has_japanese_kana

    def has_hanzi(self, text: str) -> bool:
        """
        Checks if text contains CJK Unified Ideographs.

        Args:
            text (str): Source text string to inspect.

        Returns:
            bool: True if Hanzi / Chinese characters exist in text.
        """

        # CJK unified ideographs range
        has_cjk: bool = bool(re.search(r"[\u4e00-\u9fff]", text))

        return has_cjk

    def romanize_korean(self, text: str) -> str:
        """
        Transliterates Korean Hangul into Revised Romanization.

        Args:
            text (str): Source text containing Hangul.

        Returns:
            str: Romanized Korean string.
        """

        # Transliterate Hangul syllables
        try:
            romanized: str = self._korean_transliter.translit(text)
            return romanized.strip()
        except Exception:
            return text

    def romanize_japanese(self, text: str) -> str:
        """
        Transliterates Japanese Kanji, Hiragana, and Katakana into Hepburn Romaji.

        Args:
            text (str): Source Japanese text.

        Returns:
            str: Romanized Japanese string in Hepburn Romaji.
        """

        # Convert using kakasi
        try:
            results: list[dict[str, str]] = self._kakasi.convert(text)
            romaji_tokens: list[str] = [
                token.get("hepburn", "") for token in results if token.get("hepburn")
            ]
            return " ".join(romaji_tokens).strip()
        except Exception:
            return text

    def romanize_chinese(self, text: str) -> str:
        """
        Transliterates Chinese characters into Pinyin with tone markers.

        Args:
            text (str): Source Chinese text.

        Returns:
            str: Romanized Chinese string with tone diacritics.
        """

        # Convert using pypinyin with diacritic tone marks
        try:
            pinyin_list: list[str] = pypinyin.lazy_pinyin(
                text,
                style=pypinyin.Style.TONE,
            )
            return " ".join(pinyin_list).strip()
        except Exception:
            return text

    def romanize(self, text: str, language: str = "") -> str:
        """
        Automatically selects and executes the appropriate romanization based on script or code.

        Args:
            text (str): Original transcribed text.
            language (str): Detected ISO language code (e.g. 'ko', 'ja', 'zh').

        Returns:
            str: Phonetic Roman transcription or original text if already Latin.
        """

        cleaned_text: str = text.strip()
        if not cleaned_text:
            return ""

        # Language code matching
        normalized_lang: str = language.lower().strip()

        # Check Korean
        if normalized_lang.startswith("ko") or self.has_hangul(cleaned_text):
            return self.romanize_korean(cleaned_text)

        # Check Japanese (Kana or marked as ja)
        if normalized_lang.startswith("ja") or self.has_kana(cleaned_text):
            return self.romanize_japanese(cleaned_text)

        # Check Chinese
        if normalized_lang.startswith("zh") or self.has_hanzi(cleaned_text):
            return self.romanize_chinese(cleaned_text)

        # Return unchanged for Latin scripts
        return cleaned_text
