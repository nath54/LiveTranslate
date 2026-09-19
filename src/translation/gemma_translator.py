"""
Translation client communicating with hosted translategemma4b using httpx and requests.
"""

# Import Modules
from dataclasses import dataclass
from typing import Any
import time
import re

import requests
import httpx

from src.config import TranslationConfig


@dataclass
class DialogueTurn:
    """
    Represents a single conversational turn containing source and translated text.

    Attributes:
        source_text (str): Original utterance in source language.
        source_language (str): Language code of the original text.
        translated_text (str): Translated output sentence.
        target_language (str): Destination language code.
        timestamp (float): UNIX epoch timestamp when turn occurred.
    """

    source_text: str
    source_language: str
    translated_text: str
    target_language: str
    timestamp: float


class GemmaTranslator:
    """
    Translates transcribed sentences into English or French using hosted translategemma4b.

    Attributes:
        config (TranslationConfig): Settings for endpoint URL, model, and network parameters.
        history (list[DialogueTurn]): Memory buffer of prior utterances for conversational context.
    """

    def __init__(self, config: TranslationConfig) -> None:
        """
        Initializes the translation client with httpx and requests sessions.

        Args:
            config (TranslationConfig): Client configuration settings.
        """

        # Set configuration and initialize conversation history
        self.config: TranslationConfig = config
        self.history: list[DialogueTurn] = []

        # Persistent HTTP clients for performance
        self._httpx_client: httpx.Client = httpx.Client(
            timeout=config.request_timeout,
        )
        self._requests_session: requests.Session = requests.Session()

    def add_turn(
        self,
        source_text: str,
        source_language: str,
        translated_text: str,
        target_language: str,
    ) -> None:
        """
        Appends a completed dialogue turn into the conversational memory buffer.

        Args:
            source_text (str): Original spoken text.
            source_language (str): Original language code.
            translated_text (str): Resulting translation.
            target_language (str): Target language code.
        """

        # Create dialogue turn record
        turn: DialogueTurn = DialogueTurn(
            source_text=source_text.strip(),
            source_language=source_language.strip(),
            translated_text=translated_text.strip(),
            target_language=target_language.strip(),
            timestamp=time.time(),
        )

        # Append and maintain maximum history window
        self.history.append(turn)
        if len(self.history) > self.config.history_max_turns:
            self.history.pop(0)

    def clear_history(self) -> None:
        """
        Empties the current conversational memory buffer.
        """

        self.history.clear()

    def get_history_summary(self) -> str:
        """
        Formats recent turns into a textual context block for the LLM prompt.

        Returns:
            str: Multi-line string summarizing prior dialogue.
        """

        if not self.history:
            return ""

        # Format each turn
        lines: list[str] = []
        for turn in self.history:
            entry: str = f"\"{turn.source_text}\" -> \"{turn.translated_text}\""
            line: str = f"- {turn.source_language}: {entry}"
            lines.append(line)

        formatted_context: str = "\n".join(lines)

        return formatted_context

    def _build_system_prompt(self, target_lang: str) -> str:
        """
        Builds the system instruction tailored for translategemma4b.

        Args:
            target_lang (str): Target language code or name.

        Returns:
            str: System instruction string.
        """

        lang_name: str = "French" if target_lang.lower().startswith("fr") else "English"

        prompt: str = (
            f"You are a fast, high-accuracy stream translator. "
            f"Translate the given text into {lang_name}. "
            "Output ONLY the direct translation. Do not add notes, explanations, romaji, or quotes."
        )

        return prompt

    def _build_payload(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
    ) -> dict[str, Any]:
        """
        Constructs the OpenAI-compatible chat completions payload with conversational history.

        Args:
            text (str): Source sentence to translate.
            source_lang (str): Source language code or auto.
            target_lang (str): Destination language code.

        Returns:
            dict[str, Any]: JSON payload dictionary.
        """

        _ = source_lang
        system_instruction: str = self._build_system_prompt(target_lang)
        lang_name: str = "French" if target_lang.lower().startswith("fr") else "English"

        # Build clean chat completion messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_instruction},
        ]

        # Add recent conversation turns as user/assistant pairs (capped at 2 turns)
        max_context: int = min(len(self.history), self.config.history_max_turns, 2)
        if max_context > 0:
            for turn in self.history[-max_context:]:
                messages.append({
                    "role": "user",
                    "content": f"Translate to {lang_name}:\n{turn.source_text}",
                })
                messages.append({
                    "role": "assistant",
                    "content": turn.translated_text,
                })

        # Add current sentence request
        messages.append({
            "role": "user",
            "content": f"Translate to {lang_name}:\n{text}",
        })

        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": 256,
            "stop": [
                "<|im_start|>",
                "<|im_end|>",
                "<end_of_turn>",
                "<start_of_turn>",
                "<|file_separator|>",
                "<|endoftext|>",
                "<eos>",
                "\n\n",
            ],
            "presence_penalty": 0.1,
            "frequency_penalty": 0.1,
            "stream": False,
        }

        return payload

    def _request_with_httpx(self, payload: dict[str, Any]) -> str:
        """
        Executes the translation request using httpx.

        Args:
            payload (dict[str, Any]): HTTP POST JSON payload.

        Returns:
            str: Translated text returned by server.
        """

        response: httpx.Response = self._httpx_client.post(
            self.config.server_url,
            json=payload,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        content: str = data["choices"][0]["message"]["content"]

        return content.strip()

    def _request_with_requests(self, payload: dict[str, Any]) -> str:
        """
        Executes the translation request using the requests library.

        Args:
            payload (dict[str, Any]): HTTP POST JSON payload.

        Returns:
            str: Translated text returned by server.
        """

        response: requests.Response = self._requests_session.post(
            self.config.server_url,
            json=payload,
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        data: dict[str, Any] = response.json()
        content: str = str(data["choices"][0]["message"]["content"])

        return content.strip()

    @staticmethod
    def _clean_response(raw_text: str) -> str:
        """
        Strips model control tokens and wrapping quotation marks from translated text.

        Args:
            raw_text (str): Raw model completion output.

        Returns:
            str: Cleaned translation string.
        """

        cleaned: str = raw_text.strip()

        # Remove known model template artifacts
        template_tokens: tuple[str, ...] = (
            "<|file_separator|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<start_of_turn>",
            "<end_of_turn>",
            "<eos>",
        )
        for token in template_tokens:
            cleaned = cleaned.replace(token, "")

        # Strip any residual bracketed special tokens
        cleaned = re.sub(r"<\|.*?\|>", "", cleaned)

        # Strip accidental quotes and surrounding whitespace
        cleaned = cleaned.strip("\"' \t\r\n")

        return cleaned

    def translate(
        self,
        text: str,
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> str:
        """
        Translates source text into target language using configured HTTP backend and history.

        Args:
            text (str): Spoken sentence to translate.
            source_lang (str): Spoken language code.
            target_lang (str): Desired output language ('en' or 'fr').

        Returns:
            str: Translated sentence or an error indicator if offline.
        """

        cleaned_text: str = text.strip()
        if not cleaned_text:
            return ""

        # Prepare request payload
        payload: dict[str, Any] = self._build_payload(
            text=cleaned_text,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # Dispatch via preferred HTTP client engine
        translated_result: str
        try:
            if self.config.http_client.lower() == "requests":
                translated_result = self._request_with_requests(payload)
            else:
                translated_result = self._request_with_httpx(payload)

            # Sanitize response from model template tokens and wrapping quotes
            translated_result = self._clean_response(translated_result)

            # Record turn in history
            self.add_turn(
                source_text=cleaned_text,
                source_language=source_lang,
                translated_text=translated_result,
                target_language=target_lang,
            )

            return translated_result

        except Exception as error:
            # Return descriptive offline status rather than crashing
            err_msg: str = f"[Translation Server Offline: {type(error).__name__}]"
            return err_msg

    def close(self) -> None:
        """
        Closes underlying HTTP client sessions.
        """

        try:
            self._httpx_client.close()
        except Exception:
            pass

        try:
            self._requests_session.close()
        except Exception:
            pass
