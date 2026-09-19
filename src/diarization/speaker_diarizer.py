"""
Speaker diarization module using CAM++ ONNX model for online speaker identification.
"""

# Import Modules
from typing import Any

import os
import re
import logging
import threading

import numpy as np
import torch
import torchaudio
import onnxruntime as ort
from numpy.typing import NDArray
from huggingface_hub import hf_hub_download

from src.config import DiarizationConfig


logger: logging.Logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """
    Manages online speaker tracking, acoustic embedding extraction, and turn-splitting.

    Attributes:
        config (DiarizationConfig): Diarization configuration parameters.
        _session (ort.InferenceSession | None): Loaded ONNX Runtime inference session.
        _centroids (list[tuple[str, NDArray[np.float32]]]): Active speaker cluster centroids.
        _lock (threading.Lock): Concurrency lock for thread-safe clustering.
        _active_threshold (float): Active cosine similarity threshold.
    """

    def __init__(self, config: DiarizationConfig) -> None:
        """
        Initializes the speaker diarizer with model configuration.

        Args:
            config (DiarizationConfig): Speaker diarization configuration instance.
        """

        self.config: DiarizationConfig = config
        self._session: ort.InferenceSession | None = None
        self._centroids: list[tuple[str, NDArray[np.float32]]] = []
        self._lock: threading.Lock = threading.Lock()
        self._active_threshold: float = config.similarity_threshold
        self._last_identified_speaker: str = "Speaker 1"

    def load_model(self) -> None:
        """
        Loads the CAM++ ONNX model from HuggingFace cache or downloads it if missing.
        """

        # Avoid redundant loading
        with self._lock:
            if self._session is not None:
                return

        # Disable HuggingFace symlinks on Windows
        os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

        try:
            # Download or load model weights
            model_path: str = hf_hub_download(
                repo_id=self.config.model_repo,
                filename=self.config.model_filename,
            )

            # Initialize ONNX inference session on CPU
            sess_options: ort.SessionOptions = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session: ort.InferenceSession = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

            with self._lock:
                self._session = session

        except Exception as exc:
            logger.error("Failed to load CAM++ speaker model: %s", exc)

    def extract_embedding(
        self,
        audio_slice: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> NDArray[np.float32] | None:
        """
        Extracts a 192-dimensional L2-normalized speaker embedding from audio.

        Args:
            audio_slice (NDArray[np.float32]): 1D mono float32 audio waveform.
            sample_rate (int): Audio sampling rate in Hz (expected 16000).

        Returns:
            NDArray[np.float32] | None: 192-dim normalized embedding array, or None.
        """

        # Ensure model is initialized
        if self._session is None:
            self.load_model()

        if self._session is None or audio_slice.size == 0:
            return None

        # Pad very short slices to minimum window duration (0.3s)
        min_samples: int = int(sample_rate * 0.3)
        processed_audio: NDArray[np.float32] = audio_slice
        if len(processed_audio) < min_samples:
            pad_len: int = min_samples - len(processed_audio)
            processed_audio = np.pad(processed_audio, (0, pad_len))

        # Scale waveform to 16-bit range for Kaldi fbank
        waveform: torch.Tensor = (
            torch.from_numpy(processed_audio).float().unsqueeze(0) * 32768.0
        )

        # Compute 80-bin Mel Filterbank features
        fbank: torch.Tensor = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            sample_frequency=sample_rate,
            dither=0.0,
        )

        # Apply Cepstral Mean Normalization (CMVN)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        fbank_np: NDArray[np.float32] = fbank.numpy().astype(np.float32)

        # Run ONNX inference
        emb_output: Any = self._session.run(
            None,
            {"feats": fbank_np[np.newaxis, ...]},
        )
        raw_embedding: NDArray[np.float32] = emb_output[0][0].astype(np.float32)

        # L2-normalize embedding
        norm: float = float(np.linalg.norm(raw_embedding))
        if norm > 0.0:
            raw_embedding = (raw_embedding / norm).astype(np.float32)

        return raw_embedding

    def identify_or_register(
        self,
        audio_slice: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> tuple[str, float]:
        """
        Identifies active speaker or registers a new cluster centroid.

        Args:
            audio_slice (NDArray[np.float32]): Audio waveform chunk.
            sample_rate (int): Sampling rate in Hz.

        Returns:
            tuple[str, float]: Tuple of (speaker_id, cosine_similarity).
        """

        # Check duration guard
        duration_sec: float = len(audio_slice) / float(sample_rate)
        if duration_sec < self.config.min_speech_duration:
            return self._last_identified_speaker, 1.0

        # Extract acoustic embedding
        embedding: NDArray[np.float32] | None = self.extract_embedding(
            audio_slice=audio_slice,
            sample_rate=sample_rate,
        )
        if embedding is None:
            return self._last_identified_speaker, 1.0

        with self._lock:
            # First speaker registration
            if not self._centroids:
                first_label: str = "Speaker 1"
                self._centroids.append((first_label, embedding))
                self._last_identified_speaker = first_label
                return first_label, 1.0

            # Compare against existing centroids
            sims: list[float] = [
                float(np.dot(embedding, centroid_vec))
                for _, centroid_vec in self._centroids
            ]
            best_idx: int = int(np.argmax(sims))
            best_sim: float = sims[best_idx]
            best_label: str = self._centroids[best_idx][0]

            # Match existing speaker if above threshold
            if best_sim >= self._active_threshold:
                # Update centroid with Exponential Moving Average (EMA)
                old_vec: NDArray[np.float32] = self._centroids[best_idx][1]
                updated_vec: NDArray[np.float32] = (
                    0.80 * old_vec + 0.20 * embedding
                ).astype(np.float32)
                up_norm: float = float(np.linalg.norm(updated_vec))
                if up_norm > 0.0:
                    updated_vec = (updated_vec / up_norm).astype(np.float32)
                self._centroids[best_idx] = (best_label, updated_vec)
                self._last_identified_speaker = best_label
                return best_label, best_sim

            # If dissimilar and budget permits, register new speaker
            if len(self._centroids) < self.config.max_speakers:
                new_idx: int = len(self._centroids) + 1
                new_label: str = f"Speaker {new_idx}"
                self._centroids.append((new_label, embedding))
                self._last_identified_speaker = new_label
                return new_label, best_sim

            # Fallback to nearest speaker centroid
            self._last_identified_speaker = best_label
            return best_label, best_sim

    def detect_speaker_change(
        self,
        audio_slice_a: NDArray[np.float32],
        audio_slice_b: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> tuple[bool, float]:
        """
        Determines whether two distinct audio segments originate from different speakers.

        Args:
            audio_slice_a (NDArray[np.float32]): First audio segment.
            audio_slice_b (NDArray[np.float32]): Second audio segment.
            sample_rate (int): Sampling rate in Hz.

        Returns:
            tuple[bool, float]: Tuple of (is_different_speaker, cosine_similarity).
        """

        # Extract embeddings for both segments
        emb_a: NDArray[np.float32] | None = self.extract_embedding(
            audio_slice=audio_slice_a,
            sample_rate=sample_rate,
        )
        emb_b: NDArray[np.float32] | None = self.extract_embedding(
            audio_slice=audio_slice_b,
            sample_rate=sample_rate,
        )

        if emb_a is None or emb_b is None:
            return False, 1.0

        # Calculate cosine similarity
        cos_sim: float = float(np.dot(emb_a, emb_b))
        is_change: bool = cos_sim < self._active_threshold
        return is_change, cos_sim

    def split_dialogue(
        self,
        text: str,
        words_meta: list[dict[str, Any]],
        full_audio: NDArray[np.float32],
        sample_rate: int = 16000,
    ) -> list[tuple[str, str]]:
        """
        Analyzes phrase punctuation and word timestamps to split dialogue turns.

        Args:
            text (str): Full transcribed text.
            words_meta (list[dict[str, Any]]): Word-level metadata timestamps from Whisper.
            full_audio (NDArray[np.float32]): Full recorded utterance audio waveform.
            sample_rate (int): Sampling rate in Hz.

        Returns:
            list[tuple[str, str]]: List of (turn_text, speaker_label) segments.
        """

        # Default fallback if insufficient data
        if not words_meta or full_audio.size == 0:
            spk, _ = self.identify_or_register(
                audio_slice=full_audio,
                sample_rate=sample_rate,
            )
            return [(text, spk)]

        # Find terminal punctuation marks inside words
        terminal_marks: tuple[str, ...] = ("？", "?", "。", "！", "!")
        split_candidates: list[int] = []

        for w_idx, w_item in enumerate(words_meta[:-1]):
            w_text: str = str(w_item.get("word", ""))
            if any(term in w_text for term in terminal_marks):
                split_candidates.append(w_idx)

        # If no internal terminal punctuation found, treat as single speaker
        if not split_candidates:
            spk, _ = self.identify_or_register(
                audio_slice=full_audio,
                sample_rate=sample_rate,
            )
            return [(text, spk)]

        # Inspect first major split point
        split_w_idx: int = split_candidates[0]
        words_a: list[dict[str, Any]] = words_meta[: split_w_idx + 1]
        words_b: list[dict[str, Any]] = words_meta[split_w_idx + 1 :]

        time_split: float = float(words_a[-1].get("end", 0.0))
        split_sample: int = int(time_split * sample_rate)

        # Ensure valid boundary
        if split_sample <= 0 or split_sample >= len(full_audio):
            spk, _ = self.identify_or_register(
                audio_slice=full_audio,
                sample_rate=sample_rate,
            )
            return [(text, spk)]

        audio_part_a: NDArray[np.float32] = full_audio[:split_sample]
        audio_part_b: NDArray[np.float32] = full_audio[split_sample:]

        # Verify both parts exceed minimum duration
        min_pts: int = int(sample_rate * 0.40)
        if len(audio_part_a) >= min_pts and len(audio_part_b) >= min_pts:
            is_different, _ = self.detect_speaker_change(
                audio_slice_a=audio_part_a,
                audio_slice_b=audio_part_b,
                sample_rate=sample_rate,
            )

            if is_different:
                # Identify speakers for each half
                spk_a, _ = self.identify_or_register(
                    audio_slice=audio_part_a,
                    sample_rate=sample_rate,
                )
                spk_b, _ = self.identify_or_register(
                    audio_slice=audio_part_b,
                    sample_rate=sample_rate,
                )

                text_a: str = "".join(
                    str(w.get("word", "")) for w in words_a
                ).strip()
                text_b: str = "".join(
                    str(w.get("word", "")) for w in words_b
                ).strip()

                word_pattern: str = (
                    r"[a-zA-Z0-9\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]"
                )
                if re.search(word_pattern, text_a) and re.search(word_pattern, text_b):
                    return [(text_a, spk_a), (text_b, spk_b)]

        # Same speaker throughout
        spk, _ = self.identify_or_register(
            audio_slice=full_audio,
            sample_rate=sample_rate,
        )
        return [(text, spk)]

    def set_similarity_threshold(self, threshold: float) -> None:
        """
        Updates the active cosine similarity clustering threshold.

        Args:
            threshold (float): New similarity threshold.
        """

        with self._lock:
            self._active_threshold = threshold

    def reset_speakers(self) -> None:
        """
        Clears all recorded speaker centroids and resets registry.
        """

        with self._lock:
            self._centroids.clear()
            self._last_identified_speaker = "Speaker 1"
