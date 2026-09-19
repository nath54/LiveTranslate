"""
Audio resampling and channel downmixing utilities.
"""

# Import Modules
import math

import numpy as np
from numpy.typing import NDArray
from scipy.signal import resample_poly


class AudioResampler:
    """
    Converts multi-channel audio arrays to mono and resamples to target sample rate.

    Attributes:
        source_rate (int): Input audio sampling frequency in Hz.
        target_rate (int): Desired output audio sampling frequency in Hz.
        up_factor (int): Numerator for polyphase resampling filter.
        down_factor (int): Denominator for polyphase resampling filter.
    """

    def __init__(self, source_rate: int = 48000, target_rate: int = 16000) -> None:
        """
        Initializes the AudioResampler with input and target rates.

        Args:
            source_rate (int): Sampling rate of the source audio stream.
            target_rate (int): Desired target sampling rate.
        """

        # Set sampling rates
        self.source_rate: int = source_rate
        self.target_rate: int = target_rate

        # Compute greatest common divisor to optimize resampling factors
        gcd_val: int = math.gcd(source_rate, target_rate)
        self.up_factor: int = target_rate // gcd_val
        self.down_factor: int = source_rate // gcd_val

    def to_mono(self, audio_data: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Downmixes multi-channel audio to a single mono channel.

        Args:
            audio_data (NDArray[np.float32]): Input audio array with shape (N, C) or (N,).

        Returns:
            NDArray[np.float32]: Mono audio array with shape (N,).
        """

        # Verify dimensionality and average channels if stereo or multi-channel
        if audio_data.ndim == 1:
            return audio_data

        # Compute mean across channels
        mono_array: NDArray[np.float32] = np.mean(
            audio_data,
            axis=1,
            dtype=np.float32,
        )

        return mono_array

    def resample(self, mono_audio: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Resamples a mono audio array from source_rate to target_rate.

        Args:
            mono_audio (NDArray[np.float32]): 1D mono audio array.

        Returns:
            NDArray[np.float32]: Resampled 1D mono audio array at target_rate.
        """

        # If rates match, skip resampling calculation
        if self.source_rate == self.target_rate:
            return mono_audio

        # Check for empty input array
        if mono_audio.size == 0:
            return np.empty(0, dtype=np.float32)

        # Apply polyphase FIR filtering for fast resampling
        resampled_raw: NDArray[np.float32] = resample_poly(
            mono_audio,
            up=self.up_factor,
            down=self.down_factor,
        ).astype(np.float32)

        return resampled_raw

    def process(self, audio_data: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Converts input audio to mono and resamples it to the target rate in one step.

        Args:
            audio_data (NDArray[np.float32]): Raw multi-channel audio data from soundcard.

        Returns:
            NDArray[np.float32]: Processed 16kHz mono audio slice for Whisper.
        """

        # Downmix audio to mono
        mono_audio: NDArray[np.float32] = self.to_mono(audio_data)

        # Resample mono audio to target frequency
        processed_audio: NDArray[np.float32] = self.resample(mono_audio)

        return processed_audio
