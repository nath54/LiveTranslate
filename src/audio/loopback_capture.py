"""
Audio loopback capture service with dynamic speaker hot-swapping and live AGC metrics.
"""

# Import Modules
from typing import Any
from collections.abc import Callable

import time
import threading

import numpy as np
import soundcard as sc
from numpy.typing import NDArray

from src.audio.audio_resampler import AudioResampler


class AudioLoopbackCapture:
    """
    Captures loopback audio from Windows WASAPI output devices and computes live metrics.

    Attributes:
        sample_rate (int): Source capture rate in Hz.
        target_sample_rate (int): Resampled target rate in Hz.
        buffer_frames (int): Number of frames recorded per audio slice.
        on_audio_chunk (Callable[[NDArray[np.float32]], None] | None): Receiver callback for STT.
        on_audio_metrics (Callable[[float, float, list[float]], None] | None): Metrics callback.
        is_active (bool): Current loopback listening status.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        target_sample_rate: int = 16000,
        buffer_frames: int = 1024,
        on_audio_chunk: Callable[[NDArray[np.float32]], None] | None = None,
        on_audio_metrics: Callable[[float, float, list[float]], None] | None = None,
    ) -> None:
        """
        Initializes the audio loopback monitor.

        Args:
            sample_rate (int): Sampling rate for loopback recording.
            target_sample_rate (int): Desired output sampling rate for Whisper.
            buffer_frames (int): Audio slice size in frames.
            on_audio_chunk (Callable[[NDArray[np.float32]], None] | None): Audio chunk callback.
            on_audio_metrics (Callable[[float, float, list[float]], None] | None): Metrics callback.
        """

        # Configuration variables
        self.sample_rate: int = sample_rate
        self.target_sample_rate: int = target_sample_rate
        self.buffer_frames: int = buffer_frames
        self.on_audio_chunk: Callable[[NDArray[np.float32]], None] | None = on_audio_chunk
        self.on_audio_metrics: (
            Callable[[float, float, list[float]], None] | None
        ) = on_audio_metrics

        # Audio preprocessing utility
        self._resampler: AudioResampler = AudioResampler(
            source_rate=sample_rate,
            target_rate=target_sample_rate,
        )

        # Threading and lifecycle flags
        self._is_running: bool = False
        self._switch_pending: bool = False
        self._capture_thread: threading.Thread | None = None

        # Speaker management
        self._current_speaker: Any = sc.default_speaker()
        self._pending_speaker: Any = None

    @staticmethod
    def get_available_speakers() -> list[Any]:
        """
        Enumerates all active audio output devices on the machine.

        Returns:
            list[Any]: List of SoundCard speaker instances.
        """

        # Fetch speakers from soundcard WASAPI subsystem
        speakers: list[Any] = sc.all_speakers()

        return speakers

    @staticmethod
    def get_all_audio_devices() -> list[Any]:
        """
        Enumerates both output speakers (for loopback) and input microphones.

        Returns:
            list[Any]: Combined list of soundcard audio devices.
        """

        devices: list[Any] = []

        # Add speakers
        for speaker in sc.all_speakers():
            setattr(speaker, "display_label", f"🔊 {speaker.name}")
            devices.append(speaker)

        # Add microphones
        for mic in sc.all_microphones(include_loopback=False):
            setattr(mic, "display_label", f"🎙 {mic.name}")
            devices.append(mic)

        return devices

    def get_current_speaker_name(self) -> str:
        """
        Retrieves the name of the currently monitored speaker device.

        Returns:
            str: Human-readable device name.
        """

        # Return speaker name or placeholder
        if self._current_speaker is not None:
            return str(getattr(self._current_speaker, "name", "Default Device"))

        return "Default Output Device"

    def switch_speaker(self, new_speaker: Any) -> None:
        """
        Hot-swaps the current recording loopback device to a new speaker.

        Args:
            new_speaker (Any): SoundCard speaker instance or device identifier.
        """

        # Schedule switch for the worker loop
        self._pending_speaker = new_speaker
        self._switch_pending = True

    def start(self) -> None:
        """
        Starts the background loopback audio capture thread.
        """

        # Verify whether already running
        if self._is_running:
            return

        # Initialize thread and state
        self._is_running = True
        self._switch_pending = False
        self._capture_thread = threading.Thread(
            target=self._capture_worker,
            daemon=True,
        )
        self._capture_thread.start()

    def stop(self) -> None:
        """
        Stops the loopback capture thread and releases audio resources.
        """

        # Signal termination
        self._is_running = False
        self._switch_pending = True

        # Wait for thread to finish
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=1.5)
            self._capture_thread = None

    def is_running(self) -> bool:
        """
        Checks if the capture loop is currently active.

        Returns:
            bool: True if recording, False otherwise.
        """

        return self._is_running

    def _resolve_recording_device(self, target_device: Any) -> Any:
        """
        Resolves the recording device, converting speakers into loopback microphones.

        Args:
            target_device (Any): SoundCard speaker or microphone instance.

        Returns:
            Any: SoundCard recording microphone instance.
        """

        # If already a microphone, use directly
        if hasattr(target_device, "isloopback"):
            return target_device

        device_id: str = str(getattr(target_device, "id", ""))
        device_name: str = str(getattr(target_device, "name", ""))

        # Try finding loopback by device ID
        try:
            mic: Any = sc.get_microphone(id=device_id, include_loopback=True)
            return mic
        except Exception:
            pass

        # Try finding loopback by device name
        try:
            mic = sc.get_microphone(id=device_name, include_loopback=True)
            return mic
        except Exception:
            pass

        # Fallback to default speaker loopback microphone
        default_spk: Any = sc.default_speaker()
        return sc.get_microphone(id=str(default_spk.name), include_loopback=True)

    def _compute_spectrum_bars(
        self,
        mono_audio: NDArray[np.float32],
        num_bands: int = 24,
    ) -> list[float]:
        """
        Computes normalized AC frequency energy bars for live spectrogram display in dB.

        Args:
            mono_audio (NDArray[np.float32]): DC-centered mono audio frame slice.
            num_bands (int): Number of visual frequency bands.

        Returns:
            list[float]: Normalized energy values between 0.0 and 1.0.
        """

        if mono_audio.size < num_bands + 1:
            return [0.0] * num_bands

        # Compute real FFT magnitude and skip DC bin 0 to remove phantom DC bar
        fft_mags: NDArray[np.float32] = np.abs(np.fft.rfft(mono_audio)).astype(np.float32)
        ac_mags: NDArray[np.float32] = fft_mags[1:]

        # Split into frequency bands
        band_splits: list[NDArray[np.float32]] = np.array_split(ac_mags, num_bands)
        energies: list[float] = []

        for band in band_splits:
            if band.size > 0:
                mean_mag: float = float(np.mean(band))
                # Convert magnitude to decibels with -60 dB floor
                db_energy: float = 20.0 * float(np.log10(max(mean_mag, 1e-6)))
                scaled_val: float = float(np.clip((db_energy + 60.0) / 60.0, 0.0, 1.0))
                energies.append(scaled_val)
            else:
                energies.append(0.0)

        return energies

    def _process_and_dispatch(self, raw_audio: NDArray[np.float32]) -> None:
        """
        Downmixes, calculates visualizer metrics, applies AGC, and dispatches to listeners.

        Args:
            raw_audio (NDArray[np.float32]): Captured audio block.
        """

        # Downmix multi-channel to mono
        mono_audio: NDArray[np.float32] = self._resampler.to_mono(raw_audio)
        if mono_audio.size == 0:
            return

        # Remove DC bias offset from sound hardware
        dc_offset: float = float(np.mean(mono_audio))
        centered_audio: NDArray[np.float32] = mono_audio - dc_offset

        # Calculate live visualizer metrics on centered audio
        peak_val: float = float(np.max(np.abs(centered_audio)))
        rms_val: float = float(np.sqrt(np.mean(centered_audio ** 2)))

        # Convert peak and RMS to decibel-based VU scale (0.0 to 1.0)
        db_peak: float = 20.0 * float(np.log10(max(peak_val, 1e-6)))
        vu_peak: float = float(np.clip((db_peak + 55.0) / 55.0, 0.0, 1.0))

        db_rms: float = 20.0 * float(np.log10(max(rms_val, 1e-6)))
        vu_rms: float = float(np.clip((db_rms + 55.0) / 55.0, 0.0, 1.0))

        # Compute AC frequency spectrum bars
        spectrum_bars: list[float] = self._compute_spectrum_bars(
            mono_audio=centered_audio,
            num_bands=24,
        )

        # Dispatch metrics to UI visualizer
        if self.on_audio_metrics is not None:
            self.on_audio_metrics(vu_rms, vu_peak, spectrum_bars)

        # Resample mono audio to 16kHz for Whisper
        resampled_slice: NDArray[np.float32] = self._resampler.resample(centered_audio)

        # Automatic Gain Control (AGC): amplify quiet YouTube streams to healthy speech level
        target_rms: float = 0.06
        gain_boost: float = 1.0
        if rms_val > 0.0003:
            gain_boost = float(np.clip(target_rms / rms_val, 1.0, 30.0))

        boosted_slice: NDArray[np.float32] = (resampled_slice * gain_boost).astype(np.float32)
        boosted_slice = np.clip(boosted_slice, -0.98, 0.98)

        # Dispatch slice to Whisper callback if registered
        if self.on_audio_chunk is not None and boosted_slice.size > 0:
            self.on_audio_chunk(boosted_slice)

    def _capture_worker(self) -> None:
        """
        Internal worker method continuously reading loopback audio frames.
        """

        # Main supervision loop
        while self._is_running:
            # Handle pending speaker switch
            if self._switch_pending and self._pending_speaker is not None:
                self._current_speaker = self._pending_speaker
                self._pending_speaker = None
                self._switch_pending = False

            # Ensure speaker exists
            if self._current_speaker is None:
                time.sleep(0.1)
                continue

            # Resolve loopback recording microphone for the speaker
            try:
                rec_mic: Any = self._resolve_recording_device(self._current_speaker)
            except Exception:
                time.sleep(0.2)
                continue

            # Determine channel count (use 2 channels, or device native if less)
            target_channels: int = min(2, int(getattr(rec_mic, "channels", 2)))

            # Open recorder on resolved microphone
            try:
                with rec_mic.recorder(
                    samplerate=self.sample_rate,
                    channels=target_channels,
                ) as recorder:
                    while self._is_running and not self._switch_pending:
                        raw_data: NDArray[np.float32] = recorder.record(
                            numframes=self.buffer_frames,
                        )
                        self._process_and_dispatch(raw_data)

            except Exception:
                # Brief pause before retrying upon hardware disconnect or format change
                time.sleep(0.2)
