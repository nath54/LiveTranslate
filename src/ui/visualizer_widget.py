"""
Real-time audio visualizer displaying live amplitude levels, history, and frequency spectrogram.
"""

# Import Modules
from typing import Any

from PySide6.QtGui import QColor, QPainter, QLinearGradient
from PySide6.QtCore import Qt, QRectF
from PySide6.QtWidgets import QWidget


class AudioVisualizerWidget(QWidget):
    """
    Renders a live audio spectrogram with frequency bands, RMS volume, and peak indicators.

    Attributes:
        num_bands (int): Number of frequency spectrum bars to display.
        rms_level (float): Current normalized root mean square energy (0.0 to 1.0).
        peak_level (float): Current peak audio sample amplitude (0.0 to 1.0).
        spectrum_bars (list[float]): Current smoothed frequency bar heights.
    """

    def __init__(self, parent: Any = None, num_bands: int = 24) -> None:
        """
        Initializes the visualizer with frequency bar buffers.

        Args:
            parent (Any): Optional parent Qt widget.
            num_bands (int): Number of visual frequency spectrum bars.
        """

        super().__init__(parent)

        self.num_bands: int = num_bands
        self.rms_level: float = 0.0
        self.peak_level: float = 0.0
        self.spectrum_bars: list[float] = [0.0] * num_bands
        self.amplitude_history: list[float] = [0.0] * 40

        # Configure geometry
        self.setFixedHeight(50)
        self.setMinimumWidth(180)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)

    def update_metrics(
        self,
        rms: float,
        peak: float,
        spectrum_bars: list[float],
    ) -> None:
        """
        Updates live audio metrics and triggers a redraw with exponential falloff smoothing.

        Args:
            rms (float): Root mean square amplitude.
            peak (float): Peak sample amplitude.
            spectrum_bars (list[float]): Energy per frequency band.
        """

        # Update RMS and Peak levels with fast attack, smooth decay
        decay_factor: float = 0.70
        self.rms_level = max(rms, self.rms_level * decay_factor)
        self.peak_level = max(peak, self.peak_level * decay_factor)

        # Update rolling amplitude history
        self.amplitude_history.append(self.rms_level)
        if len(self.amplitude_history) > 40:
            self.amplitude_history.pop(0)

        # Update frequency bars with decay smoothing
        for idx in range(min(len(spectrum_bars), self.num_bands)):
            target_val: float = spectrum_bars[idx]
            current_val: float = self.spectrum_bars[idx]
            self.spectrum_bars[idx] = max(target_val, current_val * decay_factor)

        # Trigger visual redraw
        self.update()

    def paintEvent(self, event: Any) -> None:
        """
        Paints the spectrum bars, level meters, and status overlay.

        Args:
            event (Any): QPaintEvent instance.
        """

        _ = event
        painter: QPainter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width_f: float = float(self.width())
        height_f: float = float(self.height())

        # 1. Background dark rounded plate
        bg_rect: QRectF = QRectF(0.0, 0.0, width_f, height_f)
        painter.fillRect(bg_rect, QColor(16, 18, 26, 180))

        # 2. Draw frequency spectrogram bars
        margin_x: float = 6.0
        avail_width: float = width_f - (margin_x * 2.0)
        bar_gap: float = 2.0
        total_gaps: float = bar_gap * float(self.num_bands - 1)
        bar_width: float = max(2.0, (avail_width - total_gaps) / float(self.num_bands))
        spectrum_bottom: float = height_f - 12.0
        max_bar_height: float = spectrum_bottom - 4.0

        for idx in range(self.num_bands):
            bar_h: float = max(1.0, self.spectrum_bars[idx] * max_bar_height)
            bar_x: float = margin_x + float(idx) * (bar_width + bar_gap)
            bar_y: float = spectrum_bottom - bar_h

            # Gradient from soft cyan to violet
            gradient: QLinearGradient = QLinearGradient(bar_x, bar_y, bar_x, spectrum_bottom)
            gradient.setColorAt(0.0, QColor(136, 192, 208, 220))
            gradient.setColorAt(1.0, QColor(94, 129, 172, 140))

            painter.setBrush(gradient)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_width, bar_h), 1.5, 1.5)

        # 3. Draw bottom horizontal RMS & Peak level meter
        meter_y: float = height_f - 7.0
        meter_h: float = 3.0
        meter_bg: QRectF = QRectF(margin_x, meter_y, avail_width, meter_h)
        painter.fillRect(meter_bg, QColor(46, 52, 64, 160))

        rms_fill_w: float = avail_width * min(1.0, max(0.0, self.rms_level))
        rms_rect: QRectF = QRectF(margin_x, meter_y, rms_fill_w, meter_h)
        meter_color: QColor = (
            QColor(163, 190, 140, 240) if self.rms_level > 0.05 else QColor(76, 86, 106, 180)
        )
        painter.fillRect(rms_rect, meter_color)

        # Peak indicator pip
        peak_x: float = margin_x + (avail_width * min(1.0, max(0.0, self.peak_level)))
        peak_rect: QRectF = QRectF(max(margin_x, peak_x - 1.0), meter_y - 1.0, 2.0, meter_h + 2.0)
        painter.fillRect(peak_rect, QColor(235, 203, 139, 255))

        painter.end()
