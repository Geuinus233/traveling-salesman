"""Matplotlib canvas embedded in PySide6."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, width: float = 6, height: float = 5, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

    def clear(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)

    def refresh(self):
        try:
            self.fig.tight_layout()
        except Exception:
            pass
        self.draw()
