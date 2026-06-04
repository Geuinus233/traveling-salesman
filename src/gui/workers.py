"""Background workers for long-running TSP/NLP jobs."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal


class WorkerThread(QThread):
    finished_ok = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int)  # current, total  (total=0 means indeterminate)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))
