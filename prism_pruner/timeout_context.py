"""PRISM - Pruning Interface for Similar Molecules."""

import threading
from ctypes import c_ulong, py_object, pythonapi
from typing import Self, cast


class Timeout:
    """Timeout context manager."""

    def __init__(self, seconds: int = 60, error_message: str = "Timeout") -> None:
        """Initialize Timeout Context Manager."""
        self.seconds = seconds
        self.error_message = error_message
        self._timer: threading.Timer | None = None

    def _raise_timeout(self) -> None:
        pythonapi.PyThreadState_SetAsyncExc(
            c_ulong(self._main_thread_id),
            py_object(TimeoutError),
        )

    def __enter__(self) -> Self:
        """Exit method."""
        self._main_thread_id = cast("int", threading.main_thread().ident)
        self._timer = threading.Timer(self.seconds, self._raise_timeout)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Exit method."""
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
