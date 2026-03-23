"""PRISM - Pruning Interface for Similar Molecules."""

import signal
from typing import Any


class Timeout:
    """Timeout context manager."""

    def __init__(self, seconds: int = 60, error_message: str = "Timeout") -> None:
        """Define the __init__ method of the context manager."""
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum: Any, frame: Any) -> None:
        """Handle the timeout signal."""
        raise TimeoutError(self.error_message)

    def __enter__(self) -> None:
        """Define the __enter__ method of the context manager."""
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        """Define the __exit__ method of the context manager."""
        signal.alarm(0)
