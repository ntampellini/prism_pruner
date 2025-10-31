"""Confguration for pytest."""

import pytest
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from pytest import Config


def pytest_addoption(parser: Parser) -> None:
    """Add max duration option."""
    parser.addoption(
        "--timing",
        action="store_true",
        help="Run timing tests [default: False].",
    )


def pytest_collection_modifyitems(config: Config, items: list[Item]) -> None:
    """Modify which tests are run."""
    # If --timing is set, only run tests marked as timing
    for item in items:
        if config.getoption("--timing"):
            if "timing" not in (marker.name for marker in item.iter_markers()):
                item.add_marker(pytest.mark.skip(reason="Skipping non-timing test due to --timing"))
        elif "timing" in (marker.name for marker in item.iter_markers()):
            item.add_marker(pytest.mark.skip(reason="Skipping timing test"))
