"""PRISM - PRuning Interface for Similar Molecules."""

from typing import Any

from periodictable import core, covalent_radius, mass


class PeriodicTableWrapper:
    """Wrapper around periodictable that handles numpy strings."""

    def __init__(self, table: core.PeriodicTable):
        self._table = table

    def __getitem__(self, key: Any) -> Any:
        """Get element by symbol, handling numpy strings."""
        # Convert numpy strings to Python strings
        if hasattr(key, "item"):
            key = key.item()
        # Use getattr for symbol access since periodictable uses attributes
        return getattr(self._table, key)

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the underlying table."""
        return getattr(self._table, name)


# Initialize the periodic table
for pt_n in range(5):
    try:
        _pt_raw = core.PeriodicTable(table=f"H={pt_n + 1}")
        covalent_radius.init(_pt_raw)
        mass.init(_pt_raw)
    except ValueError:
        continue
    break

pt = PeriodicTableWrapper(_pt_raw)
