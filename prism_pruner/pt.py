"""PRISM - PRuning Interface for Similar Molecules."""

from periodictable import core, covalent_radius, mass

for pt_n in range(5):
    try:
        pt = core.PeriodicTable(table=f"H={pt_n + 1}")
        covalent_radius.init(pt)
        mass.init(pt)
    except ValueError:
        continue
    break
