"""ConformerEnsemble class."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

import numpy as np

from prism_pruner.typing import Array1D_float, Array1D_str, Array2D_float, Array3D_float


@dataclass
class ConformerEnsemble:
    """Class representing a conformer ensemble."""

    coords: Array3D_float
    atoms: Array1D_str
    energies: Array1D_float = field(default_factory=lambda: np.array([]))

    @classmethod
    def from_xyz(cls, file: Path | str, read_energies: bool = False) -> Self:
        """Generate ensemble from a multiple conformer xyz file."""
        coords = []
        atoms = []
        energies = []
        with Path(file).open() as f:
            for num in f:
                if read_energies:
                    energy = next(re.finditer(r"-*\d+\.\d+", next(f))).group()
                    energies.append(float(energy))
                else:
                    _comment = next(f)

                conf_atoms = []
                conf_coords = []
                for _ in range(int(num)):
                    atom, *xyz = next(f).split()
                    conf_atoms.append(atom)
                    conf_coords.append([float(x) for x in xyz])

                atoms.append(conf_atoms)
                coords.append(conf_coords)

        return cls(coords=np.array(coords), atoms=np.array(atoms[0]), energies=np.array(energies))

    def to_xyz(self, file: Path | str) -> None:
        """Write ensemble to an xyz file."""

        def to_xyz(coords: Array2D_float) -> str:
            return "\n".join(
                f"{atom} {x:15.8f} {y:15.8f} {z:15.8f}"
                for atom, (x, y, z) in zip(self.atoms, coords, strict=True)
            )

        with Path(file).open("w") as f:
            f.write("\n".join(map(to_xyz, self.coords)))
