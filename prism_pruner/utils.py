"""PRISM - PRuning Interface for Similar Molecules."""

from typing import Any, Sequence

import numpy as np
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike

from prism_pruner.algebra import get_alignment_matrix, rot_mat_from_pointer
from prism_pruner.typing import Array1D_bool, Array1D_int, Array1D_str, Array2D_float, Array3D_float

EH_TO_EV = 27.211399
EH_TO_KCAL = 627.5096080305927
EV_TO_KCAL = 23.060541945329334


def align_structures(
    structures: Array3D_float, indices: Array1D_int | None = None
) -> Array3D_float:
    """Align structures.

    Aligns molecules of a structure array (shape is (n_structures, n_atoms, 3))
    to the first one, based on the indices. If not provided, all atoms are used
    to get the best alignment. Return is the aligned array.
    """
    reference = structures[0]
    targets = structures[1:]
    if isinstance(indices, (list, tuple)):
        indices = np.array(indices)

    indices = indices if indices is not None else np.array([i for i, _ in enumerate(structures[0])])

    reference -= np.mean(reference[indices], axis=0)
    for t, _ in enumerate(targets):
        targets[t] -= np.mean(targets[t, indices], axis=0)

    output = np.zeros(structures.shape)
    output[0] = reference

    for t, target in enumerate(targets):
        try:
            matrix = get_alignment_matrix(reference[indices], target[indices])

        except LinAlgError:
            # it is actually possible for the kabsch alg not to converge
            matrix = np.eye(3)

        # output[t+1] = np.array([matrix @ vector for vector in target])
        output[t + 1] = (matrix @ target.T).T

    return output


def time_to_string(total_time: float, verbose: bool = False, digits: int = 1) -> str:
    """Convert totaltime (float) to a timestring with hours, minutes and seconds."""
    timestring = ""

    names = ("days", "hours", "minutes", "seconds") if verbose else ("d", "h", "m", "s")

    if total_time > 24 * 3600:
        d = total_time // (24 * 3600)
        timestring += f"{int(d)} {names[0]} "
        total_time %= 24 * 3600

    if total_time > 3600:
        h = total_time // 3600
        timestring += f"{int(h)} {names[1]} "
        total_time %= 3600

    if total_time > 60:
        m = total_time // 60
        timestring += f"{int(m)} {names[2]} "
        total_time %= 60

    timestring += f"{round(total_time, digits):{2 + digits}} {names[3]}"

    return timestring


double_bonds_thresholds_dict = {
    "CC": 1.4,
    "CN": 1.3,
}


def get_double_bonds_indices(coords: Array2D_float, atoms: Array1D_str) -> list[tuple[int, int]]:
    """Return a list containing 2-elements tuples of indices involved in any double bond."""
    mask = atoms != "H"
    numbering = np.arange(len(coords))[mask]
    coords = coords[mask]
    atoms_masked = atoms[mask]
    output = []

    for i1, _ in enumerate(coords):
        for i2 in range(i1 + 1, len(coords)):
            dist = np.linalg.norm(coords[i1] - coords[i2])
            tag = "".join(sorted([atoms_masked[i1], atoms_masked[i2]]))

            threshold = double_bonds_thresholds_dict.get(tag)
            if threshold is not None and dist < threshold:
                output.append((numbering[i1], numbering[i2]))

    return output


def rotate_dihedral(
    coords: Array2D_float,
    dihedral: list[int] | tuple[int, ...],
    angle: float,
    mask: Array1D_bool | None = None,
    indices_to_be_moved: ArrayLike | None = None,
) -> Array2D_float:
    """Rotate a molecule around a given bond.

    Atoms that will move are the ones
    specified by mask or indices_to_be_moved.
    If both are None, only the first index of
    the dihedral iterable is moved.

    angle: angle, in degrees
    """
    i1, i2, i3, *_ = dihedral

    if indices_to_be_moved is not None:
        mask = np.isin(np.arange(len(coords)), indices_to_be_moved)

    if mask is None:
        mask = np.zeros(len(coords), dtype=bool)
        mask[i1] = True

    axis = coords[i2] - coords[i3]
    mat = rot_mat_from_pointer(axis, angle)

    center = coords[i3]
    coords[mask] = (coords[mask] - center) @ mat.T + center

    return coords


def flatten(array: Sequence[Any], typefunc: type = float) -> list[Any]:
    """Return the unraveled sequence, with items coerced into the typefunc type."""
    out = []

    def rec(_l: Any) -> None:
        """Recursive unraveling function."""
        for e in _l:
            if type(e) in [list, tuple, np.ndarray]:
                rec(e)
            else:
                out.append(typefunc(e))

    rec(array)
    return out
