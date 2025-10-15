"""PRISM - PRuning Interface for Similar Molecules."""

import time
from typing import Any, Callable, TextIO

import numpy as np
from cclib.io.ccio import ccread
from numpy.linalg import LinAlgError
from numpy.typing import ArrayLike

from prism_pruner.algebra import get_alignment_matrix, norm_of, rot_mat_from_pointer
from prism_pruner.pt import pt
from prism_pruner.typing import Array1D_bool, Array1D_int, Array2D_float, Array3D_float


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

    indices = indices or np.array([i for i, _ in enumerate(structures[0])])

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


def write_xyz(
    coords: Array2D_float, atomnos: Array1D_int, output: TextIO, title: str = "temp"
) -> None:
    """Write xyz coordinates to a TextIO file."""
    assert atomnos.shape[0] == coords.shape[0]
    assert coords.shape[1] == 3
    string = ""
    string += str(len(coords))
    string += f"\n{title}\n"
    for i, atom in enumerate(coords):
        string += "%s     % .6f % .6f % .6f\n" % (pt[atomnos[i]].symbol, atom[0], atom[1], atom[2])
    output.write(string)


def read_xyz(filename: str) -> Any:
    """Ccread wrapper. Raises an error if unsuccessful."""
    mol = ccread(filename)
    assert mol is not None, f"Reading molecule {filename} failed - check its integrity."
    return mol


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


def pretty_num(n: float) -> str:
    """Prettify float values to nice strings."""
    if n < 1e3:
        return str(n)
    if n < 1e6:
        return str(round(n / 1e3, 2)) + " k"
    return str(round(n / 1e6, 2)) + " M"


def loadbar(
    iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 50,
    fill: str = "#",
) -> None:
    """Print a loading bar."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end="\r")
    if iteration == total:
        print()


def cartesian_product(*arrays: ArrayLike) -> Any:
    """Compute the cartesian product of generic arrays."""
    return np.stack(np.meshgrid(*arrays), -1).reshape(-1, len(arrays))


double_bonds_thresholds_dict = {
    "CC": 1.4,
    "CN": 1.3,
}


def get_double_bonds_indices(coords: Array2D_float, atomnos: Array1D_int) -> list[tuple[int, int]]:
    """Return a list containing 2-elements tuples of indices involved in any double bond."""
    mask = atomnos != 1
    numbering = np.arange(len(coords))[mask]
    coords = coords[mask]
    atomnos = atomnos[mask]
    output = []

    for i1, _ in enumerate(coords):
        for i2 in range(i1 + 1, len(coords)):
            dist = norm_of(coords[i1] - coords[i2])
            tag = "".join(sorted([pt[atomnos[i1]].symbol, pt[atomnos[i2]].symbol]))

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
    i1, i2, i3, _ = dihedral

    if indices_to_be_moved is not None:
        mask = np.isin(np.arange(len(coords)), indices_to_be_moved)

    if mask is None:
        mask = np.array([[i1]])

    axis = coords[i2] - coords[i3]
    mat = rot_mat_from_pointer(axis, angle)
    center = coords[i3]

    coords[mask] = (mat @ (coords[mask] - center).T).T + center

    return coords


def timing_wrapper(
    function: Callable[..., object],
    *args: Any,
    payload: Any = None,
    **kwargs: Any,
) -> tuple[Any, Any, float] | tuple[Any, float]:
    """Time the execution of a given function, and return an optional additional payload.

    Generic function wrapper that appends the
    execution time at the end of return.
    If payload is not None, appends it at the end
    of the function return, before the elapsed time.
    """
    start_time = time.perf_counter()
    func_return = function(*args, **kwargs)
    elapsed = time.perf_counter() - start_time

    if payload is None:
        return func_return, elapsed

    return func_return, payload, elapsed
