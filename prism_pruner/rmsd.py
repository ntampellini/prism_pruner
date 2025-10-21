"""PRISM - PRuning Interface for Similar Molecules."""

import numpy as np

from prism_pruner.algebra import get_alignment_matrix, njit_typed, norm_of
from prism_pruner.typing import Array1D_float, Array2D_float


@njit_typed
def np_mean_along_axis(axis: int, arr: Array2D_float) -> Array1D_float:
    """Jittable np.mean.

    Workaround to specify axis parameters to
    numba functions, adapted from
    https://github.com/numba/numba/issues/1269
    """
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1], dtype=np.float64)
        for i in range(len(result)):
            result[i] = np.mean(arr[:, i])
    else:
        result = np.empty(arr.shape[0], dtype=np.float64)
        for i in range(len(result)):
            result[i] = np.mean(arr[i, :])
    return result


@njit_typed
def rmsd_and_max_numba(
    p: Array2D_float,
    q: Array2D_float,
    center: bool = False,
) -> tuple[float, float]:
    """Return RMSD and max deviation.

    Return a tuple with the RMSD between p and q
    and the maximum deviation of their positions.
    """
    if center:
        # p -= p.mean(axis=0)
        # q -= q.mean(axis=0)
        p -= np_mean_along_axis(0, p)
        q -= np_mean_along_axis(0, q)

    # get alignment matrix
    rot_mat = get_alignment_matrix(p, q)

    # Apply it to p
    p = np.ascontiguousarray(p) @ rot_mat

    # Calculate deviations
    diff = p - q

    # Calculate RMSD
    rmsd = np.sqrt((diff * diff).sum() / len(diff))

    # # Calculate max deviation
    # max_delta = np.linalg.norm(diff, axis=1).max()
    max_delta = max([norm_of(v) for v in diff])

    return rmsd, max_delta
