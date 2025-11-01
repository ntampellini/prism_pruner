"""PRISM - PRuning Interface for Similar Molecules."""

import numpy as np

from prism_pruner.algebra import get_alignment_matrix, norm_of
from prism_pruner.typing import Array2D_float


def rmsd_and_max(
    p: Array2D_float,
    q: Array2D_float,
    center: bool = False,
) -> tuple[float, float]:
    """Return RMSD and max deviation.

    Return a tuple with the RMSD between p and q
    and the maximum deviation of their positions.
    """
    if center:
        p -= p.mean(axis=0)
        q -= q.mean(axis=0)

    # get alignment matrix
    rot_mat = get_alignment_matrix(p, q)

    # Apply it to p
    p = np.ascontiguousarray(p) @ rot_mat

    # Calculate deviations
    diff = p - q

    # Calculate RMSD
    rmsd = np.sqrt((diff * diff).sum() / len(diff))

    # # Calculate max deviation
    max_delta = max([norm_of(v) for v in diff])

    return rmsd, max_delta
