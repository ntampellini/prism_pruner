"""PRISM - PRuning Interface for Similar Molecules."""

from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
from networkx import Graph, connected_components

from prism_pruner.algebra import all_dists, get_moi_deviation_vec
from prism_pruner.pruner import prune_by_moment_of_inertia, prune_by_rmsd, prune_by_rmsd_rot_corr
from prism_pruner.pt import pt
from prism_pruner.rmsd import rmsd_and_max_numba
from prism_pruner.torsion_module import (
    _get_hydrogen_bonds,
    _get_torsions,
    _is_nondummy,
    get_angles,
    rotationally_corrected_rmsd_and_max,
)
from prism_pruner.typing import (
    Array1D_bool,
    Array1D_float,
    Array1D_int,
    Array2D_float,
    Array3D_float,
    F,
    FloatIterable,
)
from prism_pruner.utils import flatten, get_double_bonds_indices, time_to_string

__version__ = "1.0.0"
