"""PRISM - PRuning Interface for Similar Molecules."""

from typing import Annotated, Any, Union

import numpy as np
from numpy.typing import NDArray

Array3D_float = Annotated[NDArray[np.float64], "shape: (nconfs, natoms, 3)"]
Array2D_float = Annotated[NDArray[np.float64], "shape: (natoms, 3)"]
Array2D_int = Annotated[NDArray[np.int32], "shape: (a, b)"]
Array1D_float = Annotated[NDArray[np.float64], "shape: (energy,)"]
Array1D_int = Annotated[NDArray[np.int32], "shape: (natoms,)"]
Array1D_str = Annotated[NDArray[np.str_], "shape: (natoms,)"]
Array1D_bool = Annotated[NDArray[np.bool_], "shape: (n,)"]
FloatIterable = Union[tuple[float, ...], NDArray[np.floating[Any]]]
