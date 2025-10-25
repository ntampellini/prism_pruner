"""Graph manipulation utilities for molecular structures."""

from functools import lru_cache

import numpy as np
from networkx import Graph, all_simple_paths, from_numpy_array, set_node_attributes
from scipy.spatial.distance import cdist

from prism_pruner.algebra import dihedral, norm_of
from prism_pruner.pt import pt
from prism_pruner.typing import Array1D_bool, Array1D_int, Array2D_float


@lru_cache()
def d_min_bond(a1: int, a2: int, factor: float = 1.2) -> float:
    """Return the bond distance between two atoms."""
    return factor * (pt[a1].covalent_radius + pt[a2].covalent_radius)  # type: ignore [no-any-return]


def graphize(
    atoms: Array1D_int,
    coords: Array2D_float,
    mask: Array1D_bool | None = None,
) -> Graph:
    """
    Return a NetworkX undirected graph of molecular connectivity.

    :param atoms: atomic numbers
    :param coords: atomic coordinates as 3D vectors
    :param mask: bool array, with False for atoms to be excluded in the bond evaluation
    :return: connectivity graph
    """
    mask = np.array([True for _ in atoms], dtype=bool) if mask is None else mask
    assert len(coords) == len(atoms)
    assert len(coords) == len(mask)

    matrix = np.zeros((len(coords), len(coords)))
    for i, mask_i in enumerate(mask):
        if not mask_i:
            continue

        for j, mask_j in enumerate(mask[i + 1 :], start=i + 1):
            if not mask_j:
                continue

            if norm_of(coords[i] - coords[j]) < d_min_bond(atoms[i], atoms[j]):
                matrix[i][j] = 1

    graph = from_numpy_array(matrix)
    set_node_attributes(graph, dict(enumerate(atoms)), "atomnos")

    return graph


def get_sp_n(index: int, graph: Graph) -> int | None:
    """
    Get hybridization of selected atom.

    Return n, that is the apex of sp^n hybridization for CONPS atoms.
    This is just an assimilation to the carbon geometry in relation to sp^n:
    - sp¹ is linear
    - sp² is planar
    - sp³ is tetraedral
    This is mainly used to understand if a torsion is to be rotated or not.
    """
    element = graph.nodes[index]["atomnos"]

    if element not in {6, 7, 8, 15, 16}:
        return None

    d: dict[int, dict[int, int | None]] = {
        6: {2: 1, 3: 2, 4: 3},  # C - 2 neighbors means sp, 3 nb means sp2, 4 nb sp3
        7: {2: 2, 3: None, 4: 3},  # N - 2 neighbors means sp2, 3 nb could mean sp3 or sp2, 4 nb sp3
        8: {1: 2, 2: 3, 3: 3, 4: 3},  # O
        15: {2: 2, 3: 3, 4: 3},  # P - like N
        16: {2: 2, 3: 3, 4: 3},  # S
    }
    return d[element].get(len(set(graph.neighbors(index))))


def is_amide_n(index: int, graph: Graph, mode: int = -1) -> bool:
    """
    Assess if the atom is an amide-like nitrogen.

    Note: carbamates and ureas are considered amides.

    mode:
    -1 - any amide
    0 - primary amide (CONH2)
    1 - secondary amide (CONHR)
    2 - tertiary amide (CONR2)
    """
    # Must be a nitrogen atom
    if graph.nodes[index]["atomnos"] == 7:
        nb = set(graph.neighbors(index))
        nb_atoms = [graph.nodes[j]["atomnos"] for j in nb]

        if mode != -1:
            # Primary amides need to have 1H, secondary amides none
            if nb_atoms.count(1) != (2, 1, 0)[mode]:
                return False

        for n in nb:
            # There must be at least one carbon atom next to N
            if graph.nodes[n]["atomnos"] == 6:
                nb_nb = set(graph.neighbors(n))
                # Bonded to three atoms
                if len(nb_nb) == 3:
                    # and at least one of them has to be an oxygen
                    if 8 in {graph.nodes[i]["atomnos"] for i in nb_nb}:
                        return True
    return False


def is_ester_o(index: int, graph: Graph) -> bool:
    """
    Assess if the index is an ester-like oxygen.

    Note: carbamates and carbonates return True, carboxylic acids return False.
    """
    if graph.nodes[index]["atomnos"] == 8:
        if 1 in (nb := set(graph.neighbors(index))):
            return False

        for n in nb:
            if graph.nodes[n]["atomnos"] == 6:
                nb_nb = set(graph.neighbors(n))
                if len(nb_nb) == 3:
                    nb_nb_sym = [graph.nodes[i]["atomnos"] for i in nb_nb]
                    if nb_nb_sym.count(8) > 1:
                        return True
    return False


def is_phenyl(coords: Array2D_float) -> bool:
    """
    Assess if the six atomic coords refer to a phenyl-like ring.

    Note: quinones evaluate to True

    :params coords: six coordinates of C/N atoms
    :return: bool indicating if the six atoms look like part of a phenyl/naphtyl/pyridine
             system, coordinates for the center of that ring
    """
    # if any atomic couple is more than 3 A away from each other, this is not a Ph
    if np.max(cdist(coords, coords)) > 3:
        return False

    threshold_delta: float = 1 - np.cos(10 * np.pi / 180)
    flat_delta: float = 1 - np.abs(np.cos(dihedral(coords[[0, 1, 2, 3]]) * np.pi / 180))

    return flat_delta < threshold_delta


def get_phenyl_ids(index: int, graph: Graph) -> list[int] | None:
    """If index is part of a phenyl, return the six heavy atoms ids associated with the ring."""
    for n in graph.neighbors(index):
        for path in all_simple_paths(graph, source=index, target=n, cutoff=6):
            if len(path) != 6 or any(graph.nodes[n]["atomnos"] == 1 for n in path):
                continue
            if all(len(set(graph.neighbors(i))) == 3 for i in path):
                return path  # type: ignore [no-any-return]

    return None


def find_paths(
    graph: Graph,
    u: int,
    n: int,
    exclude_set: set[int] | None = None,
) -> list[list[int]]:
    """
    Find paths in graph.

    Recursively find all paths of a NetworkX graph with length = n, starting from node u.

    :param graph: NetworkX graph
    :param u: starting node
    :param n: path length
    :param exclude_set: set of nodes to exclude from the paths
    :return: list of paths (each path is a list of node indices)
    """
    exclude_set = (exclude_set or set()) | {u}

    if n == 0:
        return [[u]]

    return [
        [u, *path]
        for neighbor in graph.neighbors(u)
        if neighbor not in exclude_set
        for path in find_paths(graph, neighbor, n - 1, exclude_set)
    ]
