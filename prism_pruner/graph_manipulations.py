"""PRISM - PRuning Interface for Similar Molecules."""

from itertools import combinations

import numpy as np
from networkx import Graph, all_simple_paths, from_numpy_array, set_node_attributes

from prism_pruner.algebra import all_dists, dihedral, norm_of
from prism_pruner.pt import pt
from prism_pruner.typing import Array1D_bool, Array1D_int, Array2D_float, Array3D_float


def d_min_bond(e1: int, e2: int, factor: float = 1.2) -> float:
    """Return the bond distance between two elements."""
    d: float = factor * (pt[e1].covalent_radius + pt[e2].covalent_radius)
    return d


def graphize(
    coords: Array2D_float, atomnos: Array1D_int, mask: Array1D_bool | None = None
) -> Graph:
    """Return a NetworkX undirected graph of molecular connectivity.

    :params coords: atomic coordinates as 3D vectors
    :params atomnos: atomic numbers as a list
    :params mask: bool array, with False for atoms
    to be excluded in the bond evaluation
    :return connectivity graph

    """
    mask = np.array([True for _ in atomnos], dtype=bool) if mask is None else mask

    matrix = np.zeros((len(coords), len(coords)))
    for i, _ in enumerate(coords):
        for j in range(i, len(coords)):
            if mask[i] and mask[j]:
                if norm_of(coords[i] - coords[j]) < d_min_bond(atomnos[i], atomnos[j]):
                    matrix[i][j] = 1

    graph: Graph = from_numpy_array(matrix)
    set_node_attributes(graph, dict(enumerate(atomnos)), "atomnos")

    return graph


def neighbors(graph: Graph, index: int) -> list[int]:
    """Return a list of neighbors of the given index."""
    neighbors = list(graph.neighbors(index))
    if index in neighbors:
        neighbors.remove(index)
    return neighbors


def is_sp_n(index: int, graph: Graph, n: int) -> bool:
    """Return True if the sp_n value of node at index matches n."""
    sp_n = get_sp_n(index, graph)
    if sp_n == n:
        return True
    return False


def get_sp_n(index: int, graph: Graph) -> int | None:
    """Get index hybridization.

    Return n, that is the apex of sp^n hybridization for CONPS atoms.
    This is just an assimilation to the carbon geometry in relation to sp^n:
    - sp(1) is linear
    - sp2 is planar
    - sp3 is tetraedral
    This is mainly used to understand if a torsion is to be rotated or not.
    """
    element = graph.nodes[index]["atomnos"]

    if element not in (6, 7, 8, 15, 16):
        return None

    d: dict[int, dict[int, int | None]] = {
        6: {2: 1, 3: 2, 4: 3},  # C - 2 neighbors means sp, 3 nb means sp2, 4 nb sp3
        7: {2: 2, 3: None, 4: 3},  # N - 2 neighbors means sp2, 3 nb could mean sp3 or sp2, 4 nb sp3
        8: {1: 2, 2: 3, 3: 3, 4: 3},  # O
        15: {2: 2, 3: 3, 4: 3},  # P - like N
        16: {2: 2, 3: 3, 4: 3},  # S
    }
    return d[element].get(len(neighbors(graph, index)))


def is_amide_n(index: int, graph: Graph, mode: int = -1) -> bool:
    """Assess if the index is an amide-like nitrogen.

    Return true if the nitrogen atom at the given
    index is a nitrogen and is part of an amide.
    Carbamates and ureas are considered amides.

    mode:
    -1 - any amide
    0 - primary amide (CONH2)
    1 - secondary amide (CONHR)
    2 - tertiary amide (CONR2)
    """
    if graph.nodes[index]["atomnos"] == 7:
        # index must be a nitrogen atom

        nb = neighbors(graph, index)
        nb_atomnos = [graph.nodes[j]["atomnos"] for j in nb]

        if mode != -1:
            if nb_atomnos.count(1) != (2, 1, 0)[mode]:
                # primary amides need to have 1H, secondary amides none
                return False

        for n in nb:
            if graph.nodes[n]["atomnos"] == 6:
                # there must be at least one carbon atom next to N

                nb_nb = neighbors(graph, n)
                if len(nb_nb) == 3:
                    # bonded to three atoms

                    nb_nb_sym = [graph.nodes[i]["atomnos"] for i in nb_nb]
                    if 8 in nb_nb_sym:
                        return True
                        # and at least one of them has to be an oxygen
    return False


def is_ester_o(index: int, graph: Graph) -> bool:
    """Assess if the index is an ester-like oxygen.

    Return true if the atom at the given
    index is an oxygen and is part of an ester.
    Carbamates and carbonates return True,
    Carboxylic acids return False.
    """
    if graph.nodes[index]["atomnos"] == 8:
        nb = neighbors(graph, index)
        if 1 not in nb:
            for n in nb:
                if graph.nodes[n]["atomnos"] == 6:
                    nb_nb = neighbors(graph, n)
                    if len(nb_nb) == 3:
                        nb_nb_sym = [graph.nodes[i]["atomnos"] for i in nb_nb]
                        if nb_nb_sym.count(8) > 1:
                            return True
    return False


def is_phenyl(coords: Array2D_float) -> bool:
    """Assess if the six atomic coords refer to a phenyl-like ring.

    :params coords: six coordinates of C/N atoms
    :return tuple: bool indicating if the six atoms look like part of a
                   phenyl/naphtyl/pyridine system, coordinates for the center of that ring

    (quinones evaluate to True)
    """
    if np.max(all_dists(coords, coords)) > 3:
        return False
    # if any atomic couple is more than 3 A away from each other, this is not a Ph

    threshold_delta = 1 - np.cos(10 * np.pi / 180)
    flat_delta = 1 - np.abs(np.cos(dihedral(coords[[0, 1, 2, 3]]) * np.pi / 180))

    if flat_delta < threshold_delta:
        flat_delta = 1 - np.abs(np.cos(dihedral(coords[[0, 1, 2, 3]]) * np.pi / 180))
        if flat_delta < threshold_delta:
            # print('phenyl center at', np.mean(coords, axis=0))
            return True

    return False


def get_phenyls(coords: Array2D_float, atomnos: Array1D_int) -> Array3D_float:
    """Return a (n, 6, 3) array where the first dimension is the aromatic rings detected."""
    if len(atomnos) < 6:
        return np.array([])

    output = []

    c_n_indices = np.fromiter(
        (i for i, a in enumerate(atomnos) if a in (6, 7)), dtype=atomnos.dtype
    )
    comb = combinations(c_n_indices, 6)

    for c in comb:
        mask = np.fromiter((i in c for i in range(len(atomnos))), dtype=bool)
        coords_ = coords[mask]
        if is_phenyl(coords_):
            output.append(coords_)

    return np.array(output)


def _get_phenyl_ids(i: int, graph: Graph) -> list[int] | None:
    """If index i is part of a phenyl, return the six heavy atoms ids associated with the ring."""
    for n in neighbors(graph, i):
        paths: list[list[int]] = all_simple_paths(graph, source=i, target=n, cutoff=6)
        for path in paths:
            if len(path) == 6:
                if all(graph.nodes[n]["atomnos"] != 1 for n in path):
                    if all(len(neighbors(graph, i)) == 3 for i in path):
                        return path

    return None


def find_paths(
    graph: Graph,
    u: int,
    n: int,
    exclude_set: set[int] | None = None,
) -> list[list[int]]:
    """Find paths in graph.

    Recursively find all paths of a NetworkX
    graph G with length = n, starting from node u
    """
    if exclude_set is None:
        exclude_set = {u}

    else:
        exclude_set.add(u)

    if n == 0:
        return [[u]]

    paths: list[list[int]] = [
        [u, *path]
        for neighbor in graph.neighbors(u)
        if neighbor not in exclude_set
        for path in find_paths(graph, neighbor, n - 1, exclude_set)
    ]
    exclude_set.remove(u)

    return paths
