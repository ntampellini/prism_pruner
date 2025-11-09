"""PRISM - PRuning Interface for Similar Molecules."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from networkx import (
    Graph,
    connected_components,
    has_path,
    is_isomorphic,
    minimum_spanning_tree,
    shortest_path,
    subgraph,
)

from prism_pruner.algebra import vec_angle
from prism_pruner.graph_manipulations import (
    get_phenyl_ids,
    get_sp_n,
    is_amide_n,
    is_ester_o,
)
from prism_pruner.rmsd import rmsd_and_max
from prism_pruner.typing import Array1D_bool, Array1D_str, Array2D_float, Array2D_int
from prism_pruner.utils import rotate_dihedral


@dataclass
class Torsion:
    """Torsion class."""

    i1: int
    i2: int
    i3: int
    i4: int
    mode: str | None = None

    @property
    def torsion(self) -> tuple[int, int, int, int]:
        """Return tuple of indices defining the torsion."""
        return (self.i1, self.i2, self.i3, self.i4)


def in_cycle(torsion: Torsion, graph: Graph) -> bool:
    """Return True if the torsion is part of a cycle."""
    graph.remove_edge(torsion.i2, torsion.i3)
    cyclical: bool = has_path(graph, torsion.i1, torsion.i4)
    graph.add_edge(torsion.i2, torsion.i3)
    return cyclical


def is_rotable(
    torsion: Torsion,
    graph: Graph,
    hydrogen_bonds: list[list[int]],
    keepdummy: bool = False,
) -> bool:
    """Return True if the Torsion object is rotatable.

    hydrogen bonds: iterable with pairs of sorted atomic indices.
    """
    if sorted((torsion.i2, torsion.i3)) in hydrogen_bonds:
        # self.n_fold = 6
        # # This has to be an intermolecular HB: rotate it
        # return True
        return False

    if _is_free(torsion.i2, graph) or (_is_free(torsion.i3, graph)):
        if keepdummy or (
            is_nondummy(torsion.i2, torsion.i3, graph)
            and (is_nondummy(torsion.i3, torsion.i2, graph))
        ):
            return True

    return False


def get_n_fold(torsion: Torsion, graph: Graph) -> int:
    """Return the n-fold of the rotation."""
    atoms = (graph.nodes[torsion.i2]["atoms"], graph.nodes[torsion.i3]["atoms"])

    if "H" in atoms:
        return 6  # H-N, H-O hydrogen bonds

    if is_amide_n(torsion.i2, graph, mode=2) or (is_amide_n(torsion.i3, graph, mode=2)):
        # tertiary amides rotations are 2-fold
        return 2

    if ("C" in atoms) or ("N" in atoms) or ("S" in atoms):  # if C, N or S atoms
        sp_n_i2 = get_sp_n(torsion.i2, graph)
        sp_n_i3 = get_sp_n(torsion.i3, graph)

        if 3 == sp_n_i2 == sp_n_i3:
            return 3

        if 3 in (sp_n_i2, sp_n_i3):  # Csp3-X, Nsp3-X, Ssulfone-X
            if torsion.mode == "csearch":
                return 3

            elif torsion.mode == "symmetry":
                return sp_n_i3 or 2

        if 2 in (sp_n_i2, sp_n_i3):
            return 2

    return 4  # O-O, S-S, Ar-Ar, Ar-CO, and everything else


def get_angles(torsion: Torsion, graph: Graph) -> tuple[int, ...]:
    """Return the angles associated with the torsion."""
    d = {
        2: (0, 180),
        3: (0, 120, 240),
        4: (0, 90, 180, 270),
        6: (0, 60, 120, 180, 240, 300),
    }

    n_fold = get_n_fold(torsion, graph)

    return d[n_fold]


def _is_free(index: int, graph: Graph) -> bool:
    """Return whether the torsion is free to rotate.

    Return True if the index specified
    satisfies all of the following:
    - Is not a sp2 carbonyl carbon atom
    - Is not the oxygen atom of an ester
    - Is not the nitrogen atom of a secondary amide (CONHR)
    """
    if all(
        (
            graph.nodes[index]["atoms"] == "C",
            2 == get_sp_n(index, graph),
            "O" in (graph.nodes[n]["atoms"] for n in graph.neighbors(index)),
        )
    ):
        return False

    if is_amide_n(index, graph, mode=1):
        return False

    if is_ester_o(index, graph):
        return False

    return True


def is_nondummy(i: int, root: int, graph: Graph) -> bool:
    """Return whether the torsion is not dummy.

    Checks that a molecular rotation along the dihedral
    angle (*, root, i, *) is non-dummy, that is the atom
    at index i, in the direction opposite to the one leading
    to root, has different substituents. i.e. methyl, CF3 and tBu
    rotations should return False.
    """
    if graph.nodes[i]["atoms"] not in ("C", "N"):
        return True
    # for now, we only discard rotations around carbon
    # and nitrogen atoms, like methyl/tert-butyl/triphenyl
    # and flat symmetrical rings like phenyl, N-pyrrolyl...

    G = deepcopy(graph)
    nb = list(G.neighbors(i))
    nb.remove(root)

    if len(nb) == 1:
        if len(list(G.neighbors(nb[0]))) == 2:
            return False
    # if node i has two bonds only (one with root and one with a)
    # and the other atom (a) has two bonds only (one with i)
    # the rotation is considered dummy: some other rotation
    # will account for its freedom (i.e. alkynes, hydrogen bonds)

    # check if it is a phenyl-like rotation
    if len(nb) == 2:
        # get the 6 indices of the aromatic atoms (i1-i6)
        phenyl_indices = get_phenyl_ids(i, G)

        # compare the two halves of the 6-membered ring (indices i2-i3 region with i5-i6 region)
        if phenyl_indices is not None:
            i1, i2, i3, i4, i5, i6 = phenyl_indices
            G.remove_edge(i3, i4)
            G.remove_edge(i4, i5)
            G.remove_edge(i1, i2)
            G.remove_edge(i1, i6)

            subgraphs = [
                subgraph(G, _set) for _set in connected_components(G) if i2 in _set or i6 in _set
            ]

            if len(subgraphs) == 2:
                return not is_isomorphic(
                    subgraphs[0],
                    subgraphs[1],
                    node_match=lambda n1, n2: n1["atoms"] == n2["atoms"],
                )

            # We should not end up here, but if we do, rotation should not be dummy
            return True

    # if not, compare immediate neighbors of i
    for n in nb:
        G.remove_edge(i, n)

    # make a set of each fragment around the chopped n-i bonds,
    # but only for fragments that are not root nor contain other random,
    # disconnected parts of the graph
    subgraphs_nodes = [
        _set for _set in connected_components(G) if root not in _set and any(n in _set for n in nb)
    ]

    if len(subgraphs_nodes) == 1:
        return True
        # if not, the torsion is likely to be rotable
        # (tetramethylguanidyl alanine C(β)-N bond)

    subgraphs = [subgraph(G, s) for s in subgraphs_nodes]
    for sub in subgraphs[1:]:
        if not is_isomorphic(
            subgraphs[0], sub, node_match=lambda n1, n2: n1["atoms"] == n2["atoms"]
        ):
            return True
    # Care should be taken because chiral centers are not taken into account: a rotation
    # involving an index where substituents only differ by stereochemistry, and where a
    # rotation is not an element of symmetry of the subsystem, the rotation is considered
    # dummy even if it would be more correct not to. For rotaionally corrected RMSD this
    # should only cause small inefficiencies and not lead to discarding any good conformer.

    return False


def get_hydrogen_bonds(
    coords: Array2D_float,
    atoms: Array1D_str,
    graph: Graph,
    d_min: float = 2.5,
    d_max: float = 3.3,
    max_angle: int = 45,
    elements: Sequence[Sequence[str]] | None = None,
    fragments: Sequence[Sequence[int]] | None = None,
) -> list[list[int]]:
    """Return a list of tuples with the indices of hydrogen bonding partners.

    An HB is a pair of atoms:
    - with one H and one X (N or O) atom
    - with an Y-X distance between d_min and d_max (i.e. N-O, Angstroms)
    - with an Y-H-X angle below max_angle (i.e. N-H-O, degrees)

    elements: iterable of two iterables with donor atomic symbols in the first
    element and acceptors in the second. default: (("N", "O"), ("N", "O"))

    If fragments is specified (iterable of iterable of indices for each fragment)
    the function only returns inter-fragment hydrogen bonds.
    """
    hbs = []
    # initializing output list

    if elements is None:
        elements = (("N", "O"), ("N", "O", "F"))

    het_idx_from = np.array([i for i, a in enumerate(atoms) if a in elements[0]], dtype=int)
    het_idx_to = np.array([i for i, a in enumerate(atoms) if a in elements[1]], dtype=int)
    # indices where N or O (or user-specified elements) atoms are present.

    for i1 in het_idx_from:
        for i2 in het_idx_to:
            # if inter-fragment HBs are requested, skip intra-HBs
            if fragments is not None:
                if any(((i1 in f and i2 in f) for f in fragments)):
                    continue

            # keep close pairs
            if d_min < np.linalg.norm(coords[i1] - coords[i2]) < d_max:
                # getting the indices of all H atoms attached to them
                Hs = [i for i in graph.neighbors(i1) if graph.nodes[i]["atoms"] == "H"]

                # versor connectring the two Heteroatoms
                versor = coords[i2] - coords[i1]
                versor = versor / np.linalg.norm(versor)

                for iH in Hs:
                    # vectors connecting heteroatoms to H
                    v1 = coords[iH] - coords[i1]
                    v2 = coords[iH] - coords[i2]

                    # lengths of these vectors
                    d1 = np.linalg.norm(v1)
                    d2 = np.linalg.norm(v2)

                    # scalar projection in the heteroatom direction
                    l1 = v1 @ versor
                    l2 = v2 @ -versor

                    # largest planar angle between Het-H and Het-Het, in degrees (0 to 90°)
                    alfa = vec_angle(v1, versor) if l1 < l2 else vec_angle(v2, -versor)

                    # if the three atoms are not too far from being in line
                    if alfa < max_angle:
                        # adding the correct pair of atoms to results
                        if d1 < d2:
                            hbs.append(sorted((iH, i2)))
                        else:
                            hbs.append(sorted((iH, i1)))

                        break

    return hbs


def _get_rotation_mask(graph: Graph, torsion: Iterable[int]) -> Array1D_bool:
    """Return the rotation mask to be applied to coordinates before rotation.

    Get mask for the atoms that will rotate in a torsion:
    all the ones in the graph reachable from the last index
    of the torsion but not going through the central two
    atoms in the torsion quadruplet.
    """
    _, i2, i3, i4 = torsion

    graph.remove_edge(i2, i3)
    reachable_indices = shortest_path(graph, i4).keys()
    # get all indices reachable from i4 not going through i2-i3

    graph.add_edge(i2, i3)
    # restore modified graph

    mask = np.array([i in reachable_indices for i in graph.nodes], dtype=bool)
    # generate boolean mask

    # if np.count_nonzero(mask) > int(len(mask)/2):
    #     mask = ~mask
    # if we want to rotate more than half of the indices,
    # invert the selection so that we do less math

    mask[i3] = False
    # do not rotate i3: it would not move,
    # since it lies on the rotation axis

    return mask


def _get_quadruplets(graph: Graph) -> Array2D_int:
    """Return list of quadruplets that indicate potential torsions."""
    # Step 1: Find spanning tree
    spanning_tree = minimum_spanning_tree(graph)

    # Step 2: Add dihedrals for spanning tree
    dihedrals = []

    # For each edge in the spanning tree, we can potentially define a dihedral
    # We need edges that have at least 2 neighbors each to form a 4-point dihedral
    for edge in spanning_tree.edges():
        i, j = edge

        # Find neighbors of i and j in the original graph
        i_neighbors = [n for n in graph.neighbors(i) if n not in (i, j)]
        j_neighbors = [n for n in graph.neighbors(j) if n not in (i, j)]

        if len(i_neighbors) > 0 and len(j_neighbors) > 0:
            # Form dihedral: neighbor_of_i - i - j - neighbor_of_j
            k = i_neighbors[0]  # Choose first available neighbor
            m = j_neighbors[0]  # Choose first available neighbor
            dihedrals.append((k, i, j, m))

    return np.array(dihedrals)


def get_torsions(
    graph: Graph,
    hydrogen_bonds: list[list[int]],
    double_bonds: list[tuple[int, int]],
    keepdummy: bool = False,
    mode: str = "csearch",
) -> list[Torsion]:
    """Return list of Torsion objects."""
    torsions = []
    for path in _get_quadruplets(graph):
        _, i2, i3, _ = path
        bt = tuple(sorted((i2, i3)))

        if bt not in double_bonds:
            t = Torsion(*path)
            t.mode = mode

            if (not in_cycle(t, graph)) and is_rotable(
                t, graph, hydrogen_bonds, keepdummy=keepdummy
            ):
                torsions.append(t)
    # Create non-redundant torsion objects
    # Rejects (4,3,2,1) if (1,2,3,4) is present
    # Rejects torsions that do not represent a rotable bond

    return torsions


def rotationally_corrected_rmsd_and_max(
    ref: Array2D_float,
    coord: Array2D_float,
    atoms: Array1D_str,
    torsions: Array2D_int,
    graph: Graph,
    angles: Sequence[Sequence[int]],
    heavy_atoms_only: bool = True,
    debugfunction: Callable[..., object] | None = None,
    return_type: str = "rmsd",
) -> tuple[float, float] | Array2D_float:
    """Return RMSD and max deviation, corrected for degenerate torsions.

    Return a tuple with the RMSD between p and q
    and the maximum deviation of their positions.
    """
    assert return_type in ("rmsd", "coords")

    torsion_corrections = [0 for _ in torsions]

    mask = (
        np.array([a != "H" for a in atoms]) if heavy_atoms_only else np.ones(len(atoms), dtype=bool)
    )

    # Now rotate every dummy torsion by the appropriate increment until we minimize local RMSD
    for i, torsion in enumerate(torsions):
        best_rmsd = 1e10

        # Look for the rotational angle set that minimizes the torsion RMSD and save it for later
        for angle in angles[i]:
            coord = rotate_dihedral(coord, torsion, angle, indices_to_be_moved=[torsion[3]])

            locally_corrected_rmsd, _ = rmsd_and_max(ref[torsion], coord[torsion])

            if locally_corrected_rmsd < best_rmsd:
                best_rmsd = locally_corrected_rmsd
                torsion_corrections[i] = angle

            # it is faster to undo the rotation rather than working with a copy of coordss
            coord = rotate_dihedral(coord, torsion, -angle, indices_to_be_moved=[torsion[3]])

        # now rotate that angle to the desired orientation before going to the next angle
        if torsion_corrections[i] != 0:
            coord = rotate_dihedral(
                coord, torsion, torsion_corrections[i], mask=_get_rotation_mask(graph, torsion)
            )

        if debugfunction is not None:
            global_rmsd = rmsd_and_max(ref[mask], coord[mask])[0]
            debugfunction(
                f"    Torsion {i + 1} - {torsion}: best θ = {torsion_corrections[i]}°, "
                + f"4-atom RMSD: {best_rmsd:.3f} Å, global RMSD: {global_rmsd:.3f} Å"
            )

    # we should have the optimal orientation on all torsions now:
    # calculate the RMSD
    rmsd, maxdev = rmsd_and_max(ref[mask], coord[mask])

    # since we could have segmented graphs, and therefore potentially only rotate
    # subsets of the graph where the torsion last two indices are,
    # we have to undo the final rotation too (would not be needed for connected graphs)
    for torsion, optimal_angle in zip(
        reversed(torsions), reversed(torsion_corrections), strict=False
    ):
        coord = rotate_dihedral(
            coord, torsion, -optimal_angle, mask=_get_rotation_mask(graph, torsion)
        )

    if return_type == "rmsd":
        return rmsd, maxdev

    return coord
