"""PRISM - PRuning Interface for Similar Molecules."""

from typing import Any, Callable, Sequence

import numpy as np
from networkx import Graph, connected_components

from prism_pruner.algebra import all_dists, get_moi_deviation_vec
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
from prism_pruner.utils import get_double_bonds_indices

__version__ = "1.0.0"


class Pruner:
    """Pruner class."""

    def __init__(
        self,
        structures: Array3D_float,
        atomnos: Array1D_int,
        energies: Array1D_float | None = None,
        ewin: float | None = None,
        debugfunction: F | None = None,
        mode: str | None = None,
    ):
        """Pruner class to remove duplicates.

        structures: a shape (c, n, 3) numpy array of atomic cooordinates, with c conformers
            of n atoms each.
        atomnos: a shape (n,) numpy array with atomic numbers
        energies: an optional shape (n,) numpy array with energies (arbitrary UOM)
        ewin: an optional float specifying the energy window for structural comparison: if energies
            are provided, the structures are only considered similar is their energy difference is
            less than ewin.
        debugfunction: an optional function that will be passed debug strings (i.e. print)
        mode: one of self.standard_dict's keys: ('rmsd_rot_corr', 'rmsd', 'moi')
        """
        self.structures = structures
        self.atomnos = atomnos
        self.debugfunction = debugfunction

        if energies is not None:
            assert ewin is not None, (
                "If you provide energies, please also provide an appropriate energy window ewin."
            )

        self.energies = energies or np.zeros(structures.shape[0])
        self.ewin = ewin or 1

        self.calls = 0
        self.cache_calls = 0

        self.defaults_dict: dict[
            str, tuple[Callable[..., FloatIterable], list[str], dict[str, Any], list[str]]
        ] = {
            "rmsd_rot_corr": (
                rotationally_corrected_rmsd_and_max,
                [
                    "atomnos",
                    "torsions",
                    "graph",
                    "angles",
                ],  # args
                {},  # kwargs
                ["max_rmsd", "max_dev"],  # thresholds
            ),
            "rmsd": (
                rmsd_and_max_numba,
                [],  # args
                {},  # kwargs
                ["max_rmsd", "max_dev"],  # thresholds
            ),
            "moi": (
                get_moi_deviation_vec,
                [
                    "masses",
                ],  # args
                {},  # kwargs
                ["max_dev", "max_dev", "max_dev"],  # thresholds
            ),
        }

        self.max_rmsd: float | None = None
        self.max_dev: float | None = None
        self.angles: Sequence[Sequence[int]] | None = None
        self.torsions: Sequence[Sequence[int]] | None = None
        self.masses: Array1D_float | None = None
        self.graph: Graph | None = None

        if mode is not None:
            self.set_mode(mode)

    def set_mode(self, mode: str) -> None:
        """Set the pruning mode of the Prune object.

        mode: one of the modes in self.defaults_dict.
        ("rmsd_rot_corr", "rmsd", "moi)
        """
        if mode not in self.defaults_dict.keys():
            raise NameError(f'pruning mode "{mode}" not recognized.')
        self.mode = mode

        self.eval_func, args_names, kwargs_names, thresholds_names = self.defaults_dict[self.mode]
        self.args = [getattr(self, name) for name in args_names]
        self.kwargs = {name: getattr(self, value) for name, value in kwargs_names.items()}
        self.thresholds: list[float] = [getattr(self, name) for name in thresholds_names]

        for name, value in zip(thresholds_names, self.thresholds, strict=False):
            if value is None:
                raise UnboundLocalError(
                    f'Class Pruner({self.mode}) does not have a "{name}" attribute. '
                    + "Please set it as:\n"
                    + f">>> pruner.{name} = foobar"
                )

    def _main_eval_similarity(self, coords1: Array2D_float, coords2: Array2D_float) -> bool:
        """Evaluate the similarity of two structures.

        Return True if coords1 and coords2 are deemed similar by self.eval_func.
        """
        results = self.eval_func(coords1, coords2, *self.args, **self.kwargs)
        for r, t in zip(results, self.thresholds, strict=False):
            if r > t:
                return False
        return True

    def _main_compute_subrow(
        self,
        ref: Array2D_float,
        structures: Array3D_float,
        in_mask: Array1D_bool,
        first_abs_index: int,
    ) -> bool:
        """Evaluate the similarity of a subrow of the similarity matrix.

        Return True if ref is similar to any
        structure in structures, returning at the first instance of a match.
        Ignores structures that are False (0) in in_mask and does not perform
        the comparison if the energy difference between the structures is less
        than self.ewin. Saves dissimilar structural pairs (i.e. that evaluate to
        False (0)) by adding them to self.cache, avoiding redundant calcaulations.
        """
        # iterate over target structures
        for i, structure in enumerate(structures):
            # only compare active structures
            if in_mask[i]:
                # check if we have performed this comparison already:
                # if so, we already know that those two structures are not similar,
                # since the in_mask attribute is not False for ref nor for i
                i1 = first_abs_index
                i2 = first_abs_index + 1 + i
                hash_value = (i1, i2)

                self.calls += 1
                if hash_value in self.cache:
                    self.cache_calls += 1

                # if we have not computed the value before, check if the two
                # structures have close enough energy before running the comparison
                elif np.abs(self.energies[i1] - self.energies[i2]) < self.ewin:
                    # function will return True if the structures are similar,
                    # and will stop iterating on this row, returning
                    if self._main_eval_similarity(ref, structure):
                        return True

                # if structures are not similar, add the result to the
                # cache, because they will return here,
                # while similar structures are discarded and won't come back
                self.cache.add(hash_value)

        return False

    def _main_compute_row(
        self,
        structures: Array3D_float,
        in_mask: Array1D_bool,
        first_abs_index: int,
    ) -> Array1D_bool:
        """Evaluate the similarity of a row of the similarity matrix.

        For a given set of structures, check if each is similar
        to any other after itself. Return a boolean mask to slice
        the array, only retaining the structures that are dissimilar.
        The inner subrow function caches computed non-similar pairs.

        """
        # initialize the result container
        out_mask = np.ones(shape=in_mask.shape, dtype=np.bool_)

        # loop over the structures
        for i, ref in enumerate(structures):
            # only check for similarity if the structure is active
            if in_mask[i]:
                # reject structure i if it is similar to any other after itself
                similar = self._main_compute_subrow(
                    ref,
                    structures[i + 1 :],
                    in_mask[i + 1 :],
                    first_abs_index=first_abs_index + i,
                )
                out_mask[i] = not similar

            else:
                out_mask[i] = 0

        return out_mask

    def _main_compute_group(
        self,
        structures: Array2D_float,
        in_mask: Array1D_bool,
        k: int,
    ) -> Array1D_bool:
        """Evaluate the similarity of each chunk of the similarity matrix.

        Acts individually on k chunks of the structures array,
        returning the updated mask.
        """
        # initialize final result container
        out_mask = np.ones(shape=structures.shape[0], dtype=np.bool_)

        # calculate the size of each chunk
        chunksize = int(len(structures) // k)

        # iterate over chunks (multithreading here?)
        for chunk in range(int(k)):
            first = chunk * chunksize
            if chunk == k - 1:
                last = len(structures)
            else:
                last = chunksize * (chunk + 1)

            # get the structure chunk
            structures_chunk = structures[first:last]

            # compare structures within that chunk and save results to the out_mask
            out_mask[first:last] = self._main_compute_row(
                structures_chunk,
                in_mask[first:last],
                first_abs_index=first,
            )
        return out_mask

    def prune(self) -> None:
        """Perform the similarity pruning.

        Remove similar structures by repeatedly grouping them into k
        subgroups and removing similar ones. A cache is present to avoid
        repeating RMSD computations.

        Similarity occurs for structures with both rmsd < self.max_rmsd and
        maximum absolute atomic deviation < self.max_dev.

        Sets the self.structures and the corresponding self.mask attributes.
        """
        if self.mode is None:
            known = tuple(self.defaults_dict.keys())
            raise Exception(
                'Please set the Pruner object "mode" attribute '
                + f"before pruning. Known modes: {known}"
            )

        if self.mode in ("rmsd_rot_corr"):
            # all atoms are passed to the functions, but only the
            # heavy ones are used for the rot. corr. RMSD calcs
            structures = self.structures

        else:
            # only feed non-hydrogen atoms to eval funcs
            heavy_atoms = self.atomnos != 1
            structures = np.array([structure[heavy_atoms] for structure in self.structures])

        # initialize the output mask
        out_mask = np.ones(shape=self.structures.shape[0], dtype=np.bool_)
        self.cache: set[tuple[int, int]] = set()

        # split the structure array in subgroups and prune them internally
        for k in (
            5e5,
            2e5,
            1e5,
            5e4,
            2e4,
            1e4,
            5000,
            2000,
            1000,
            500,
            200,
            100,
            50,
            20,
            10,
            5,
            2,
            1,
        ):
            # choose only k values such that every subgroup
            # has on average at least twenty active structures in it
            if k == 1 or 20 * k < np.count_nonzero(out_mask):
                before = np.count_nonzero(out_mask)

                # compute similarities and get back the out_mask
                # and the pairings to be added to cache
                out_mask = self._main_compute_group(
                    structures,
                    out_mask,
                    k=int(k),
                )

                after = np.count_nonzero(out_mask)
                newly_discarded = before - after

                if self.debugfunction is not None:
                    self.debugfunction(
                        f"DEBUG: Pruner({self.mode}) - k={k}, rejected {newly_discarded} "
                        + f"(keeping {after}/{len(out_mask)})"
                    )

        del self.cache
        self.mask = out_mask
        self.structures = self.structures[self.mask]


def prune_by_rmsd(
    structures: Array3D_float,
    atomnos: Array1D_int,
    max_rmsd: float = 0.25,
    max_dev: float | None = None,
    debugfunction: F | None = None,
) -> tuple[Array3D_float, Array1D_bool]:
    """Remove duplicate structures using a heavy-atom RMSD metric.

    Remove similar structures by repeatedly grouping them into k
    subgroups and removing similar ones. A cache is present to avoid
    repeating RMSD computations.

    Similarity occurs for structures with both RMSD < max_rmsd and
    maximum deviation < max_dev. max_dev by default is 2 * max_rmsd.
    """
    # set default max_dev if not provided
    max_dev = max_dev or 2 * max_rmsd

    pruner = Pruner(structures, atomnos, debugfunction=debugfunction)
    pruner.max_rmsd = max_rmsd
    pruner.max_dev = max_dev
    pruner.set_mode("rmsd")
    pruner.prune()
    final_mask = pruner.mask

    if debugfunction is not None:
        fraction = 0 if pruner.calls == 0 else pruner.cache_calls / pruner.calls
        debugfunction(
            f"DEBUG: prune_by_rmsd - Used cached data {pruner.cache_calls}/"
            + f"{pruner.calls} times, {100 * fraction:.2f}% of total calls"
        )

    return structures[final_mask], final_mask


def prune_by_rmsd_rot_corr(
    structures: Array3D_float,
    atomnos: Array1D_int,
    graph: Graph,
    max_rmsd: float = 0.25,
    max_dev: float | None = None,
    logfunction: F | None = None,
    debugfunction: F | None = None,
) -> tuple[Array3D_float, Array1D_bool]:
    """Remove duplicates using a heavy-atom RMSD metric, corrected for degenerate torsions.

    Remove similar structures by repeatedly grouping them into k
    subgroups and removing similar ones. A cache is present to avoid
    repeating RMSD computations.

    Similarity occurs for structures with both RMSD < max_rmsd and
    maximum deviation < max_dev. max_dev by default is 2 * max_rmsd.

    The RMSD and maximum deviation metrics used are the lowest ones
    of all the degenerate rotamers of the input structure.
    """
    # center structures
    structures = np.array([s - s.mean(axis=0) for s in structures])
    ref = structures[0]

    # get the number of molecular fragments
    subgraphs = list(connected_components(graph))

    # if they are more than two, give up on pruning by rot corr rmsd
    if len(subgraphs) > 2:
        return structures, np.ones(structures.shape[0], dtype=bool)

    # if they are two, we can add a fictitious bond between the closest
    # atoms on the two molecular fragment in the provided graph, and
    # then removing it before returning
    if len(subgraphs) == 2:
        subgraphs = [list(set) for set in connected_components(graph)]
        all_dists_array = all_dists(ref[list(subgraphs[0])], ref[list(subgraphs[1])])
        min_d = np.min(all_dists_array)
        s1, s2 = np.where(all_dists_array == min_d)
        i1, i2 = subgraphs[0][s1[0]], subgraphs[1][s2[0]]
        graph.add_edge(i1, i2)

        if debugfunction is not None:
            debugfunction(
                f"DEBUG: prune_by_rmsd_rot_corr - temporarily added "
                f"edge {i1}-{i2} to the graph (will be removed before returning)"
            )

    # set default max_dev if not provided
    max_dev = max_dev or 2 * max_rmsd

    # add hydrogen bonds to molecular graph
    hydrogen_bonds = _get_hydrogen_bonds(ref, atomnos, graph)
    for hb in hydrogen_bonds:
        graph.add_edge(*hb)

    # get all rotable bonds in the molecule, including dummy rotations
    torsions = _get_torsions(
        graph,
        hydrogen_bonds=_get_hydrogen_bonds(ref, atomnos, graph),
        double_bonds=get_double_bonds_indices(ref, atomnos),
        keepdummy=True,
        mode="symmetry",
    )

    # only keep dummy rotations (checking both directions)
    torsions = [
        t
        for t in torsions
        if not (_is_nondummy(t.i2, t.i3, graph) and (_is_nondummy(t.i3, t.i2, graph)))
    ]

    # since we only compute RMSD based on heavy atoms, discard
    # quadruplets that involve hydrogen atoms
    torsions = [t for t in torsions if 1 not in [atomnos[i] for i in t.torsion]]

    # get torsions angles
    angles = [get_angles(t, graph) for t in torsions]

    # Used specific directionality of torsions so that we always
    # rotate the dummy portion (the one attached to the last index)
    torsions_ids = [
        list(t.torsion) if _is_nondummy(t.i2, t.i3, graph) else list(reversed(t.torsion))
        for t in torsions
    ]

    # Set up final mask and cache
    final_mask = np.ones(structures.shape[0], dtype=bool)

    # Halt the run if there are too many structures or no subsymmetrical bonds
    if len(torsions_ids) == 0:
        if debugfunction is not None:
            debugfunction(
                "DEBUG: prune_by_rmsd_rot_corr - No subsymmetrical torsions found: skipping "
                "symmetry-corrected RMSD pruning"
            )

        return structures[final_mask], final_mask

    # Print out torsion information
    if logfunction is not None:
        logfunction("\n >> Dihedrals considered for rotamer corrections:")
        for i, (torsion, angle) in enumerate(zip(torsions_ids, angles, strict=False)):
            logfunction(
                " {:2s} - {:21s} : {}{}{}{} : {}-fold".format(
                    str(i + 1),
                    str(torsion),
                    pt[atomnos[torsion[0]]].symbol,
                    pt[atomnos[torsion[1]]].symbol,
                    pt[atomnos[torsion[2]]].symbol,
                    pt[atomnos[torsion[3]]].symbol,
                    len(angle),
                )
            )
        logfunction("\n")

    pruner = Pruner(structures, atomnos, debugfunction=debugfunction)
    pruner.graph = graph
    pruner.torsions = torsions_ids
    pruner.angles = angles
    pruner.max_rmsd = max_rmsd
    pruner.max_dev = max_dev
    pruner.set_mode("rmsd_rot_corr")
    pruner.prune()
    final_mask = pruner.mask

    # remove the extra bond in the molecular graph
    if len(subgraphs) == 2:
        graph.remove_edge(i1, i2)

    if debugfunction is not None:
        fraction = 0 if pruner.calls == 0 else pruner.cache_calls / pruner.calls
        debugfunction(
            f"DEBUG: prune_by_rmsd_rot_corr - Used cached data {pruner.cache_calls}/"
            + f"{pruner.calls} times, {100 * fraction:.2f}% of total calls"
        )

    return structures[final_mask], final_mask


def prune_by_moment_of_inertia(
    structures: Array3D_float,
    atomnos: Array1D_int,
    max_deviation: float = 1e-2,
    debugfunction: F | None = None,
) -> tuple[Array3D_float, Array1D_bool]:
    """Remove duplicate structures using a moments of inertia-based metric.

    Remove duplicate structures (enantiomeric or rotameric) based on the
    moments of inertia on the principal axes. If all three MOI
    deviate less than max_deviation percent from another structure,
    they are classified as rotamers or enantiomers and therefore only one
    of them is kept (i.e. max_deviation = 0.1 is 10% relative deviation).
    """
    pruner = Pruner(structures, atomnos, debugfunction=debugfunction)
    pruner.max_dev = max_deviation
    pruner.masses = np.array([pt[a].mass for a in atomnos])
    pruner.set_mode("moi")
    pruner.prune()
    mask = pruner.mask

    if debugfunction is not None:
        fraction = 0 if pruner.calls == 0 else pruner.cache_calls / pruner.calls
        debugfunction(
            f"DEBUG: prune_by_moment_of_inertia - Used cached data {pruner.cache_calls}/"
            + f"{pruner.calls} times, {100 * fraction:.2f}% of total calls"
        )

    return structures[mask], mask
