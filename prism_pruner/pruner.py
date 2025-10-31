"""PRISM - PRuning Interface for Similar Molecules."""

from copy import deepcopy
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np
from networkx import Graph, connected_components
from scipy.spatial.distance import cdist

from prism_pruner.algebra import get_moi_deviation_vec
from prism_pruner.pt import pt
from prism_pruner.rmsd import rmsd_and_max
from prism_pruner.torsion_module import (
    get_angles,
    get_hydrogen_bonds,
    get_torsions,
    is_nondummy,
    rotationally_corrected_rmsd_and_max,
)
from prism_pruner.typing import (
    Array1D_bool,
    Array1D_float,
    Array1D_int,
    Array2D_float,
    Array2D_int,
    Array3D_float,
)
from prism_pruner.utils import flatten, get_double_bonds_indices, time_to_string

__version__ = "1.0.0"


@dataclass
class PrunerConfig:
    """Configuration dataclass for Pruner."""

    structures: Array3D_float

    # Optional parameters that get initialized
    energies: Array1D_float = field(default_factory=lambda: np.array([]))
    ewin: float = field(default=0.0)
    debugfunction: Callable[[str], None] | None = field(default=None)

    # Computed fields
    calls: int = field(default=0, init=False)
    cache_calls: int = field(default=0, init=False)
    cache: set[tuple[int, int]] = field(default_factory=lambda: set(), init=False)

    def __post_init__(self) -> None:
        """Validate inputs and initialize computed fields."""
        self.mask = np.ones(shape=(self.structures.shape[0],), dtype=np.bool_)

        if len(self.energies) != 0:
            assert self.ewin > 0.0, (
                "If you provide energies, please also provide an appropriate energy window ewin."
            )

        # Set defaults for optional parameters
        if len(self.energies) == 0:
            self.energies = np.zeros(self.structures.shape[0])

        if self.ewin == 0.0:
            self.ewin = 1.0

    def evaluate_sim(self, *args: Any, **kwargs: Any) -> bool:
        """Stub method - override in subclasses as needed."""
        raise NotImplementedError


@dataclass
class RMSDRotCorrPrunerConfig(PrunerConfig):
    """Configuration dataclass for Pruner."""

    atomnos: Array1D_int = field(kw_only=True)
    max_rmsd: float = field(kw_only=True)
    max_dev: float = field(kw_only=True)
    angles: Sequence[Sequence[int]] = field(kw_only=True)
    torsions: Array2D_int = field(kw_only=True)
    graph: Graph = field(kw_only=True)
    heavy_atoms_only: bool = True

    def evaluate_sim(self, coord1: Array2D_float, coord2: Array2D_float) -> bool:
        """Return if the structures are similar."""
        rmsd, max_dev = rotationally_corrected_rmsd_and_max(
            coord1,
            coord2,
            atomnos=self.atomnos,
            torsions=self.torsions,
            graph=self.graph,
            angles=self.angles,
            debugfunction=self.debugfunction,
            heavy_atoms_only=self.heavy_atoms_only,
        )

        if rmsd > self.max_rmsd:
            return False

        if max_dev > self.max_dev:
            return False

        return True


@dataclass
class RMSDPrunerConfig(PrunerConfig):
    """Configuration dataclass for Pruner."""

    atomnos: Array1D_int = field(kw_only=True)
    max_rmsd: float = field(kw_only=True)
    max_dev: float = field(kw_only=True)
    heavy_atoms_only: bool = True

    def evaluate_sim(self, coord1: Array2D_float, coord2: Array2D_float) -> bool:
        """Return if the structures are similar."""
        if self.heavy_atoms_only:
            mask = self.atomnos != 1
        else:
            mask = np.ones(self.structures[0].shape[0], dtype=bool)

        rmsd, max_dev = rmsd_and_max(
            coord1[mask],
            coord2[mask],
            center=True,
        )

        if rmsd > self.max_rmsd:
            return False

        if max_dev > self.max_dev:
            return False

        return True


@dataclass
class MOIPrunerConfig(PrunerConfig):
    """Configuration dataclass for Pruner."""

    masses: Array1D_float = field(kw_only=True)
    max_dev: float = 0.01

    def evaluate_sim(self, coord1: Array2D_float, coord2: Array2D_float) -> bool:
        """Return if the structures are similar."""
        dev_vec = get_moi_deviation_vec(
            coord1,
            coord2,
            masses=self.masses,
        )

        return bool((dev_vec < self.max_dev).all())


def _main_compute_subrow(
    prunerconfig: PrunerConfig,
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

            prunerconfig.calls += 1
            if hash_value in prunerconfig.cache:
                prunerconfig.cache_calls += 1

            # if we have not computed the value before, check if the two
            # structures have close enough energy before running the comparison
            elif np.abs(prunerconfig.energies[i1] - prunerconfig.energies[i2]) < prunerconfig.ewin:
                # function will return True if the structures are similar,
                # and will stop iterating on this row, returning
                if prunerconfig.evaluate_sim(ref, structure):
                    return True

            # if structures are not similar, add the result to the
            # cache, because they will return here,
            # while similar structures are discarded and won't come back
            prunerconfig.cache.add(hash_value)

    return False


def _main_compute_row(
    prunerconfig: PrunerConfig,
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
            similar = _main_compute_subrow(
                prunerconfig,
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
    prunerconfig: PrunerConfig,
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
        out_mask[first:last] = _main_compute_row(
            prunerconfig,
            structures_chunk,
            in_mask[first:last],
            first_abs_index=first,
        )
    return out_mask


def prune(prunerconfig: PrunerConfig) -> tuple[Array2D_float, Array1D_bool]:
    """Perform the similarity pruning.

    Remove similar structures by repeatedly grouping them into k
    subgroups and removing similar ones. A cache is present to avoid
    repeating RMSD computations.

    Similarity occurs for structures with both rmsd < self.max_rmsd and
    maximum absolute atomic deviation < self.max_dev.

    Sets the self.structures and the corresponding self.mask attributes.
    """
    start_t = perf_counter()

    structures = deepcopy(prunerconfig.structures)

    # initialize the output mask
    out_mask = np.ones(shape=prunerconfig.structures.shape[0], dtype=np.bool_)
    prunerconfig.cache = set()

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

            start_t_k = perf_counter()

            # compute similarities and get back the out_mask
            # and the pairings to be added to cache
            out_mask = _main_compute_group(
                prunerconfig,
                structures,
                out_mask,
                k=int(k),
            )

            after = np.count_nonzero(out_mask)
            newly_discarded = before - after

            if prunerconfig.debugfunction is not None:
                elapsed = start_t_k - perf_counter()
                prunerconfig.debugfunction(
                    f"DEBUG: {prunerconfig.__class__.__name__} - k={k}, rejected {newly_discarded} "
                    + f"(keeping {after}/{len(out_mask)}), in {time_to_string(elapsed)}"
                )

    del prunerconfig.cache

    if prunerconfig.debugfunction is not None:
        elapsed = start_t - perf_counter()
        prunerconfig.debugfunction(
            f"DEBUG: {prunerconfig.__class__.__name__} - keeping "
            + f"{after}/{len(out_mask)} "
            + f"({time_to_string(elapsed)})"
        )

        fraction = 0 if prunerconfig.calls == 0 else prunerconfig.cache_calls / prunerconfig.calls
        prunerconfig.debugfunction(
            f"DEBUG: {prunerconfig.__class__.__name__} - Used cached data "
            + f"{prunerconfig.cache_calls}/{prunerconfig.calls} times, "
            + f"{100 * fraction:.2f}% of total calls"
        )

    return prunerconfig.structures[out_mask], out_mask


def prune_by_rmsd(
    structures: Array3D_float,
    atomnos: Array1D_int,
    max_rmsd: float = 0.25,
    max_dev: float | None = None,
    debugfunction: Callable[[str], None] | None = None,
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

    # set up PrunerConfig dataclass
    prunerconfig = RMSDPrunerConfig(
        structures=structures,
        atomnos=atomnos,
        max_rmsd=max_rmsd,
        max_dev=max_dev,
        debugfunction=debugfunction,
    )

    # run the pruning
    return prune(prunerconfig)


def prune_by_rmsd_rot_corr(
    structures: Array3D_float,
    atomnos: Array1D_int,
    graph: Graph,
    max_rmsd: float = 0.25,
    max_dev: float | None = None,
    logfunction: Callable[[str], None] | None = None,
    debugfunction: Callable[[str], None] | None = None,
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
        subgraphs = [list(vals) for vals in connected_components(graph)]
        all_dists_array = cdist(ref[list(subgraphs[0])], ref[list(subgraphs[1])])
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
    hydrogen_bonds = get_hydrogen_bonds(ref, atomnos, graph)
    for hb in hydrogen_bonds:
        graph.add_edge(*hb)

    # keep an unraveled set of atoms in hbs
    flat_hbs = set(flatten(hydrogen_bonds))

    # get all rotable bonds in the molecule, including dummy rotations
    torsions = get_torsions(
        graph,
        hydrogen_bonds=hydrogen_bonds,
        double_bonds=get_double_bonds_indices(ref, atomnos),
        keepdummy=True,
        mode="symmetry",
    )

    # only keep dummy rotations (checking both directions)
    torsions = [
        t
        for t in torsions
        if not (is_nondummy(t.i2, t.i3, graph) and (is_nondummy(t.i3, t.i2, graph)))
    ]

    # since we only compute RMSD based on heavy atoms, discard
    # quadruplets that involve hydrogen atom as termini, unless
    # they are involved in hydrogen bonding
    torsions = [
        t
        for t in torsions
        if (1 not in [atomnos[i] for i in t.torsion])
        or (t.torsion[0] in flat_hbs or t.torsion[3] in flat_hbs)
    ]

    # get torsions angles
    angles = [get_angles(t, graph) for t in torsions]

    # Used specific directionality of torsions so that we always
    # rotate the dummy portion (the one attached to the last index)
    torsions_ids = np.asarray(
        [
            list(t.torsion) if is_nondummy(t.i2, t.i3, graph) else list(reversed(t.torsion))
            for t in torsions
        ]
    )

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

    # Initialize PrunerConfig
    prunerconfig = RMSDRotCorrPrunerConfig(
        structures=structures,
        atomnos=atomnos,
        graph=graph,
        torsions=torsions_ids,
        debugfunction=debugfunction,
        angles=angles,
        max_rmsd=max_rmsd,
        max_dev=max_dev,
    )

    # run pruning
    structures_out, mask = prune(prunerconfig)

    # remove the extra bond in the molecular graph
    if len(subgraphs) == 2:
        graph.remove_edge(i1, i2)

    return structures_out, mask


def prune_by_moment_of_inertia(
    structures: Array3D_float,
    atomnos: Array1D_int,
    max_deviation: float = 1e-2,
    debugfunction: Callable[[str], None] | None = None,
) -> tuple[Array3D_float, Array1D_bool]:
    """Remove duplicate structures using a moments of inertia-based metric.

    Remove duplicate structures (enantiomeric or rotameric) based on the
    moments of inertia on the principal axes. If all three MOI
    deviate less than max_deviation percent from another structure,
    they are classified as rotamers or enantiomers and therefore only one
    of them is kept (i.e. max_deviation = 0.1 is 10% relative deviation).
    """
    # set up PrunerConfig dataclass
    prunerconfig = MOIPrunerConfig(
        structures=structures,
        debugfunction=debugfunction,
        max_dev=max_deviation,
        masses=np.array([pt[a].mass for a in atomnos]),
    )

    return prune(prunerconfig)
