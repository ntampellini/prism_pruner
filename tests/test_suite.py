"""Tests for the prism_pruner package."""

from pathlib import Path

import numpy as np

from prism_pruner.conformer_ensemble import ConformerEnsemble
from prism_pruner.graph_manipulations import graphize
from prism_pruner.pruner import (
    prune,
    prune_by_moment_of_inertia,
    prune_by_rmsd,
    prune_by_rmsd_rot_corr,
)

HERE = Path(__file__).resolve().parent


def test_two_identical() -> None:
    """Test that two identical structures evaluate as similar under all metrics."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "P4_folded.xyz")
    coords = np.stack((ensemble.coords[0], ensemble.coords[0]))

    pruned, _ = prune_by_moment_of_inertia(coords, ensemble.atoms)
    assert len(pruned) == 1

    pruned, _ = prune_by_rmsd(coords, ensemble.atoms)
    assert len(pruned) == 1

    graph = graphize(ensemble.atoms, ensemble.coords[0])
    pruned, _ = prune_by_rmsd_rot_corr(coords, ensemble.atoms, graph)
    assert len(pruned) == 1


def test_two_different() -> None:
    """Test that two different structures evaluate as different under all metrics."""
    ensemble1 = ConformerEnsemble.from_xyz(HERE / "P4_folded.xyz")
    ensemble2 = ConformerEnsemble.from_xyz(HERE / "P4_hairpin.xyz")
    coords = np.stack((ensemble1.coords[0], ensemble2.coords[0]))

    pruned, _ = prune_by_moment_of_inertia(coords, ensemble1.atoms)
    assert len(pruned) == 2

    pruned, _ = prune_by_rmsd(coords, ensemble1.atoms)
    assert len(pruned) == 2

    graph1 = graphize(ensemble1.atoms, ensemble1.coords[0])
    pruned, _ = prune_by_rmsd_rot_corr(coords, ensemble1.atoms, graph1)
    assert len(pruned) == 2


def test_ensemble_moi() -> None:
    """Assert that an ensemble of structures is reduced in size after MOI pruning."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "ensemble_100.xyz")

    pruned, _ = prune_by_moment_of_inertia(
        ensemble.coords,
        ensemble.atoms,
    )

    assert pruned.shape[0] < ensemble.coords.shape[0]


def test_ensemble_rmsd() -> None:
    """Assert that an ensemble of structures is reduced in size after RMSD pruning."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "ensemble_100.xyz")

    pruned, _ = prune_by_rmsd(
        ensemble.coords,
        ensemble.atoms,
        max_rmsd=1.0,
    )

    assert pruned.shape[0] < ensemble.coords.shape[0]


def test_ensemble_rmsd_rot_corr() -> None:
    """Assert that an ensemble of structures is reduced in size after rot. corr. RMSD pruning."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "ensemble_100.xyz")

    graph = graphize(ensemble.atoms, ensemble.coords[0])

    pruned, _ = prune_by_rmsd_rot_corr(
        ensemble.coords,
        ensemble.atoms,
        graph,
        max_rmsd=1.0,
    )

    assert pruned.shape[0] < ensemble.coords.shape[0]


def test_rmsd_rot_corr_segmented_graph_2_mols() -> None:
    """Assert that an ensemble of structures is reduced in size after rot. corr. RMSD pruning.

    The provided ensemble has four different rotamers and two
    connected components in its graph (i.e. two separate molecules).
    The expected behavior is that this fact should not stump the
    rotamer-invariant function.
    """
    ensemble = ConformerEnsemble.from_xyz(HERE / "MTBE_tBuOH_ens.xyz")

    graph = graphize(ensemble.atoms, ensemble.coords[0])

    pruned, _ = prune_by_rmsd_rot_corr(
        ensemble.coords,
        ensemble.atoms,
        graph,
        max_rmsd=0.1,
    )

    assert pruned.shape[0] == 1


def test_chained_pruning_1() -> None:
    """Assert that chained pruning works and masking is consistent."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "ensemble_100.xyz")

    n = 50

    pruned, mask = prune(
        ensemble.coords[0:n],
        ensemble.atoms,
    )

    np.testing.assert_array_equal(ensemble.coords[0:n][mask], pruned)


def test_chained_pruning_2() -> None:
    """Assert that chained pruning works and masking is consistent."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "ensemble_100.xyz")

    n = 20

    pruned, mask = prune(
        ensemble.coords[0:n],
        ensemble.atoms,
        rot_corr_rmsd_pruning=True,
    )

    np.testing.assert_array_equal(ensemble.coords[0:n][mask], pruned)
