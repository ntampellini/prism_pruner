"""Tests for the prism_pruner package."""

from pathlib import Path

import numpy as np

from prism_pruner.conformer_ensemble import ConformerEnsemble
from prism_pruner.graph_manipulations import graphize
from prism_pruner.pruner import (
    prune_by_moment_of_inertia,
    prune_by_rmsd,
    prune_by_rmsd_rot_corr,
)

HERE = Path(__file__).resolve().parent


def test_two_identical() -> None:
    """Test that two identical structures evaluate as similar under all metrics."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "P4_folded.xyz")
    coords = np.stack((ensemble.coords[0], ensemble.coords[0]))
    graph = graphize(ensemble.atomnos, ensemble.coords[0])

    pruned, _ = prune_by_rmsd(coords, ensemble.atomnos)
    assert len(pruned) == 1

    pruned, _ = prune_by_rmsd_rot_corr(coords, ensemble.atomnos, graph)
    assert len(pruned) == 1

    pruned, _ = prune_by_moment_of_inertia(coords, ensemble.atomnos)
    assert len(pruned) == 1


def test_two_different() -> None:
    """Test that two different structures evaluate as different under all metrics."""
    ensemble1 = ConformerEnsemble.from_xyz(HERE / "P4_folded.xyz")
    ensemble2 = ConformerEnsemble.from_xyz(HERE / "P4_hairpin.xyz")

    graph1 = graphize(ensemble1.atomnos, ensemble1.coords[0])
    coords = np.stack((ensemble1.coords[0], ensemble2.coords[0]))

    pruned, _ = prune_by_rmsd(coords, ensemble1.atomnos)
    assert len(pruned) == 2

    pruned, _ = prune_by_rmsd_rot_corr(coords, ensemble1.atomnos, graph1)
    assert len(pruned) == 2

    pruned, _ = prune_by_moment_of_inertia(coords, ensemble1.atomnos)
    assert len(pruned) == 2


def test_ensemble_moi() -> None:
    """Assert that an ensemble of structures is reduced in size after MOI pruning."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "ensemble_100.xyz")

    pruned, _ = prune_by_moment_of_inertia(
        ensemble.coords,
        ensemble.atomnos,
    )

    assert pruned.shape[0] < ensemble.coords.shape[0]


def test_ensemble_rmsd() -> None:
    """Assert that an ensemble of structures is reduced in size after RMSD pruning."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "ensemble_100.xyz")

    pruned, _ = prune_by_rmsd(
        ensemble.coords,
        ensemble.atomnos,
        max_rmsd=1.0,
    )

    assert pruned.shape[0] < ensemble.coords.shape[0]


def test_ensemble_rmsd_rot_corr() -> None:
    """Assert that an ensemble of structures is reduced in size after rot. corr. RMSD pruning."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "ensemble_100.xyz")

    graph = graphize(ensemble.atomnos, ensemble.coords[0])

    pruned, _ = prune_by_rmsd_rot_corr(
        ensemble.coords,
        ensemble.atomnos,
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

    graph = graphize(ensemble.atomnos, ensemble.coords[0])

    pruned, _ = prune_by_rmsd_rot_corr(
        ensemble.coords,
        ensemble.atomnos,
        graph,
        max_rmsd=0.1,
    )

    assert pruned.shape[0] == 1
