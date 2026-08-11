"""Tests for the prism_pruner package."""

from pathlib import Path
from time import perf_counter

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
        debugfunction=lambda x: None,
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


def test_timeout() -> None:
    """Test timeout function."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "ensemble_100.xyz")
    ensemble.coords = np.concatenate([ensemble.coords] * 100)

    t_start = perf_counter()

    prune(
        ensemble.coords,
        ensemble.atoms,
        moi_pruning=False,
        rmsd_pruning=False,
        rot_corr_rmsd_pruning=True,
        timeout_s=1,
    )

    elapsed = perf_counter() - t_start
    assert elapsed < 2


def test_to_xyz_roundtrip_energies() -> None:
    """Assert that to_xyz writes energies to the comment lines and they round-trip."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "crest_conformers.xyz", read_energies=True)

    sub = ConformerEnsemble(
        coords=ensemble.coords[:10],
        atoms=ensemble.atoms,
        energies=ensemble.energies[:10],
    )

    outfile = HERE / "test_energies_roundtrip.xyz"
    try:
        sub.to_xyz(outfile)

        # each block's comment line must contain the corresponding energy
        lines = outfile.read_text().splitlines()
        natoms = len(sub.atoms)
        for i, energy in enumerate(sub.energies):
            comment = lines[i * (natoms + 2) + 1]
            assert comment == f"{energy}"

        # re-reading the file must recover the same energies and coordinates
        # (coords are written with 8 decimals, so allow a small tolerance)
        reread = ConformerEnsemble.from_xyz(outfile, read_energies=True)
        np.testing.assert_allclose(reread.energies, sub.energies)
        np.testing.assert_allclose(reread.coords, sub.coords, atol=1e-6)
    finally:
        outfile.unlink(missing_ok=True)


def test_pruned_energies_aligned() -> None:
    """Assert that pruning with energies keeps one matching energy per structure."""
    ensemble = ConformerEnsemble.from_xyz(HERE / "crest_conformers.xyz", read_energies=True)

    n = 50
    # mirror the CLI: sort by energy up front so the output is energy-ordered
    order = np.argsort(ensemble.energies[:n])
    coords = ensemble.coords[:n][order]
    energies = ensemble.energies[:n][order]

    pruned, mask = prune(
        coords,
        ensemble.atoms,
        energies=energies,
        max_dE=1.0,
        debugfunction=lambda x: None,
    )

    # the returned mask must be consistent with the (energy-sorted) input
    np.testing.assert_array_equal(coords[mask], pruned)

    # kept energies are ascending (pruned structures are energy-sorted)
    # and one per kept structure
    kept = energies[mask]
    assert len(kept) == len(pruned)
    assert np.all(np.diff(kept) >= 0)
