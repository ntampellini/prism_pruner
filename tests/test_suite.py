"""Tests for the prism_pruner package."""

import os

import numpy as np

from prism_pruner import (
    prune_by_moment_of_inertia,
    prune_by_rmsd,
    prune_by_rmsd_rot_corr,
)
from prism_pruner.graph_manipulations import graphize
from prism_pruner.utils import read_xyz

test_dir = os.path.dirname(os.path.realpath(__file__))


def test_two_identical() -> None:
    """Test that two identical structures evaluate as similar under all metrics."""
    os.chdir(test_dir)
    mol = read_xyz("P4_folded.xyz")
    coords = np.stack((mol.atomcoords[0], mol.atomcoords[0]))
    graph = graphize(mol.atomcoords[0], mol.atomnos)

    pruned, _ = prune_by_rmsd(coords, mol.atomnos)
    assert len(pruned) == 1

    pruned, _ = prune_by_rmsd_rot_corr(coords, mol.atomnos, graph)
    assert len(pruned) == 1

    pruned, _ = prune_by_moment_of_inertia(coords, mol.atomnos)
    assert len(pruned) == 1


def test_two_different() -> None:
    """Test that two different structures evaluate as different under all metrics."""
    os.chdir(test_dir)

    mol1 = read_xyz("P4_folded.xyz")
    mol2 = read_xyz("P4_hairpin.xyz")

    graph1 = graphize(mol1.atomcoords[0], mol1.atomnos)
    coords = np.stack((mol1.atomcoords[0], mol2.atomcoords[0]))

    pruned, _ = prune_by_rmsd(coords, mol1.atomnos)
    assert len(pruned) == 2

    pruned, _ = prune_by_rmsd_rot_corr(coords, mol1.atomnos, graph1)
    assert len(pruned) == 2

    pruned, _ = prune_by_moment_of_inertia(coords, mol1.atomnos)
    assert len(pruned) == 2
