# Prism Pruner

[![License](https://img.shields.io/github/license/ntampellini/prism_pruner)](https://github.com/ntampellini/prism_pruner/blob/master/LICENSE)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ntampellini/prism_pruner/test.yml?branch=master&logo=github-actions)](https://github.com/ntampellini/prism_pruner/actions/)
[![Codecov](https://img.shields.io/codecov/c/github/ntampellini/prism_pruner)](https://codecov.io/gh/ntampellini/prism_pruner)
[![PyPI - Version](https://img.shields.io/pypi/v/prism_pruner)](https://pypi.org/project/prism-pruner/)

PRISM (PRuning Interface for Similar Molecules) is the modular similarity pruning code originally from [FIRECODE](https://github.com/ntampellini/FIRECODE/tree/main), in a polished standalone package. It filters out duplicate structures from conformational ensembles, leaving behind non-redundant states.

The code implements a cached, iterative, divide-and conquer approach on increasingly large subsets of the ensemble and removes duplicates as assessed by one of three metrics:
- Relative deviation of the moments of inertia on the principal axes
- Heavy-atom RMSD and maximum deviation
- Rotamer-corrected heavy-atom RMSD and maximum deviation

## Installation
The package is distributed through PyPI.

    pip install prism_pruner

## Usage
The main pruning functions are in prism_pruner.pruning, and a wrapper that chains up to all three is also available. The functions return the pruned ensemble structures and the relative boolean mask.

```python
from prism_pruner.conformer_ensemble import ConformerEnsemble
from prism_pruner.pruner import prune

ensemble = ConformerEnsemble.from_xyz("ensemble.xyz")

ensemble.coords.shape # (1086, 136, 3)

pruned, mask = prune(
    ensemble.coords,
    ensemble.atoms,

    # the third pruning routine can be
    # slow and is often not necessary,
    # so it's off by default
    rot_corr_rmsd_pruning=False,

    debugfunction=print,
)

pruned.shape # (387, 136, 3)
mask.shape # (1086,)
# where pruned is ensemble.coords[mask]
```
For additional performance, it is also possible to read/provide energies to only evaluate the similarity of structures that are energetically close.

For additional usage, see the [examples folder](https://github.com/ntampellini/prism_pruner/tree/master/examples).

## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.
