# Prism Pruner

[![License](https://img.shields.io/github/license/ntampellini/prism_pruner)](https://github.com/ntampellini/prism_pruner/blob/master/LICENSE)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ntampellini/prism_pruner/test.yml?branch=master&logo=github-actions)](https://github.com/ntampellini/prism_pruner/actions/)
[![Codecov](https://img.shields.io/codecov/c/github/ntampellini/prism_pruner)](https://codecov.io/gh/ntampellini/prism_pruner)

PRISM (PRuning Interface for Similar Molecules) is the modular similarity pruning code from [FIRECODE](https://github.com/ntampellini/FIRECODE/tree/main), in a standalone package. It filters out duplicate structures from conformational ensembles, leaving behind non-redundant states.

The code implements a cached, iterative, divide-and conquer approach on increasingly large subsets of the ensemble and removes duplicates as assessed by one of three metrics:
- Heavy-atom RMSD and maximum deviation
- Rotamer-corrected heavy-atom RMSD and maximum deviation
- Relative deviation of the moments of inertia on the principal axes

## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.
