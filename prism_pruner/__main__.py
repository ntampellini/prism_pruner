"""CLI interface."""

import argparse
from pathlib import Path

from prism_pruner.conformer_ensemble import ConformerEnsemble
from prism_pruner.pruner import prune


def cli_main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "inputfile",
        help="Multimolecular .xyz file.",
        action="store",
    )

    parser.add_argument(
        "-e",
        "--energies",
        help="Attempt reading energies from the .xyz file to speed up pruning.",
        action="store_true",
        required=False,
        default=False,
    )

    parser.add_argument(
        "-t",
        "--timeout",
        help="Set maximum time (in seconds) for each pruning step.",
        action="store",
        required=False,
        default=60,
    )

    args = parser.parse_args()

    # read input file
    ens = ConformerEnsemble.from_xyz(args.inputfile, read_energies=args.energies)
    print(f"--> Read {len(ens.coords)} structures from {args.inputfile}.")

    # perform pruning
    pruned_coords, _ = prune(
        ens.coords,
        ens.atoms,
        energies=ens.energies if args.energies else None,
        max_dE=1.0 if args.energies else 0.0,
        timeout_s=args.timeout,
        logfunction=print,
        debugfunction=print,
    )

    # update ens coordinates
    ens.coords = pruned_coords

    # write new ensemble to file
    outname = f"{Path(args.inputfile).stem}_pruned.xyz"
    ens.to_xyz(outname)

    print(f"--> Wrote {len(ens.coords)} pruned structures to {outname}.")
