import time

from prism_pruner.conformer_ensemble import ConformerEnsemble
from prism_pruner.pruner import prune

ensemble = ConformerEnsemble.from_xyz("iodine_ensemble.xyz")

start_time = time.time()

pruned, mask = prune(
    ensemble.coords,
    ensemble.atoms,
    rot_corr_rmsd_pruning=True,
    debugfunction=print,
)

end_time = time.time()

print(end_time - start_time)
