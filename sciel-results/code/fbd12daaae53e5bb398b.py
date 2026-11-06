"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Worst-Fit: Prefer bins with the most remaining capacity.
    # This strategy preserves large contiguous spaces, which is beneficial
    # for Weibull distributions where large items may arrive later.
    # By keeping bins 'fresh', we increase the chance that large items
    # fit into existing bins rather than requiring new ones.
    return bins
# EVOLVE-BLOCK-END
