"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    fits = pair <= capacity
    # Reward pairs that fill a large fraction of the bin capacity
    # Using squared fraction to emphasize good utilization without being too peaked
    fraction = np.where(fits, pair / capacity, 0.0)
    eta = np.where(fits, fraction ** 2, 1e-9)
    return eta
# EVOLVE-BLOCK-END
