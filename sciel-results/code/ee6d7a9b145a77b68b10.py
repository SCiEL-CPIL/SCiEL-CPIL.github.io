"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    fits = pair <= capacity
    # Reward tight packing: smaller remaining space after placing i+j is more attractive
    remaining = np.where(fits, capacity - pair, capacity)
    # Inverse of (remaining + 1) strongly favors tight fits
    eta = np.where(fits, 1.0 / (remaining + 1.0), 1e-9)
    return eta
# EVOLVE-BLOCK-END
