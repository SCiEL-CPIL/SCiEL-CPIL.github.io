"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    pair = demands[:, None] + demands[None, :]
    mask = pair <= capacity
    ratio = pair / capacity
    # Strongly favor pairs that nearly fill the bin
    heuristic = np.where(mask, ratio ** 16, 1e-9)
    return heuristic
# EVOLVE-BLOCK-END
