"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    n = 200
    h = np.zeros(N, dtype=np.float64)
    h[:n] = 1.0
    h[n:] = -1.0
    return h
# EVOLVE-BLOCK-END
