"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    h = np.zeros(N)
    h[0] = 0.5
    for i in range(1, N):
        s = h[i-1]
        h[i] = 1.0 - s
    return h
# EVOLVE-BLOCK-END
