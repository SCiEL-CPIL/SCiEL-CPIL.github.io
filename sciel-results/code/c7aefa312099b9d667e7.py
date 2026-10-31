"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    h = np.zeros(N)
    h[25:50] = 1.0
    h[75:100] = 1.0
    return h
# EVOLVE-BLOCK-END
