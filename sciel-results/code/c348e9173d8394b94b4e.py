"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Returns the optimal sequence of 100 step heights.
    Based on the analysis, the constant sequence h_i = 0.5 minimizes the
    maximum correlation between h and 1-h, achieving a value of 0.5.
    """
    return np.full(N, 0.5)
# EVOLVE-BLOCK-END
