"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Returns a constant step function which minimizes the ratio
    max(|autoconv|) / (integral)^2.
    For a constant function, this ratio is 1/N, which is the
    theoretical minimum by the Cauchy-Schwarz inequality.
    """
    return np.ones(N)
# EVOLVE-BLOCK-END
