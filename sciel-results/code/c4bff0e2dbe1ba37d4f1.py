"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math

def construct():
    # Exact algebraic solution
    # The optimal coefficients correspond to the basis function with the smallest
    # Rayleigh quotient R_i = Mx_ii / M_ii.
    # Based on analysis, i=0 (H0) is optimal.
    # The vector is [1, 0, 0] normalized.
    return np.array([1.0, 0.0, 0.0])

# EVOLVE-BLOCK-END
