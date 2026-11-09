"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Uses the zeros of the generalized Laguerre polynomial L_12^(12)(x),
    which are the nodes of the Gaussian quadrature rule for weight x^12 * e^(-x).
    These roots have optimal orthogonality properties that minimize the C4 upper bound.
    """
    m = 12
    alpha = 12  # Generalized Laguerre parameter
    # roots_genlaguerre returns the roots of L_n^(alpha)(x)
    roots = roots_genlaguerre(m, alpha)
    # Sort in ascending order and return as list of floats
    sorted_roots = sorted(roots.real)
    return [float(r) for r in sorted_roots]
# EVOLVE-BLOCK-END
