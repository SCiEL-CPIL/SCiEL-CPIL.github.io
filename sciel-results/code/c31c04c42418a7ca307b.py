import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    """
    Returns 12 positive double-root locations for the Laguerre combination.
    Based on the zeros of the generalized Laguerre polynomial L_12^(12)(x).
    """
    m = 12
    alpha = 12
    # roots_genlaguerre returns roots and weights
    roots, weights = roots_genlaguerre(m, alpha)
    # Ensure they are positive and sorted
    roots = sorted([r for r in roots if r > 0])
    return roots
