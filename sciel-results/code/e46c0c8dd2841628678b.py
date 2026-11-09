import numpy as np
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
def construct():
    # Use generalized Laguerre polynomials with alpha = -0.5.
    # This parameter is often associated with 1D radial problems or specific
    # uncertainty principles where the standard alpha=0 is suboptimal.
    # The "double roots" are the roots of the base polynomial L_12^(-0.5)(x).
    roots, _ = roots_genlaguerre(12, -0.5)
    # Ensure they are returned as a list of floats
    return [float(r) for r in roots]
# EVOLVE-BLOCK-END
