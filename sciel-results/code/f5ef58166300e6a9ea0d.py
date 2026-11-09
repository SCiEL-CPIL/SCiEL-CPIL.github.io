# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # The Cohn-Goncalves construction for the Fourier sign uncertainty
    # upper bound with m=12 and n=1 uses a specific Laguerre parameter.
    # The standard "double root" structure in this context corresponds to
    # the zeros of the generalized Laguerre polynomial L_m^(1)(x),
    # i.e., alpha = 1.
    roots, _ = roots_genlaguerre(12, 1)
    # Ensure positive and sort
    roots = sorted([float(r) for r in roots])
    return roots
# EVOLVE-BLOCK-END
