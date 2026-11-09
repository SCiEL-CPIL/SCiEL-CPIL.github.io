"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_genlaguerre

def construct():
    # Cohn-Goncalves construction for m=12, n=1
    # The relevant Laguerre parameter for dimension n=1 is alpha = 2*n = 2
    # Using zeros of generalized Laguerre polynomial L_12^(2)(x)
    roots, _ = roots_genlaguerre(12, 2)
    # Ensure positive and sort
    roots = sorted([float(r) for r in roots])
    return roots
# EVOLVE-BLOCK-END
