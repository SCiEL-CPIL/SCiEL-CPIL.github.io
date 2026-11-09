"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_genlaguerre

# EVOLVE-BLOCK-START
def construct():
    # Refined Cohn-Goncalves Laguerre-polynomial construction
    # m=12 double roots, n=1 dimension
    # Try alpha=0 generalized Laguerre roots
    m = 12
    alpha = 0.0
    roots = roots_genlaguerre(m, alpha)[0]
    roots_sorted = sorted(float(r) for r in roots)
    return roots_sorted
# EVOLVE-BLOCK-END
