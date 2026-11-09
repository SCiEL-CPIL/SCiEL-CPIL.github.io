"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_genlaguerre
# EVOLVE-BLOCK-START
def construct():
    # Use roots of the generalized Laguerre polynomial L_12^(1)(x)
    # n=1 suggests alpha=1, m=12 is the degree
    roots = roots_genlaguerre(12, 1)
    # roots_genlaguerre returns complex array; take real parts (they should be real)
    real_roots = np.real(roots)
    # Sort in ascending order
    real_roots = np.sort(real_roots)
    # Convert to list of floats
    return [float(r) for r in real_roots]
# EVOLVE-BLOCK-END
