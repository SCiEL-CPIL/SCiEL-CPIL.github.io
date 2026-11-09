import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    # Compute the 12 roots of the Laguerre polynomial L_12(x)
    # In the context of the Cohn-Goncalves construction for 1D Fourier uncertainty,
    # the Laguerre polynomial argument often corresponds to x^2 (or r^2).
    # Therefore, the actual spatial locations (double roots of the extremal polynomial in x)
    # are the square roots of the Laguerre roots.
    roots, _ = roots_laguerre(12)
    # Ensure they are returned as a list of floats
    return [float(np.sqrt(r)) for r in roots]
# EVOLVE-BLOCK-END
