"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.special import roots_laguerre

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the 12 positive double-root locations using the roots of the
    Laguerre polynomial L_12(x). This is the standard optimal distribution
    for Laguerre-based uncertainty bounds, concentrating roots near the origin
    according to the Laguerre weight function.
    """
    m = 12
    # Get the roots of the Laguerre polynomial of degree m
    # roots_laguerre returns (roots, weights)
    roots, _ = roots_laguerre(m)
    
    # The roots are already positive and sorted in ascending order by SciPy
    # Ensure they are floats and return as a list
    return [float(x) for x in roots]
# EVOLVE-BLOCK-END
