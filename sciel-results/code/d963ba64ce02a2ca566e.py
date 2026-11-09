"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre

def construct():
    # The task mentions "12 double roots with m=12".
    # In the context of Laguerre polynomials and uncertainty principles,
    # the optimal nodes are often the roots of the Laguerre polynomial L_m.
    # For m=12, we compute the roots of L_12(x).
    # roots_laguerre returns the roots in ascending order.
    x, w = roots_laguerre(12)
    # The roots are positive for Laguerre polynomials.
    # We return them as a list of floats.
    return [float(r) for r in x]
# EVOLVE-BLOCK-END
