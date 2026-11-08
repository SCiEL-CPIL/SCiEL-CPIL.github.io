import numpy as np
from numpy.polynomial.laguerre import laggauss

# EVOLVE-BLOCK-START
def construct():
    """
    Returns 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Based on the Cohn-Goncalves construction, the optimal roots are hypothesized
    to be the roots of the Laguerre polynomial L_12^(1)(x).
    """
    m = 12
    n = 1
    
    # The roots of L_m^(n)(x) are the nodes of the Gauss-Laguerre quadrature rule.
    nodes, _ = laggauss(m, alpha=n)
    
    # Ensure they are returned as a list of floats, sorted.
    roots = sorted([float(r) for r in nodes])
    
    return roots
# EVOLVE-BLOCK-END
