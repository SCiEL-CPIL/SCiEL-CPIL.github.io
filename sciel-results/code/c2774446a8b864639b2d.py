"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def construct():
    # Use Gauss-Legendre quadrature nodes and weights on [0, 1]
    # These are optimal for smooth integrands and provide better accuracy
    # than the composite midpoint rule for the luminosity distance integrand.
    n = 8
    nodes, weights = np.polynomial.legendre.leggauss(n)
    # Transform from [-1, 1] to [0, 1]
    nodes = 0.5 * (nodes + 1.0)
    weights = 0.5 * weights
    # Ensure weights sum to 1 (they should already, but normalize for safety)
    weights = weights / weights.sum()
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
