"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Use 8-point Gauss-Legendre quadrature rule on [0, 1].
    # This rule is exact for polynomials up to degree 15 and provides
    # high accuracy for smooth functions like 1/E(a) in cosmological models.
    
    # Get nodes and weights for [-1, 1]
    nodes_gl, weights_gl = np.polynomial.legendre.leggauss(8)
    
    # Transform to [0, 1]
    # x = (y + 1) / 2
    # weights are scaled by dx/dy = 1/2
    nodes = (nodes_gl + 1.0) / 2.0
    weights = weights_gl / 2.0
    
    # Ensure nodes are strictly increasing and in (0, 1)
    # leggauss nodes are already sorted and in (-1, 1)
    
    # Verify constraints
    # Weights should sum to 1
    # Nodes should be positive
    
    return {"nodes": nodes, "weights": weights}
# EVOLVE-BLOCK-END
