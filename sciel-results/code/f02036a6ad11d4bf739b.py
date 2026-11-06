import numpy as np

def construct():
    # Generate 8-point Gauss-Legendre quadrature rule on [-1, 1]
    x, w = np.polynomial.legendre.leggauss(8)
    
    # Map nodes from [-1, 1] to [0, 1]
    nodes = (x + 1.0) / 2.0
    
    # Scale weights to sum to 1 on [0, 1]
    # Integral over [-1, 1] is approx by sum w_i f(x_i)
    # Integral over [0, 1] is 0.5 * Integral over [-1, 1]
    # So new weights are w_i / 2
    weights = w / 2.0
    
    # Ensure strict increasing order (leggauss returns sorted x)
    # nodes are already strictly increasing
    
    return {"nodes": nodes, "weights": weights}
