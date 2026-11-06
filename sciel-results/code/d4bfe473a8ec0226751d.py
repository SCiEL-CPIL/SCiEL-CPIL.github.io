import numpy as np

def construct():
    # Get Gauss-Legendre nodes and weights for N=8 on interval [-1, 1]
    nodes_old, weights_old = np.polynomial.legendre.leggauss(8)
    
    # Map nodes from [-1, 1] to [0, 1]
    nodes = (nodes_old + 1) / 2
    # Map weights: integral_0^1 f(u) du = 0.5 * integral_-1^1 f((v+1)/2) dv
    # So new weights are old weights * 0.5
    weights = weights_old * 0.5
    
    # Ensure strict increasing order (leggauss returns sorted)
    # Ensure positivity (leggauss weights are positive)
    
    return {"nodes": nodes, "weights": weights}
