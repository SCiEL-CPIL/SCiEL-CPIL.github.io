"""Program using local spectral (Katz) centrality for Maximum Independent Set priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    """
    Assign scores to vertices based on local spectral (Katz) centrality.
    Vertices with lower local centrality (less connected to immediate neighbors) 
    get higher priority.
    
    We use power iteration to first estimate the spectral radius of the adjacency matrix,
    then compute the Katz centrality vector. Katz centrality with a small damping factor 
    emphasizes local neighborhood structure, which is more relevant for the 
    Maximum Independent Set problem than global eigenvector centrality.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Step 1: Estimate the spectral radius (largest eigenvalue) of A
    # Use power iteration
    rng = np.random.RandomState(42)
    v = rng.rand(n)
    v = v / np.linalg.norm(v)
    
    lambda_max = 0.0
    for _ in range(100):
        v_new = adjacency @ v
        norm = np.linalg.norm(v_new)
        if norm < 1e-12:
            lambda_max = 0.0
            break
        lambda_max = norm
        v_new = v_new / norm
        if np.linalg.norm(v_new - v) < 1e-10:
            v = v_new
            break
        v = v_new
    
    # Step 2: Compute Katz Centrality
    # x = (I - alpha * A)^(-1) * ones
    # Iterate: x_new = ones + alpha * A @ x_old
    # Choose alpha to be small enough for convergence and locality
    if lambda_max > 1e-10:
        alpha = 0.2 / lambda_max
    else:
        alpha = 0.1
        
    x = np.ones(n)
    # Iterate until convergence
    for _ in range(1000):
        x_new = 1.0 + alpha * (adjacency @ x)
        # Check convergence
        if np.linalg.norm(x_new - x) < 1e-8:
            x = x_new
            break
        x = x_new
        
    # Lower Katz centrality => higher priority
    scores = -x
    
    return scores
# EVOLVE-BLOCK-END
