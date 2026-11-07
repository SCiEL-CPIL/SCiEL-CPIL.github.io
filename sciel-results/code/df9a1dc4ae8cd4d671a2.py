"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]
    if n < 2:
        return np.ones_like(d, dtype=np.float64)
    
    # Set diagonal to a large value so 1/d^alpha becomes very small
    d_safe = d.copy()
    np.fill_diagonal(d_safe, 1e10)
    
    # Use inverse cubic distance for sharper guidance toward short edges
    alpha = 3.0
    eta = 1.0 / (d_safe ** alpha + 1e-30)
    
    # Degree-based penalty: penalize edges to nodes that are already
    # highly connected to many other close nodes, to avoid over-concentration
    # and encourage more balanced tour structure
    k = 5.0
    popularity = np.sum(np.exp(-k * d_safe), axis=0)
    pop_mean = np.mean(popularity)
    if pop_mean > 0:
        popularity_norm = popularity / pop_mean
    else:
        popularity_norm = np.ones(n)
    
    beta = 0.5
    penalty = 1.0 + beta * (popularity_norm - 1.0)
    penalty = np.where(penalty > 0, penalty, 1e-9)
    eta = eta / penalty[np.newaxis, :]
    
    # Spectral Gap Penalty using Fiedler Vector
    # Construct a weighted adjacency matrix based on the heuristic weights
    # We use the inverse distance weights to define the graph structure
    W = eta.copy()
    np.fill_diagonal(W, 0)
    
    # Compute Laplacian L = D - W
    degrees = np.sum(W, axis=1)
    D = np.diag(degrees)
    L = D - W
    
    # Compute eigenvalues and eigenvectors
    # We only need the second smallest eigenvalue's vector (Fiedler vector)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # eigh returns eigenvalues in ascending order
        # The smallest is usually 0 (for connected graphs)
        # The Fiedler vector is the eigenvector corresponding to eigenvalues[1]
        fiedler_vec = eigenvectors[:, 1]
        
        # Normalize the Fiedler vector to have zero mean and unit variance
        # This makes the differences comparable
        fiedler_vec = fiedler_vec - np.mean(fiedler_vec)
        std = np.std(fiedler_vec)
        if std > 1e-9:
            fiedler_vec = fiedler_vec / std
            
        # Compute pairwise differences in Fiedler coordinates
        diff = np.abs(fiedler_vec[:, np.newaxis] - fiedler_vec[np.newaxis, :])
        
        # Apply exponential damping to edges with large spectral distance
        # gamma controls the strength of the penalty
        # A larger gamma means stronger penalty for "long" spectral edges
        gamma = 0.5
        spectral_penalty = np.exp(-gamma * diff)
        
        # Multiply heuristic by spectral penalty
        eta = eta * spectral_penalty
        
    except Exception:
        # If spectral computation fails for any reason, fall back to no spectral penalty
        pass
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta
# EVOLVE-BLOCK-END
