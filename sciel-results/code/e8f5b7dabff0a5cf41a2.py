"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree


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
    
    # Compute MST to identify edges that are likely part of good tours
    # MST edges get a boost to encourage connectivity
    try:
        mst = minimum_spanning_tree(d_safe)
        mst_dense = mst.toarray()
        # MST edges get a multiplicative boost
        mst_boost = 1.0 + 0.5 * (mst_dense > 0)
        eta = eta * mst_boost
    except Exception:
        pass
    
    # Spectral gap penalty using Fiedler vector
    try:
        # Build a weighted adjacency from inverse distances
        # Use a moderate power to get a smooth weight matrix
        w = 1.0 / (d_safe ** 1.5 + 1e-30)
        np.fill_diagonal(w, 0.0)
        
        # Compute degree matrix
        degree = w.sum(axis=1)
        
        # Build Laplacian L = D - W
        L = np.diag(degree) - w
        
        # Compute eigenvalues and eigenvectors
        # We need the second smallest eigenvalue's eigenvector (Fiedler vector)
        eigvals, eigvecs = np.linalg.eigh(L)
        
        # Sort by eigenvalue (eigh already returns sorted)
        # Fiedler vector is eigvecs[:, 1] (index 1, since index 0 is the constant vector)
        fiedler = eigvecs[:, 1]
        
        # Compute pairwise differences in Fiedler coordinates
        # diff[i,j] = |fiedler[i] - fiedler[j]|
        fiedler_col = fiedler[:, np.newaxis]
        fiedler_row = fiedler[np.newaxis, :]
        diff = np.abs(fiedler_col - fiedler_row)
        
        # Normalize differences to [0, 1]
        diff_max = diff.max()
        if diff_max > 1e-15:
            diff_norm = diff / diff_max
        else:
            diff_norm = np.zeros_like(diff)
        
        # Dampen edges with large Fiedler coordinate differences
        # Use a smooth penalty: 1 - penalty * diff_norm^power
        penalty_strength = 0.3
        penalty_power = 2.0
        spectral_penalty = 1.0 - penalty_strength * (diff_norm ** penalty_power)
        
        # Ensure penalty stays positive
        spectral_penalty = np.maximum(spectral_penalty, 0.1)
        
        # Apply spectral penalty multiplicatively
        eta = eta * spectral_penalty
        
        # Zero out diagonal
        np.fill_diagonal(eta, 0.0)
    except Exception:
        pass
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    return eta
# EVOLVE-BLOCK-END
