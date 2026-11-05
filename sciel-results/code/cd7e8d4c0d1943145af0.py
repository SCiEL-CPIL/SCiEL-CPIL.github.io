"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # Approach: Spectral Centrality
    # Vertices with low eigenvector centrality are "peripheral" and
    # generally better candidates for the Maximum Independent Set,
    # as they are less likely to be part of the dense core.
    try:
        # Compute eigenvalues and eigenvectors for the symmetric adjacency matrix
        eigvals, eigvecs = np.linalg.eigh(adj)
        # The largest eigenvalue is the last one
        v = eigvecs[:, -1]
        
        # Ensure the eigenvector is non-negative
        if np.sum(v) < 0:
            v = -v
            
        # Score is negative eigenvector value:
        # Lower centrality -> Higher score -> Selected earlier
        score = -v
    except Exception:
        # Fallback to simple degree heuristic if spectral decomposition fails
        degrees = adj.sum(axis=1)
        score = -degrees

    return score
# EVOLVE-BLOCK-END
