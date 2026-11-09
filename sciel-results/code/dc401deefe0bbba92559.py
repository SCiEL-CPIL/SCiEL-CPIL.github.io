"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]

    # Compute degrees
    degrees = adj.sum(axis=1)

    # Check time
    if time.time() > deadline - 10:
        # Not enough time, fall back to simple degree
        return -degrees.astype(float)

    # Structural Change: Use Local Clustering Coefficient
    # Clustering coefficient C_i = (2 * edges_in_N(i)) / (d_i * (d_i - 1))
    # edges_in_N(i) is the number of edges between neighbors of i.
    # This is given by the diagonal of A^2? No, (A^2)_{ii} is the number of triangles involving i, which is 2 * edges_in_N(i).
    # So edges_in_N(i) = (A^2)_{ii} / 2.
    
    # Compute A^2
    A2 = adj @ adj
    
    # Diagonal of A^2 is 2 * (number of edges in the neighborhood of i)
    # For i with degree d_i, the max possible edges in N(i) is d_i*(d_i-1)/2.
    # Clustering Coeff C_i = (A^2)_{ii} / (d_i * (d_i - 1))
    
    # Avoid division by zero
    denom = degrees * (degrees - 1)
    denom[denom == 0] = 1.0 # Avoid div by zero, these will have C=0 anyway
    
    clustering_coeff = A2.diagonal() / denom
    
    # Set clustering coeff to 0 for nodes with degree < 2 (undefined/0)
    clustering_coeff[degrees < 2] = 0.0

    # Score: We want to minimize Degree and Clustering Coefficient.
    # Vertices with low degree and low clustering are preferred.
    # Clustering coefficient is in [0, 1]. Degree can be large.
    # We need to balance them.
    # Let's normalize both to [0, 1] and combine.
    
    def normalize(x):
        mn, mx = np.min(x), np.max(x)
        if mx - mn > 1e-10:
            return (x - mn) / (mx - mn)
        return np.zeros_like(x)

    deg_norm = normalize(degrees)
    clus_norm = normalize(clustering_coeff)

    # Weighted sum: Degree is primary, Clustering is secondary structural signal
    # High clustering means the neighborhood is a clique-like structure, which is bad for MIS expansion.
    score = -(1.0 * deg_norm + 0.8 * clus_norm)

    return score.astype(float)
# EVOLVE-BLOCK-END
