"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit

    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]

    # Compute degrees - primary signal
    degrees = adj.sum(axis=1)

    # Check time for additional computation
    if time.time() > deadline - 10:
        # Not enough time, return simple degree-based score
        return -degrees.astype(float)

    # Compute local clustering coefficient as tie-breaker
    # For each vertex, count edges among its neighbors
    # A = adj, A[i,j] = 1 if edge
    # For vertex i, neighbors are where adj[i,:] == 1
    # Edges among neighbors: sum over pairs of neighbors j,k of adj[j,k]
    # This is equivalent to (A @ A @ A)[i] / 2 for undirected graphs
    # But more efficiently: for each vertex i, count = sum_j sum_k adj[i,j]*adj[i,k]*adj[j,k]
    # Which is the trace of A[i,:].T @ A @ A[i,:]
    # Simpler: (A @ A @ A)[i] gives sum of length-3 paths from i back to i
    # For undirected: number of triangles containing i = (A^3)[i] / 2
    # But we want edges among neighbors, which is slightly different.
    
    # Actually, edges among neighbors of i = sum_{j in N(i)} sum_{k in N(i), k > j} adj[j,k]
    # = (1/2) * sum_{j in N(i)} (A @ degrees_of_neighbors)[j] ... this is getting complex.
    
    # Simpler approach: use A^2 diagonal as a proxy
    # (A^2)[i,i] = degree of i, not helpful
    # (A^3)[i,i] = number of length-3 closed walks = 2 * (number of triangles containing i)
    # So triangles[i] = (A^3)[i,i] / 2
    
    # But we want edges among neighbors, which includes non-triangle edges too.
    # Let's use a simpler metric: sum of degrees of neighbors (already in current program)
    # But keep it as a very small perturbation
    
    neighbor_degree_sums = adj @ degrees
    
    # Check time again
    if time.time() > deadline - 5:
        return -degrees.astype(float)

    # Primary: negative degree (lower degree = higher score)
    # Secondary: small negative perturbation based on neighbor degree sum
    # Use a very small weight so degree dominates
    score = -degrees - 0.01 * neighbor_degree_sums

    return score.astype(float)
# EVOLVE-BLOCK-END
