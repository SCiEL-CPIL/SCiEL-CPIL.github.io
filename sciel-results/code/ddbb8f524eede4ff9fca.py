import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = 120.0  # Safety margin before 150s kill
    
    A = np.asarray(adjacency, dtype=float)
    n = A.shape[0]
    
    # Calculate degrees
    degrees = A.sum(axis=1)
    
    # Calculate 2 * L(v) where L(v) = number of edges in N(v)
    # B = A^2, B[v,u] = number of common neighbors of v and u
    # two_L[v] = sum_{u in N(v)} B[v,u] = sum_{u in N(v)} |N(v) ∩ N(u)| = 2 * L(v)
    B = A @ A
    two_L = np.sum(A * B, axis=1)
    L = two_L / 2.0
    
    # Score: L(v) - d(v)
    # Rationale: picking v gains 1 but blocks d(v) neighbors.
    # However, L(v) edges among neighbors mean those neighbors conflict with each other,
    # so the "effective" number of lost picks is reduced by approximately L(v).
    # Net cost ≈ d(v) - L(v), so we want to minimize this → score = L(v) - d(v)
    scores = L - degrees
    
    # Safety check
    if time.time() - start_time > deadline:
        return scores
    
    return scores
