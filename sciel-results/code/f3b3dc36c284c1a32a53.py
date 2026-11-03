"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]
import time
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Schoolbook solution
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_sb
    
    # Define the tensor entries that must be 1
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_entries.append((i * m + j, j * p + k, k * n + i))
    
    # Try to find a better solution
    # We'll try ranks from best_rank-1 down to some minimum
    # For each rank, try a few random initializations
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W with given rank correctly decompose the tensor."""
        # Check target entries
        for (a, b, c) in target_entries:
            val = sum(U[a, s] * V[b, s] * W[c, s] for s in range(rank))
            if abs(val - 1) > 1e-9:
                return False
        # We also need to check that non-target entries are 0, but that's expensive
        # For now, just check target entries (necessary but not sufficient)
        return True
    
    # Try a simple heuristic: for each rank, try a few random solutions
    # This is very unlikely to work for high ranks, but let's try
    
    # Actually, this approach is too slow. Let me just return the schoolbook solution.
    
    return best
# EVOLVE-BLOCK-END
