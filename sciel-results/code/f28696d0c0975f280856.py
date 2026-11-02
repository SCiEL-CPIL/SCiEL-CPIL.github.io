"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    rank_school = n * m * p  # 60

    def verify_numpy(U, V, W, rank):
        """Fast verification using numpy vectorization."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i,j,k), check sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
        # Build index arrays
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                    if err > 1e-9:
                        return False
        return True

    # Schoolbook construction (always valid)
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Try to find a rank-48 decomposition using a structured approach.
    # Idea: group the 4 middle indices into 2 pairs, and for each pair of (i,k),
    # use shared computation across the 2 j-values in the pair.
    #
    # For a fixed (i,k), we need: sum_{j=0}^{3} A[i,j]*B[j,k] = C[i,k]
    # = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #
    # If we could compute this with fewer than 4 terms per (i,k), we'd save.
    # But the terms must be shared across different (i,k) pairs.
    #
    # Let's try a different structured approach: use the fact that we can have
    # fractional entries. Consider a rank-48 construction where we merge 12 terms.
    #
    # Actually, let's try a systematic search with better initialization.
    # Key insight: the entries don't need to be integers. They can be any rational.
    # 
    # Strategy: For each target rank, try to find a decomposition by:
    # 1. Initialize U, V, W with random sparse patterns
    # 2. Use a local search / coordinate descent to satisfy the constraints
    #
    # But this is complex. Let's try a simpler approach first:
    # Try rank values that are multiples of smaller numbers and use
    # a "block" construction.

    # Try rank 48: we can potentially save 12 terms by sharing computation
    # across some (i,j,k) triples.
    
    # Let's try a constructive rank-48 decomposition:
    # For each i (3 values) and k (5 values), we need to compute sum_j A[i,j]*B[j,k]
    # Total: 15 output entries, each requiring 4 multiplications = 60 terms.
    # 
    # If we can share some multiplications, we can reduce the rank.
    # For example, if for a fixed i, we compute:
    #   s0 = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #   s1 = A[i,2]*B[2,k] + A[i,3]*B[3,k]
    # Then C[i,k] = s0 + s1
    # But this still uses 4 terms per (i,k).
    #
    # The key is to share across different i values for the same k, or
    # across different k values for the same i.
    
    # Let's try a random search with a better structure:
    # Initialize with the schoolbook, then try to "merge" pairs of terms
    # by adjusting entries.
    
    # Alternative: use a known result. For 3x4x5 matrix multiplication,
    # the best known upper bound might be 48 or lower.
    
    # Let's try a direct construction for rank 48:
    # We'll use 48 terms. For 12 of the (i,j,k) triples, we'll have
    # a single term. For the remaining 48, we'll use shared terms.
    
    # Actually, let's just do a more thorough random search with
    # a wider range of entry values and more attempts.
    
    best_result = None
    
    # Try ranks from 48 down to 20
    for target_rank in [48, 47, 46, 45, 44, 43, 42, 40, 38, 36, 35, 34, 32, 30, 28, 26, 24, 22, 20]:
        if time.time() > deadline - 15:
            break
        
        # Multiple random attempts
        for attempt in range(20):
            if time.time() > deadline - 10:
                break
            
            random.seed(attempt * 7919 + target_rank * 13)
            
            # Initialize with sparse random entries
            U = [[0] * target_rank for _ in range(n * m)]
            V = [[0] * target_rank for _ in range(m * p)]
            W = [[0] * target_rank for _ in range(p * n)]
            
            # Fill with random sparse patterns
            for c in range(target_rank):
                # Each term c covers some subset of (i,j,k)
                # Assign random entries
                for idx_u in range(n * m):
                    if random.random() < 0.3:
                        U[idx_u][c] = random.choice([-1, 0, 1, 0.5, -0.5])
                for idx_v in range(m * p):
                    if random.random() < 0.3:
                        V[idx_v][c] = random.choice([-1, 0, 1, 0.5, -0.5])
                for idx_w in range(p * n):
                    if random.random() < 0.3:
                        W[idx_w][c] = random.choice([-1, 0, 1, 0.5, -0.5])
            
            if verify_numpy(U, V, W, target_rank):
                return {"U": U, "V": V, "W": W}
        
        # If we found a valid one at this rank, return it
        # (but we only return if verify passes)
    
    # Return schoolbook as fallback
    return make_schoolbook()
# EVOLVE-BLOCK-END
