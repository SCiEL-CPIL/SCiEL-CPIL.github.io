"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

import time
import random
import numpy as np
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # The tensor T[i,j,k] = 1 for all i in range(n), j in range(m), k in range(p)
    # We need to find U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # such that for all i,k: sum_j A[i,j]*B[j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # More precisely: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Try ranks from low to high
    for rank in range(15, 31):
        if time.time() > deadline - 30:
            break
        
        # Try to find a decomposition at this rank
        result = _try_find_decomposition(n, m, p, rank, deadline)
        if result is not None:
            return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    rank = len(U[0]) if U else 0
    # For each (i, j, k), check that sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for c in range(rank):
                    total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(total - 1) > 1e-10:
                    return False
    return True


def _try_find_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-`rank` decomposition using structured search."""
    
    # Strategy: Use a semi-structured approach based on known efficient algorithms
    # For 2x3x5, try to exploit the structure
    
    # Approach: Try to build the decomposition using a greedy/iterative method
    # Initialize with random values and try to optimize
    
    if rank < n * m * p // 2:  # sanity check, shouldn't be too low
        return None
    
    # Try a structured approach: 
    # The matrix multiplication tensor can be viewed as T[i,j,k] = 1
    # We want: 1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # Use a different strategy: start from schoolbook and try to merge terms
    # Or: use a known construction for small matrices
    
    # For 2x3x5, let's try a specific construction
    # C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    # One idea: use the fact that we can share some computations
    # Let's try rank = 15 first with a structured approach
    
    if rank == 15:
        result = _try_rank_15(n, m, p)
        if result and _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    if rank == 16:
        result = _try_rank_16(n, m, p)
        if result and _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    if rank == 17:
        result = _try_rank_17(n, m, p)
        if result and _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    if rank == 18:
        result = _try_rank_18(n, m, p)
        if result and _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    # Generic search for other ranks
    if rank in [19, 20, 21, 22]:
        result = _generic_search(n, m, p, rank, deadline)
        if result and _verify_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
    
    return None


def _try_rank_15(n, m, p):
    """
    Attempt a rank-15 decomposition for 2x3x5.
    
    Idea: Split the 5 columns into 3+2, use 2x3x3 (rank ~11-12) + 2x3x2 (rank ~10) 
    but that's additive. Instead, try a more clever construction.
    
    Actually, for M_{2,3,5}, let me think about it as:
    C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k]
    
    For each i (2 values) and k (5 values), we have 3 terms.
    Total bilinear terms if done naively: 2*5*3 = 30.
    
    To reduce rank, we need to share multiplications across different (i,k) pairs.
    
    Let's try: for each j (3 values), we compute A[i,j]*B[j,k] for all i,k.
    That's 3 * (2*5) = 30 terms. No sharing yet.
    
    Alternative: Group the k indices. For k in {0,1,2,3,4}:
    - Compute partial sums for groups of k's
    
    Let me try a different approach: use the structure of 2x3x5.
    
    C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0]
    C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1]
    C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2] + A[0,2]B[2,2]
    C[0,3] = A[0,0]B[0,3] + A[0,1]B[1,3] + A[0,2]B[2,3]
    C[0,4] = A[0,0]B[0,4] + A[0,1]B[1,4] + A[0,2]B[2,4]
    C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0] + A[1,2]B[2,0]
    C[1,1] = A[1,0]B[0,1] + A[1,1]B[1,1] + A[1,2]B[2,1]
    C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2] + A[1,2]B[2,2]
    C[1,3] = A[1,0]B[0,3] + A[1,1]B[1,3] + A[1,2]B[2,3]
    C[1,4] = A[1,0]B[0,4] + A[1,1]B[1,4] + A[1,2]B[2,4]
    
    Key observation: A[0,0]*B[0,k] appears in C[0,k] for all k.
    A[1,0]*B[0,k] appears in C[1,k] for all k.
    
    So for each (i,j) pair, the product A[i,j]*B[j,k] contributes to multiple C[i,k].
    
    The rank is the minimum number of "scalar multiplications" where each multiplication 
    x*y can be used to contribute to multiple output entries via linear combinations.
    
    For rank 15, we need 15 scalar multiplications.
    
    Let me try: 
    - 5 multiplications for A[0,0]*B[0,k] for k=0..4? No, that's 5 separate mults.
    
    Better: Think of it as matrix multiplication where we can precompute.
    
    Actually, the bilinear complexity is the rank of the tensor. Let me just try 
    a numerical optimization approach for small ranks.
    """
    return None


def _try_rank_16(n, m, p):
    return None


def _try_rank_17(n, m, p):
    return None


def _try_rank_18(n, m, p):
    return None


def _generic_search(n, m, p, rank, deadline):
    """
    Generic search: try to find a valid decomposition at the given rank.
    Uses a simple iterative approach.
    """
    # For now, fall back to schoolbook if rank >= n*m*p
    if rank >= n * m * p:
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    return None

# EVOLVE-BLOCK-END
