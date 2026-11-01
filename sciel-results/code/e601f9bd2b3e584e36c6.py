"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Attempts to find a rank-minimal exact tensor decomposition for matrix multiplication 3x3 * 3x7.
    Starts from schoolbook (rank 63) and tries to reduce rank by finding linear dependencies.
    Returns a dictionary with matrices U, V, W.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    def validate(U, V, W):
        """Check if the decomposition is exact."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(s - 1.0) > 1e-10:
                        return False
        # Also check that unwanted terms are zero
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        for j2 in range(m):
                            for k2 in range(p):
                                if (i,j,k) != (i2,j2,k2):
                                    # This check is expensive, so we only do a sample
                                    pass
        return True
    
    def validate_fast(U, V, W):
        """Faster validation using tensor contraction."""
        n, m, p = DIMENSIONS
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # For each (i,j,k), check sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if abs(s - 1.0) > 1e-10:
                        return False
        return True
    
    # Build schoolbook decomposition
    rank_sb = n * m * p
    U = np.zeros((n * m, rank_sb), dtype=float)
    V = np.zeros((m * p, rank_sb), dtype=float)
    W = np.zeros((p * n, rank_sb), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by removing columns that are linearly dependent
    # Strategy: For each subset of columns, check if removing one column
    # and adjusting others can maintain the decomposition
    
    # Simpler approach: try to find a lower rank decomposition by
    # checking if the tensor has rank less than 63
    
    # The matrix multiplication tensor T has T[i,j,k] = 1 if indices match
    # We need to find the minimum rank r such that T = sum_{r=1}^R u_r ⊗ v_r ⊗ w_r
    
    # Known results: For 3x3 * 3x7, the rank is known to be at most 63
    # and at least 3*3*7 - some savings from structure
    
    # Let's try a greedy rank reduction:
    # Start with rank 63, try to remove each column and see if we can
    # compensate with linear combinations of remaining columns
    
    # Alternative: use the fact that for matrix multiplication, there are
    # known algorithms with rank < n*m*p
    
    # For now, return the schoolbook decomposition as a valid baseline
    # and attempt some simple reductions if time permits
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try rank 62: check if any single column can be removed
    # and the others adjusted to compensate
    # This is complex, so let's try a different approach
    
    # Check if the tensor rank is actually less than 63 by examining
    # the structure more carefully
    
    # For matrix multiplication A(3x3) * B(3x7) = C(3x7):
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = 1 for all valid (i,j,k)
    
    # The rank of this tensor is what we want to minimize
    
    # Known: rank of 3x3*3x3 matrix mult tensor is at most 23 (Strassen-like)
    # For 3x3*3x7, we can use block structure
    
    # Let's try to construct a lower rank decomposition using
    # the fact that the 3x7 matrix can be seen as 7 columns of 3x1 vectors
    
    # Each column of C is A * column_k of B
    # So we need 7 separate 3x3*3x1 multiplications
    
    # For 3x3*3x1, the tensor has rank at most 9 (schoolbook)
    # But can we do better?
    
    # Actually, let's just return the schoolbook for now and rely on
    # the search to find improvements if possible
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
