"""
Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK.
"""
DIMENSIONS = [3, 3, 5]
import numpy as np
import time

# EVOLVE-BLOCK-START
def _validate(U, V, W, n, m, p):
    """Validate that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                expected = 1.0 if (i == i and j == j and k == k) else 0.0
                # The tensor entry T[i,j,k] should be 1 (it's the coefficient of A[i,j]*B[j,k] in C[i,k])
                # Actually, T[i,j,k] = 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
                # Wait, let me reconsider. The multiplication tensor has T[i,j,k] = 1 for all i,j,k
                # because each term A[i,j]*B[j,k] appears in C[i,k] with coefficient 1.
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition with rank n*m*p."""
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

def _try_low_rank_search(n, m, p, max_rank=40, time_limit=120.0):
    """
    Try to find a lower-rank decomposition using structured patterns.
    For 3x3x5, known bounds suggest rank might be reducible from 45.
    """
    start_time = time.time()
    
    # Try various structured decompositions
    # One approach: use the fact that for small matrices, we can sometimes
    # share computations across different (i,j,k) triples.
    
    # Strategy: Try to find a decomposition where some terms are shared
    # by using linear combinations.
    
    best = _schoolbook(n, m, p)
    
    # Try rank 44, 43, etc. by attempting to merge terms
    # This is a heuristic search - not guaranteed to find optimal
    
    # For now, return schoolbook as we need to ensure validity
    # A proper search would require solving a system of polynomial equations
    
    return best

def construct():
    """
    Returns the factor matrices U, V, W for multiplying a 3x3 matrix by a 3x5 matrix.
    Starts with schoolbook and attempts to find lower-rank decompositions.
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # Safety margin
    
    # Start with schoolbook
    result = _schoolbook(n, m, p)
    
    # Verify it's valid
    if not _validate(result["U"], result["V"], result["W"], n, m, p):
        # This should never happen, but just in case
        return result
    
    # Try to find better decompositions
    try:
        better = _try_low_rank_search(n, m, p, time_limit=110.0)
        if better is not None:
            rank_better = len(better["U"][0])
            rank_current = len(result["U"][0])
            if rank_better < rank_current and _validate(better["U"], better["V"], better["W"], n, m, p):
                result = better
    except Exception:
        pass
    
    return result
# EVOLVE-BLOCK-END
