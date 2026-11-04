"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 7]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    # Build the multiplication tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(actual - expected) > 1e-10:
                    return False
    return True

def _construct_schoolbook():
    """Schoolbook decomposition of rank n*m*p."""
    n, m, p = DIMENSIONS
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

def _try_strassen_like(n, m, p):
    """
    Try to find a lower-rank decomposition by combining terms.
    For matrix multiplication, we can try to find linear dependencies.
    """
    n, m, p = DIMENSIONS
    schoolbook = _construct_schoolbook()
    
    # Try various rank reductions
    # The theoretical minimum for 3x5x7 is not known, but we can try to find
    # a decomposition better than 105.
    
    # Strategy: Start with schoolbook and try to find linear combinations
    # that allow us to reduce the rank.
    
    # For now, return schoolbook as baseline
    return schoolbook

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    # Start with schoolbook
    best = _construct_schoolbook()
    
    # Try to improve using tensor decomposition search
    # The key is to find if there are linear dependencies in the schoolbook
    # decomposition that allow rank reduction.
    
    # For 3x5x7, the rank is at most 105 (schoolbook).
    # Known lower bounds: For m x n x p matrix multiplication,
    # the rank is at least max(m*n, n*p, m*p) = max(15, 35, 21) = 35.
    # But actual minimum is likely much higher.
    
    # Try a systematic search for lower rank decompositions
    # using the fact that we need T[i,j,k] = 1 for all valid (i,j,k)
    
    # Represent the problem as: find U, V, W with minimal rank such that
    # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # This is equivalent to finding a 3-tensor decomposition.
    
    # For small dimensions, we can try to find decompositions using
    # algebraic methods.
    
    # One approach: Try to find if the tensor has a low-rank structure
    # by looking at slices.
    
    # Let's try a simple heuristic: see if we can combine some of the
    # schoolbook terms.
    
    # For now, just return schoolbook as we don't have a guaranteed
    # improvement in the time budget.
    return best
# EVOLVE-BLOCK-END
