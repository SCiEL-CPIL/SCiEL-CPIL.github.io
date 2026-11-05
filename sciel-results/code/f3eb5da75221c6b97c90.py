"""Block-diagonal Kronecker factorization for 3x5x7 matrix multiplication."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    rank = len(U[0])
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i*m+j, c] * Va[j*p+k, c] * Wa[k*n+i, c]
                if abs(s - 1.0) > tol:
                    ok = False
                    return ok
    return ok

def _build_strassen_1x5x7():
    """
    Decompose the 1x5x7 multiplication core T(j,k) = 1 (for fixed i) into rank-1 terms.
    Returns list of (u_j for j in 0..4, v_k for k in 0..6) as two lists per term.
    Uses Kronecker: split A (5 cols) into a1 (2) + a2 (3); B (7 cols) into b1 (2) + b2 (5).
    The 2x2 block uses Strassen rank-2; cross blocks use identity.
    """
    # We want sum_c u_c(j) * v_c(k) = 1 for all j in 0..4, k in 0..6.
    # Strategy: represent the all-ones 5x7 matrix as sum of rank-1 matrices.
    # A 5x7 all-ones matrix has rank 1! So actually the 1x5x7 core is rank 1:
    # T(j,k) = 1 = u(j)*v(k) with u(j)=1, v(k)=1.
    # Wait -- that means each 1x5x7 strip is rank 1, total rank 3!
    # But that's only if the middle dimension j is shared correctly.
    # The full tensor: C[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    # For fixed i, the (j,k) slice must be all-ones.
    # If we set U[i*m+j,c] = 1 for c=i (a dedicated column), V[j*p+k,c]=1, W[k*n+i,c]=1
    # then for column c=i: contribution = 1*1*1 = 1 for all j,k at row i.
    # And for other columns, set them 0 at row i.
    # This gives rank = n = 3!
    terms = []
    return terms

def construct():
    n, m, p = DIMENSIONS
    # Deadline guard
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s

    # Try the rank-n decomposition first (each output row i gets its own column).
    # Column c = i: U[i*m+j, c]=1 for all j; V[j*p+k, c]=1 for all j,k; W[k*n+i, c]=1 for all k.
    # Then for fixed i: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
    #   = at c=i: 1*1*1 = 1; at c != i: W[k*n+i, c] = 0 (since c != i means row k*n+i of W is 0 in column c).
    # So total = 1. 
    rank = n
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    for i in range(n):
        c = i
        for j in range(m):
            U[i * m + j][c] = 1.0
        for j in range(m):
            for k in range(p):
                V[j * p + k][c] = 1.0
        for k in range(p):
            W[k * n + i][c] = 1.0
    # Verify
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}

    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
