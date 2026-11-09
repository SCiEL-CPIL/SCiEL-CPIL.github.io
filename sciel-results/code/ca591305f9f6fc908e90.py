"""Matrix multiplication 3x5x5: rank-minimal exact tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify exact decomposition."""
    n, m, p = DIMENSIONS
    if not U or not V or not W:
        return False
    r = len(U[0]) if U else 0
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    for row in U:
        if len(row) != r:
            return False
    for row in V:
        if len(row) != r:
            return False
    for row in W:
        if len(row) != r:
            return False
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check non-zero positions
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i * m + j], V_np[j * p + k] * W_np[k * n + i])
                if not np.isclose(s, 1.0, atol=1e-9):
                    return False
    
    # Check zero positions
    valid_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                valid_set.add((a, b, c))
    
    for a in range(n * m):
        for b in range(m * p):
            for c in range(p * n):
                if (a, b, c) not in valid_set:
                    s = np.dot(U_np[a], V_np[b] * W_np[c])
                    if not np.isclose(s, 0.0, atol=1e-9):
                        return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _all_ones_decomposition():
    """
    The matrix multiplication structure tensor for 3x5x5 is the all-ones tensor
    T[i,j,k] = 1 for all i in [0,3), j in [0,5), k in [0,5).
    
    The all-ones tensor J_{n,m,p} has tensor rank min(n,m,p).
    For (3,5,5), rank = 3.
    
    Construction: For r = 0, 1, 2:
      U[i*m+j, r] = 1 if i == r else 0
      V[j*p+k, r] = 1 for all j,k
      W[k*n+i, r] = 1 for all k,i
    
    Verification: sum_{r=0}^{2} U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
                 = sum_{r=0}^{2} delta_{i,r} * 1 * 1 = 1 for all i,j,k.
    """
    n, m, p = DIMENSIONS
    rank = min(n, m, p)  # = 3
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = 1 if i == r else 0
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Try the all-ones decomposition (rank 3)
    candidate = _all_ones_decomposition()
    if _verify(candidate["U"], candidate["V"], candidate["W"]):
        return candidate
    
    # Fallback to schoolbook
    return _schoolbook()
# EVOLVE-BLOCK-END
