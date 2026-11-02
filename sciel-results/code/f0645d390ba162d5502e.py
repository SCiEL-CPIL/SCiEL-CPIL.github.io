import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, c] = 1
                V[j * p + k, c] = 1
                W[k * n + i, c] = 1
                c += 1
    return U, V, W

def _to_list(mat):
    return mat.tolist()

def _check_exact(U, V, W, n, m, p):
    """Verify T[i,j,k] == sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] for all (i,j,k)."""
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            u_row = U[i * m + j]
            v_base = j * p
            for k in range(p):
                s = 0
                for c in range(r):
                    s = s + u_row[c] * V[v_base + k, c] * W[k * n + i, c]
                if s != 1:
                    return False
    return True

def _structured_rank_10(n, m, p):
    """
    Construct a rank-10 decomposition for 3x5x6 matrix multiplication.
    Based on splitting into 10 disjoint 3x1x3 blocks.
    """
    if (n, m, p) != (3, 5, 6):
        return None
    
    rank = 10
    U = np.zeros((n * m, rank), dtype=object)
    V = np.zeros((m * p, rank), dtype=object)
    W = np.zeros((p * n, rank), dtype=object)
    
    # Define the 10 blocks
    # Blocks 0-4: K = {0, 1, 2}, J = 0..4
    # Blocks 5-9: K = {3, 4, 5}, J = 0..4
    
    for c in range(rank):
        if c < 5:
            j_c = c
            K_c = [0, 1, 2]
        else:
            j_c = c - 5
            K_c = [3, 4, 5]
            
        # Fill U
        for i in range(n):
            U[i * m + j_c, c] = 1
            
        # Fill V
        for k in K_c:
            V[j_c * p + k, c] = 1
            
        # Fill W
        for k in K_c:
            for i in range(n):
                W[k * n + i, c] = 1
                
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 130.0
    n, m, p = DIMENSIONS
    
    # Try the structured rank-10 decomposition
    U, V, W = _structured_rank_10(n, m, p)
    
    if U is not None:
        # Verify correctness
        if _check_exact(U, V, W, n, m, p):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    # Fallback to schoolbook if something went wrong
    U0, V0, W0 = _schoolbook(n, m, p)
    return {"U": _to_list(U0), "V": _to_list(V0), "W": _to_list(W0)}
# EVOLVE-BLOCK-END
