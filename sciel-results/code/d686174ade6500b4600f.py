"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition exactly."""
    def scale2(mat):
        out = np.array(mat, dtype=np.float64) * 2
        if np.max(np.abs(out - np.round(out))) > 1e-9:
            return None
        return np.round(out).astype(np.int64)
    U2 = scale2(U)
    V2 = scale2(V)
    W2 = scale2(W)
    if U2 is None or V2 is None or W2 is None:
        return False
    target = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for a in range(n):
        for b in range(m):
            for c in range(p):
                target[a*m + b, b*p + c, c*n + a] = 1
    T_computed = np.einsum('ic,jc,kc->ijk', U2, V2, W2, optimize=True)
    if np.max(np.abs(T_computed - target * 8)) > 1e-6:
        return False
    return True

def _schoolbook(n, m, p):
    """Standard O(n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for a in range(n):
        for b in range(m):
            for c in range(p):
                U[a*m + b][col] = 1
                V[b*p + c][col] = 1
                W[c*n + a][col] = 1
                col += 1
    return U, V, W

def _construct_hybrid_5x5(n, m, p):
    """
    Hybrid decomposition for 5x5x5:
    - Strassen (rank 7) for top-left 2x2
    - Schoolbook for remaining blocks
    Total: 7 + 12 + 12 + 27 = 58
    """
    if (n, m, p) != (5, 5, 5):
        return None
    
    max_rank = 60
    U = np.zeros((n*m, max_rank), dtype=np.int64)
    V = np.zeros((m*p, max_rank), dtype=np.int64)
    W = np.zeros((p*n, max_rank), dtype=np.int64)
    current_rank = 0
    
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal current_rank
        if current_rank >= max_rank:
            return
        U[u_idx, current_rank] += coeff
        V[v_idx, current_rank] += coeff
        W[w_idx, current_rank] += coeff
        current_rank += 1
    
    # Strassen terms for 2x2 block (indices 0,1,5,6 in each matrix)
    strassen_terms = [
        ([0, 6], [1, 1], [0, 6], [1, 1], [0, 6], [1, 1]),
        ([5, 6], [1, 1], [0], [1], [5, 6], [1, -1]),
        ([0], [1], [1, 6], [1, -1], [1, 6], [1, 1]),
        ([6], [1], [5, 0], [1, -1], [0, 5], [1, 1]),
        ([0, 1], [1, 1], [6], [1], [0, 1], [-1, 1]),
        ([5, 0], [1, -1], [0, 1], [1, 1], [6], [1]),
        ([1, 6], [1, -1], [5, 6], [1, 1], [0], [1]),
    ]
    
    for term in strassen_terms:
        a_idxs, a_coefs, b_idxs, b_coefs, c_idxs, c_coefs = term
        col = current_rank
        for idx, coef in zip(a_idxs, a_coefs):
            U[idx, col] = coef
        for idx, coef in zip(b_idxs, b_coefs):
            V[idx, col] = coef
        for idx, coef in zip(c_idxs, c_coefs):
            W[idx, col] = coef
        current_rank += 1
    
    # Top-Right 2x3 block
    for i in range(2):
        for k in range(2, 5):
            for j in range(2):
                a_idx = i * 5 + j
                b_idx = j * 5 + k
                c_idx = i * 5 + k
                add_term(a_idx, b_idx, c_idx, 1)
    
    # Bottom-Left 3x2 block
    for i in range(2, 5):
        for k in range(2):
            for j in range(2):
                a_idx = i * 5 + j
                b_idx = j * 5 + k
                c_idx = i * 5 + k
                add_term(a_idx, b_idx, c_idx, 1)
    
    # Bottom-Right 3x3 block
    for i in range(2, 5):
        for k in range(2, 5):
            for j in range(2, 5):
                a_idx = i * 5 + j
                b_idx = j * 5 + k
                c_idx = i * 5 + k
                add_term(a_idx, b_idx, c_idx, 1)
    
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    deadline = time.time() + 135.0
    n, m, p = DIMENSIONS
    
    # Try hybrid decomposition
    result = _construct_hybrid_5x5(n, m, p)
    if result:
        U, V, W = result
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
