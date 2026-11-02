import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 4]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify exact decomposition: sum_r W[k*n+i,r]*U[i*m+j,r]*V[j'*p+k,r] == delta[j,j'].
    U: (n*m, R), V: (m*p, R), W: (p*n, R)
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    R = U.shape[1]
    for i in range(n):
        for k in range(p):
            M = np.zeros((m, m))
            for r in range(R):
                w_val = W[k * n + i, r]
                if w_val == 0:
                    continue
                u_vec = U[i * m:(i + 1) * m, r]
                v_col = V[k::p, r]
                M += w_val * np.outer(u_vec, v_col)
            if not np.allclose(M, np.eye(m), atol=1e-10):
                return False
    return True

def _build_strassen_2x4x4(n, m, p):
    """
    Build a rank-28 decomposition of 2x4x4 matrix multiplication using
    four independent 2x2x2 Strassen products.
    """
    R = 28
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    def _strassen_block(base_col, i_block, j_block, k_block):
        a_rows = [0, 1]
        a_cols = [j_block * 2, j_block * 2 + 1]
        b_rows = [j_block * 2, j_block * 2 + 1]
        b_cols = [k_block * 2, k_block * 2 + 1]
        c_rows = [0, 1]
        c_cols = [k_block * 2, k_block * 2 + 1]
        
        # M1 = (A00+A11)*(B00+B11)
        U[0 * m + a_cols[0], base_col + 0] = 1
        U[1 * m + a_cols[1], base_col + 0] = 1
        V[b_rows[0] * p + b_cols[0], base_col + 0] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 0] = 1
        W[c_cols[0] * n + c_rows[0], base_col + 0] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 0] = 1
        
        # M2 = (A10+A11)*B00
        U[1 * m + a_cols[0], base_col + 1] = 1
        U[1 * m + a_cols[1], base_col + 1] = 1
        V[b_rows[0] * p + b_cols[0], base_col + 1] = 1
        W[c_cols[0] * n + c_rows[1], base_col + 1] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 1] = -1
        
        # M3 = A00*(B01-B11)
        U[0 * m + a_cols[0], base_col + 2] = 1
        V[b_rows[0] * p + b_cols[1], base_col + 2] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 2] = -1
        W[c_cols[1] * n + c_rows[0], base_col + 2] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 2] = 1
        
        # M4 = A11*(B10-B00)
        U[1 * m + a_cols[1], base_col + 3] = 1
        V[b_rows[1] * p + b_cols[0], base_col + 3] = 1
        V[b_rows[0] * p + b_cols[0], base_col + 3] = -1
        W[c_cols[0] * n + c_rows[0], base_col + 3] = 1
        W[c_cols[0] * n + c_rows[1], base_col + 3] = 1
        
        # M5 = (A00+A10)*B11
        U[0 * m + a_cols[0], base_col + 4] = 1
        U[1 * m + a_cols[0], base_col + 4] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 4] = 1
        W[c_cols[0] * n + c_rows[0], base_col + 4] = -1
        W[c_cols[1] * n + c_rows[0], base_col + 4] = 1
        
        # M6 = (A01-A00)*(B00+B10)
        U[0 * m + a_cols[1], base_col + 5] = 1
        U[0 * m + a_cols[0], base_col + 5] = -1
        V[b_rows[0] * p + b_cols[0], base_col + 5] = 1
        V[b_rows[1] * p + b_cols[0], base_col + 5] = 1
        W[c_cols[1] * n + c_rows[1], base_col + 5] = 1
        
        # M7 = (A10-A11)*(B01+B11)
        U[1 * m + a_cols[0], base_col + 6] = 1
        U[1 * m + a_cols[1], base_col + 6] = -1
        V[b_rows[0] * p + b_cols[1], base_col + 6] = 1
        V[b_rows[1] * p + b_cols[1], base_col + 6] = 1
        W[c_cols[0] * n + c_rows[0], base_col + 6] = 1

    _strassen_block(0, 0, 0, 0)
    _strassen_block(7, 0, 1, 0)
    _strassen_block(14, 0, 0, 1)
    _strassen_block(21, 0, 1, 1)
    
    return U, V, W

def _schoolbook_decomposition(n, m, p):
    """Build the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_limit=10):
    """
    Try to reduce the rank of the decomposition by setting some terms to zero
    and adjusting others. This is a heuristic search.
    """
    start_time = time.time()
    U = U.copy()
    V = V.copy()
    W = W.copy()
    R = U.shape[1]
    
    # We try to eliminate terms one by one.
    # This is computationally expensive and may not succeed.
    # For now, we just return the input if we don't have a specific algorithm.
    # A real implementation would use gradient descent or random sampling.
    
    # Given the complexity, we will not attempt a full search in this version
    # to ensure stability, but we structure the code to allow it.
    
    return U, V, W, R

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110  # Safety margin
    
    # Try Strassen-based rank-28 decomposition first
    U, V, W = _build_strassen_2x4x4(n, m, p)
    
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Schoolbook
    U, V, W = _schoolbook_decomposition(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
