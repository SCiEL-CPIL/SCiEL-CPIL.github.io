import time
from functools import lru_cache
import numpy as np

DIMENSIONS = [4, 5, 5]
_start_time = time.time()
TIME_LIMIT = 145  # safety margin before 150s kill

def _strassen_2x2x2():
    """Return (U, V, W) for 2x2 * 2x2 using Strassen rank 7."""
    U = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ], dtype=object)
    V = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ], dtype=object)
    W = np.array([
        [1, 0, 0, 1],
        [0, -1, 0, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 0, 0, 1],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ], dtype=object)
    return U, V, W

def _schoolbook(m, n, p):
    """Return schoolbook decomposition as numpy arrays."""
    rank = m * n * p
    U = np.zeros((m * n, rank), dtype=object)
    V = np.zeros((n * p, rank), dtype=object)
    W = np.zeros((p * m, rank), dtype=object)
    col = 0
    for i in range(m):
        for j in range(n):
            for k in range(p):
                U[i * n + j, col] = 1
                V[j * p + k, col] = 1
                W[k * m + i, col] = 1
                col += 1
    return U, V, W

def _place_block(U_sub, V_sub, W_sub, m_i, n_i, p_i, off_m, off_n, off_p, m, n, p):
    """Place a sub-decomposition into the full-size arrays."""
    U_full = np.zeros((m * n, len(U_sub)), dtype=object)
    V_full = np.zeros((n * p, len(U_sub)), dtype=object)
    W_full = np.zeros((p * m, len(U_sub)), dtype=object)
    
    for r in range(len(U_sub)):
        # U: (m_i * n_i) -> (m * n)
        for ii in range(m_i):
            for jj in range(n_i):
                U_full[(off_m + ii) * n + (off_n + jj), r] = U_sub[r, ii * n_i + jj]
        # V: (n_i * p_i) -> (n * p)
        for jj in range(n_i):
            for kk in range(p_i):
                V_full[(off_n + jj) * p + (off_p + kk), r] = V_sub[r, jj * p_i + kk]
        # W: (p_i * m_i) -> (p * m), W[k*m+i, r]
        for kk in range(p_i):
            for ii in range(m_i):
                W_full[(off_p + kk) * m + (off_m + ii), r] = W_sub[r, kk * m_i + ii]
    
    return U_full, V_full, W_full

@lru_cache(maxsize=None)
def _decompose(m, n, p):
    """
    Return (U, V, W) as numpy arrays for m×n * n×p matrix multiplication.
    U: shape (m*n, rank)
    V: shape (n*p, rank)
    W: shape (p*m, rank), W[k*m+i, r] convention
    """
    if time.time() - _start_time > TIME_LIMIT:
        raise TimeoutError()
    
    if m == 0 or n == 0 or p == 0:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 0))
    
    if m == 1 and n == 1 and p == 1:
        return np.array([[1]]), np.array([[1]]), np.array([[1]])
    
    # Strassen for 2×2 * 2×2
    if m == 2 and n == 2 and p == 2:
        return _strassen_2x2x2()
    
    # Try all possible splits and pick the best
    best = None
    best_rank = m * n * p  # schoolbook upper bound
    
    # Pre-compute sub-problem ranks for pruning
    # We'll try splits in order of expected benefit
    
    for m1 in range(1, m):
        m2 = m - m1
        for n1 in range(1, n):
            n2 = n - n1
            for p1 in range(1, p):
                p2 = p - p1
                
                if time.time() - _start_time > TIME_LIMIT:
                    raise TimeoutError()
                
                # Get sub-decompositions
                U1, V1, W1 = _decompose(m1, n1, p1)
                U2, V2, W2 = _decompose(m1, n1, p2)
                U3, V3, W3 = _decompose(m1, n2, p1)
                U4, V4, W4 = _decompose(m1, n2, p2)
                U5, V5, W5 = _decompose(m2, n1, p1)
                U6, V6, W6 = _decompose(m2, n1, p2)
                U7, V7, W7 = _decompose(m2, n2, p1)
                U8, V8, W8 = _decompose(m2, n2, p2)
                
                total_rank = (U1.shape[1] + U2.shape[1] + U3.shape[1] + U4.shape[1] +
                             U5.shape[1] + U6.shape[1] + U7.shape[1] + U8.shape[1])
                
                if total_rank >= best_rank:
                    continue
                
                best_rank = total_rank
                
                # Build combined U, V, W using numpy
                U_new = np.zeros((m * n, total_rank), dtype=object)
                V_new = np.zeros((n * p, total_rank), dtype=object)
                W_new = np.zeros((p * m, total_rank), dtype=object)
                
                col_offset = 0
                
                def place(U_s, V_s, W_s, m_i, n_i, p_i, off_m, off_n, off_p):
                    nonlocal col_offset
                    r_s = U_s.shape[1]
                    for r in range(r_s):
                        for ii in range(m_i):
                            for jj in range(n_i):
                                U_new[(off_m + ii) * n + (off_n + jj), col_offset + r] = U_s[r, ii * n_i + jj]
                        for jj in range(n_i):
                            for kk in range(p_i):
                                V_new[(off_n + jj) * p + (off_p + kk), col_offset + r] = V_s[r, jj * p_i + kk]
                        for kk in range(p_i):
                            for ii in range(m_i):
                                W_new[(off_p + kk) * m + (off_m + ii), col_offset + r] = W_s[r, kk * m_i + ii]
                    col_offset += r_s
                
                place(U1, V1, W1, m1, n1, p1, 0, 0, 0)
                place(U3, V3, W3, m1, n2, p1, 0, n1, 0)
                place(U2, V2, W2, m1, n1, p2, 0, 0, p1)
                place(U4, V4, W4, m1, n2, p2, 0, n1, p1)
                place(U5, V5, W5, m2, n1, p1, m1, 0, 0)
                place(U6, V6, W6, m2, n1, p2, m1, 0, p1)
                place(U7, V7, W7, m2, n2, p1, m1, n1, 0)
                place(U8, V8, W8, m2, n2, p2, m1, n1, p1)
                
                best = (U_new, V_new, W_new)
    
    if best is not None:
        return best
    else:
        return _schoolbook(m, n, p)

def construct():
    global _start_time
    _start_time = time.time()
    _decompose.cache_clear()
    n, m, p = DIMENSIONS  # n=4, m=5, p=5
    U, V, W = _decompose(n, m, p)
    
    # Convert to list of lists for the required interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
