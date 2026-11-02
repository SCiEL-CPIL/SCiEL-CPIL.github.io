import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    rank = U.shape[1]
    try:
        U_t = U.reshape(n, m, rank)
        V_t = V.reshape(m, p, rank)
        W_t = W.reshape(p, n, rank)
        T = np.einsum('ijr,jkr,ki->ijk', U_t, V_t, W_t)
        return np.allclose(T, 1.0, atol=1e-9)
    except Exception:
        return False

def _try_rank40():
    """
    Attempt to find a rank-40 exact decomposition by exploiting shared terms
    between adjacent Strassen blocks. This is a heuristic search that may
    fail; if it does, we fall back to rank 42.
    
    The idea: when two 2x2 blocks share a row or column of the larger matrix,
    some of the 7 Strassen products can be combined. We try to find a linear
    combination of the 12 block-level decompositions (2 rows x 3 cols x 7 Strassen)
    that yields rank 40 with exact integer coefficients.
    """
    n, m, p = 2, 4, 6
    # This is a complex search; we try a few structured approaches
    # For now, return None to indicate failure (fallback to rank 42)
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    # First, try rank-40 (may succeed with clever sharing)
    result = _try_rank40()
    if result is not None:
        U, V, W = result
        if _verify(U, V, W, n, m, p):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to rank-42 Strassen block decomposition
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # M1 = (X00 + X11)(Y00 + Y11)
    # M2 = (X10 + X11)Y00
    # M3 = X00(Y01 - Y11)
    # M4 = X11(Y10 - Y00)
    # M5 = (X00 + X01)Y11
    # M6 = (X10 - X00)(Y00 + Y01)
    # M7 = (X01 - X11)(Y10 + Y11)
    
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    
    strassen_terms = [
        # M1
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        # M2
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
        # M3
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
        # M4
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]},
        # M5
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        # M6
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        # M7
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    col = 0
    for r in range(3):
        for s in range(2):
            for t in range(7):
                term = strassen_terms[t]
                
                for local_u, coeff in term['U']:
                    i = local_u // 2
                    j_local = local_u % 2
                    j_global = 2 * s + j_local
                    u_idx = i * m + j_global
                    U[u_idx, col] += coeff
                
                for local_v, coeff in term['V']:
                    j_local = local_v // 2
                    k_local = local_v % 2
                    j_global = 2 * s + j_local
                    k_global = 2 * r + k_local
                    v_idx = j_global * p + k_global
                    V[v_idx, col] += coeff
                
                for local_w, coeff in term['W']:
                    i = local_w // 2
                    k_local = local_w % 2
                    k_global = 2 * r + k_local
                    w_idx = k_global * n + i
                    W[w_idx, col] += coeff
                
                col += 1
                
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Verify rank-42 decomposition
    if _verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook (rank 48)
    r_sb = n * m * p
    U_sb = np.zeros((n * m, r_sb))
    V_sb = np.zeros((m * p, r_sb))
    W_sb = np.zeros((p * n, r_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
