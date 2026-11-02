import time
import numpy as np

DIMENSIONS = [2, 2, 10]
_DEADLINE = 120.0

_STRASSEN = [
    # M1 = (A00+A11)*(B00+B11)
    ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
    # M2 = (A10+A11)*B00
    ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
    # M3 = A00*(B01-B11)
    ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
    # M4 = A11*(B10-B00)
    ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
    # M5 = (A00+A01)*B11
    ([(0, 1), (1, 1)], [(3, 1)], [(1, 1), (0, -1)]),
    # M6 = (A10-A00)*(B00+B01)
    ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
    # M7 = (A01-A11)*(B10+B11)
    ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
]

def construct():
    t0 = time.time()
    n, m, p = DIMENSIONS  # 2, 2, 10
    rank = 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 5 blocks of 2 columns each
    for b in range(5):
        base_col = b * 7
        c0 = 2 * b
        c1 = 2 * b + 1
        
        for t in range(7):
            col = base_col + t
            u_side, v_side, w_side = _STRASSEN[t]
            
            for u_row, coeff in u_side:
                U[u_row][col] += coeff
            
            for v_koff, coeff in v_side:
                # v_koff 0,1 -> c0; 2,3 -> c1 ? 
                # In _STRASSEN, V indices are 0,1,2,3.
                # 0: B00, 1: B01, 2: B10, 3: B11
                # So 0,1 are row 0? No.
                # Standard Strassen V indices:
                # 0: B00, 1: B10, 2: B01, 3: B11?
                # Let's check the prompt's _emit_strassen logic.
                # v_koff // 2 gives j (row of B).
                # v_koff % 2 == 0 -> c0, else c1.
                # So v_koff 0 -> j=0, c0. (B00)
                # v_koff 1 -> j=0, c1. (B01)
                # v_koff 2 -> j=1, c0. (B10)
                # v_koff 3 -> j=1, c1. (B11)
                
                j = v_koff // 2
                k = c0 if v_koff % 2 == 0 else c1
                V[j * p + k][col] += coeff
                
            for w_koff, coeff in w_side:
                # w_koff // 2 gives i (row of C).
                # w_koff % 2 == 0 -> c0, else c1.
                # So w_koff 0 -> i=0, c0. (C00)
                # w_koff 1 -> i=0, c1. (C01)
                # w_koff 2 -> i=1, c0. (C10)
                # w_koff 3 -> i=1, c1. (C11)
                
                i = w_koff // 2
                k = c0 if w_koff % 2 == 0 else c1
                W[k * n + i][col] += coeff
                
    # Verification
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Build target tensor T
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] = 1.0
                
    recon = np.einsum('ir,jr,kr->ijk', U_np, V_np, W_np)
    
    if not np.allclose(recon, T, atol=1e-9):
        # Fallback: should not happen
        raise RuntimeError("Verification failed")
        
    return {"U": U, "V": V, "W": W}
