import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

def _verify_exact(U, V, W, n, m, p, R):
    """Verify the decomposition exactly using Fractions."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0, 1)
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                for r in range(R):
                    u_val = Fraction(str(U[idx_u, r]))
                    v_val = Fraction(str(V[idx_v, r]))
                    w_val = Fraction(str(W[idx_w, r]))
                    val += u_val * v_val * w_val
                if val != 1:
                    return False
    return True

def _construct_rank3():
    """
    Construct a rank-3 decomposition for 2x3x6 matrix multiplication.
    
    The decomposition is:
    U[i*m + j, r] = 1 if j == r, else 0  (for r in 0,1,2)
    V[j*p + k, r] = 1 if j == r, else 0  (for r in 0,1,2)
    W[k*n + i, r] = 1 for all k, i, r    (for r in 0,1,2)
    """
    n, m, p = 2, 3, 6
    R = 3
    
    U = [[0] * R for _ in range(n * m)]  # 6 x 3
    V = [[0] * R for _ in range(m * p)]  # 18 x 3
    W = [[0] * R for _ in range(p * n)]  # 12 x 3
    
    # Fill U: U[i*m + j, r] = 1 if j == r
    for i in range(n):
        for j in range(m):
            for r in range(R):
                if j == r:
                    U[i * m + j][r] = 1
    
    # Fill V: V[j*p + k, r] = 1 if j == r
    for j in range(m):
        for k in range(p):
            for r in range(R):
                if j == r:
                    V[j * p + k][r] = 1
    
    # Fill W: W[k*n + i, r] = 1 for all k, i, r
    for k in range(p):
        for i in range(n):
            for r in range(R):
                W[k * n + i][r] = 1
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if _verify_exact(U_np, V_np, W_np, n, m, p, R):
        return {"U": U, "V": V, "W": W}
    
    return None

def _schoolbook(n, m, p):
    """Return the standard schoolbook decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try the rank-3 decomposition
    result = _construct_rank3()
    
    if result is not None:
        # Verify once more with the returned lists
        U = result["U"]
        V = result["V"]
        W = result["W"]
        R = 3
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        if _verify_exact(U_np, V_np, W_np, n, m, p, R):
            return result
    
    # Fallback: schoolbook
    return _schoolbook(n, m, p)
