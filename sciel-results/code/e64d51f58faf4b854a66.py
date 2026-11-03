import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 4]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification using einsum."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    U_3d = U_arr.reshape(n, m, rank)
    V_3d = V_arr.reshape(m, p, rank)
    W_3d = W_arr.reshape(p, n, rank)
    
    T = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
    
    max_err = np.max(np.abs(T - 1.0))
    return max_err < tol, max_err

def _schoolbook():
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

def _strassen_44():
    n, m, p = DIMENSIONS
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen steps definition
    # Each step is a tuple of (U_coeffs, V_coeffs, W_coeffs)
    # Coeffs are dicts mapping local (i, j) to value
    # Local indices are 0 or 1
    
    strassen_steps = [
        # M1 = (X00 + X11)(Y00 + Y11) -> Z00 += 1, Z11 += 1
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (X10 + X11)Y00 -> Z10 += 1, Z11 -= 1
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = X00(Y01 - Y11) -> Z01 += 1, Z11 += 1
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = X11(Y10 - Y00) -> Z00 += 1, Z10 += 1
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (X00 + X01)Y11 -> Z01 += 1, Z00 -= 1
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        # M6 = (X10 - X00)(Y00 + Y01) -> Z11 += 1
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (X01 - X11)(Y10 + Y11) -> Z00 += 1
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    # Define blocks
    # (X_offset_i, X_offset_j, Y_offset_j, Y_offset_k, Z_offset_i, Z_offset_k)
    blocks = [
        # M1 N1 -> C[0:2, 0:2]
        (0, 0, 0, 0, 0, 0),
        # M2 N3 -> C[0:2, 0:2]
        (0, 2, 2, 0, 0, 0),
        # M1 N2 -> C[0:2, 2:4]
        (0, 0, 0, 2, 0, 2),
        # M2 N4 -> C[0:2, 2:4]
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
            # Fill U
            for (li, lj), val in u_coeffs.items():
                gi = li + xi
                gj = lj + xj
                U[gi * m + gj][col] += val
            
            # Fill V
            for (lj, lk), val in v_coeffs.items():
                gj = lj + yj
                gk = lk + yk
                V[gj * p + gk][col] += val
            
            # Fill W
            for (li, lk), val in w_coeffs.items():
                gi = li + zi
                gk = lk + zk
                W[gk * n + gi][col] += val
            
            col += 1
    
    # Naive part for bottom row (i=2)
    # A[2, j] * B[j, k] -> C[2, k]
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + 2][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the rank 44 decomposition
    result_44 = _strassen_44()
    ok, err = _verify_fast(result_44["U"], result_44["V"], result_44["W"], n, m, p)
    
    if ok:
        return result_44
    
    # Fallback to schoolbook
    return _schoolbook()
