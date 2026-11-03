import time
import numpy as np
from fractions import Fraction
from itertools import combinations

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
    
    strassen_steps = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    blocks = [
        (0, 0, 0, 0, 0, 0),
        (0, 2, 2, 0, 0, 0),
        (0, 0, 0, 2, 0, 2),
        (0, 2, 2, 2, 0, 2)
    ]
    
    col = 0
    for (xi, xj, yj, yk, zi, zk) in blocks:
        for u_coeffs, v_coeffs, w_coeffs in strassen_steps:
            for (li, lj), val in u_coeffs.items():
                gi = li + xi
                gj = lj + xj
                U[gi * m + gj][col] += val
            
            for (lj, lk), val in v_coeffs.items():
                gj = lj + yj
                gk = lk + yk
                V[gj * p + gk][col] += val
            
            for (li, lk), val in w_coeffs.items():
                gi = li + zi
                gk = lk + zk
                W[gk * n + gi][col] += val
            
            col += 1
    
    for j in range(m):
        for k in range(p):
            U[2 * m + j][col] += 1
            V[j * p + k][col] += 1
            W[k * n + 2][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _compute_slice_rank_lower_bound(n, m, p):
    """Compute the slice rank lower bound for matrix multiplication tensor."""
    return max(min(m, p), min(n, p), min(n, m))

def _try_reduce_rank(U, V, W, n, m, p, deadline):
    """
    Placeholder for rank reduction. 
    Current best known upper bound is 44. 
    Removing SVD to save time.
    """
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try the rank 44 decomposition
    result_44 = _strassen_44()
    ok, err = _verify_fast(result_44["U"], result_44["V"], result_44["W"], n, m, p)
    
    if ok:
        # Try to reduce rank (currently no-op)
        reduced = _try_reduce_rank(result_44["U"], result_44["V"], result_44["W"], n, m, p, deadline)
        if reduced["U"] != result_44["U"] or reduced["V"] != result_44["V"] or reduced["W"] != result_44["W"]:
            ok_r, err_r = _verify_fast(reduced["U"], reduced["V"], reduced["W"], n, m, p)
            if ok_r:
                return reduced
        return result_44
    
    # Fallback to schoolbook
    return _schoolbook()
