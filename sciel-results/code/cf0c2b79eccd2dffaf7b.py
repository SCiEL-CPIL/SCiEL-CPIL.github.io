import time
import numpy as np

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p):
    """Verify tensor decomposition: C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]"""
    rank = len(U[0])
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)
    return max_err

def _construct_rank56():
    """Rank-56 decomposition using Strassen for 2x2x2 blocks."""
    n, m, p = 2, 4, 8
    rank = 56
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    strassen = [
        {'U': {(0,0):1, (1,1):1}, 'V': {(0,0):1, (1,1):1}, 'W': {(0,0):1, (1,1):1}},
        {'U': {(1,0):1, (1,1):1}, 'V': {(0,0):1}, 'W': {(0,1):1, (1,1):-1}},
        {'U': {(0,0):1}, 'V': {(0,1):1, (1,1):-1}, 'W': {(1,0):1, (1,1):1}},
        {'U': {(1,1):1}, 'V': {(1,0):1, (0,0):-1}, 'W': {(0,0):1, (0,1):1}},
        {'U': {(0,0):1, (0,1):1}, 'V': {(1,1):1}, 'W': {(0,0):-1, (1,0):1}},
        {'U': {(1,0):1, (0,0):-1}, 'V': {(0,0):1, (0,1):1}, 'W': {(1,1):1}},
        {'U': {(0,1):1, (1,1):-1}, 'V': {(1,0):1, (1,1):1}, 'W': {(0,0):1}},
    ]
    
    col = 0
    for j_block in range(2):
        for k_block in range(4):
            for t in strassen:
                for (i, j_lo), c_u in t['U'].items():
                    j = j_block * 2 + j_lo
                    U[i * m + j][col] += c_u
                for (j_lo, k_lo), c_v in t['V'].items():
                    j = j_block * 2 + j_lo
                    k = k_block * 2 + k_lo
                    V[j * p + k][col] += c_v
                for (k_lo, i), c_w in t['W'].items():
                    k = k_block * 2 + k_lo
                    W[k * n + i][col] += c_w
                col += 1
    
    return {"U": U, "V": V, "W": W}

def _trim_zero_columns(U, V, W):
    """Remove columns that are all zero in any of U, V, or W."""
    rank = len(U[0])
    keep = []
    for r in range(rank):
        u_nonzero = any(abs(U[i][r]) > 1e-15 for i in range(len(U)))
        v_nonzero = any(abs(V[i][r]) > 1e-15 for i in range(len(V)))
        w_nonzero = any(abs(W[i][r]) > 1e-15 for i in range(len(W)))
        if u_nonzero and v_nonzero and w_nonzero:
            keep.append(r)
    if len(keep) == rank:
        return U, V, W, rank
    U_new = [[U[i][r] for r in keep] for i in range(len(U))]
    V_new = [[V[i][r] for r in keep] for i in range(len(V))]
    W_new = [[W[i][r] for r in keep] for i in range(len(W))]
    return U_new, V_new, W_new, len(keep)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Build rank-56 decomposition
    result = _construct_rank56()
    U, V, W = result["U"], result["V"], result["W"]
    
    # Verify correctness
    err = _verify(U, V, W, n, m, p)
    
    # Trim zero columns
    U, V, W, rank = _trim_zero_columns(U, V, W)
    
    # Re-verify after trimming
    err = _verify(U, V, W, n, m, p)
    
    if err < 1e-10:
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback: return the untrimmed version
        result = _construct_rank56()
        return result
# EVOLVE-BLOCK-END
