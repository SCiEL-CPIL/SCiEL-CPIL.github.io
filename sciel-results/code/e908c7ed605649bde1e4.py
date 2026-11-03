"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_BUDGET = 125.0

def _time_left():
    return max(0.0, TIME_BUDGET - (time.time() - _start_time))

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_64():
    """Try to construct a rank-64 decomposition."""
    n, m, p = 3, 3, 8
    r = 64
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    
    # Use the fact that we can group the 8 columns
    # B has shape 3x8, split into two 3x4 blocks
    # A@B1 and A@B2 each need 3*3*4=36 mults in schoolbook
    # But we can potentially share: A is the same, so compute intermediate values
    # Let M1 = A @ e_j for j in 0..2 (3 vectors of dim 3)
    # Then C[:, 4l:t] = M1 @ B1[:, l] etc.
    # This doesn't directly help with rank reduction in the tensor sense.
    
    # Alternative: use a 2x2 Strassen-like block decomposition
    # Split A into 2x2 blocks (with one row/col padded), B into 2x2 blocks
    # This gets complicated for 3x3 x 3x8.
    
    # For now, return None to indicate failure
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    
    # Try lower rank decompositions
    candidates = [_try_rank_64()]
    for cand in candidates:
        if cand is not None and _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            r = cand["U"][0].__len__()
            if r < best["U"][0].__len__():
                best = cand
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
