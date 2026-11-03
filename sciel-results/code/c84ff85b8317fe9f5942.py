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
    """
    Try to construct a rank-64 decomposition using a structured approach.
    
    Key insight: For 3x3x8 matrix multiplication, we can exploit the linearity
    in the 8-dimension. Consider that B is 3x8, so we can write B = [B1, B2]
    where B1 and B2 are 3x4. Then C = A@B = [A@B1, A@B2].
    
    However, this doesn't directly reduce rank. Instead, let's try to find
    a decomposition where some rank-1 terms contribute to multiple (i,j,k) triples.
    
    Approach: Start with schoolbook (rank 72) and try to merge 8 pairs of terms
    into single terms by adjusting U, V, W.
    """
    n, m, p = 3, 3, 8
    r = 64
    
    # Initialize with schoolbook pattern for first 64 terms
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if c >= r:
                    break
                U[i*m+j, c] = 1.0
                V[j*p+k, c] = 1.0
                W[k*n+i, c] = 1.0
                c += 1
                if c >= r:
                    break
            if c >= r:
                break
        if c >= r:
            break
    
    # The remaining 8 (i,j,k) triples are not covered. We need to modify the
    # decomposition to cover all 72 triples using only 64 terms.
    
    # This is a non-trivial optimization problem. For now, return None.
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
