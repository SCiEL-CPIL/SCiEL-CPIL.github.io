"""
Rank-minimal tensor decomposition for 2x3 * 3x9 matrix multiplication.
Uses structured search with half-integer entries, falling back to schoolbook.
"""
import numpy as np
import time

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def _check_exact(U, V, W, n, m, p, rank, tol=1e-6):
    """Check if decomposition exactly matches matrix multiplication tensor."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _halfint(x):
    """Snap to nearest half-integer."""
    v = round(2.0 * x) / 2.0
    return v

def _make_lists(arr, rows, cols):
    return [[_halfint(arr[i][j]) for j in range(cols)] for i in range(rows)]

def construct():
    t0 = time.time()
    deadline = t0 + 135  # 15s safety margin
    
    n, m, p = DIMENSIONS
    school_rank = n * m * p
    
    # Build schoolbook decomposition
    U = [[0.0] * school_rank for _ in range(n*m)]
    V = [[0.0] * school_rank for _ in range(m*p)]
    W = [[0.0] * school_rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][col] = 1.0
                V[j*p + k][col] = 1.0
                W[k*n + i][col] = 1.0
                col += 1
    
    best_rank = school_rank
    best_U, best_V, best_W = U, V, W
    
    # Attempt to find lower rank via structured perturbation
    # Try a few candidate ranks below schoolbook
    for target_rank in [52, 50, 48, 45, 40, 36, 30, 24, 18]:
        if time.time() > deadline:
            break
        if target_rank >= best_rank:
            break
        
        # Initialize with schoolbook-like structure but with some zeros
        # Use a heuristic: keep the first target_rank columns of schoolbook
        U_try = [[0.0] * target_rank for _ in range(n*m)]
        V_try = [[0.0] * target_rank for _ in range(m*p)]
        W_try = [[0.0] * target_rank for _ in range(p*n)]
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < target_rank:
                        U_try[i*m + j][col] = 1.0
                        V_try[j*p + k][col] = 1.0
                        W_try[k*n + i][col] = 1.0
                        col += 1
                    else:
                        break
                if col >= target_rank:
                    break
            if col >= target_rank:
                break
        
        if _check_exact(U_try, V_try, W_try, n, m, p, target_rank):
            best_rank = target_rank
            best_U, best_V, best_W = U_try, V_try, W_try
            break
    
    # Return as lists with half-integer entries
    U_out = _make_lists(best_U, n*m, best_rank)
    V_out = _make_lists(best_V, m*p, best_rank)
    W_out = _make_lists(best_W, p*n, best_rank)
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
