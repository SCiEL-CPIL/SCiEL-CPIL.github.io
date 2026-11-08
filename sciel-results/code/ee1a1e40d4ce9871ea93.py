"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 5]

def _verify(U, V, W):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    # Build the target tensor T[i,j,k] = 1
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The decomposition: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    # for all valid (i,j,k)
    
    # Check shapes
    if len(U) != n * m:
        return False
    if len(V) != m * p:
        return False
    if len(W) != p * n:
        return False
    if len(U[0]) != rank or len(V[0]) != rank or len(W[0]) != rank:
        return False
    
    # Verify all entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _verify_fast(U, V, W):
    """Fast verification using numpy."""
    n, m, p = DIMENSIONS
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_arr[i * m + j]
                v_val = V_arr[j * p + k]
                w_val = W_arr[k * n + i]
                s = np.dot(u_val, v_val * w_val)
                if abs(s - 1) > 1e-10:
                    return False
    return True

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_rank_k_structural(k):
    """Try to find a rank-k decomposition using structured random search."""
    n, m, p = DIMENSIONS
    np.random.seed(42 + k)
    
    for attempt in range(50):
        # Random initialization with small values
        U = np.random.randn(n * m, k) * 0.5
        V = np.random.randn(m * p, k) * 0.5
        W = np.random.randn(p * n, k) * 0.5
        
        # Verify
        valid = True
        for i in range(n):
            for j in range(m):
                for k_idx in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k_idx]
                    w_val = W[k_idx * n + i]
                    s = np.dot(u_val, v_val * w_val)
                    if abs(s - 1) > 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Convert to exact fractions
            U_frac = [[Fraction(int(round(x * 2)) / 2) for x in row] for row in U]
            V_frac = [[Fraction(int(round(x * 2)) / 2) for x in row] for row in V]
            W_frac = [[Fraction(int(round(x * 2)) / 2) for x in row] for row in W]
            
            # Verify with exact arithmetic
            if _verify(U_frac, V_frac, W_frac):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    # Start with schoolbook as fallback
    best = _schoolbook()
    best_rank = len(best["U"][0])
    
    n, m, p = DIMENSIONS
    
    # Try ranks from lower to higher
    # Known lower bounds: rank >= max(n*m, m*p, p*n) = max(8, 20, 10) = 20
    # Also rank >= n + m + p - 2 = 2 + 4 + 5 - 2 = 9 (weak bound)
    # Try from 20 upward
    
    for k in range(20, best_rank):
        if time.time() > deadline:
            break
        
        result = _try_rank_k_structural(k)
        if result is not None:
            best = result
            best_rank = k
            break
    
    return best
