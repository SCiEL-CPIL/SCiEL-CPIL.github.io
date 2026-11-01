"""
Matrix multiplication 2x5 x 5x5: rank-minimal exact tensor decomposition.

Strategy:
1. Start with schoolbook rank 50.
2. Use structured parametric families and exact rational arithmetic.
3. Try various rank values with structured initializations.
4. Validate exactly using Fraction arithmetic.
"""

import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def validate_exact(U, V, W, n, m, p):
    """Validate using exact Fraction arithmetic."""
    r = len(W[0]) if isinstance(W, list) and len(W) > 0 else W.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = Fraction(0, 1)
                for c in range(r):
                    u_val = Fraction(U[idx_u][c] if isinstance(U[idx_u][c], (int, float)) else U[idx_u][c])
                    v_val = Fraction(V[idx_v][c] if isinstance(V[idx_v][c], (int, float)) else V[idx_v][c])
                    w_val = Fraction(W[idx_w][c] if isinstance(W[idx_w][c], (int, float)) else W[idx_w][c])
                    val += u_val * v_val * w_val
                if val != Fraction(1, 1):
                    return False
    return True

def validate_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast validation using floating point."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    r = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                val = np.dot(U[idx_u, :], V[idx_v, :] * W[idx_w, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def schoolbook_decomposition(n, m, p):
    """Return the schoolbook rank n*m*p decomposition."""
    r = n * m * p
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _clean_val(x):
    """Clean a float value to integer or half-integer if possible."""
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    elif abs(2 * x - round(2 * x)) < 1e-9:
        return round(2 * x) / 2.0
    return x

def try_rank_structured(r, n, m, p, timeout):
    """Try to find rank-r decomposition using structured patterns."""
    start_time = time.time()
    
    # Pattern 1: Block-structured (like schoolbook but merged)
    if time.time() - start_time > timeout:
        return None
    
    # Try various structured initializations
    patterns = []
    
    # Pattern: Use Strassen-like combinations
    for seed in range(10):
        if time.time() - start_time > timeout:
            break
        
        rng = random.Random(seed * 7919 + 13)
        
        U = [[0] * r for _ in range(n * m)]
        V = [[0] * r for _ in range(m * p)]
        W = [[0] * r for _ in range(p * n)]
        
        # For each column, create a structured pattern
        for c in range(r):
            # Random signs for Strassen-like
            u_signs = [rng.choice([-1, 0, 1]) for _ in range(n * m)]
            v_signs = [rng.choice([-1, 0, 1]) for _ in range(m * p)]
            w_signs = [rng.choice([-1, 0, 1]) for _ in range(p * n)]
            
            for i in range(n * m):
                U[i][c] = u_signs[i]
            for j in range(m * p):
                V[j][c] = v_signs[j]
            for k in range(p * n):
                W[k][c] = w_signs[k]
        
        # Validate
        if validate_fast(U, V, W, n, m, p):
            return [U, V, W]
    
    return None

def try_rank_randomized(r, n, m, p, timeout, max_restarts=20):
    """Try to find rank-r decomposition using randomized search."""
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() - start_time > timeout:
            break
        
        # Random initialization with small values
        U = np.zeros((n * m, r), dtype=np.float64)
        V = np.zeros((m * p, r), dtype=np.float64)
        W = np.zeros((p * n, r), dtype=np.float64)
        
        # Set random non-zero entries
        for c in range(r):
            for _ in range(max(1, n * m // 4)):
                row = random.randint(0, n * m - 1)
                U[row, c] = random.choice([-1, 1])
            for _ in range(max(1, m * p // 4)):
                row = random.randint(0, m * p - 1)
                V[row, c] = random.choice([-1, 1])
            for _ in range(max(1, p * n // 4)):
                row = random.randint(0, p * n - 1)
                W[row, c] = random.choice([-1, 1])
        
        # Quick validation
        if validate_fast(U, V, W, n, m, p):
            U_list = [[_clean_val(U[i, c]) for c in range(r)] for i in range(n * m)]
            V_list = [[_clean_val(V[i, c]) for c in range(r)] for i in range(m * p)]
            W_list = [[_clean_val(W[i, c]) for c in range(r)] for i in range(p * n)]
            return U_list, V_list, W_list
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 145  # Safety margin before 150s kill
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = schoolbook_decomposition(n, m, p)
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for r in range(best_rank - 1, 24, -1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 10:
            break
        
        remaining = deadline - elapsed - 3
        if remaining <= 0:
            break
        
        # Try structured approach first
        timeout_structured = min(2.0, remaining * 0.4)
        if timeout_structured < 0.3:
            break
        
        result = try_rank_structured(r, n, m, p, timeout_structured)
        
        if result is None and time.time() - start_time < deadline - 5:
            # Try randomized approach
            timeout_randomized = min(2.0, remaining * 0.3)
            if timeout_randomized > 0.3:
                result = try_rank_randomized(r, n, m, p, timeout_randomized, max_restarts=15)
        
        if result is not None:
            U_r, V_r, W_r = result
            # Final exact validation
            if validate_exact(U_r, V_r, W_r, n, m, p):
                best_result = {"U": U_r, "V": V_r, "W": W_r}
                best_rank = r
    
    # Final validation
    elapsed = time.time() - start_time
    if elapsed < deadline - 2:
        U_arr = np.array(best_result["U"])
        V_arr = np.array(best_result["V"])
        W_arr = np.array(best_result["W"])
        if not validate_fast(U_arr, V_arr, W_arr, n, m, p):
            best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    return best_result
