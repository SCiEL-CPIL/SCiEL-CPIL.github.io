"""Combined program for matrix multiplication 2x3x8 tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START

def _check_exact(U, V, W, n, m, p):
    """Check if decomposition is exact using Fraction arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                if s != 1:
                    return False
    return True

def _check_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast float-based check."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W_arr if False else V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i*m+j, c] * V_arr[j*p+k, c] * W_arr[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _try_rank_r(n, m, p, r, deadline, rng, max_trials=200):
    """
    Try to find a rank-r decomposition.
    Uses structured random initialization with patterns that share indices.
    """
    if r < max(n*m, m*p, p*n):
        return None
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        
        U = [[0] * r for _ in range(U_rows)]
        V = [[0] * r for _ in range(V_rows)]
        W = [[0] * r for _ in range(W_rows)]
        
        # Each column is a "pattern" that can cover multiple (i,j,k) entries
        # Pattern types:
        # 0: single (i,j,k) - schoolbook
        # 1: two entries sharing j: (i1,j,k1), (i2,j,k2)
        # 2: two entries sharing k: (i1,j1,k), (i2,j2,k)
        # 3: two entries sharing i: (i,j1,k1), (i,j2,k2)
        # 4: three entries in a line
        
        for c in range(r):
            if time.time() > deadline:
                return None
            
            mode = rng.random()
            
            if mode < 0.3:
                # Single term
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
            elif mode < 0.55:
                # Share j: (i1,j,k1) and (i2,j,k2)
                j = rng.randint(0, m-1)
                i1 = rng.randint(0, n-1)
                i2 = rng.randint(0, n-1)
                k1 = rng.randint(0, p-1)
                k2 = rng.randint(0, p-1)
                if i1 == i2 and k1 == k2:
                    continue
                U[i1*m+j][c] = 1
                V[j*p+k1][c] = 1
                W[k1*n+i1][c] = 1
                U[i2*m+j][c] = 1
                V[j*p+k2][c] = 1
                W[k2*n+i2][c] = 1
            elif mode < 0.75:
                # Share k: (i1,j1,k) and (i2,j2,k)
                k = rng.randint(0, p-1)
                i1 = rng.randint(0, n-1)
                i2 = rng.randint(0, n-1)
                j1 = rng.randint(0, m-1)
                j2 = rng.randint(0, m-1)
                if i1 == i2 and j1 == j2:
                    continue
                U[i1*m+j1][c] = 1
                V[j1*p+k][c] = 1
                W[k*n+i1][c] = 1
                U[i2*m+j2][c] = 1
                V[j2*p+k][c] = 1
                W[k*n+i2][c] = 1
            elif mode < 0.9:
                # Share i: (i,j1,k1) and (i,j2,k2)
                i = rng.randint(0, n-1)
                j1 = rng.randint(0, m-1)
                j2 = rng.randint(0, m-1)
                k1 = rng.randint(0, p-1)
                k2 = rng.randint(0, p-1)
                if j1 == j2 and k1 == k2:
                    continue
                U[i*m+j1][c] = 1
                V[j1*p+k1][c] = 1
                W[k1*n+i][c] = 1
                U[i*m+j2][c] = 1
                V[j2*p+k2][c] = 1
                W[k2*n+i][c] = 1
            else:
                # Three entries: (i,j1,k), (i,j2,k), (i,j3,k) sharing i and k
                i = rng.randint(0, n-1)
                k = rng.randint(0, p-1)
                js = rng.sample(range(m), min(3, m))
                for j in js:
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
        
        # Quick float check
        if _check_fast(U, V, W, n, m, p):
            if _check_exact(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
    
    return None

def _try_half_integer_rank(n, m, p, r, deadline, rng, max_trials=100):
    """Try with half-integer entries (scaled by 2)."""
    if r < max(n*m, m*p, p*n):
        return None
    
    U_rows = n * m
    V_rows = m * p
    W_rows = p * n
    
    for trial in range(max_trials):
        if time.time() > deadline:
            return None
        
        # Use integer representation (multiply by 2)
        U_int = np.zeros((U_rows, r), dtype=int)
        V_int = np.zeros((V_rows, r), dtype=int)
        W_int = np.zeros((W_rows, r), dtype=int)
        
        for c in range(r):
            if time.time() > deadline:
                return None
            
            # Random sparse pattern
            n_nonzero = rng.randint(1, 3)
            for _ in range(n_nonzero):
                i = rng.randint(0, n-1)
                j = rng.randint(0, m-1)
                k = rng.randint(0, p-1)
                val = rng.choice([-2, -1, 1, 2])
                U_int[i*m+j, c] = val
                V_int[j*p+k, c] = val
                W_int[k*n+i, c] = 2  # = 1.0 in half-integers
        
        U_half = (U_int / 2).tolist()
        V_half = (V_int / 2).tolist()
        W_half = (W_int / 2).tolist()
        
        if _check_fast(U_half, V_half, W_half, n, m, p):
            if _check_exact(U_half, V_half, W_half, n, m, p):
                return {"U": U_half, "V": V_half, "W": W_half}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 48
    
    rng = random.Random(12345)
    
    lower_bound = max(n*m, m*p, p*n)  # 24
    
    # Try candidate ranks, prioritizing known-plausible values
    # For M_{2,3,8}, literature suggests rank around 40-44
    candidate_ranks = [40, 41, 42, 43, 44, 45, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24]
    candidate_ranks = [r for r in candidate_ranks if lower_bound <= r < best_rank]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline:
            break
        
        # Try integer patterns
        result = _try_rank_r(n, m, p, target_rank, deadline, rng, max_trials=30)
        if result is not None:
            return result
        
        # Try half-integer patterns
        result = _try_half_integer_rank(n, m, p, target_rank, deadline, rng, max_trials=20)
        if result is not None:
            return result
    
    return best

# EVOLVE-BLOCK-END
