"""Matrix multiplication 2,3,3 - rank-minimal tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import sys

DIMENSIONS = [2, 3, 3]

def _to_rational_approx(matrix, max_den=1000):
    """Convert a float matrix to a rational matrix with small denominators."""
    result = []
    for row in matrix:
        r = []
        for val in row:
            # Try to find a good rational approximation
            best = Fraction(0)
            best_err = abs(val)
            for d in range(1, max_den + 1):
                n = round(val * d)
                if n == 0:
                    continue
                err = abs(val - n / d)
                if err < best_err:
                    best_err = err
                    best = Fraction(n, d)
                if best_err < 1e-12:
                    break
            r.append(best)
        result.append(r)
    return result

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _verify_decomposition_exact(U, V, W, n, m, p):
    """Verify with exact rational arithmetic."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if s != 1:
                    return False
    return True

def _random_decomposition(rank, n, m, p, rng):
    """Generate a random rank-r decomposition and try to solve for it."""
    U = rng.random((n*m, rank))
    V = rng.random((m*p, rank))
    
    num_eqs = n*m*p
    num_unk = p*n*rank
    
    if num_unk < num_eqs:
        return None
    
    A = np.zeros((num_eqs, num_unk))
    b = np.ones(num_eqs)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                eq_idx = (i * m + j) * p + k
                for c in range(rank):
                    unk_idx = (k * n + i) * rank + c
                    A[eq_idx, unk_idx] = U[i*m+j, c] * V[j*p+k, c]
    
    try:
        w_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    
    if len(residuals) > 0 and residuals[0] > 1e-8:
        return None
    
    W = w_flat.reshape((p*n, rank))
    
    if not _verify_decomposition(U, V, W, n, m, p, tol=1e-7):
        return None
    
    return U, V, W

def _try_rank(rank, n, m, p, max_attempts=50, time_limit=None):
    """Try to find a rank-r decomposition."""
    rng = np.random.default_rng(42 + rank)
    
    for attempt in range(max_attempts):
        if time_limit is not None:
            if time.time() > time_limit:
                break
        
        result = _random_decomposition(rank, n, m, p, rng)
        if result is not None:
            U, V, W = result
            U_r = _to_rational_approx(U, max_den=100)
            V_r = _to_rational_approx(V, max_den=100)
            W_r = _to_rational_approx(W, max_den=100)
            
            if _verify_decomposition_exact(U_r, V_r, W_r, n, m, p):
                return U_r, V_r, W_r
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    schoolbook_rank = n * m * p
    
    # Try ranks from low to high
    for rank in range(9, schoolbook_rank + 1):
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(rank, n, m, p, max_attempts=100, time_limit=deadline - 15)
        if result is not None:
            U, V, W = result
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    rank = schoolbook_rank
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
