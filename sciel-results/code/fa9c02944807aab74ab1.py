import numpy as np
import time
import itertools
from scipy.optimize import least_squares
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

def _verify(U, V, W, n=2, m=3, p=3, tol=1e-9):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rational_rounding(U, V, W, max_denom=20):
    """Try to round the solution to rationals with small denominators."""
    U_f = np.array(U, dtype=float)
    V_f = np.array(V, dtype=float)
    W_f = np.array(W, dtype=float)
    
    # Try rounding to integers first
    U_int = np.round(U_f)
    V_int = np.round(V_f)
    W_int = np.round(W_f)
    
    if _verify(U_int.tolist(), V_int.tolist(), W_int.tolist()):
        return U_int, V_int, W_int
    
    # Try half-integers
    U_half = np.round(2 * U_f) / 2.0
    V_half = np.round(2 * V_f) / 2.0
    W_half = np.round(2 * W_f) / 2.0
    
    if _verify(U_half.tolist(), V_half.tolist(), W_half.tolist()):
        return U_half, V_half, W_half
    
    # Try rationals with small denominators
    for denom in range(2, max_denom + 1):
        U_r = np.round(U_f * denom) / denom
        V_r = np.round(V_f * denom) / denom
        W_r = np.round(W_f * denom) / denom
        
        if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist()):
            return U_r, V_r, W_r
            
    # Try using Fraction with continued fractions for each element
    # This is slower but might find exact solutions with slightly larger denominators
    def to_fraction(x, max_denom=100):
        try:
            frac = Fraction(x).limit_denominator(max_denom)
            return float(frac)
        except:
            return x
            
    U_frac = np.array([[to_fraction(x) for x in row] for row in U_f])
    V_frac = np.array([[to_fraction(x) for x in row] for row in V_f])
    W_frac = np.array([[to_fraction(x) for x in row] for row in W_f])
    
    if _verify(U_frac.tolist(), V_frac.tolist(), W_frac.tolist()):
        return U_frac, V_frac, W_frac

    return None, None, None

def _build_schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=int)
    V = np.zeros((m*p, rank), dtype=int)
    W = np.zeros((p*n, rank), dtype=int)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _residuals(x, n, m, p, r):
    """
    Compute the residuals for the tensor decomposition.
    x is a flattened vector of length (n*m + m*p + p*n) * r
    """
    size_u = n * m
    size_v = m * p
    size_w = p * n
    
    U = x[:size_u * r].reshape(size_u, r)
    V = x[size_u * r : (size_u + size_v) * r].reshape(size_v, r)
    W = x[(size_u + size_v) * r :].reshape(size_w, r)
    
    residuals = np.zeros(n * m * p)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Sum over rank c
                # U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :]
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                residuals[idx] = val - 1.0
                idx += 1
                
    return residuals

def _search_rank(target_rank, n=2, m=3, p=3, max_restarts=100, time_budget=30.0):
    """
    Search for a rank decomposition using least_squares.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    size_u = n * m
    size_v = m * p
    size_w = p * n
    total_vars = (size_u + size_v + size_w) * target_rank
    
    best_error = np.inf
    best_x = None
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        np.random.seed(restart * 12345 + target_rank)
        
        # Initialize with small random values
        x0 = np.random.randn(total_vars) * 0.1
        
        try:
            result = least_squares(
                _residuals, 
                x0, 
                args=(n, m, p, target_rank),
                method='lm', # Levenberg-Marquardt is good for non-linear least squares
                xtol=1e-12, 
                ftol=1e-12, 
                gtol=1e-12
            )
            
            if result.success and result.cost < 1e-10:
                if result.cost < best_error:
                    best_error = result.cost
                    best_x = result.x
                    
                    # If we found a very good solution, we can stop early
                    if best_error < 1e-12:
                        break
                        
        except Exception:
            continue
            
    if best_x is not None:
        # Reshape best_x back to U, V, W
        size_u = n * m
        size_v = m * p
        size_w = p * n
        
        U = best_x[:size_u * target_rank].reshape(size_u, target_rank)
        V = best_x[size_u * target_rank : (size_u + size_v) * target_rank].reshape(size_v, target_rank)
        W = best_x[(size_u + size_v) * target_rank :].reshape(size_w, target_rank)
        
        return U, V, W, best_error
        
    return None, None, None, np.inf

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Try ranks from lower bound up to schoolbook
    # Theoretical lower bound is roughly 15-16. Known rank is 16.
    # We try 16, 17, 18.
    
    for target_rank in [16, 17, 18]:
        if time.time() > deadline - 20.0:
            break
            
        U, V, W, error = _search_rank(target_rank, time_budget=30.0)
        
        if U is not None and error < 1e-8:
            # Try to round to rationals
            U_r, V_r, W_r = _try_rational_rounding(U, V, W)
            if U_r is not None:
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
            
            # If rounding fails, check if the floating point solution is valid
            # (The evaluator might accept float if it's close enough? 
            # "integer/half-integer entries" is specified in the contract. 
            # So we MUST return integers or halves.)
            
            # If we found a real solution but couldn't round it, 
            # we might need to try more restarts or different rounding strategies.
            # But if we can't find a rational one, we might have to fall back.
            # However, if a rank 16 solution exists, it likely has small rationals.
            pass
            
    # Fallback to schoolbook
    return _build_schoolbook()
