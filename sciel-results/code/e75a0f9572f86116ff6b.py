# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_START = time.time()
_DEADLINE = 600.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _shannon_bits(p):
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 1e-18]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())

def _objective(params):
    A, B, C = params
    x = np.arange(_N)
    y = np.arange(_N)
    X, Y = np.meshgrid(x, y, indexing='xy') # P[y, x] so Y is rows, X is cols
    # Note: meshgrid(x, y) with indexing='xy' gives X with shape (N,N) where X[i,j] = x[j]
    # and Y[i,j] = y[i]. This matches P[y, x].
    
    log_w = A * X + B * Y + C * X * Y
    # Shift for stability
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total == 0 or not np.isfinite(total):
        return 0.0
    P = w / total
    
    # Projections
    # D: x - y
    val_d = X - Y
    # M components
    val_m0 = X
    val_m1 = X + Y
    val_m2 = X + 2 * Y
    
    def _proj_entropy(val, P):
        # Aggregate probabilities for each value in val
        # Since val is integer grid, we can use np.unique or bincount
        # Flatten
        flat_val = val.flatten()
        flat_P = P.flatten()
        # unique values and inverse indices
        unique_vals, inverse = np.unique(flat_val, return_inverse=True)
        # Sum probabilities
        proj_p = np.bincount(inverse, weights=flat_P)
        return _shannon_bits(proj_p)
    
    D = _proj_entropy(val_d, P)
    M0 = _proj_entropy(val_m0, P)
    M1 = _proj_entropy(val_m1, P)
    M2 = _proj_entropy(val_m2, P)
    M = max(M0, M1, M2)
    
    if M < 1e-9:
        return 0.0
        
    return D / M

def arithmetic_kakeya_distribution() -> np.ndarray:
    # Initial guess: uniform (A=B=C=0)
    x0 = [0.0, 0.0, 0.0]
    
    # We want to maximize, so minimize negative
    def _neg_obj(params):
        val = _objective(params)
        if not np.isfinite(val):
            return 0.0
        return -val
        
    best_params = x0
    best_val = _objective(best_params)
    
    # Try a few random starts and local optimizations
    rng = np.random.default_rng(42)
    
    starts = [x0]
    for _ in range(10):
        # Random start in a reasonable range
        A = rng.uniform(-2, 2)
        B = rng.uniform(-2, 2)
        C = rng.uniform(-2, 2)
        starts.append([A, B, C])
        
    for s in starts:
        if _remaining() < 30:
            break
        try:
            res = minimize(_neg_obj, s, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < -best_val: # -val is lower, so val is higher
                best_val = -res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Generate final P from best params
    A, B, C = best_params
    x = np.arange(_N)
    y = np.arange(_N)
    X, Y = np.meshgrid(x, y, indexing='xy')
    log_w = A * X + B * Y + C * X * Y
    log_w -= np.max(log_w)
    w = np.exp(log_w)
    total = np.sum(w)
    if total == 0 or not np.isfinite(total):
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P = w / total
        
    # Final safety checks
    P = np.clip(P, 0.0, None)
    s = np.sum(P)
    if s < 1e-9:
        P = np.ones((_N, _N)) / (_N * _N)
    else:
        P = P / s
        
    return P

# EVOLVE-BLOCK-END
