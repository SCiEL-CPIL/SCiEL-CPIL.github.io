import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill

def _compute_C(h):
    """
    Compute C_k for all k.
    C_k = sum_{i=0}^{N-1-k} h_i - h_i h_{i+k}
    """
    n = len(h)
    C = np.zeros(n)
    for k in range(n):
        # Sum part
        s = np.sum(h[:n-k])
        # Product part
        p = np.sum(h[:n-k] * h[k:])
        C[k] = s - p
    return C

def _compute_C_grad(h):
    """
    Compute gradient of C_k with respect to h.
    Returns a matrix G where G[k, j] = dC_k/dh_j.
    dC_k/dh_j = I(j <= N-1-k) - I(j+k < N)h_{j+k} - I(j-k >= 0)h_{j-k}
    """
    n = len(h)
    G = np.zeros((n, n))
    for k in range(n):
        # Term 1: I(j <= N-1-k)
        G[k, :n-k] += 1.0
        # Term 2: -I(j+k < N)h_{j+k}
        if k > 0:
            # j ranges from 0 to n-1-k
            # h_{j+k} ranges from h_k to h_{n-1}
            G[k, :n-k] -= h[k:]
        # Term 3: -I(j-k >= 0)h_{j-k}
        # j ranges from k to n-1
        # h_{j-k} ranges from h_0 to h_{n-1-k}
        G[k, k:] -= h[:n-k]
    return G

def _smooth_max_and_grad(h, beta):
    """
    Compute smooth max of C(h) and its gradient.
    """
    C = _compute_C(h)
    # Shift for numerical stability
    max_C = np.max(C)
    exp_C = np.exp(beta * (C - max_C))
    sum_exp = np.sum(exp_C)
    val = max_C + np.log(sum_exp) / beta
    weights = exp_C / sum_exp
    
    G = _compute_C_grad(h)
    grad = weights @ G
    return val, grad

def _penalized_obj_and_grad(h, beta, lambda_pen):
    """
    Objective: smooth_max(C(h)) + lambda * (sum(h) - N/2)^2
    """
    val, grad = _smooth_max_and_grad(h, beta)
    sum_h = np.sum(h)
    diff = sum_h - N / 2.0
    val += lambda_pen * diff * diff
    grad += 2 * lambda_pen * diff * np.ones(N)
    return val, grad

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        C = _compute_C(h)
        s = float(np.max(C))
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    def _run_optimization(h0, beta, lambda_pen, maxiter, ftol):
        nonlocal best_h, best_score
        
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            val, _ = _penalized_obj_and_grad(x, beta, lambda_pen)
            return val
            
        def jac_func(x):
            _, grad = _penalized_obj_and_grad(x, beta, lambda_pen)
            return grad
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-8})
            if res.fun < best_score + 1e-6: # Allow slight penalty in objective if raw score is better
                _update_best(res.x)
        except Exception:
            pass

    # Initial candidates
    candidates = []
    
    # 1. Cosine
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    candidates.append(_fix_sum(h_cos))
    
    # 2. Random
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        candidates.append(_fix_sum(h_rand))
        
    # 3. Block pattern (alternating high/low)
    h_block = np.zeros(N)
    h_block[::2] = 1.0
    h_block[1::2] = 0.0
    # Adjust to sum N/2. This already sums to N/2 if N is even.
    candidates.append(_fix_sum(h_block))
    
    # 4. Linear ramp
    h_ramp = np.linspace(0, 1, N)
    candidates.append(_fix_sum(h_ramp))

    # Evaluate initial candidates
    for h in candidates:
        _update_best(h)
        
    # Optimization Phase 1: Moderate beta, low penalty
    if time.time() - t0 < _DEADLINE - 40:
        for h in candidates:
            if time.time() - t0 > _DEADLINE - 30:
                break
            _run_optimization(h, beta=20.0, lambda_pen=10.0, maxiter=100, ftol=1e-6)
            
    # Optimization Phase 2: Higher beta, higher penalty (tighter sum constraint)
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        _run_optimization(best_h, beta=50.0, lambda_pen=100.0, maxiter=200, ftol=1e-8)
        
    # Optimization Phase 3: Very high beta, refine
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        _run_optimization(best_h, beta=100.0, lambda_pen=1000.0, maxiter=300, ftol=1e-10)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _fix_sum(best_h)
    return best_h
