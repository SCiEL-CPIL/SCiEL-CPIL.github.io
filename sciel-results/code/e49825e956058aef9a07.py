"""Optimized program for autocorrelation inequality task using direct 400D gradient optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 400

def _compute_value(h):
    """Compute the exact evaluation value."""
    s = np.sum(h)
    if s == 0:
        return float('inf')
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def _smooth_obj_and_grad(h, beta=50.0):
    """
    Compute smooth objective and its gradient in the full 400D space.
    
    Objective: smooth_max(conv) / sum(h)^2
    where smooth_max uses log-sum-exp with temperature 1/beta.
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = np.convolve(h, h)
    m = np.max(conv)
    exp_conv = np.exp(conv - m)
    sum_exp = np.sum(exp_conv)
    S = m + np.log(sum_exp) / beta
    
    # Weights: dS/d(conv[i]) = w[i] / beta
    w = exp_conv / sum_exp
    
    # dS_dh[k] = (2/beta) * sum_j w[j+k] * h[j]
    # Use np.correlate for efficiency
    corr = np.correlate(w, h, mode='full')
    dS_dh = (2.0 / beta) * corr[N-1:N-1+N]
    
    obj = S / (s ** 2)
    grad = dS_dh / (s ** 2) - 2.0 * S / (s ** 3)
    
    return obj, grad

def _diverse_init(i, rng):
    """Generate diverse initializations."""
    if i == 0:
        return np.ones(N)
    elif i == 1:
        return rng.standard_normal(N) * 0.1 + 1.0
    elif i == 2:
        t = np.arange(N)
        return 1.0 + 0.3 * np.cos(2 * np.pi * 5 * t / N)
    elif i == 3:
        t = np.arange(N)
        return 1.0 + 0.3 * np.cos(2 * np.pi * 50 * t / N)
    elif i == 4:
        t = np.arange(N)
        return 1.0 + 0.3 * np.sin(2 * np.pi * 10 * t / N)
    elif i == 5:
        t = np.arange(N)
        return 1.0 + 0.5 * np.cos(2 * np.pi * 2 * t / N)
    elif i == 6:
        t = np.arange(N)
        return 1.0 + 0.4 * np.sin(2 * np.pi * 20 * t / N)
    elif i == 7:
        t = np.arange(N)
        return 1.0 + 0.3 * np.cos(2 * np.pi * 100 * t / N)
    else:
        return rng.standard_normal(N) * 0.2 + 1.0

def construct():
    """Construct step heights minimizing autocorrelation peak."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    rng = np.random.RandomState(42)
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    n_starts = 20
    n_iter_per_start = 300
    
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        h0 = _diverse_init(i, rng)
        
        try:
            result = minimize(
                _smooth_obj_and_grad,
                h0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': n_iter_per_start, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50},
                args=(30.0,)
            )
            
            if result.fun < best_val:
                val = _compute_value(result.x)
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
                    
        except Exception:
            continue
    
    # Phase 2: Refine with higher beta
    if time.time() < deadline - 20 and best_val < 2.0:
        try:
            result = minimize(
                _smooth_obj_and_grad,
                best_h,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50},
                args=(100.0,)
            )
            
            if result.fun < best_val:
                val = _compute_value(result.x)
                if val < best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: More restarts from best with perturbations
    if time.time() < deadline - 10 and best_val < 2.0:
        for i in range(10):
            if time.time() > deadline:
                break
            
            h_perturbed = best_h + rng.standard_normal(N) * 0.05
            
            try:
                result = minimize(
                    _smooth_obj_and_grad,
                    h_perturbed,
                    jac=True,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 50},
                    args=(80.0,)
                )
                
                if result.fun < best_val:
                    val = _compute_value(result.x)
                    if val < best_val:
                        best_val = val
                        best_h = result.x.copy()
            except Exception:
                continue
    
    # Ensure finite and sum not zero
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N)
    
    return best_h
