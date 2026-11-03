"""Refined uncertainty inequality: Direct optimization of Laguerre roots for C4 upper bound."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

_START = time.time()
_DEADLINE = 630.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - _MARGIN - time.time()

def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 given 12 positive root locations.
    The bound is 1 / sqrt(||P||^2), where P(x) = prod (x - r_i)^2.
    ||P||^2 is the integral of P(x)^2 * x^2 * exp(-x) dx.
    """
    if not np.all(np.isfinite(roots)) or np.any(roots <= 0):
        return 1e100
    
    # Build Q(x) = prod (x - r_i)
    q_coeff = np.array([1.0])
    for r in roots:
        q_coeff = np.convolve(q_coeff, np.array([1.0, -r]))
        
    # P(x) = Q(x)^2
    p_coeff = np.convolve(q_coeff, q_coeff)
    
    # Compute norm squared: sum_{i,j} p_i p_j Gamma(i+j+3)
    # alpha = 2, so weight is x^2 exp(-x).
    # Integral x^{i+j+2} exp(-x) = Gamma(i+j+3)
    n = len(p_coeff)
    indices = np.arange(n)
    gamma_args = indices[:, None] + indices[None, :] + 3.0
    
    # Use log-gamma for stability
    log_gamma_vals = gammaln(gamma_args)
    log_gamma_vals = np.clip(log_gamma_vals, -700, 700)
    gamma_vals = np.exp(log_gamma_vals)
    
    # norm_sq = p_coeff^T G p_coeff
    norm_sq = p_coeff @ (gamma_vals @ p_coeff)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return 1e100
        
    return 1.0 / np.sqrt(norm_sq)

def _objective(log_roots):
    """
    Objective function for optimization.
    Takes log of roots, returns the C4 bound.
    """
    try:
        roots = np.exp(log_roots)
        # Basic sanity checks
        if np.any(roots <= 0) or np.any(roots > 1e6) or np.any(roots < 1e-6):
            return 1e100
        return _compute_c4_bound(roots)
    except Exception:
        return 1e100

def construct():
    """
    Construct 12 positive double-root locations for the Laguerre combination
    (m=12, n=1) that minimize the upper bound on C4.
    Returns a list of 12 positive root locations.
    """
    deadline = _START + _DEADLINE - _MARGIN
    
    # Initial guess: Geometric progression similar to previous best
    # Previous best was around r0=4, q=1.25
    # Let's use a few different initializations
    np.random.seed(42)
    
    best_roots = None
    best_score = 1e100
    
    # Start points for log-roots
    # 1. Geometric fit to previous best
    r0 = 4.0
    q = 1.25
    roots_init = r0 * (q ** np.arange(12))
    log_roots_init = np.log(roots_init)
    
    # 2. Arithmetic progression
    roots_arith = np.linspace(1.0, 10.0, 12)
    log_roots_arith = np.log(roots_arith)
    
    # 3. Random starts
    num_starts = 20
    
    starts = [log_roots_init, log_roots_arith]
    for _ in range(num_starts):
        # Log-uniform distribution between 0.1 and 100
        log_roots_rand = np.random.uniform(np.log(0.1), np.log(100.0), size=12)
        starts.append(log_roots_rand)
    
    bounds = [(np.log(0.01), np.log(1000.0))] * 12
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline:
            break
            
        try:
            result = minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_roots = np.exp(result.x)
                
        except Exception:
            continue
    
    # If we have time, do a second pass with Nelder-Mead or finer L-BFGS on the best result
    if time.time() < deadline and best_roots is not None:
        try:
            log_best = np.log(best_roots)
            result2 = minimize(
                _objective,
                log_best,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result2.fun < best_score:
                best_score = result2.fun
                best_roots = np.exp(result2.x)
        except Exception:
            pass
            
    if best_roots is None:
        # Fallback
        best_roots = roots_init
        
    return [float(r) for r in best_roots]
