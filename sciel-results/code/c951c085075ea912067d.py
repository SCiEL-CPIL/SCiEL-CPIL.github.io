import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize
import time
import math

_START_TIME = time.time()
_TIME_LIMIT = 580.0  # Safety margin before 600s limit

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _TIME_LIMIT - _elapsed())

# Precompute factorials up to 24!
_MAX_DEG = 24
_FACTS = np.array([math.factorial(k) for k in range(_MAX_DEG + 1)], dtype=float)

def _compute_l2_norm_sq(roots):
    """
    Compute the square of the L2 norm of P(x) = prod (x - r_i)^2 with weight e^-x.
    Uses analytic expansion: integral = sum b_k * k!
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e100
    if np.any(roots <= 0):
        return 1e100
        
    try:
        # Coefficients of Q(x) = prod (x - r_i)
        coeffs_q = np.poly(roots)
        
        # Coefficients of P(x) = Q(x)^2
        coeffs_p = np.convolve(coeffs_q, coeffs_q)
        
        # coeffs_p is [b_24, ..., b_0]
        # We want sum b_k * k!
        result = np.dot(coeffs_p, _FACTS[::-1])
        
        return result
    except Exception:
        return 1e100

def _compute_l2_from_log_params(y):
    """
    Compute L2 norm squared from log-parameterized roots.
    y_i = log(r_i), so r_i = exp(y_i).
    This removes positivity constraints.
    """
    roots = np.exp(y)
    return _compute_l2_norm_sq(roots)

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    m = 12
    
    # Initial guess: roots of generalized Laguerre polynomial L_12^(0)(x)
    try:
        roots0 = roots_genlaguerre(m, 0)
        roots0 = np.sort(roots0.real)
        if len(roots0) != 12 or np.any(roots0 <= 0):
            roots0 = np.array([4.0 * (i + 1) for i in range(12)])
    except Exception:
        roots0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Calculate baseline score
    base_score = _compute_l2_norm_sq(roots0)
    
    best_roots = roots0.copy()
    best_score = base_score
    
    # Log-parameterize: y_i = log(r_i)
    log_roots0 = np.log(roots0)
    
    # Objective in log-space (unconstrained)
    def objective_log(y):
        return _compute_l2_from_log_params(y)
    
    # Multiple restarts with different strategies
    rng = np.random.default_rng(42)
    
    # Strategy 1: Start from Laguerre roots
    candidates = [log_roots0.copy()]
    
    # Strategy 2: Perturbed versions
    for _ in range(5):
        perturbation = rng.normal(0, 0.1, size=12)
        candidates.append(log_roots0 + perturbation)
    
    # Strategy 3: Different initial spreads
    for spread in [0.5, 1.0, 2.0, 5.0]:
        base = np.linspace(0.5, 10.0, 12) * spread
        if np.all(base > 0):
            candidates.append(np.log(base))
    
    # Optimize each candidate
    for idx, y0 in enumerate(candidates):
        if _remaining() < 5.0:
            break
        
        try:
            # First pass: L-BFGS-B (unconstrained in log-space)
            result = minimize(
                objective_log,
                y0,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
            )
            
            if result.success and result.fun < best_score:
                best_roots = np.exp(result.x)
                best_score = result.fun
            
            # Second pass: Polish with Nelder-Mead for fine-tuning
            if _remaining() > 10.0:
                result2 = minimize(
                    objective_log,
                    result.x,
                    method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                
                if result2.success and result2.fun < best_score:
                    best_roots = np.exp(result2.x)
                    best_score = result2.fun
                    
        except Exception:
            continue
    
    # Final polish: use SLSQP in original space with tight constraints
    if _remaining() > 10.0:
        try:
            # Sort roots for SLSQP
            best_roots_sorted = np.sort(best_roots)
            
            def objective_orig(x):
                return _compute_l2_norm_sq(x)
            
            bounds = [(1e-6, None) for _ in range(12)]
            constraints = []
            for i in range(11):
                constraints.append({'type': 'ineq', 'fun': lambda x, i=i: x[i+1] - x[i] - 1e-8})
            
            result_final = minimize(
                objective_orig,
                best_roots_sorted,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False}
            )
            
            if result_final.success and result_final.fun < best_score:
                best_roots = result_final.x
                best_score = result_final.fun
                
        except Exception:
            pass
    
    # Ensure final result is valid
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0) or np.any(np.diff(best_roots) < 1e-8):
        best_roots = roots0
    
    return [float(r) for r in best_roots]
