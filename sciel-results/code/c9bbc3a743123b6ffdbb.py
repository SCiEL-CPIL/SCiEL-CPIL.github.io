import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize
import time

def _objective(roots):
    """
    Compute a proxy for the C4 upper bound.
    We minimize the weighted maximum of the squared polynomial P(x) = prod (x-r_i)^2.
    Weight w(x) is chosen to reflect the n=1 dimension context, likely x * exp(-x/2) or exp(-x/2).
    We use x * exp(-x/2) as a strong prior based on Laguerre weight structures.
    """
    roots = np.asarray(roots, dtype=np.float64)
    # Ensure roots are positive and sorted for stability, though order doesn't matter for product
    roots = np.maximum(roots, 1e-8)
    
    # Grid for evaluation
    # The roots of L_12 are roughly in [0.5, 25]. 
    # If scaled or different alpha, they might shift.
    # Let's use a grid that covers the likely range of roots and tail.
    max_root = np.max(roots)
    x_max = max(100.0, 3.0 * max_root)
    x = np.linspace(0.01, x_max, 2000)
    
    # Calculate log |P(x)| = 2 * sum log |x - r_i|
    log_abs_P = np.zeros_like(x)
    for r in roots:
        log_abs_P += 2.0 * np.log(np.abs(x - r) + 1e-300)
        
    # Weight: x * exp(-x/2)
    log_w = np.log(x) - 0.5 * x
    
    log_obj = log_abs_P + log_w
    # Return the max value (in log space to avoid overflow, but we compare values)
    # To be safe with minimize, return the actual max value if it's not too big, 
    # or just the log max if we treat it as a monotonic transform.
    # However, the score is seed/value. If value is huge, score is small.
    # The objective should be the raw value.
    # If exp(log_obj) overflows, we return a large number.
    max_log_val = np.max(log_obj)
    if max_log_val > 700:
        return 1e300
    return np.exp(max_log_val)

def construct():
    deadline = time.time() + 570  # Safety margin
    
    # Candidate 1: Standard Laguerre (alpha=0) - Current best
    roots0 = roots_genlaguerre(12, 0.0)[0]
    
    # Candidate 2: Generalized Laguerre (alpha=0.5)
    roots_half = roots_genlaguerre(12, 0.5)[0]
    
    # Candidate 3: Generalized Laguerre (alpha=1.0)
    roots_one = roots_genlaguerre(12, 1.0)[0]
    
    # Candidate 4: Generalized Laguerre (alpha=1.5)
    roots_15 = roots_genlaguerre(12, 1.5)[0]
    
    candidates = [roots0, roots_half, roots_one, roots_15]
    
    best_roots = None
    best_val = float('inf')
    
    # Evaluate initial candidates
    for r in candidates:
        if time.time() > deadline:
            break
        try:
            val = _objective(r)
            if val < best_val:
                best_val = val
                best_roots = r.copy()
        except Exception:
            pass
    
    if best_roots is None:
        # Fallback to seed if all fail
        return [4.0 * (i + 1) for i in range(12)]
        
    # Refine the best candidate using Nelder-Mead
    # Nelder-Mead is derivative-free and handles local noise well.
    if time.time() < deadline - 30:
        try:
            res = minimize(
                _objective,
                x0=best_roots,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            if res.fun < best_val:
                candidate = np.maximum(res.x, 1e-8)
                val = _objective(candidate)
                if val < best_val:
                    best_val = val
                    best_roots = candidate
        except Exception:
            pass

    # One more attempt with L-BFGS-B if time permits, using gradients?
    # _objective is not smooth due to max, so gradient methods are tricky.
    # Stick with Nelder-Mead.
    
    return best_roots.tolist()
