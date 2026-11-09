import numpy as np
from scipy.special import roots_genlaguerre
from scipy.optimize import minimize
import time

_START_TIME = time.time()
_TIME_LIMIT = 550.0  # Safety margin before 600s limit

def _elapsed():
    return time.time() - _START_TIME

def _remaining():
    return max(0.0, _TIME_LIMIT - _elapsed())

# Pre-compute Gauss-Laguerre nodes and weights for high accuracy
# Degree of P(x)^2 is 24. N=50 is overkill but fast. N=100 is safer.
_GL_N = 100
try:
    _gl_nodes, _gl_weights = np.polynomial.laguerre.laggauss(_GL_N)
except Exception:
    _gl_nodes = np.linspace(0, 100, _GL_N)
    _gl_weights = np.ones(_GL_N) / _GL_N

def _compute_l2_norm_sq(roots):
    """
    Compute the square of the L2 norm of P(x) = prod (x - r_i)^2 with weight e^-x.
    Uses Gauss-Laguerre quadrature for speed.
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != 12:
        return 1e100
    if np.any(roots <= 0):
        return 1e100
    # Check for distinct roots (optional, but good for stability if we assume distinct)
    # If roots are very close, the polynomial is still well-defined.
    # We don't strictly enforce distinctness in the objective, but the optimum likely has distinct roots.
    
    # P(x) = prod (x - r_i)^2
    # P(x)^2 = prod (x - r_i)^4
    # We want integral of P(x)^2 * e^-x dx
    # Using GL quadrature: sum w_j * P(x_j)^2
    
    # Vectorized calculation:
    # x_j is shape (N,), roots is shape (12,)
    # diff is shape (N, 12)
    try:
        diff = _gl_nodes[:, np.newaxis] - roots[np.newaxis, :]
        # P(x_j)^2 = prod_i (diff_{j,i}^4)
        # log(P^2) = sum_i 4 * log(|diff|)
        # To avoid under/overflow, compute log first
        # Note: if diff is 0, log is -inf, exp is 0.
        # We can compute directly if values are reasonable.
        # Roots are around 1-20. Nodes are 0-100.
        # diff^4 can be large.
        # Let's use log-sum-exp style or just direct if stable.
        # With N=100 and degree 24, direct multiplication might overflow if roots are large.
        # But roots are typically O(10). 10^4 = 10000. 10000^12 = 10^48. exp(-x) handles decay.
        # However, e^-x is in the weight.
        # Integral approx = sum w_j * prod (x_j - r_i)^4
        
        # Let's compute log_val to be safe
        # log_prod = sum_i 4 * log(|x_j - r_i|)
        # val = exp(log_prod - x_j) * w_j ? No, w_j is part of quadrature for e^-x.
        # GL quadrature: int f(x) e^-x dx ~ sum w_j f(x_j)
        # Here f(x) = P(x)^2.
        # So we need sum w_j * P(x_j)^2.
        
        # Calculate P(x_j)^2
        # If we do direct:
        # p_sq = np.prod(diff**4, axis=1)
        # result = np.sum(_gl_weights * p_sq)
        
        # This might overflow. Let's use log.
        # log_p_sq = 4 * sum(log(|diff|))
        # term = w_j * exp(log_p_sq)
        # This is still risky if log_p_sq > 700.
        # But if log_p_sq > 700, the value is huge, which is bad for minimization.
        # We can clamp.
        
        abs_diff = np.abs(diff)
        abs_diff[abs_diff == 0] = 1e-300 # Avoid log(0)
        log_p_sq = 4.0 * np.sum(np.log(abs_diff), axis=1)
        
        # We want sum w_j * exp(log_p_sq)
        # To avoid overflow in exp, we can scale?
        # Or just clamp.
        log_terms = log_p_sq
        # w_j are positive.
        # max_log_term = max(log_terms + log(w_j))
        # sum = exp(max_log_term) * sum(exp(log_terms + log(w_j) - max_log_term))
        
        log_w = np.log(_gl_weights)
        log_terms_w = log_p_sq + log_w
        
        max_log = np.max(log_terms_w)
        if np.isnan(max_log):
             return 1e100
             
        # If max_log is very large, the score is huge.
        if max_log > 700:
             return 1e100
             
        sum_exp = np.sum(np.exp(log_terms_w - max_log))
        result = np.exp(max_log) * sum_exp
        return result
        
    except Exception:
        return 1e100

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
    
    best_roots = roots0.copy()
    best_score = _compute_l2_norm_sq(best_roots)
    
    # Optimize in log-space to ensure positivity
    # y = log(r) -> r = exp(y)
    # Objective: f(y) = _compute_l2_norm_sq(exp(y))
    
    def objective_log(y):
        if _remaining() < 5.0:
            return best_score
        r = np.exp(y)
        return _compute_l2_norm_sq(r)
    
    y0 = np.log(roots0)
    
    # Bounds for y: r in [1e-6, 1e6] -> y in [-13.8, 13.8]
    bounds_y = [(-15.0, 15.0) for _ in range(12)]
    
    try:
        if _remaining() > 10.0:
            # L-BFGS-B is efficient for box constraints
            result = minimize(
                objective_log,
                y0,
                method='L-BFGS-B',
                bounds=bounds_y,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            
            if result.fun < best_score:
                best_roots = np.exp(result.x)
                best_score = result.fun
            
            # Second pass with tighter tolerance if time allows
            if _remaining() > 20.0:
                result2 = minimize(
                    objective_log,
                    result.x,
                    method='L-BFGS-B',
                    bounds=bounds_y,
                    options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'disp': False}
                )
                if result2.fun < best_score:
                    best_roots = np.exp(result2.x)
                    best_score = result2.fun
                    
    except Exception:
        pass
    
    # Final validation
    best_roots = np.sort(best_roots)
    if len(best_roots) != 12 or np.any(best_roots <= 0) or np.any(np.diff(best_roots) < 1e-8):
        # Fallback to initial guess if optimization produced invalid roots
        best_roots = roots0
    
    return [float(r) for r in best_roots]
