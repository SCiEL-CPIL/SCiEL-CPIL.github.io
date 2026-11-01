"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _compute_value(h):
    """
    Compute the raw score (upper bound on C3) for a given height vector h.
    Metric: max_k |conv(h, h)_k| / (sum_i |h_i|)^2
    """
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(np.abs(conv))
    sum_abs_h = np.sum(np.abs(h))
    if sum_abs_h == 0:
        return np.inf
    return max_conv / (sum_abs_h ** 2)


def _objective(x, T=5.0, eps=1e-8):
    """
    Smooth objective function for optimization.
    x is the parameter vector, h = tanh(x).
    """
    h = np.tanh(x)
    conv = np.convolve(h, h, mode='full')
    abs_conv = np.sqrt(conv ** 2 + eps)
    
    # Soft max approximation
    max_abs = np.max(abs_conv)
    log_sum_exp = np.log(np.sum(np.exp((abs_conv - max_abs) / T))) + max_abs / T
    log_max = T * log_sum_exp
    
    sum_abs_h = np.sum(np.sqrt(h ** 2 + eps))
    value = log_max / (sum_abs_h ** 2)
    return value


def _optimize_from_start(x0, maxiter=500):
    """
    Run L-BFGS-B optimization from a specific start point.
    Returns the optimized h and the true (non-smooth) value.
    """
    try:
        res = minimize(_objective, x0, method='L-BFGS-B', 
                       options={'maxiter': maxiter, 'ftol': 1e-10})
        h_opt = np.tanh(res.x)
        val = _compute_value(h_opt)
        return h_opt, val
    except Exception:
        return None, np.inf


def construct():
    # Multi-start strategy to find a better global minimum
    # The problem is non-convex. Starting from all-ones is one local min.
    # Random starts or structured starts (e.g., alternating signs) might yield better results.
    
    best_h = None
    best_val = np.inf
    
    # Start 1: All ones (seed equivalent)
    x0_1 = np.full(N, 5.0)
    h1, v1 = _optimize_from_start(x0_1)
    if h1 is not None and v1 < best_val:
        best_val = v1
        best_h = h1

    # Start 2: Alternating signs (Rademacher-like)
    # This creates cancellation in convolution.
    x0_2 = np.array([5.0 if i % 2 == 0 else -5.0 for i in range(N)])
    h2, v2 = _optimize_from_start(x0_2)
    if h2 is not None and v2 < best_val:
        best_val = v2
        best_h = h2

    # Start 3: Random small perturbations around 0
    # tanh(0)=0, so this starts near zero.
    # We need to ensure the optimizer moves away from zero if it's a saddle point.
    # Actually, if h is small, value is 0/0 -> inf. 
    # Let's try a random initialization with values in [-1, 1] mapped to tanh space.
    rng = np.random.default_rng(42)
    x0_3 = rng.uniform(-5, 5, size=N)
    h3, v3 = _optimize_from_start(x0_3)
    if h3 is not None and v3 < best_val:
        best_val = v3
        best_h = h3

    # Start 4: Another random start
    x0_4 = rng.uniform(-5, 5, size=N)
    h4, v4 = _optimize_from_start(x0_4)
    if h4 is not None and v4 < best_val:
        best_val = v4
        best_h = h4

    # If all optimizations failed or returned inf, fallback to seed
    if best_h is None or not np.isfinite(best_val):
        best_h = np.ones(N)
        best_val = _compute_value(best_h)
        
    return best_h
# EVOLVE-BLOCK-END
