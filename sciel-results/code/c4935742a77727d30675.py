"""Refined uncertainty inequality - Cohn-Goncalves Laguerre construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _compute_bound(roots):
    """
    Compute an upper bound related to the Cohn-Goncalves Laguerre construction.
    roots: array of 12 positive double-root locations.
    Returns a scalar bound (lower is better).
    """
    roots = np.asarray(roots, dtype=np.float64)
    if len(roots) != 12:
        return 1e10
    if np.any(roots <= 0):
        return 1e10
    
    # Sort roots
    roots = np.sort(roots)
    
    # Check for reasonable spacing (no overlaps)
    if np.any(np.diff(roots) < 1e-8):
        return 1e10
    
    # Compute the Laguerre-weighted L2 norm of P(x) = prod (x - r_i)^2
    # Integral I = ∫_0^∞ |P(x)|^2 e^{-x} dx = ∫_0^∞ [prod (x-r_i)^4] e^{-x} dx
    
    # Use Gauss-Laguerre quadrature
    # To handle the large dynamic range, we compute in log space
    # log(I) = log( sum w_i * prod (x_i - r_j)^4 )
    # This is numerically unstable if done directly.
    # However, for the purpose of optimization, we can approximate the behavior.
    # The dominant term in the integral for large roots is determined by the largest root.
    # Specifically, if roots are large, the integral is dominated by the region x ~ max(roots).
    # For a product polynomial, the L2 norm is heavily influenced by the spread.
    
    # Let's use a robust quadrature with fewer points for speed, but enough for accuracy.
    # 100 points is usually sufficient for smooth integrands like this.
    _N_QUAD = 100
    _X_NODES, _X_WEIGHTS = np.polynomial.laguerre.laggauss(_N_QUAD)
    
    # log|P(x)|^2 = 4 * sum_j log|x - r_j|
    log_abs_P_sq = np.zeros_like(_X_NODES)
    for r in roots:
        # Avoid log(0)
        log_abs_P_sq += 4.0 * np.log(np.abs(_X_NODES - r) + 1e-300)
    
    log_terms = log_abs_P_sq + np.log(_X_WEIGHTS + 1e-300)
    max_log = np.max(log_terms)
    
    # If the value is astronomically large, return a large penalty
    if max_log > 700:
        return np.exp(700)
    
    # Compute sum of exp(log_terms - max_log) * exp(max_log)
    integral = np.exp(max_log) * np.sum(np.exp(log_terms - max_log))
    
    return integral

def construct():
    """
    Return 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    Score is the upper bound on C4 (lower is better).
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # Seed roots
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    seed_bound = _compute_bound(seed_roots)
    
    best_roots = seed_roots.copy()
    best_bound = 1.0  # Normalized score (raw / seed_bound)

    def objective(x):
        if time.time() > deadline - 5:
            return best_bound
        if np.any(x <= 0):
            return 1e10
        raw = _compute_bound(x)
        if raw >= 1e9:
            return 1e10
        return raw / seed_bound

    # Strategy: Log-space optimization
    # The roots are positive. Let r_i = exp(y_i).
    # This ensures positivity and allows for multiplicative steps.
    # The seed roots are roughly arithmetic: 4, 8, ..., 48.
    # In log space, this is not linear.
    
    # Initial guess in log space
    y_seed = np.log(seed_roots)
    
    def objective_log(y):
        r = np.exp(y)
        # Check for overflow
        if np.any(r > 1e300):
            return 1e10
        return objective(r)
    
    # Use Nelder-Mead which doesn't require gradients and works well in log space
    # for non-linear transformations.
    
    # Multi-start in log space
    # Start 1: Seed
    # Start 2: Uniform spread in log space over a wider range
    # Start 3: Clustered at high values (to maximize geometric mean)
    
    starts = [
        y_seed,
        np.linspace(np.log(1), np.log(100), 12),
        np.linspace(np.log(10), np.log(500), 12),
        np.log(np.linspace(1, 100, 12)),
        np.log(np.linspace(5, 200, 12))
    ]
    
    for y0 in starts:
        if time.time() > deadline - 10:
            break
        try:
            result = minimize(
                objective_log,
                y0,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_roots = np.exp(result.x).copy()
        except Exception:
            pass

    # Refine with L-BFGS-B in linear space if we have time
    if time.time() < deadline - 50:
        try:
            result = minimize(
                objective,
                best_roots,
                method='L-BFGS-B',
                bounds=[(0.01, 1000)] * 12,
                options={'maxiter': 200, 'maxfun': 5000}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_roots = result.x.copy()
        except Exception:
            pass

    # Ensure roots are sorted and positive
    best_roots = np.sort(best_roots)
    if np.any(best_roots <= 0):
        best_roots = seed_roots.copy()
    
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
