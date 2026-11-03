"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

def _compute_c4_bound(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # Use Gauss-Laguerre quadrature (includes e^{-x} weight)
    n_quad = 150
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # P(x) = prod (x - x_i)^2
    # Compute log_P for numerical stability
    log_P = np.zeros(n_quad)
    for r in roots:
        log_P += 2.0 * np.log(np.abs(x_quad - r) + 1e-300)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _compute_c4_bound_fast(roots):
    """Faster version using vectorized operations where possible."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    n_quad = 150
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Vectorized: compute log_P for all roots at once
    # Shape: (150, 12)
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    P = np.exp(log_P)
    P2 = P * P
    
    int_P = np.sum(w_quad * P)
    int_P2 = np.sum(w_quad * P2)
    
    if int_P <= 0 or np.isinf(int_P) or np.isinf(int_P2):
        return np.inf
    
    c4 = int_P2 / (int_P * int_P)
    return c4

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound_fast(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points - emphasize uniform and geometric spacing
    initial_points = []
    
    # Uniform spacing in log-space (key insight: uniform in log = geometric in linear)
    for log_start in [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0]:
        for log_end in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                roots = np.exp(log_roots)
                if roots[0] > 1e-3 and roots[-1] < 100:
                    initial_points.append(roots.copy())
    
    # Pure uniform spacing
    for start in [0.1, 0.5, 1.0, 2.0, 5.0]:
        for end in [5.0, 10.0, 20.0, 50.0]:
            if end > start:
                uniform_roots = np.linspace(start, end, 12)
                initial_points.append(uniform_roots.copy())
    
    # Geometric spacing with various ratios
    for start in [0.1, 0.5, 1.0]:
        for ratio in [1.2, 1.3, 1.5, 1.8, 2.0]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 100:
                initial_points.append(geom.copy())
    
    # Standard Laguerre roots (for reference)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    initial_points.append(x0.copy())
    
    # Start with the best uniform initialization
    best_roots = initial_points[0].copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    # Try L-BFGS-B from each initial point, prioritizing diverse spacings
    bounds = [(1e-6, 100)] * 12
    
    for x_init in initial_points:
        if time.time() > deadline - 30:
            break
        
        try:
            result = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            candidate = result.x
            if np.all(candidate > 0) and np.all(candidate < 100) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Final polish on best roots with tighter tolerance
    if time.time() < deadline - 5:
        try:
            result_final = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(result_final.x > 0) and np.all(result_final.x < 100) and not np.any(np.isnan(result_final.x)):
                candidate = np.sort(result_final.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
