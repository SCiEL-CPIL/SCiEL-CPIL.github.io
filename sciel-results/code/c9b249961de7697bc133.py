"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution
import time

def _compute_c4_bound_fast(roots):
    """
    Compute the C4 upper bound for given double-root positions using 
    Gauss-Laguerre quadrature with log-space stability.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    # Use a sufficient number of quadrature points for accuracy
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Compute log|P(x)| = 2 * sum(log|x - r_i|)
    diff = x_quad[:, np.newaxis] - roots[np.newaxis, :]
    log_abs_diff = np.log(np.abs(diff) + 1e-300)
    log_P = 2.0 * np.sum(log_abs_diff, axis=1)
    
    log_I1_terms = log_P - x_quad
    log_I2_terms = 2.0 * log_P - x_quad
    
    def log_sum_exp_with_weights(A, W):
        log_W = np.log(W)
        B = A + log_W
        m = np.max(B)
        if np.isinf(m):
            return m
        return m + np.log(np.sum(np.exp(B - m)))
    
    log_I1 = log_sum_exp_with_weights(log_I1_terms, w_quad)
    log_I2 = log_sum_exp_with_weights(log_I2_terms, w_quad)
    
    if np.isinf(log_I1) or np.isinf(log_I2) or np.isnan(log_I1) or np.isnan(log_I2):
        return np.inf
        
    log_c4 = log_I2 - 2.0 * log_I1
    c4 = np.exp(log_c4)
    
    return c4

def _objective(roots):
    """Objective function for optimization."""
    return _compute_c4_bound_fast(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580
    
    # Start from standard Laguerre roots as a baseline
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 60:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # ============================================================
    # PHASE 1: Global Search with Differential Evolution
    # ============================================================
    bounds = [(1e-4, 100.0)] * 12
    
    de_configs = [
        {'maxiter': 60, 'popsize': 20, 'seed': 42, 'tol': 1e-8},
        {'maxiter': 40, 'popsize': 25, 'seed': 123, 'tol': 1e-8},
        {'maxiter': 30, 'popsize': 30, 'seed': 777, 'tol': 1e-8},
    ]
    
    for cfg in de_configs:
        if time.time() > deadline - 120:
            break
        
        try:
            result_de = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=cfg['maxiter'],
                popsize=cfg['popsize'],
                tol=cfg['tol'],
                seed=cfg['seed'],
                polish=False,
                updating='immediate',
                workers=1
            )
            
            if result_de.fun < best_score:
                candidate = np.sort(result_de.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
            
            if np.all(result_de.x > 0):
                result_polish = minimize(
                    _objective,
                    result_de.x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if np.all(result_polish.x > 0) and not np.any(np.isnan(result_polish.x)):
                    candidate = np.sort(result_polish.x)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
                        
        except Exception:
            pass
    
    # ============================================================
    # PHASE 2: Local Refinement with Multiple Restarts
    # ============================================================
    initial_points = [best_roots.copy(), x0.copy()]
    
    for scale in [0.1, 0.2, 0.5, 1.0]:
        perturbed = x0 * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-4)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    for log_start in [-2.0, -1.0, 0.0]:
        for log_end in [1.0, 2.0, 3.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                roots = np.exp(log_roots)
                if roots[0] > 1e-4 and roots[-1] < 100:
                    initial_points.append(roots.copy())
    
    for _ in range(10):
        log_roots = rng.uniform(-2.0, 3.0, size=12)
        log_roots = np.sort(log_roots)
        roots = np.exp(log_roots)
        initial_points.append(roots)
    
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
            if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                candidate = np.sort(candidate)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # ============================================================
    # PHASE 3: Final High-Precision Polish
    # ============================================================
    if time.time() < deadline - 5:
        try:
            for _ in range(3):
                if time.time() > deadline - 2:
                    break
                result_final = minimize(
                    _objective,
                    best_roots,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-14}
                )
                if np.all(result_final.x > 0) and not np.any(np.isnan(result_final.x)):
                    candidate = np.sort(result_final.x)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
# EVOLVE-BLOCK-END
