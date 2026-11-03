import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre
import time

def _compute_c4_bound_fast(roots):
    """Compute the C4 upper bound for given double-root positions using Gauss-Laguerre quadrature."""
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    
    n_quad = 200
    x_quad, w_quad = np.polynomial.laguerre.laggauss(n_quad)
    
    # Vectorized: compute log_P for all roots at once
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
    start_time = time.time()
    deadline = start_time + 570.0  # Safety margin before 630s kill
    
    # Get the standard Laguerre roots of degree 12 (these are the equilibrium points)
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 60:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initial points around the Laguerre roots
    initial_points = [x0.copy()]
    
    # Perturbed Laguerre roots with various small scales (most important)
    for scale in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        perturbed = x0 * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-4)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    # Multiple random perturbations at small scale
    for _ in range(5):
        perturbed = x0 * (1 + 0.05 * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-4)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    # Scaled Laguerre roots
    for scale in [0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0, 3.0]:
        scaled = x0 * scale
        initial_points.append(scaled.copy())
    
    # Random uniform in various ranges
    for lo, hi in [(0.1, 10), (0.5, 20), (1.0, 30), (0.1, 5)]:
        rand_roots = rng.uniform(lo, hi, size=12)
        rand_roots = np.sort(rand_roots)
        initial_points.append(rand_roots)
    
    # Log-uniform spacing
    for log_lo, log_hi in [(-1, 2), (0, 3), (-0.5, 2.5)]:
        log_roots = rng.uniform(log_lo, log_hi, size=12)
        log_roots = np.sort(log_roots)
        initial_points.append(np.exp(log_roots))
    
    # Geometric spacing variations
    for start in [0.3, 0.5, 1.0]:
        for ratio in [1.2, 1.3, 1.4, 1.5, 1.7]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 200 and geom[0] > 0.05:
                initial_points.append(geom.copy())
    
    bounds = [(1e-6, None)] * 12
    
    # Stage 1: Coarse optimization from each initial point
    for x_init in initial_points:
        if time.time() > deadline - 90:
            break
        
        try:
            result = minimize(
                _objective,
                x_init,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
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
    
    # Stage 2: Try Nelder-Mead from best and a few other good candidates
    if time.time() < deadline - 30:
        # Collect top candidates from stage 1 results
        nm_starts = [best_roots.copy()]
        
        # Add slightly perturbed versions of best
        for _ in range(3):
            pert = best_roots * (1 + 0.01 * rng.standard_normal(12))
            pert = np.maximum(pert, 1e-4)
            pert = np.sort(pert)
            nm_starts.append(pert)
        
        for x_init in nm_starts:
            if time.time() > deadline - 20:
                break
            try:
                result_nm = minimize(
                    _objective,
                    x_init,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                candidate = result_nm.x
                if np.all(candidate > 0) and not np.any(np.isnan(candidate)):
                    candidate = np.sort(candidate)
                    score = _compute_c4_bound_fast(candidate)
                    if score < best_score:
                        best_score = score
                        best_roots = candidate.copy()
            except Exception:
                pass
    
    # Stage 3: Final high-precision polish on best roots
    if time.time() < deadline - 10:
        try:
            result_final = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if np.all(result_final.x > 0) and not np.any(np.isnan(result_final.x)):
                candidate = np.sort(result_final.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    # Stage 4: One more very tight polish
    if time.time() < deadline - 5:
        try:
            result_tight = minimize(
                _objective,
                best_roots,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-18, 'gtol': 1e-16}
            )
            if np.all(result_tight.x > 0) and not np.any(np.isnan(result_tight.x)):
                candidate = np.sort(result_tight.x)
                score = _compute_c4_bound_fast(candidate)
                if score < best_score:
                    best_score = score
                    best_roots = candidate.copy()
        except Exception:
            pass
    
    return [float(xi) for xi in best_roots]
