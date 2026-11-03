"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import roots_laguerre
from scipy.optimize import minimize, differential_evolution, basinhopping
import time
from math import factorial

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

def _lognormal_to_roots(mu, sigma):
    """Convert log-normal parameters (mu, sigma) to 12 sorted root locations.
    The roots are the quantiles of a log-normal distribution at points
    p_i = (i + 0.5) / 12 for i = 0, ..., 11."""
    from scipy.stats import norm
    if sigma < 1e-6:
        sigma = 1e-6
    # Quantile points: 1/12, 2/12, ..., 12/12
    p = (np.arange(1, 13)) / 12.0
    # Inverse CDF of log-normal: exp(mu + sigma * z) where z = Phi^{-1}(p)
    z = norm.ppf(p)
    roots = np.exp(mu + sigma * z)
    return np.sort(roots)

def _lognormal_objective(params):
    """Objective for log-normal parameter optimization."""
    mu, sigma = params
    if sigma < 1e-6:
        sigma = 1e-6
    if mu < -5 or mu > 5:
        return np.inf
    roots = _lognormal_to_roots(mu, sigma)
    if np.any(roots <= 0) or np.any(np.isnan(roots)):
        return np.inf
    return _compute_c4_bound_fast(roots)

def construct():
    """Return 12 positive double-root locations for the Laguerre combination (m=12, n=1)."""
    deadline = time.time() + 580  # Safety margin before 630s kill
    
    # Start from standard Laguerre roots
    x0, _ = roots_laguerre(12)
    x0 = np.sort(x0)
    
    best_roots = x0.copy()
    best_score = _compute_c4_bound_fast(best_roots)
    
    if time.time() > deadline - 120:
        return [float(xi) for xi in best_roots]
    
    rng = np.random.default_rng(42)
    
    # ============================================================
    # PHASE 1: 2D log-normal parameter search
    # ============================================================
    # The hypothesis: optimal roots follow a geometric/log-normal pattern.
    # We search over (mu, sigma) in 2D, which is much more efficient.
    
    # Grid search over (mu, sigma)
    mu_grid = np.linspace(-3.0, 3.0, 60)
    sigma_grid = np.linspace(0.1, 2.5, 50)
    
    best_ln_score = np.inf
    best_ln_params = (0.0, 0.5)
    
    for mu in mu_grid:
        if time.time() > deadline - 200:
            break
        for sigma in sigma_grid:
            score = _lognormal_objective([mu, sigma])
            if score < best_ln_score:
                best_ln_score = score
                best_ln_params = (mu, sigma)
    
    if best_ln_score < best_score:
        best_score = best_ln_score
        best_roots = _lognormal_to_roots(*best_ln_params)
    
    # Refine with local optimization in (mu, sigma) space
    if time.time() < deadline - 150:
        try:
            result_ln = minimize(
                _lognormal_objective,
                best_ln_params,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
            )
            if result_ln.fun < best_score:
                candidate_roots = _lognormal_to_roots(*result_ln.x)
                score = _compute_c4_bound_fast(candidate_roots)
                if score < best_score:
                    best_score = score
                    best_roots = candidate_roots
        except Exception:
            pass
    
    # Basin hopping in (mu, sigma) space
    if time.time() < deadline - 100:
        try:
            def _ln_wrapper(params):
                return _lognormal_objective(params)
            
            bh_result = basinhopping(
                _ln_wrapper,
                x0=best_ln_params,
                niter=50,
                minimizer_kwargs={'options': {'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}},
                stepsize=0.5,
                seed=42
            )
            if bh_result.fun < best_score:
                candidate_roots = _lognormal_to_roots(*bh_result.x)
                score = _compute_c4_bound_fast(candidate_roots)
                if score < best_score:
                    best_score = score
                    best_roots = candidate_roots
        except Exception:
            pass
    
    # ============================================================
    # PHASE 2: Direct coordinate optimization (fallback + refinement)
    # ============================================================
    # Generate diverse initial points for direct optimization
    
    initial_points = [x0.copy(), best_roots.copy()]
    
    # Perturbed Laguerre roots
    for scale in [0.1, 0.2, 0.3, 0.5]:
        perturbed = x0 * (1 + scale * rng.standard_normal(12))
        perturbed = np.maximum(perturbed, 1e-3)
        perturbed = np.sort(perturbed)
        initial_points.append(perturbed)
    
    # Log-uniform spacing
    for log_start in [-2.0, -1.5, -1.0, -0.5, 0.0]:
        for log_end in [1.0, 1.5, 2.0, 2.5, 3.0]:
            if log_end > log_start:
                log_roots = np.linspace(log_start, log_end, 12)
                roots = np.exp(log_roots)
                if roots[0] > 1e-3 and roots[-1] < 100:
                    initial_points.append(roots.copy())
    
    # Pure uniform spacing
    for start in [0.1, 0.5, 1.0, 2.0]:
        for end in [5.0, 10.0, 20.0, 50.0]:
            if end > start:
                roots = np.linspace(start, end, 12)
                initial_points.append(roots.copy())
    
    # Random log-uniform
    for _ in range(8):
        log_roots = rng.uniform(-2.0, 3.0, size=12)
        log_roots = np.sort(log_roots)
        roots = np.exp(log_roots)
        initial_points.append(roots.copy())
    
    # Structured: geometric with various ratios
    for start in [0.3, 0.5, 1.0]:
        for ratio in [1.3, 1.5, 1.8, 2.0]:
            geom = start * (ratio ** np.arange(12))
            if geom[-1] < 100 and geom[0] > 1e-3:
                initial_points.append(geom.copy())
    
    bounds = [(1e-6, None)] * 12
    
    for x_init in initial_points:
        if time.time() > deadline - 80:
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
    
    # Differential evolution in 12D
    de_configs = [
        {'bounds': [(0.01, 50)] * 12, 'maxiter': 40, 'popsize': 15, 'seed': 42},
        {'bounds': [(0.05, 30)] * 12, 'maxiter': 40, 'popsize': 15, 'seed': 123},
        {'bounds': [(0.1, 20)] * 12, 'maxiter': 35, 'popsize': 18, 'seed': 777},
        {'bounds': [(0.01, 80)] * 12, 'maxiter': 30, 'popsize': 20, 'seed': 999},
    ]
    
    for cfg in de_configs:
        if time.time() > deadline - 50:
            break
        try:
            de_bounds = cfg['bounds']
            
            result_de = differential_evolution(
                _objective,
                bounds=de_bounds,
                maxiter=cfg['maxiter'],
                popsize=cfg['popsize'],
                tol=1e-10,
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
            
            # Polish the DE result with tight tolerances
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
    
    # Final high-precision polish on best roots
    if time.time() < deadline - 10:
        try:
            for ftol, gtol, maxit in [(1e-14, 1e-12, 300), (1e-15, 1e-13, 300), (1e-16, 1e-14, 200)]:
                if time.time() > deadline - 5:
                    break
                result_final = minimize(
                    _objective,
                    best_roots,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxit, 'ftol': ftol, 'gtol': gtol}
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
