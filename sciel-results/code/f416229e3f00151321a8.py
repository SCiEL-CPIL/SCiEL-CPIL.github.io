import time
import numpy as np
from numpy.polynomial import laguerre
from scipy.optimize import minimize, differential_evolution
from scipy.special import eval_laguerre, gammaln, gamma
import math

_START = time.time()
_DEADLINE = _START + 600.0
_MARGIN = 30.0

def _time_left():
    return _DEADLINE - _MARGIN - time.time()

def _compute_c4_bound(roots, m=12, n_dim=1):
    """
    Compute the upper bound on C4 from the Laguerre polynomial construction.
    
    The construction: P(x) = prod_{i=1}^{m} (x - r_i)^2
    The norm is computed via the Laguerre inner product:
        ||P||^2 = integral_0^inf P(x)^2 * x^{alpha} * exp(-x) dx
    where alpha = 2 * n_dim = 2.
    
    The upper bound on C4 is proportional to 1/sqrt(||P||^2).
    """
    roots = np.asarray(roots, dtype=float)
    if len(roots) != m:
        return 1e9
    if np.any(roots <= 0):
        return 1e9
    if np.any(np.diff(roots) <= 0):
        return 1e9
    
    alpha = 2 * n_dim  # = 2
    
    # Build Q(x) = prod_{i=0}^{m-1} (x - r_i)
    q_coeff = np.array([1.0])
    for r in roots:
        q_coeff = np.convolve(q_coeff, np.array([1.0, -r]))
    
    # P(x) = Q(x)^2
    p_coeff = np.convolve(q_coeff, q_coeff)
    
    # ||P||^2 = sum_{i,j} p_coeff[i] * p_coeff[j] * Gamma(i + j + alpha + 1)
    indices = np.arange(len(p_coeff))
    gamma_args = indices[:, None] + indices[None, :] + alpha + 1
    log_gamma_vals = gammaln(gamma_args)
    gamma_vals = np.exp(log_gamma_vals)
    
    norm_sq = p_coeff @ (gamma_vals @ p_coeff)
    
    if norm_sq <= 0 or not np.isfinite(norm_sq):
        return np.inf
    
    return 1.0 / np.sqrt(norm_sq)

def _objective_logspace(log_roots):
    """Objective in log-space: roots = exp(log_roots), must be sorted and positive"""
    roots = np.exp(log_roots)
    roots = np.sort(roots)
    
    if np.any(roots <= 0) or np.any(roots > 1e10) or np.any(roots < 1e-10):
        return 1e10
    
    if np.any(np.diff(roots) <= 1e-10):
        return 1e10
    
    return _compute_c4_bound(roots)

def _construct():
    m = 12
    n_dim = 1
    best = None
    best_val = 1e18
    
    # Phase 1: Start from Laguerre nodes
    try:
        x0, _ = laguerre.laggauss(m)
        x0 = np.sort(x0)
        best = x0.copy()
        best_val = _compute_c4_bound(best, m, n_dim)
    except Exception:
        best = np.linspace(0.5, 20.0, m)
        best_val = _compute_c4_bound(best, m, n_dim)
    
    # Phase 2: Differential Evolution in log-space for global exploration
    if _time_left() > 120.0:
        try:
            # Log-space bounds: roots in [0.01, 100]
            log_bounds = [(np.log(0.01), np.log(100.0))] * m
            
            def de_objective(log_roots):
                roots = np.exp(log_roots)
                roots = np.sort(roots)
                if np.any(roots <= 0) or np.any(roots > 1e10) or np.any(roots < 1e-10):
                    return 1e10
                if np.any(np.diff(roots) <= 1e-10):
                    return 1e10
                return _compute_c4_bound(roots, m, n_dim)
            
            # Use DE to explore the 12-dimensional log-space
            de_result = differential_evolution(
                de_objective,
                bounds=log_bounds,
                maxiter=50,
                popsize=15,
                tol=1e-10,
                seed=42,
                polish=True,
                init='sobol',
                mutation=(0.5, 1.5),
                recombination=0.7,
                constraints=[],
                callback=None,
                update_interval=10,
            )
            
            if de_result.fun < best_val:
                cand = np.sort(np.exp(de_result.x))
                if np.all(cand > 1e-8) and np.all(np.diff(cand) > 1e-10):
                    best_val = de_result.fun
                    best = cand
        except Exception:
            pass
    
    # Phase 3: Full 12-dimensional local refinement from best
    if _time_left() > 60.0:
        try:
            def objective(roots):
                roots = np.sort(roots)
                if np.any(roots <= 1e-8):
                    return 1e9
                if np.any(np.diff(roots) <= 1e-10):
                    return 1e9
                return _compute_c4_bound(roots, m, n_dim)
            
            res = minimize(
                objective, best,
                method='L-BFGS-B',
                bounds=[(1e-8, 100.0)] * m,
                options={'maxiter': 3000, 'maxfun': 50000, 'ftol': 1e-12},
            )
            if res.fun < best_val:
                cand = np.sort(res.x)
                if np.all(cand > 1e-8) and np.all(np.diff(cand) > 1e-10):
                    best_val = res.fun
                    best = cand
        except Exception:
            pass
    
    # Phase 4: Multistart full-dimensional optimization
    if _time_left() > 45.0:
        inits = []
        
        # Geometric spacing
        inits.append(np.geomspace(0.5, 25.0, m))
        
        # Linear spacing
        inits.append(np.linspace(1.0, 20.0, m))
        
        # Clustered near origin
        inits.append(np.sort(np.random.uniform(0.5, 10.0, m)))
        
        # Spread out
        inits.append(np.sort(np.random.uniform(1.0, 50.0, m)))
        
        # Fit geometric to current best
        if best is not None:
            logs = np.log(best)
            A = np.vstack([np.ones(m), np.arange(m)]).T
            try:
                coeffs, _, _, _ = np.linalg.lstsq(A, logs, rcond=None)
                r0_fit = np.exp(coeffs[0])
                q_fit = np.exp(coeffs[1])
                if 0.1 < q_fit < 10 and r0_fit > 0:
                    inits.append(r0_fit * (q_fit ** np.arange(m)))
            except Exception:
                pass
        
        for init in inits:
            if _time_left() < 10.0:
                break
            try:
                res = minimize(
                    objective, init,
                    method='L-BFGS-B',
                    bounds=[(1e-8, 100.0)] * m,
                    options={'maxiter': 2000, 'maxfun': 30000, 'ftol': 1e-12},
                )
                if res.fun < best_val:
                    cand = np.sort(res.x)
                    if np.all(cand > 1e-8) and np.all(np.diff(cand) > 1e-10):
                        best_val = res.fun
                        best = cand
            except Exception:
                pass
    
    # Phase 5: Final polish
    if _time_left() > 15.0:
        try:
            res = minimize(
                objective, best,
                method='L-BFGS-B',
                bounds=[(1e-8, 100.0)] * m,
                options={'maxiter': 5000, 'maxfun': 100000, 'ftol': 1e-14},
            )
            if res.fun < best_val:
                cand = np.sort(res.x)
                if np.all(cand > 1e-8) and np.all(np.diff(cand) > 1e-10):
                    best_val = res.fun
                    best = cand
        except Exception:
            pass
    
    best = np.sort(np.maximum(best, 1e-8))
    return [float(v) for v in best]

def construct():
    """Return 12 positive double-root locations for the refined
    Cohn-Goncalves Laguerre construction (m=12, n=1)."""
    try:
        return _construct()
    except Exception:
        x, _ = laguerre.laggauss(12)
        return [float(v) for v in x]
