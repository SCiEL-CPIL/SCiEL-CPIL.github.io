"""Direct construction using the Cohn-Goncalves Laguerre polynomial root structure."""
import numpy as np
import time
from numpy.polynomial.laguerre import laggauss
from scipy.special import genlaguerre, eval_genlaguerre, gammaln
from scipy.optimize import minimize, minimize_scalar

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 600.0
_SAFETY_MARGIN = 30.0

def _deadline_reached():
    return (time.time() - _START_TIME) >= (_DEADLINE - _SAFETY_MARGIN)

def _compute_c4_bound(roots, m=12, n=1, num_quad=400):
    """
    Compute the C4 uncertainty upper bound for given double-root locations.
    C4 = (∫₀^∞ |P(x)|² x^(2n-1) e^(-x) dx)^(1/2) / (∫₀^∞ x^(2n-1) e^(-x) dx)^(1/2)
    where P(x) = ∏ᵢ (x - rᵢ)²
    """
    try:
        x_quad, w_quad = np.polynomial.laguerre.laggauss(num_quad)
        
        log_abs_P_sq = np.zeros_like(x_quad)
        for r in roots:
            log_abs_P_sq += 4.0 * np.log(np.abs(x_quad - r) + 1e-300)
        
        alpha = 2 * n - 1
        weight_factor = np.power(x_quad, alpha)
        
        integrand = np.exp(log_abs_P_sq) * weight_factor
        integral_P = np.sum(w_quad * integrand)
        
        log_norm = gammaln(2 * n)
        norm = np.exp(log_norm)
        
        c4_bound = np.sqrt(integral_P / norm)
        
        return c4_bound
    except Exception:
        return np.inf

def _get_gauss_laguerre_nodes(m, alpha):
    """
    Get m Gauss-Laguerre quadrature nodes for weight x^alpha * e^(-x).
    These are the roots of the generalized Laguerre polynomial L_m^(alpha).
    """
    try:
        L = genlaguerre(m, alpha)
        roots = L.roots()
        roots = np.real(roots)
        roots = roots[roots > 0]
        roots = np.sort(roots)
        if len(roots) < m:
            # Pad if necessary
            while len(roots) < m:
                roots = np.append(roots, roots[-1] + 1.0)
        return roots[:m]
    except Exception:
        return np.array([4.0 * (i + 1) for i in range(m)], dtype=float)

def _objective_alpha(alpha, m=12, n=1):
    """
    Objective function: compute C4 bound using Gauss-Laguerre nodes
    with tunable weight parameter alpha.
    """
    try:
        roots = _get_gauss_laguerre_nodes(m, alpha)
        if np.any(roots <= 0) or np.any(~np.isfinite(roots)):
            return 1e10
        return _compute_c4_bound(roots, m, n)
    except Exception:
        return 1e10

def _refine_roots(roots, m=12, n=1):
    """
    Refine root positions using local optimization starting from
    Gauss-Laguerre node initializations.
    """
    try:
        bounds = [(1e-6, None)] * m
        
        def obj(roots_flat):
            r = np.asarray(roots_flat)
            if np.any(r <= 0) or np.any(~np.isfinite(r)):
                return 1e10
            return _compute_c4_bound(r, m, n)
        
        best_roots = roots.copy()
        best_score = obj(best_roots)
        
        # Try L-BFGS-B for refinement
        result = minimize(
            obj,
            best_roots,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.fun < best_score:
            best_roots = result.x
            best_score = result.fun
        
        # Try Nelder-Mead for further refinement
        result = minimize(
            obj,
            best_roots,
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        if result.fun < best_score:
            best_roots = result.x
            best_score = result.fun
        
        best_roots = np.maximum(best_roots, 1e-6)
        best_roots = np.sort(best_roots)
        
        return best_roots, best_score
    except Exception:
        return roots, _compute_c4_bound(roots, m, n)

def construct():
    """
    Construct the 12 positive double-root locations for the Cohn-Goncalves
    Laguerre polynomial construction with m=12, n=1.
    
    Uses a two-stage approach:
    1. Optimize the weight parameter alpha of Gauss-Laguerre quadrature (1D search)
    2. Refine individual root positions using local optimization
    """
    m = 12
    n = 1
    
    # Stage 1: Optimize alpha for Gauss-Laguerre weight
    # alpha controls the spread of nodes; typical range is -1 to 20
    alpha_bounds = (-1.0, 25.0)
    
    best_alpha = 1.0
    best_score_alpha = _objective_alpha(1.0, m, n)
    
    # Coarse grid search first
    alpha_grid = np.linspace(-1.0, 25.0, 50)
    for a in alpha_grid:
        if _deadline_reached():
            break
        score = _objective_alpha(a, m, n)
        if score < best_score_alpha:
            best_score_alpha = score
            best_alpha = a
    
    # Fine-grained 1D optimization around best alpha
    if not _deadline_reached():
        try:
            result = minimize_scalar(
                _objective_alpha,
                args=(m, n),
                bounds=(max(alpha_bounds[0], best_alpha - 5.0), min(alpha_bounds[1], best_alpha + 5.0)),
                method='bounded',
                options={'xatol': 1e-6, 'maxiter': 100}
            )
            if result.fun < best_score_alpha:
                best_score_alpha = result.fun
                best_alpha = result.x
        except Exception:
            pass
    
    # Get roots from best alpha
    best_roots = _get_gauss_laguerre_nodes(m, best_alpha)
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    # Stage 2: Refine individual root positions
    if not _deadline_reached():
        refined_roots, refined_score = _refine_roots(best_roots, m, n)
        if refined_score < best_score_alpha:
            best_roots = refined_roots
            best_score_alpha = refined_score
    
    # Ensure roots are positive and sorted
    best_roots = np.maximum(best_roots, 1e-6)
    best_roots = np.sort(best_roots)
    
    return [float(x) for x in best_roots]

def _fallback_construct(m, n):
    """Fallback: use Laguerre-Gauss quadrature nodes as proxy."""
    try:
        x, w = laggauss(m)
        return np.sort(x)
    except Exception:
        return np.array([4.0 * (i + 1) for i in range(m)], dtype=float)

# EVOLVE-BLOCK-END
